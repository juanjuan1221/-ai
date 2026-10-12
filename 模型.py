# -*- coding: utf-8 -*-
"""
============================================================
  模型.py —— 3D U-Net 去噪网络
============================================================

【它干什么】
输入：一个"部分方块被遮住"的 3D 结构（遮住的格子编号 = 遮罩），
      + 当前遮住的比例 t（0~1），+ 标签（比如"房子"）。
输出：每一格"原本应该是什么方块"的打分（logits），形状 [批, 词表大小, X, Y, Z]。

【结构示意】（以 32³、通道倍数 (1,2,4) 为例）

  方块编号 ─► 嵌入(每种方块变成32个数) + 坐标(x,y,z 3个数)
      │
  32³ ×48通道  ── 残差块×2 ───────────────────────────┐ 跳连
      │ 缩小                                          │
  16³ ×96通道  ── 残差块×2 ────────────────────┐      │
      │ 缩小                                   │      │
   8³ ×192通道 ── 残差块×2 + 自注意力 ──┐      │      │
      │                                 │      │      │
   中间层：残差块 + 自注意力 + 残差块   │      │      │
      │                                 ▼      │      │
   8³  ── 残差块×3 + 自注意力 ◄─────────┘      │      │
      │ 放大                                   ▼      │
  16³ ── 残差块×3 ◄────────────────────────────┘      │
      │ 放大                                          ▼
  32³ ── 残差块×3 ◄───────────────────────────────────┘
      │
  输出卷积 ─► 每格对每种方块的打分

- 残差块：两层 3×3×3 卷积，看周围 27 格的邻居关系（墙挨着墙、屋顶在上面…）
- 缩小/放大：让模型能看到更大范围（整体形状）
- 自注意力：最小那层每个格子都能"看到"所有格子，负责整体布局
- 时间 t 和标签 通过"缩放+偏移"注入到每个残差块里
- 坐标通道：告诉模型"这格在底部还是顶部"，方便学会"地面在下、屋顶在上"
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def 时间正弦编码(t: torch.Tensor, 维度: int) -> torch.Tensor:
    """把 0~1 的 t 变成一串正弦/余弦数字，网络更容易理解"现在去噪到哪一步了"。"""
    半 = 维度 // 2
    频率 = torch.exp(-math.log(10000) * torch.arange(半, device=t.device, dtype=torch.float32) / 半)
    角 = (t.float() * 1000)[:, None] * 频率[None]
    return torch.cat([角.sin(), 角.cos()], dim=1)


class 残差块(nn.Module):
    def __init__(self, 输入通道, 输出通道, 条件维度, 丢弃率):
        super().__init__()
        self.归一1 = nn.GroupNorm(8, 输入通道)
        self.卷积1 = nn.Conv3d(输入通道, 输出通道, 3, padding=1)
        self.条件投影 = nn.Linear(条件维度, 输出通道 * 2)      # 产生 缩放 和 偏移
        self.归一2 = nn.GroupNorm(8, 输出通道)
        self.丢弃 = nn.Dropout(丢弃率)
        self.卷积2 = nn.Conv3d(输出通道, 输出通道, 3, padding=1)
        nn.init.zeros_(self.卷积2.weight)                       # 初始时残差块≈直通，训练更稳
        nn.init.zeros_(self.卷积2.bias)
        self.捷径 = nn.Conv3d(输入通道, 输出通道, 1) if 输入通道 != 输出通道 else nn.Identity()

    def forward(self, x, 条件):
        h = self.卷积1(F.silu(self.归一1(x)))
        缩放, 偏移 = self.条件投影(F.silu(条件))[:, :, None, None, None].chunk(2, dim=1)
        h = self.归一2(h) * (1 + 缩放) + 偏移
        h = self.卷积2(self.丢弃(F.silu(h)))
        return h + self.捷径(x)


class 自注意力(nn.Module):
    def __init__(self, 通道, 头数=4):
        super().__init__()
        self.头数 = 头数
        self.归一 = nn.GroupNorm(8, 通道)
        self.qkv = nn.Conv3d(通道, 通道 * 3, 1)
        self.输出 = nn.Conv3d(通道, 通道, 1)
        nn.init.zeros_(self.输出.weight)
        nn.init.zeros_(self.输出.bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        q, k, v = self.qkv(self.归一(x)).reshape(B, 3, self.头数, C // self.头数, X * Y * Z).unbind(1)
        # [B, 头, 格子数, 每头通道]
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2).reshape(B, C, X, Y, Z)
        return x + self.输出(h)


class 结构UNet(nn.Module):
    def __init__(self, 词表大小, 标签数, 基础通道=48, 通道倍数=(1, 2, 4), 每层残差块数=2,
                 注意力层=(2,), 方块嵌入维度=32, 丢弃率=0.1):
        super().__init__()
        # 记下构造参数，存档时一起保存，生成时按同样结构重建
        self.结构参数 = dict(词表大小=词表大小, 标签数=标签数, 基础通道=基础通道, 通道倍数=tuple(通道倍数),
                         每层残差块数=每层残差块数, 注意力层=tuple(注意力层),
                         方块嵌入维度=方块嵌入维度, 丢弃率=丢弃率)
        self.词表大小 = 词表大小
        self.遮罩编号 = 词表大小                 # 最后一个编号留给"遮罩"
        self.标签数 = 标签数
        self.空标签编号 = 标签数                 # 最后一个编号表示"不指定标签"

        条件维度 = 基础通道 * 4
        self.方块嵌入 = nn.Embedding(词表大小 + 1, 方块嵌入维度)
        self.标签嵌入 = nn.Embedding(标签数 + 1, 条件维度)
        self.时间维度 = 基础通道
        self.时间网络 = nn.Sequential(nn.Linear(基础通道, 条件维度), nn.SiLU(), nn.Linear(条件维度, 条件维度))

        self.输入卷积 = nn.Conv3d(方块嵌入维度 + 3, 基础通道, 3, padding=1)

        # ---------- 下采样路径 ----------
        self.下层 = nn.ModuleList()
        通道记录 = [基础通道]
        当前 = 基础通道
        for 层, 倍 in enumerate(通道倍数):
            for _ in range(每层残差块数):
                块 = nn.ModuleList([残差块(当前, 基础通道 * 倍, 条件维度, 丢弃率)])
                当前 = 基础通道 * 倍
                if 层 in 注意力层:
                    块.append(自注意力(当前))
                self.下层.append(块)
                通道记录.append(当前)
            if 层 != len(通道倍数) - 1:
                self.下层.append(nn.ModuleList([nn.Conv3d(当前, 当前, 3, stride=2, padding=1)]))
                通道记录.append(当前)

        # ---------- 中间 ----------
        self.中间1 = 残差块(当前, 当前, 条件维度, 丢弃率)
        self.中间注意力 = 自注意力(当前)
        self.中间2 = 残差块(当前, 当前, 条件维度, 丢弃率)

        # ---------- 上采样路径 ----------
        self.上层 = nn.ModuleList()
        for 层, 倍 in reversed(list(enumerate(通道倍数))):
            for i in range(每层残差块数 + 1):
                块 = nn.ModuleList([残差块(当前 + 通道记录.pop(), 基础通道 * 倍, 条件维度, 丢弃率)])
                当前 = 基础通道 * 倍
                if 层 in 注意力层:
                    块.append(自注意力(当前))
                if 层 != 0 and i == 每层残差块数:
                    块.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    块.append(nn.Conv3d(当前, 当前, 3, padding=1))
                self.上层.append(块)

        self.输出归一 = nn.GroupNorm(8, 当前)
        self.输出卷积 = nn.Conv3d(当前, 词表大小, 3, padding=1)
        self._坐标缓存 = {}
        # False = 不告诉模型水平位置（X/Z 坐标通道填 0），由 配置.使用水平坐标 控制
        self.水平坐标 = True

    def _坐标(self, 形状, 设备, 类型):
        键 = (形状, 设备, 类型, self.水平坐标)
        if 键 not in self._坐标缓存:
            X, Y, Z = 形状
            轴 = [torch.linspace(-1, 1, n, device=设备, dtype=类型) for n in (X, Y, Z)]
            坐标 = torch.stack(torch.meshgrid(*轴, indexing="ij"), dim=0)[None]
            if not self.水平坐标:
                坐标[:, 0] = 0      # X
                坐标[:, 2] = 0      # Z（高度 Y 保留，模型需要知道哪里是地面）
            self._坐标缓存[键] = 坐标
        return self._坐标缓存[键]

    def forward(self, 方块编号, t, 标签):
        """
        方块编号: LongTensor [B, X, Y, Z]，被遮住的格子 = self.遮罩编号
        t:        FloatTensor [B]，遮住比例 0~1
        标签:     LongTensor [B]，不指定 = self.空标签编号
        返回:     logits [B, 词表大小, X, Y, Z]
        """
        B, X, Y, Z = 方块编号.shape
        条件 = self.时间网络(时间正弦编码(t, self.时间维度)) + self.标签嵌入(标签)

        h = self.方块嵌入(方块编号).permute(0, 4, 1, 2, 3)            # [B, 嵌入, X, Y, Z]
        h = torch.cat([h, self._坐标((X, Y, Z), h.device, h.dtype).expand(B, -1, -1, -1, -1)], dim=1)
        h = self.输入卷积(h)

        跳连 = [h]
        for 块 in self.下层:
            for 模块 in 块:
                h = 模块(h, 条件) if isinstance(模块, 残差块) else 模块(h)
            跳连.append(h)

        h = self.中间1(h, 条件)
        h = self.中间注意力(h)
        h = self.中间2(h, 条件)

        for 块 in self.上层:
            h = torch.cat([h, 跳连.pop()], dim=1)
            for 模块 in 块:
                h = 模块(h, 条件) if isinstance(模块, 残差块) else 模块(h)

        return self.输出卷积(F.silu(self.输出归一(h)))

    # ------------------------------------------------------------
    # 接着训练时如果采集了新方块 / 新标签，需要把词表变大
    # 旧方块、旧标签的参数原样保留，只给新的随机初始化
    # ------------------------------------------------------------
    @torch.no_grad()
    def 扩展(self, 新词表大小, 新标签数):
        if 新词表大小 > self.词表大小:
            旧V = self.词表大小
            新嵌入 = nn.Embedding(新词表大小 + 1, self.方块嵌入.embedding_dim).to(self.方块嵌入.weight.device)
            新嵌入.weight[:旧V] = self.方块嵌入.weight[:旧V]
            新嵌入.weight[新词表大小] = self.方块嵌入.weight[旧V]      # 遮罩挪到新的最后一位
            self.方块嵌入 = 新嵌入
            旧卷积 = self.输出卷积
            新卷积 = nn.Conv3d(旧卷积.in_channels, 新词表大小, 3, padding=1).to(旧卷积.weight.device)
            新卷积.weight[:旧V] = 旧卷积.weight
            新卷积.bias[:旧V] = 旧卷积.bias
            新卷积.bias[旧V:] = 旧卷积.bias.min()                     # 新方块初始概率低一些
            self.输出卷积 = 新卷积
            self.词表大小 = self.遮罩编号 = 新词表大小
            self.结构参数["词表大小"] = 新词表大小
        if 新标签数 > self.标签数:
            旧L = self.标签数
            新 = nn.Embedding(新标签数 + 1, self.标签嵌入.embedding_dim).to(self.标签嵌入.weight.device)
            新.weight[:旧L] = self.标签嵌入.weight[:旧L]
            新.weight[新标签数] = self.标签嵌入.weight[旧L]
            self.标签嵌入 = 新
            self.标签数 = self.空标签编号 = 新标签数
            self.结构参数["标签数"] = 新标签数
