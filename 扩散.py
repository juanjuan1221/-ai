# -*- coding: utf-8 -*-
"""
============================================================
  扩散.py —— "遮罩式离散扩散模型"的加噪、损失、采样
============================================================

【用大白话讲扩散模型】
普通图片扩散模型：往图片上一点点加高斯噪声直到变成雪花，再训练网络一步步把噪声去掉。
但 Minecraft 方块是"种类"（石头/木板/空气），不是连续的颜色，不能"加一点点噪声"。
所以这里用的是"离散扩散"里最常用、最稳的一种：遮罩扩散（Absorbing / Masked Diffusion，
和 MDLM、MaskGIT 是同一家族）：

  前向（加噪）：随机把一部分方块换成"？"（遮罩）。t=0 完全没遮，t=1 全部是"？"
      原结构 ──t=0.3──► 30% 的格子变成 ？ ──t=1──► 全是 ？

  训练：给网络看"遮住了 t 比例的结构"，让它猜出被遮住的格子原来是什么方块。
        损失 = 被遮住格子上的交叉熵（猜错越离谱，损失越大）

  反向（生成）：从"全是？"开始，分 N 步：
        每一步网络给所有 ？ 格子猜一个方块，
        挑其中最有把握的一部分"定下来"，其余继续保持 ？，
        下一步再根据已经定下来的方块继续猜……直到没有 ？。
        早期定下的是大轮廓（哪里是空气、哪里有墙），后期补细节。

【续写（无限延伸）是怎么做到的】
生成时不从"全是？"开始，而是：窗口里已经存在的方块原样填进去，只有空白部分是"？"。
模型看着已有的那一半，把另一半补出来，所以新旧部分是连在一起的，没有接缝。
窗口每次往外挪半个身位，重复这个过程，就可以一直延伸下去。

为了让模型擅长这件事，训练时有一部分样本会做"续写练习"：
随机选一块区域（一半 / 一个角 / L 形）完整给出，永远不遮，只让它猜其余部分。
（配置.续写训练概率 控制这种练习占多少）

【无分类器引导 (CFG)】
训练时有 10% 概率把标签换成"不指定"，于是同一个网络既会"按标签生成"也会"随便生成"。
生成时： 最终打分 = 无标签打分 + 引导强度 × (有标签打分 − 无标签打分)
引导强度越大，越像那个标签，但多样性下降。
"""
import math
import random

import torch
import torch.nn.functional as F


def 随机已知区域(B, X, Y, Z, 设备, 概率):
    """
    续写练习用：给每个样本随机决定一块"已知区域"。
    返回 BoolTensor [B,X,Y,Z]，True = 这格直接给模型看，不遮。
    七种形状，对应游戏里用掩码续写时会遇到的情况：
        0: X 方向的一半（左边或右边已知）
        1: Z 方向的一半
        2: 一个角已知
        3: L 形已知（只剩一个角要补）
        4: 上下分（下半已知补上半，或上半已知补下半）
        5: 只有一个盒子已知，其余都要补
        6: 一个盒子要补，其余都已知（在已有结构中间挖一块重新生成）
    """
    已知 = torch.zeros(B, X, Y, Z, dtype=torch.bool)

    def 一段(n):
        分界 = random.randint(max(1, n // 8), n - max(1, n // 8))
        低 = torch.arange(n) < 分界
        return 低 if random.random() < 0.5 else ~低

    def 区间(n):
        """随机一段连续区间（可以在中间），长度 n/4 ~ 3n/4"""
        长 = random.randint(max(1, n // 4), max(1, n * 3 // 4))
        起 = random.randint(0, n - 长)
        i = torch.arange(n)
        return (i >= 起) & (i < 起 + 长)

    for b in range(B):
        if random.random() >= 概率:
            continue
        kx = 一段(X)[:, None, None]      # [X,1,1]
        ky = 一段(Y)[None, :, None]      # [1,Y,1]
        kz = 一段(Z)[None, None, :]      # [1,1,Z]
        形状 = random.randint(0, 6)
        if 形状 == 0:
            区域 = kx
        elif 形状 == 1:
            区域 = kz
        elif 形状 == 2:
            区域 = kx & kz
        elif 形状 == 3:
            区域 = kx | kz
        elif 形状 == 4:
            区域 = ky
        else:
            盒子 = 区间(X)[:, None, None] & 区间(Y)[None, :, None] & 区间(Z)[None, None, :]
            区域 = 盒子 if 形状 == 5 else ~盒子
        已知[b] = 区域.expand(X, Y, Z)
    return 已知.to(设备)


def 计算损失(模型, x0, 标签, 标签丢弃概率=0.1, 类别权重=None, 续写概率=0.0):
    """
    x0:   LongTensor [B,X,Y,Z] 真实结构（词表编号）
    标签: LongTensor [B]
    返回: (损失, 被遮住格子上的猜对率)
    """
    B, X, Y, Z = x0.shape
    设备 = x0.device
    # 1. 每个样本随机一个遮罩比例 t ∈ (0, 1]
    t = torch.rand(B, device=设备).clamp(min=1e-3)
    # 2. 每个格子以概率 t 被遮住
    遮罩 = torch.rand(x0.shape, device=设备) < t[:, None, None, None]
    # 2b. 续写练习：已知区域永远不遮
    if 续写概率 > 0:
        遮罩 = 遮罩 & ~随机已知区域(B, X, Y, Z, 设备, 续写概率)
    xt = x0.masked_fill(遮罩, 模型.遮罩编号)
    # 3. 随机丢弃标签（为了 CFG）
    丢 = torch.rand(B, device=设备) < 标签丢弃概率
    标签 = torch.where(丢, torch.full_like(标签, 模型.空标签编号), 标签)
    # 4. 用"实际遮住比例"作为 t 输入（和生成时保持一致）
    实际t = 遮罩.flatten(1).float().mean(1)
    logits = 模型(xt, 实际t, 标签)
    # 5. 只在被遮住的格子上算交叉熵
    交叉熵 = F.cross_entropy(logits.float(), x0, weight=类别权重, reduction="none")
    m = 遮罩.float()
    损失 = (交叉熵 * m).sum() / m.sum().clamp(min=1)
    with torch.no_grad():
        猜对率 = ((logits.argmax(1) == x0).float() * m).sum() / m.sum().clamp(min=1)
    return 损失, 猜对率


def _gumbel(形状, 设备):
    u = torch.rand(形状, device=设备).clamp(1e-10, 1 - 1e-10)
    return -torch.log(-torch.log(u))


@torch.no_grad()
def 采样(模型, 数量, 尺寸, 标签编号=None, 步数=48, 温度=1.0, 引导强度=2.0, 揭示随机性=1.0,
       设备="cuda", 混合精度=True, 打印进度=True, 初始=None):
    """
    逐步揭示，生成 数量 个结构。
    标签编号: None = 不指定标签
    初始:     None = 从"全部是遮罩"开始（凭空生成）
              LongTensor [数量,X,Y,Z] = 续写模式：已知格子填词表编号，要生成的格子填 模型.遮罩编号
    返回: LongTensor [数量, X, Y, Z]（词表编号）。续写模式下已知格子原样保留。
    """
    X, Y, Z = 尺寸
    N = X * Y * Z
    if 初始 is None:
        x = torch.full((数量, X, Y, Z), 模型.遮罩编号, dtype=torch.long, device=设备)
    else:
        x = 初始.to(设备).clone()
    初始剩余 = (x == 模型.遮罩编号).flatten(1).sum(1)      # 每个样本一开始有多少个 ？
    无标签 = torch.full((数量,), 模型.空标签编号, dtype=torch.long, device=设备)
    有标签 = 无标签 if 标签编号 is None else torch.full((数量,), 标签编号, dtype=torch.long, device=设备)
    用引导 = 标签编号 is not None and abs(引导强度 - 1.0) > 1e-6

    for 步 in range(步数):
        遮 = x == 模型.遮罩编号
        剩余 = 遮.flatten(1).sum(1)                         # 每个样本还剩多少个 ？
        if int(剩余.sum()) == 0:
            break
        t = 剩余.float() / N

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=混合精度 and 设备 == "cuda"):
            if 用引导:
                两份 = 模型(torch.cat([x, x]), torch.cat([t, t]), torch.cat([有标签, 无标签])).float()
                有, 无 = 两份.chunk(2)
                logits = 无 + 引导强度 * (有 - 无)
            else:
                logits = 模型(x, t, 有标签).float()

        对数概率 = F.log_softmax(logits / max(温度, 1e-4), dim=1)          # [B,V,X,Y,Z]
        # 每个格子按概率抽一个方块（Gumbel-Max 技巧 = 按概率随机抽样）
        抽到 = (对数概率 + _gumbel(对数概率.shape, 设备)).argmax(1)       # [B,X,Y,Z]
        把握 = 对数概率.gather(1, 抽到[:, None]).squeeze(1)              # 抽到的方块的 log 概率

        # 余弦进度表：这一步结束后还应该剩多少个 ？（前期揭示慢、后期揭示快）
        进度 = (步 + 1) / 步数
        余弦 = math.cos(math.pi / 2 * 进度) if 步 < 步数 - 1 else 0.0

        # 把握 + 随机扰动（越早期越随机），已揭示的格子不参与
        打分 = 把握 + 揭示随机性 * (1 - 进度) * _gumbel(把握.shape, 设备)
        打分 = 打分.masked_fill(~遮, -float("inf")).flatten(1)
        for b in range(数量):
            目标剩余 = int(math.floor(int(初始剩余[b]) * 余弦))
            揭示数 = int(剩余[b]) - 目标剩余
            if 揭示数 <= 0:
                continue
            选中 = 打分[b].topk(揭示数).indices
            x[b].view(-1)[选中] = 抽到[b].view(-1)[选中]

        if 打印进度 and (步 % max(1, 步数 // 8) == 0 or 步 == 步数 - 1):
            print(f"    去噪 {步 + 1:>3}/{步数}  剩余未确定格子 {int((x == 模型.遮罩编号).sum()) // 数量:>6}")
    return x
