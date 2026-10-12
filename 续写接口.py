# -*- coding: utf-8 -*-
r"""
============================================================
  续写接口.py —— 续写 API：你给"方块 + 掩码"，模型只补掩码部分
============================================================

【核心思想】
一个窗口（32×32×32）里的每一格，只有两种身份：
    已知   = 这格是什么方块你说了算，模型只"看"，绝对不改
    要生成 = 掩码，模型根据周围已知的方块把它补出来
哪些格子是掩码完全由调用者决定：上半部分、一个盒子、零散的几格……都可以。

【Python 里直接调用】
    import torch, 续写接口
    M = 续写接口.载入模型("cuda")
    方块   = 续写接口.名字转编号(M, 名字网格)          # 或者自己准备 LongTensor[X,Y,Z]
    要生成 = torch.zeros(32, 32, 32, dtype=torch.bool)
    要生成[:, 16:, :] = True                            # 上半部分是掩码（第 2 维是高度 Y）
    结果, 说明 = 续写接口.续写(M, 方块, 要生成)
    # 结果: LongTensor[X,Y,Z] 词表编号；已知格子和输入完全一样
    # M["词表"][编号] = 方块名字

【游戏 ↔ Python 的请求文件格式】（输出\请求\xxx.json，由 Java 写、4_无限续写.py 读）
    {
      "size":    [32, 32, 32],
      "label":   "房子",                      ← 可选
      "palette": ["minecraft:air", ...],      ← 调色板
      "blocks":  [0, 3, 1, ...],              ← 每格在调色板里的编号，顺序 (x*Y + y)*Z + z
      "known":   "1101..."                    ← 每格一个字符：1 = 已知，0 = 要生成（掩码）
    }
    也可以不写 known 而写 "mask"（意思相反：1 = 要生成，0 = 已知）。
结果文件（输出\结果\同名.json）和训练样本格式一样，是补全后的完整窗口。
"""
import torch

import 配置
import 数据集
import 存档
from 扩散 import 采样


def 载入模型(设备="cuda"):
    """读取 存档/最新.pt，返回一个字典 M；没有存档返回 None。"""
    档, _, EMA模型 = 存档.读取(存档.最新存档, 设备)
    if 档 is None:
        return None
    EMA模型.eval()
    return {
        "模型": EMA模型,
        "词表": 档["词表"],
        "查表": {b: i for i, b in enumerate(档["词表"])},
        "标签表": 档["标签表"],
        "尺寸": tuple(档["结构尺寸"]),
        "轮数": 档["轮数"],
        "时间": 存档.最新存档.stat().st_mtime,
        "设备": 设备,
    }


def 名字转编号(M, 名字网格):
    """
    名字网格: 三层嵌套列表 [X][Y][Z]，每格是方块名字，如 "minecraft:stone"
    返回 LongTensor[X,Y,Z]。模型没见过的方块 = -1（续写时会被当成掩码）。
    """
    return torch.tensor([[[M["查表"].get(数据集.规范化方块(s), -1) for s in 列] for 列 in 面] for 面 in 名字网格],
                        dtype=torch.long)


def 续写(M, 方块, 要生成, 标签=None, 步数=None, 温度=None, 引导强度=None, 揭示随机性=None):
    """
    M:      载入模型() 的返回值
    方块:   LongTensor[X,Y,Z] 词表编号。要生成的格子里填什么都行（会被忽略）；-1 = 模型没见过的方块
    要生成: BoolTensor[X,Y,Z]  True = 掩码（让模型生成），False = 已知（原样保留）
    标签:   标签名字（如 "房子"），None = 不指定
    其余参数不填就用 配置.py 里的生成设置
    返回:   (结果 LongTensor[X,Y,Z], 说明文字)
    """
    尺寸 = M["尺寸"]
    模型 = M["模型"]
    if tuple(方块.shape) != 尺寸 or tuple(要生成.shape) != 尺寸:
        raise ValueError(f"方块/掩码的尺寸 {tuple(方块.shape)} / {tuple(要生成.shape)} 必须等于模型尺寸 {尺寸}")
    N = 尺寸[0] * 尺寸[1] * 尺寸[2]
    要生成 = 要生成.bool()
    没见过 = (~要生成) & (方块 < 0)             # 已知但模型不认识的方块 → 只能也当成掩码
    遮 = 要生成 | 没见过
    初始 = torch.where(遮, torch.full_like(方块, 模型.遮罩编号), 方块)[None]

    标签编号 = M["标签表"].index(标签) if 标签 in M["标签表"] else None
    if 标签编号 is None and 配置.默认生成标签 in M["标签表"]:
        标签编号 = M["标签表"].index(配置.默认生成标签)

    结果 = 采样(模型, 1, 尺寸, 标签编号,
              步数 or 配置.生成步数,
              配置.生成温度 if 温度 is None else 温度,
              配置.引导强度 if 引导强度 is None else 引导强度,
              配置.揭示随机性 if 揭示随机性 is None else 揭示随机性,
              M["设备"], 配置.使用混合精度, 打印进度=False, 初始=初始)[0]
    说明 = f"掩码 {int(要生成.sum())} 格（占窗口 {int(要生成.sum()) / N * 100:.0f}%）"
    if int(没见过.sum()):
        说明 += f"，另有 {int(没见过.sum())} 格是模型没见过的方块（当成掩码处理）"
    return 结果, 说明


def 处理请求(请求: dict, M: dict):
    """处理一个请求文件的内容（格式见本文件顶部）。返回 (结果网格, 标签名, 说明文字)"""
    尺寸 = tuple(int(v) for v in 请求["size"])
    if 尺寸 != M["尺寸"]:
        raise ValueError(f"游戏里的尺寸 {尺寸} 和模型的尺寸 {M['尺寸']} 不一致，请检查 Java 的 SIZE_X/Y/Z")
    N = 尺寸[0] * 尺寸[1] * 尺寸[2]
    调色板编号 = torch.tensor([M["查表"].get(数据集.规范化方块(s), -1) for s in 请求["palette"]], dtype=torch.long)
    方块 = 调色板编号[torch.tensor(请求["blocks"], dtype=torch.long)]
    if "mask" in 请求:
        要生成 = torch.tensor([c == "1" for c in 请求["mask"]], dtype=torch.bool)
    else:
        要生成 = torch.tensor([c != "1" for c in 请求["known"]], dtype=torch.bool)
    if len(方块) != N or len(要生成) != N:
        raise ValueError("请求文件里的方块数量 / 掩码长度不对")
    标签 = 请求.get("label")
    结果, 说明 = 续写(M, 方块.reshape(尺寸), 要生成.reshape(尺寸), 标签)
    return 结果, (标签 if 标签 in M["标签表"] else "默认"), 说明
