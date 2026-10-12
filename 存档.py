# -*- coding: utf-8 -*-
"""
============================================================
  存档.py —— 保存 / 读取模型存档
============================================================
存档文件：存档/最新.pt  （每次保存都会覆盖）
          存档/第XXXX轮.pt（每 每几轮保存×10 轮额外留一份备份）
存档里有：
  模型参数、EMA参数（生成用这个）、优化器状态、训练到第几轮、
  词表（方块编号对照）、标签表、结构尺寸、模型结构参数
所以 3_生成.py 只需要存档文件，不需要训练数据。
"""
import copy

import torch

import 配置
from 模型 import 结构UNet

最新存档 = 配置.存档目录 / "最新.pt"


def 保存(路径, 模型, EMA模型, 优化器, 轮数, 步数, 词表, 标签表):
    路径.parent.mkdir(parents=True, exist_ok=True)
    临时 = 路径.with_suffix(".tmp")
    torch.save({
        "模型": 模型.state_dict(),
        "EMA": EMA模型.state_dict(),
        "优化器": 优化器.state_dict() if 优化器 is not None else None,
        "轮数": 轮数,
        "步数": 步数,
        "词表": 词表,
        "标签表": 标签表,
        "结构尺寸": tuple(配置.结构尺寸),
        "保留方块状态": 配置.保留方块状态,
        "结构参数": 模型.结构参数,
    }, 临时)
    临时.replace(路径)          # 先写临时文件再改名，防止保存一半被中断导致存档损坏


def 读取(路径=最新存档, 设备="cpu"):
    """返回 (存档字典, 模型, EMA模型)；没有存档返回 (None, None, None)。"""
    if not 路径.exists():
        return None, None, None
    档 = torch.load(路径, map_location=设备, weights_only=False)
    模型 = 结构UNet(**档["结构参数"]).to(设备)
    模型.load_state_dict(档["模型"])
    EMA模型 = copy.deepcopy(模型)
    EMA模型.load_state_dict(档["EMA"])
    模型.水平坐标 = EMA模型.水平坐标 = bool(getattr(配置, "使用水平坐标", True))
    return 档, 模型, EMA模型
