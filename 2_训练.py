# -*- coding: utf-8 -*-
r"""
============================================================
  2_训练.py —— 训练扩散模型（可随时中断，下次自动接着训练）
============================================================
运行：PyCharm 里右键 → 运行。解释器请选 MC_AI\.mc_gpu_py314_cu128（带 CUDA 的那个）
中断：在运行窗口按 停止 或 Ctrl+C，会先保存再退出。
接着训练：直接再运行一次。会读取 存档/最新.pt 继续。
采集了新数据：直接再运行，新样本 / 新方块 / 新标签会自动加进来。
想从头训练：把 存档 文件夹里的 最新.pt 删除或改名。

【屏幕上的数字怎么看】
  损失 loss：越低越好。刚开始约 3~6，降到 0.5 以下说明学得不错，
             如果降到接近 0.05 且一直往下，可能是在"背答案"（数据太少）。
  猜对率：   被遮住的格子猜对的比例。空气多的数据一开始就会很高，主要看趋势。
训练日志同时写到 存档/训练日志.csv，可以用 Excel 打开画曲线。
"""
import 自动切换GPU
自动切换GPU.确保GPU()

import copy
import math
import random
import time

import torch
from torch.utils.data import DataLoader

import 配置
import 数据集
import 存档
from 模型 import 结构UNet
from 扩散 import 计算损失, 采样


def main():
    配置.检查配置()
    random.seed(配置.随机种子)
    torch.manual_seed(配置.随机种子)
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    if 设备 == "cpu":
        print("⚠ 没检测到 CUDA 显卡，将用 CPU 训练（非常慢）。请确认 PyCharm 解释器选的是 .mc_gpu_py314_cu128")
    else:
        print(f"使用显卡：{torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    混合精度 = 配置.使用混合精度 and 设备 == "cuda"

    # ---------- 1. 读数据 ----------
    样本们 = 数据集.读取全部样本()
    if not 样本们:
        print("没有样本，先去游戏里采集。")
        return

    # ---------- 2. 读旧存档（如果有） ----------
    档, 模型, EMA模型 = 存档.读取(存档.最新存档, 设备)
    扩展过 = False
    if 档 is not None:
        if tuple(档["结构尺寸"]) != tuple(配置.结构尺寸):
            print(f"✗ 存档的结构尺寸 {档['结构尺寸']} 和 配置 {配置.结构尺寸} 不一致。要么改回配置，要么移走 最新.pt 重新训练。")
            return
        if 档.get("保留方块状态") != 配置.保留方块状态:
            print("✗ 存档的 保留方块状态 和 配置 不一致。要么改回配置，要么移走 最新.pt 重新训练。")
            return
        起始轮 = 档["轮数"] + 1
        步数 = 档["步数"]
        词表 = 数据集.构建词表(样本们, 档["词表"])
        标签表 = 数据集.构建标签表(样本们, 档["标签表"])
        print(f"读取存档：已训练 {档['轮数']} 轮 / {步数} 步")
        if len(词表) > len(档["词表"]) or len(标签表) > len(档["标签表"]):
            print(f"  发现新方块 {len(词表) - len(档['词表'])} 种、新标签 {len(标签表) - len(档['标签表'])} 个，扩展模型")
            模型.扩展(len(词表), len(标签表))
            EMA模型.扩展(len(词表), len(标签表))
            扩展过 = True
    else:
        起始轮, 步数 = 1, 0
        词表 = 数据集.构建词表(样本们)
        标签表 = 数据集.构建标签表(样本们)
        模型 = 结构UNet(len(词表), len(标签表), 配置.基础通道, 配置.通道倍数, 配置.每层残差块数,
                     配置.注意力层, 配置.方块嵌入维度, 配置.丢弃率).to(设备)
        模型.水平坐标 = bool(getattr(配置, "使用水平坐标", True))
        EMA模型 = copy.deepcopy(模型)
        print("没有存档，从头开始训练")
    EMA模型.eval().requires_grad_(False)

    参数量 = sum(p.numel() for p in 模型.parameters()) / 1e6
    print(f"词表大小 {len(词表)}，标签 {标签表}，模型参数 {参数量:.1f}M")

    # ---------- 3. 数据加载器 ----------
    数据 = 数据集.结构数据集(样本们, 词表, 标签表)
    加载器 = DataLoader(数据, batch_size=配置.批大小, shuffle=True, drop_last=len(数据) >= 配置.批大小,
                     num_workers=0, pin_memory=(设备 == "cuda"))
    print(f"训练项数（含旋转）{len(数据)}，每轮 {len(加载器)} 步")
    print(f"续写训练概率 {getattr(配置, '续写训练概率', 0.0)}，使用水平坐标 {模型.水平坐标}")

    # ---------- 4. 优化器 ----------
    优化器 = torch.optim.AdamW(模型.parameters(), lr=配置.学习率, weight_decay=0.01)
    if 档 is not None and 档.get("优化器"):
        if 扩展过:
            print("  （模型扩展过，优化器状态重置，属正常现象）")
        else:
            try:
                优化器.load_state_dict(档["优化器"])
            except Exception:
                print("  （优化器状态读取失败，已重置，不影响模型）")

    类别权重 = None
    if 配置.空气损失权重 != 1.0:
        类别权重 = torch.ones(len(词表), device=设备)
        类别权重[0] = 配置.空气损失权重

    def 当前学习率(步):
        return 配置.学习率 * min(1.0, (步 + 1) / 配置.预热步数)

    def 存(轮):
        存档.保存(存档.最新存档, 模型, EMA模型, 优化器, 轮, 步数, 词表, 标签表)
        if 轮 % (配置.每几轮保存 * 10) == 0:
            存档.保存(配置.存档目录 / f"第{轮:05d}轮.pt", 模型, EMA模型, None, 轮, 步数, 词表, 标签表)

    日志 = 配置.存档目录 / "训练日志.csv"
    if not 日志.exists():
        日志.write_text("轮数,步数,损失,猜对率,学习率,用时秒\n", encoding="utf-8-sig")

    # ---------- 5. 训练循环 ----------
    print("=" * 60)
    print(f"开始训练：第 {起始轮} 轮 → 第 {配置.训练到第几轮} 轮（Ctrl+C 可随时中断并保存）")
    print("=" * 60)
    轮 = 起始轮 - 1
    try:
        for 轮 in range(起始轮, 配置.训练到第几轮 + 1):
            模型.train()
            开始 = time.time()
            总损失 = 总对 = 计数 = 0.0
            for x0, 标签 in 加载器:
                x0 = x0.to(设备, non_blocking=True)
                标签 = 标签.to(设备, non_blocking=True)
                for g in 优化器.param_groups:
                    g["lr"] = 当前学习率(步数)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=混合精度):
                    损失, 猜对率 = 计算损失(模型, x0, 标签, 配置.标签丢弃概率, 类别权重,
                                         getattr(配置, "续写训练概率", 0.0))
                if not torch.isfinite(损失):
                    print("⚠ 损失变成 NaN/Inf，跳过这一步。如果经常出现，把 配置.使用混合精度 改成 False 或调小学习率")
                    优化器.zero_grad(set_to_none=True)
                    continue
                优化器.zero_grad(set_to_none=True)
                损失.backward()
                torch.nn.utils.clip_grad_norm_(模型.parameters(), 1.0)
                优化器.step()
                步数 += 1

                # EMA：生成用的"平滑版"权重
                衰减 = min(配置.EMA衰减, (1 + 步数) / (10 + 步数))
                with torch.no_grad():
                    for pe, p in zip(EMA模型.parameters(), 模型.parameters()):
                        pe.lerp_(p, 1 - 衰减)

                总损失 += 损失.item(); 总对 += 猜对率.item(); 计数 += 1

            用时 = time.time() - 开始
            平均损失 = 总损失 / max(计数, 1)
            平均对 = 总对 / max(计数, 1)
            剩余 = (配置.训练到第几轮 - 轮) * 用时
            print(f"第 {轮:>5} 轮 | 损失 {平均损失:.4f} | 猜对率 {平均对 * 100:5.1f}% | "
                  f"学习率 {当前学习率(步数):.1e} | {用时:.1f}秒/轮 | 预计还要 {剩余 / 60:.0f} 分钟")
            with open(日志, "a", encoding="utf-8") as f:
                f.write(f"{轮},{步数},{平均损失:.5f},{平均对:.5f},{当前学习率(步数):.2e},{用时:.2f}\n")

            if 轮 % 配置.每几轮保存 == 0:
                存(轮)
                print(f"  ✔ 已保存存档（第 {轮} 轮）")
            if 配置.每几轮预览 and 轮 % 配置.每几轮预览 == 0:
                print("  生成预览结构…")
                EMA模型.eval()
                结果 = 采样(EMA模型, 1, 配置.结构尺寸, None, 配置.生成步数, 配置.生成温度, 1.0,
                          配置.揭示随机性, 设备, 混合精度, 打印进度=False)
                路径 = 配置.待放置目录 / f"预览_第{轮:05d}轮.json"
                n = 数据集.保存结构(结果[0], 词表, 路径, "预览")
                print(f"  ✔ 预览已写入 {路径.name}（非空气方块 {n} 个），进游戏用 AI生成器 挥动即可放置")
    except KeyboardInterrupt:
        print("\n收到中断，正在保存…")
    except torch.cuda.OutOfMemoryError:
        print("\n✗ 显存不够！请把 配置.py 里的 批大小 改小（例如 2 或 1），或把 基础通道 改小。正在保存…")
    存(轮)
    print(f"✔ 存档已保存到 {存档.最新存档}（第 {轮} 轮）。下次运行会接着训练。")


if __name__ == "__main__":
    main()
