# -*- coding: utf-8 -*-
r"""
============================================================
  4_无限续写.py —— 续写服务：开着它，游戏里就能按"掩码"补全方块
============================================================
用法：
  1. 运行本文件，看到"等待游戏请求…"后【不要关】，让它一直开着（只开一个！）
  2. 进游戏：
       · 把想让 AI 生成的地方用【海绵】填上（海绵 = 掩码），其它方块 = 已知，AI 不会动
           例：/fill ~-8 ~ ~-8 ~8 ~12 ~8 sponge
       · 站在窗口中心，拿 AI生成器 左键挥动 → AI 只把海绵的位置补出来
           （窗口范围和采样器一样：拿采样器 Shift+挥动 可以看到粒子框）
       · Shift + 挥动 = 撤销上一次
  3. 不用了就点停止（或 Ctrl+C）。关掉以后 AI生成器 会变回"放置 待放置 文件夹里的文件"

【它和游戏怎么配合】
  游戏   → 把窗口里的方块 + 掩码写到 输出\请求\
  本程序 → 已知方块原样保留，掩码部分用扩散模型补全，写到 输出\结果\
  游戏   → 读结果，只放置掩码的那些格子
请求文件格式、以及在 Python 里直接调用的方法，见 续写接口.py 顶部。

训练时也可以开着本服务，但两个程序会抢显卡，都会变慢。
存档 最新.pt 更新后，本服务下一次请求会自动换成新模型。
"""
import 自动切换GPU
自动切换GPU.确保GPU()

import json
import time

import torch

import 配置
import 数据集
import 存档
import 续写接口
from 续写接口 import 载入模型


def 处理请求(请求: dict, M: dict, 设备=None):
    return 续写接口.处理请求(请求, M)


def main():
    配置.检查配置()
    for d in (配置.请求目录, 配置.结果目录):
        d.mkdir(parents=True, exist_ok=True)
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    M = 载入模型(设备)
    if M is None:
        print(f"没有找到存档 {存档.最新存档}，请先运行 2_训练.py")
        return
    print(f"模型：已训练 {M['轮数']} 轮，尺寸 {M['尺寸']}，词表 {len(M['词表'])}，标签 {M['标签表']}，设备 {设备}")

    # 清掉上次遗留的请求（游戏那边早就等超时了）
    for f in list(配置.请求目录.glob("*")) + list(配置.结果目录.glob("*")):
        f.unlink(missing_ok=True)

    print("=" * 60)
    print("续写服务已启动，等待游戏请求…（海绵 = 掩码，拿 AI生成器 挥动；点停止或 Ctrl+C 结束）")
    print("=" * 60)
    计数 = 0
    try:
        while True:
            配置.服务心跳文件.write_text(str(time.time()), encoding="utf-8")   # 告诉游戏"我在线"
            请求们 = sorted(配置.请求目录.glob("*.json"))
            if not 请求们:
                time.sleep(0.2)
                continue
            路径 = 请求们[0]
            结果路径 = 配置.结果目录 / 路径.name
            临时 = 结果路径.with_suffix(".tmp")
            开始 = time.time()
            try:
                # 存档更新了就换新模型
                if 存档.最新存档.exists() and 存档.最新存档.stat().st_mtime != M["时间"]:
                    try:
                        新 = 载入模型(设备)
                        if 新 is not None:
                            M = 新
                            print(f"  （存档已更新，换成第 {M['轮数']} 轮的模型）")
                    except Exception as e:
                        print(f"  （新存档读取失败，继续用旧模型：{e}）")
                请求 = json.loads(路径.read_text(encoding="utf-8"))
                网格, 标签, 说明 = 处理请求(请求, M)
                数据集.保存结构(网格, M["词表"], 临时, 标签)
                计数 += 1
                print(f"第 {计数} 次 ✔ {说明}，用时 {time.time() - 开始:.1f} 秒")
            except torch.cuda.OutOfMemoryError:
                临时.write_text(json.dumps({"error": "显存不够，请关掉训练或其它占显卡的程序"}, ensure_ascii=False), encoding="utf-8")
                print("✗ 显存不够（是不是同时在训练，或者开了两个续写服务？）")
            except FileNotFoundError:
                continue                    # 请求被别的进程拿走了（开了两个服务）
            except Exception as e:
                临时.write_text(json.dumps({"error": str(e)}, ensure_ascii=False), encoding="utf-8")
                print(f"✗ 处理 {路径.name} 失败：{e}")
            临时.replace(结果路径)        # 先写临时文件再改名，游戏不会读到写了一半的文件
            路径.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\n续写服务已停止。")
    finally:
        配置.服务心跳文件.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
