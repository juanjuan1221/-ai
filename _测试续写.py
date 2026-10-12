# -*- coding: utf-8 -*-
# 临时测试：各种掩码形状的续写 + 一步续写训练。不影响存档。
import 自动切换GPU
自动切换GPU.确保GPU()
import time, torch
import 数据集, 续写接口
from 扩散 import 计算损失, 随机已知区域

设备 = "cuda"
M = 续写接口.载入模型(设备)
s = 数据集.读取全部样本(打印=False)[0]
X, Y, Z = s["网格"].shape
原 = torch.tensor([M["查表"].get(数据集.规范化方块(b), -1) for b in s["调色板"]])[s["网格"]]


def 试(名, 要生成):
    t = time.time()
    结果, 说明 = 续写接口.续写(M, 原, 要生成)
    结果 = 结果.cpu()
    保留 = bool((结果[~要生成] == 原[~要生成]).all())
    没遮罩残留 = bool((结果 < len(M["词表"])).all())
    print(f"{名}: {说明} | {time.time() - t:.1f}s | 已知部分原样保留={保留} | 全部补完={没遮罩残留} | "
          f"新部分非空气={int((结果[要生成] != 0).sum())} (原本 {int((原[要生成] != 0).sum())})")


上半 = torch.zeros(X, Y, Z, dtype=torch.bool); 上半[:, Y // 2:, :] = True
试("上半部分掩码", 上半)
盒子 = torch.zeros(X, Y, Z, dtype=torch.bool); 盒子[8:24, 4:20, 8:24] = True
试("中间一个盒子", 盒子)
零散 = torch.rand(X, Y, Z) < 0.05
试("零散 5%", 零散)

# 请求文件格式（游戏用的那条路）
请求 = {"size": [X, Y, Z], "palette": s["调色板"], "blocks": s["网格"].flatten().tolist(),
      "known": "".join("0" if k else "1" for k in 上半.flatten().tolist())}
网格, 标签, 说明 = 续写接口.处理请求(请求, M)
print("请求文件格式:", 说明, "| 已知保留 =", bool((网格.cpu()[~上半] == 原[~上半]).all()))

# 训练用的随机已知区域：7 种形状都能出来、形状正确
k = 随机已知区域(64, X, Y, Z, "cpu", 1.0)
print("训练掩码 已知比例 最小/平均/最大:", [round(float(v), 2) for v in (k.flatten(1).float().mean(1).min(), k.float().mean(), k.flatten(1).float().mean(1).max())])

模型 = M["模型"].train().requires_grad_(True)
x0 = 原.clamp(min=0)[None].to(设备).repeat(2, 1, 1, 1)
with torch.autocast("cuda", torch.bfloat16):
    l, a = 计算损失(模型, x0, torch.zeros(2, dtype=torch.long, device=设备), 0.1, None, 0.5)
l.backward(); print("训练一步 OK 损失", float(l))
