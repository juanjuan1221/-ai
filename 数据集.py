# -*- coding: utf-8 -*-
"""
============================================================
  数据集.py —— 读取 Java 采样器导出的样本，变成模型能吃的数字
============================================================

【一个样本文件长什么样】（Java 采样器写出的 .json）
{
  "format":  "mofangzhe-v1",
  "label":   "房子",                 ← 标签（采样器在铁砧改名 = 标签；没改名 = "默认"）
  "size":    [32, 32, 32],           ← X, Y, Z 尺寸
  "origin":  [100, 64, -20],         ← 采样时的世界坐标（最小角），只做记录
  "palette": ["minecraft:air", "minecraft:oak_planks", ...],  ← 调色板：本样本用到的方块
  "blocks":  [0, 0, 1, 1, 0, ...]    ← 每格方块在调色板里的编号，共 X*Y*Z 个
}
blocks 的顺序：编号 = (x * Y尺寸 + y) * Z尺寸 + z
也就是先 z 变化最快，其次 y，最后 x。

【词表】
不同样本的调色板不一样，所以要把所有出现过的方块合成一张"总词表"：
  词表[0] = "minecraft:air"（空气永远是 0 号）
  词表[1] = "minecraft:stone"  ...
模型内部只认识 0,1,2,3... 这些编号。
另外还有一个特殊编号 "遮罩" = len(词表)，表示"这一格还不知道是什么"（扩散模型用）。
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

import 配置

空气 = "minecraft:air"


# ------------------------------------------------------------
# 方块字符串 <-> (名字, 属性字典)
#   "minecraft:oak_stairs[facing=east,half=bottom]"
#   <-> ("minecraft:oak_stairs", {"facing": "east", "half": "bottom"})
# ------------------------------------------------------------
def 解析方块(s: str):
    if "[" in s and s.endswith("]"):
        名字, 属性串 = s[:-1].split("[", 1)
        属性 = {}
        for 对 in 属性串.split(","):
            if "=" in 对:
                k, v = 对.split("=", 1)
                属性[k.strip()] = v.strip()
        return 名字, 属性
    return s, {}


def 组装方块(名字: str, 属性: dict) -> str:
    if not 属性:
        return 名字
    # 属性按名字排序，保证同一个方块状态永远是同一个字符串（Java 端也排序）
    return 名字 + "[" + ",".join(f"{k}={属性[k]}" for k in sorted(属性)) + "]"


def 规范化方块(s: str) -> str:
    """统一写法；如果配置里不保留方块状态，就去掉 [...] 部分。"""
    名字, 属性 = 解析方块(s)
    if 名字 in ("minecraft:cave_air", "minecraft:void_air"):
        return 空气
    if not 配置.保留方块状态:
        return 名字
    return 组装方块(名字, 属性)


# ------------------------------------------------------------
# 旋转：把结构从上往下看，顺时针转 90 度（北→东→南→西→北）
# ------------------------------------------------------------
_顺时针 = {"north": "east", "east": "south", "south": "west", "west": "north"}
_铁轨形状 = {
    "north_south": "east_west", "east_west": "north_south",
    "ascending_north": "ascending_east", "ascending_east": "ascending_south",
    "ascending_south": "ascending_west", "ascending_west": "ascending_north",
    "north_east": "south_east", "south_east": "south_west",
    "south_west": "north_west", "north_west": "north_east",
}


def 旋转方块一次(s: str) -> str:
    名字, 属性 = 解析方块(s)
    if not 属性:
        return s
    新 = {}
    for k, v in 属性.items():
        if k == "facing":                       # 大多数有朝向的方块
            v = _顺时针.get(v, v)
        elif k == "axis":                       # 原木、柱子：x 轴 <-> z 轴
            v = {"x": "z", "z": "x"}.get(v, v)
        elif k == "rotation" and v.isdigit():   # 告示牌、旗帜、头颅 0~15
            v = str((int(v) + 4) % 16)
        elif k == "shape":                      # 铁轨形状（楼梯的 shape 不受影响）
            v = _铁轨形状.get(v, v)
        k = _顺时针.get(k, k)                   # 栅栏/墙/玻璃板/红石的 north=true 这类连接属性
        新[k] = v
    return 组装方块(名字, 新)


def 旋转方块(s: str, 次数: int) -> str:
    for _ in range(次数 % 4):
        s = 旋转方块一次(s)
    return s


def 旋转网格(网格: torch.Tensor, 次数: int) -> torch.Tensor:
    """网格形状 [X, Y, Z]，绕 Y 轴顺时针转 次数×90°（和 旋转方块一次 的方向一致）。"""
    for _ in range(次数 % 4):
        # 新[x', y, z'] = 旧[z', y, X-1-x']
        网格 = 网格.permute(2, 1, 0).flip(0)
    return 网格


# ------------------------------------------------------------
# 读取样本文件
# ------------------------------------------------------------
def 读取样本文件(路径: Path):
    """返回 dict(文件, 标签, 调色板[list[str]], 网格[LongTensor X,Y,Z])，坏文件返回 None。"""
    try:
        d = json.loads(路径.read_text(encoding="utf-8"))
        尺寸 = tuple(int(v) for v in d["size"])
        调色板 = [规范化方块(s) for s in d["palette"]]
        方块 = d["blocks"]
        if len(方块) != 尺寸[0] * 尺寸[1] * 尺寸[2]:
            print(f"  [跳过] {路径.name}: blocks 数量 {len(方块)} 和 size {尺寸} 对不上")
            return None
        网格 = torch.tensor(方块, dtype=torch.long).reshape(尺寸)
        return {"文件": 路径.name, "标签": str(d.get("label") or "默认"), "调色板": 调色板, "网格": 网格}
    except Exception as e:
        print(f"  [跳过] {路径.name}: 读取失败 {e}")
        return None


def 读取全部样本(目录: Path = None, 打印=True):
    目录 = Path(目录 or 配置.数据目录)
    文件们 = sorted(目录.glob("*.json"))
    样本们 = []
    for f in 文件们:
        s = 读取样本文件(f)
        if s is None:
            continue
        if tuple(s["网格"].shape) != tuple(配置.结构尺寸):
            print(f"  [跳过] {f.name}: 尺寸 {tuple(s['网格'].shape)} ≠ 配置.结构尺寸 {配置.结构尺寸}")
            continue
        样本们.append(s)
    if 打印:
        print(f"从 {目录} 读到 {len(样本们)} 个有效样本（共 {len(文件们)} 个文件）")
    return 样本们


def 可用旋转次数():
    """X 和 Z 一样长才能转 90°，否则只能转 180°。"""
    if not 配置.旋转增强:
        return [0]
    X, _, Z = 配置.结构尺寸
    return [0, 1, 2, 3] if X == Z else [0, 2]


def 构建词表(样本们, 旧词表=None):
    """
    旧词表：接着训练时，存档里的词表。新词表 = 旧词表 + 新出现的方块（追加在后面），
    这样旧编号不变，模型可以继续用。
    """
    词表 = list(旧词表) if 旧词表 else [空气]
    已有 = set(词表)
    新增 = set()
    for s in 样本们:
        for r in 可用旋转次数():
            for b in s["调色板"]:
                b = 旋转方块(b, r)
                if b not in 已有:
                    新增.add(b)
    词表 += sorted(新增)
    return 词表


def 构建标签表(样本们, 旧标签表=None):
    标签表 = list(旧标签表) if 旧标签表 else []
    for 名 in sorted({s["标签"] for s in 样本们}):
        if 名 not in 标签表:
            标签表.append(名)
    return 标签表


class 结构数据集(Dataset):
    """
    把每个样本（以及它的旋转版本）提前转换成"词表编号网格"，训练时直接取。
    每一项返回： (网格 LongTensor[X,Y,Z], 标签编号 int)
    """

    def __init__(self, 样本们, 词表, 标签表):
        查表 = {b: i for i, b in enumerate(词表)}
        标签查表 = {b: i for i, b in enumerate(标签表)}
        self.项目 = []
        for s in 样本们:
            for r in 可用旋转次数():
                调色板转词表 = torch.tensor([查表[旋转方块(b, r)] for b in s["调色板"]], dtype=torch.long)
                网格 = 调色板转词表[旋转网格(s["网格"], r)].contiguous()
                self.项目.append((网格, 标签查表[s["标签"]]))

    def __len__(self):
        return len(self.项目)

    def __getitem__(self, i):
        return self.项目[i]


# ------------------------------------------------------------
# 把生成结果写成和样本一样格式的 .json（Java 生成器读取）
# ------------------------------------------------------------
def 保存结构(网格: torch.Tensor, 词表, 路径: Path, 标签="默认"):
    网格 = 网格.cpu()
    用到的 = torch.unique(网格).tolist()
    if 0 in 用到的:                      # 空气放调色板 0 号
        用到的.remove(0)
    用到的 = [0] + 用到的
    映射 = torch.zeros(len(词表), dtype=torch.long)
    for 新号, 旧号 in enumerate(用到的):
        映射[旧号] = 新号
    数据 = {
        "format": "mofangzhe-v1",
        "label": 标签,
        "size": list(网格.shape),
        "palette": [词表[i] for i in 用到的],
        "blocks": 映射[网格].flatten().tolist(),
    }
    路径.parent.mkdir(parents=True, exist_ok=True)
    路径.write_text(json.dumps(数据, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    非空气 = int((网格 != 0).sum())
    return 非空气
