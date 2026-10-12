# -*- coding: utf-8 -*-
r"""
如果当前 Python 没有 CUDA（比如误用了 MC_AI\.venv 里的 CPU 版 torch），
自动改用 MC_AI\.mc_gpu_py314_cu128 里的 GPU 版 Python 重新运行当前脚本。
"""
import os
import subprocess
import sys
from pathlib import Path

GPU解释器 = Path(__file__).resolve().parent.parent / ".mc_gpu_py314_cu128" / "Scripts" / "python.exe"


def 确保GPU():
    try:
        import torch
        if torch.cuda.is_available():
            return
    except ImportError:
        pass
    if os.environ.get("MOFANGZHE_已切换") or not GPU解释器.exists():
        return  # 已经切换过或找不到 GPU 环境，就继续用当前环境
    if Path(sys.executable).resolve() == GPU解释器.resolve():
        return
    print(f"当前 Python 没有 CUDA，自动切换到：{GPU解释器}")
    环境 = dict(os.environ, MOFANGZHE_已切换="1")
    sys.exit(subprocess.call([str(GPU解释器), sys.argv[0], *sys.argv[1:]], env=环境))
