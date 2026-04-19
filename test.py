import torch
import cellpose
print(f"Cellpose 版本: {cellpose.__version__}")
print(f"MPS 加速是否可用: {torch.backends.mps.is_available()}")
from cellpose import models
try:
    m = models.Cellpose(model_type='cyto3', gpu=True)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")