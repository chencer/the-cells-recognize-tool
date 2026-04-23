import os
import torch
from cellpose import models as cp_models


def load_model_cached(model_path, use_gpu=False):
    """Load CellposeModel, using a .pkl cache to skip re-initialization."""
    cache_path = model_path + '.pkl'

    if (os.path.exists(cache_path) and
            os.path.getmtime(cache_path) > os.path.getmtime(model_path)):
        try:
            m = torch.load(cache_path, map_location='cpu', weights_only=False)
            if use_gpu and torch.cuda.is_available():
                m.net = m.net.cuda()
            print(f"✅ 从缓存加载模型 ({os.path.basename(cache_path)})")
            return m
        except Exception as e:
            print(f"⚠️ 缓存读取失败，重新加载: {e}")

    m = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    try:
        torch.save(m, cache_path)
        print(f"✅ 模型缓存已保存: {cache_path}")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")
    return m
