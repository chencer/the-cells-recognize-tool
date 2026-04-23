# CLAUDE.md

## 项目简介

`cells_find.py` 是项目的入口文件。技术栈包括 Python、ttkbootstrap、Cellpose v3.0、OpenCV 和 PyTorch，使用本地模型文件 `cyto3`。

## 识别参数

- `diameter=120`，固定不自动估算
- `channels=[0, 0]`，灰度模式
- `flow_threshold=0.95`
- `cellprob_threshold=1.0`
- `min_size=200`

## 过滤链

1. **Step 1 完整度**：touches=Y 用 `circle_comp >= 0.65`；touches=N 用 `hull_comp >= 0.85`
2. **Step 2 面积**：面积 < 中位数 × 15% 过滤
3. **Step 3 圆形度**：`circularity < 0.3` 过滤

## 亮度计算

取 80% 内圈范围像素，选前 5% 最亮像素求均值（peak brightness）。

## UI Design Tokens

| Token | 值 |
|---|---|
| BG_WIN | `#020203` |
| BG_TITLEBAR | `#08010F` |
| TOOLBAR_BG | `#120920` |
| ACCENT_PUR | `#7C3AED` |
| ACCENT_GREEN | `#00E676` |
| TEXT_SEC | `#9E9E9E` |

窗口固定大小：1400×750

## 工作流规则

- 文档文字内容（README、CHANGELOG、done.md、注释）必须由 `ollama run qwen2.5:3b` 生成
- 测试通过才允许 commit
- 不打包，除非明确说"打包"或"发布"

## 任务单规范

- 指令必须直接可复制执行
- 提供改前/改后代码块
- 不贴冗长验证脚本
