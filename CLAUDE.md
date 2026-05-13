# CLAUDE.md

## 项目简介

`cells_find.py` 是项目的主入口文件（CLI 模式）。技术栈：Python、Cellpose v3.0、OpenCV、PyTorch，使用本地模型文件 `cyto3`。

`cells_recognize.py` 是独立验证脚本，结构与 cells_find.py 一致，额外包含 MPS 缓存清理和详细 tile 日志。

## 配置文件

所有参数通过 `settings.txt` 管理，`load_settings()` 读取，缺失键用代码默认值兜底。**禁止在代码里硬编码可配置参数。**

## 识别参数（默认值）

- `diameter=120`，`channels=[0, 0]` 灰度模式
- `flow_threshold=0.95`，`cellprob_threshold=1.0`
- `min_size=200`，`niter=200`
- 大图分块：`tile_w=2048`，`tile_h=1080`，`tile_overlap=0.2`

## 过滤链

1. **Step 1 完整度**：`hull_comp < 0.85` OR `circle_comp < 0.65` 则过滤
2. **Step 2 面积**：面积 < 中位数 × `area_ratio(0.15)` 过滤
3. **Step 3 圆形度**：`circularity < 0.5` 过滤
4. **Step 4 亮度计算**：80% 内圈范围，前 `brightness_top_pct(5%)` 最亮像素均值

## 输出结构

```
results/{stem}/
├── {stem}_result.png         标注结果图
├── {stem}_raw_detections.png 诊断图（过滤前检测点）
├── {stem}_top{n}.csv         Top N 细胞数据
├── {stem}_data.csv           全部细胞数据
├── top1.png / top2.png ...   Top N 裁剪图
```

## 大图路径

- `w * h > 3000 * 3000` 判定为大图
- `large_mode=1`：`_tile_and_merge()` 分块识别 → IoU 去重 → `_filter_and_rank_tile()`
- `large_mode=0`：整图 `model.eval()` → `_filter_and_rank_mask()`

## 工作流规则

- 文档文字内容（README、CHANGELOG、MANUAL、注释）必须由 `ollama run qwen2.5:3b` 生成
- 测试通过才允许 commit
- 不打包，除非明确说"打包"或"发布"

## 任务单规范

- 指令必须直接可复制执行
- 提供改前/改后代码块
- 不贴冗长验证脚本
