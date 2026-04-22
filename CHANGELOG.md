# Changelog

## v2.1.0 — 2026-04-22

- redesign: OLED dark cinema + glassmorphism 全局视觉重设计
- feat: 环形采样区域可视化，直观展示亮度计算范围
- feat: 原图 / 结果图切换按钮

## v2.0.0 — 2026-04-21

- feat: 亮度 ROI 优化，改为 80% 内圈前 10% 最亮像素均值
- feat: Top 3 标注，最亮三个细胞分级高亮显示
- feat: Windows 原生风格标题栏 + 浮动工具栏
- fix: queue 轮询机制修复，消除 UI 线程阻塞
- feat: 异步加载 Cellpose 模型，启动时显示进度动画

## v1.5.0 — 2026-04-21

- feat: 重构 UI 为黑紫沉浸风格（Variant A）
- feat: Variant B 沉浸式 UI 重构，macOS 风格标题栏 + 动效

## v1.4.0 — 2026-04-21

- fix: 边缘完整度阈值从 0.85 提高到 0.90（>=90% 才保留）

## v1.3.0 — 2026-04-21

- fix: 修复边界截断细胞过滤逻辑
- fix: 完整度阈值从 0.70 提高到 0.85
- fix: 新增面积过滤 + 圆形度过滤，去除噪声和异形物体
- fix: 轮廓描边 + 边缘过滤联合使用

## v1.2.0 — 2026-04-21

- fix: 改用轮廓描边替代圆圈标注，视觉更准确
- fix: 移除边界接触过滤，改为纯凸包完整度判断

## v1.1.0 — 2026-04-21

- fix: 重写边缘过滤逻辑，改用边界像素接触检测（原凸包比方法无法识别圆弧截断）
- fix: 修复标注逻辑，最亮细胞绿色轮廓，其余细胞黄色轮廓

## v1.0.4 — 2026-04-20

- ci: 新增 Windows 真实荧光图测试 workflow
- 优化细胞筛选和标注逻辑，用真实图 0058.tif 验证

## v1.0.3 — 2026-04-20

- fix: 替换弯引号为直引号，修复 PyInstaller 打包语法错误
- fix: 改用独立 .py 文件替代 heredoc，兼容 PowerShell
- fix: 将中文控制台输出改为英文，修复 Windows cp1252 UnicodeEncodeError

## v1.0.2 — 2026-04-20

- feat: 过滤边缘截断细胞（边界接触检测）
- feat: 输出最亮细胞坐标与亮度值到状态栏

## v1.0.1 — 2026-04-20

- fix: 修复兼容性问题
- chore: exe 文件改名
- tune: Cellpose 参数微调

## v1.0.0 — 2026-04-20

- 初始提交，搭建项目结构
- requirements.txt、GitHub Actions build_exe.yml
- 集成本地 cyto3 模型
