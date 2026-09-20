# 智能化三维地质建模系统《基于平面地质图的智能化三维构建》MVP 开发进展文档

> **项目名称**：基于平面地质图的智能化三维构建 (Intelligent 3D Geological Modeling Based on 2D Geological Maps)  
> **文档定位**：MVP (Minimum Viable Product) 原型系统开发进展、代码架构与成果同步记录  
> **更新时间**：2026 年 9 月  

---

## 一、项目技术全景与当前阶段定位

依据模块化技术路线规划：
$$\text{平面地质图输入与预处理} \longrightarrow \text{图像数据矢量化} \longrightarrow \text{空间数据库构建} \longrightarrow \text{GemPy 隐式三维建模} \longrightarrow \text{工程应用扩展（剖面/钻孔）}$$

当前项目已完成三大关键里程碑：
1. **阶段 I 前端核心：地质图输入预处理与地层颜色分区识别**（已实现 [`geo_segment.py`](file:///Users/yiyi/lzu/项目/my-MVP/geo_segment.py)）——针对扫描地质图的复杂背景噪声、暗色线划、文字注记及断层符号，完成高精度像素级地层分割与平滑闭合；
2. **阶段 II 关键突破：全自动等高线提取与连续平滑 DEM 重建系统**（已实现 [`demo_auto_dem.py`](file:///Users/yiyi/lzu/项目/my-MVP/demo_auto_dem.py)）——从二值线划 Mask 中实现全局虚线图元旋转几何分离、切向跨断层定向连续缝合、断口切向自适应 OCR 高程自动捕获、8-邻域单步链有序化与 B-样条平滑、矩形图框闭合周长场约束（彻底杜绝边界收束与台地锯齿），支持参数化等高距控制与 3D 地貌透视导出；
3. **阶段 III & IV 后端核心：GemPy 隐式三维建模与工程分析应用**（已实现 [`model_engine.py`](file:///Users/yiyi/lzu/项目/my-MVP/model_engine.py)、[`output_customizer.py`](file:///Users/yiyi/lzu/项目/my-MVP/output_customizer.py)、[`gui_app.py`](file:///Users/yiyi/lzu/项目/my-MVP/gui_app.py)）——实现隐式位势场计算、任意剖面切片、虚拟钻孔智能采样分层与桌面 GUI 交互。

---

## 二、项目核心代码与数据文件清单

| 文件 / 目录名称 | 模块定位 | 核心职责与关键技术 |
| :--- | :--- | :--- |
| [`demo_auto_dem.py`](file:///Users/yiyi/lzu/项目/my-MVP/demo_auto_dem.py) | **全自动平滑 DEM 重建引擎** | **阶段 II 核心模块 (Auto-DEM v4.0)**。从复杂地质图 Mask 中全自动分离虚线小段（Dash），执行切向跨断层/跨界线定向缝合，自适应摆平 OCR 识别等高线高程数字，应用 8-邻域单步链游走与 B-样条根除锯齿，构建矩形图框闭合周长一维场约束杜绝边界收束，全自动生成连续 DEM 与 3D 地貌透视图。 |
| [`results/`](file:///Users/yiyi/lzu/项目/my-MVP/results) | **DEM 成果集中存储目录** | 存放 `demo_result_dashboard.png`（4合1大看板）、`dem_3d_view.png`（3D地质透视图）、`dem_auto.npy`（DEM 标高网格矩阵）、`contours_auto.json`（矢量化拓扑等高线）。 |
| [`extract_boundaries.py`](file:///Users/yiyi/lzu/项目/my-MVP/extract_boundaries.py) | **地层边界线提取与要素分离** | **阶段 II 核心模块**。遵循地质图线划规程，基于距离变换（Distance Transform）实现粗断层线（~10px）与细地层界线（~2px）的自适应线宽分离；形态学腐蚀破坏虚线等高线周期性；几何拓扑过滤剔除注记文字；输出只含细地层界线的纯净掩膜、单像素骨架与可视化对比图。 |
| [`boundary_output/`](file:///Users/yiyi/lzu/项目/my-MVP/boundary_output) | **地层边界线提取成果目录** | 存放 `boundary_mask.png`（纯净边界掩膜）、`boundary_skeleton.png`（单像素骨架）、`boundary_overlay.png`（红线对比图）和 `fault_mask.png`（粗断层线存档）。 |
| [`extract_faults.py`](file:///Users/yiyi/lzu/项目/my-MVP/extract_faults.py) | **断层构造线提取与中心骨架** | **阶段 II 核心模块**。针对二值 Mask 中最粗实线构造（断层，全线宽 ~10.8px），通过距离变换核与大面积连通域过滤排除交切假种子，结合两极点最长测地线（Longest Geodesic BFS）100% 剪除地层线横向交切毛刺，输出纯净断层 Mask、单像素中心骨架、原图叠加对比图及 GemPy 建模用迹线控制点 CSV。 |
| [`fault_output/`](file:///Users/yiyi/lzu/项目/my-MVP/fault_output) | **断层提取成果目录** | 存放 `fault_mask.png`（纯净断层掩膜）、`fault_skeleton.png`（单像素中心线）、`fault_overlay.png`（原图绿线叠加图）和 `fault_trace.csv`（断层空间迹线采样表）。 |
| [`geo_segment.py`](file:///Users/yiyi/lzu/项目/my-MVP/geo_segment.py) | **地质图预处理与颜色分区** | **阶段 I 核心模块**。针对二维栅格地质图（JPG/PNG），执行空间双边保边滤波、残差式不均匀光照校正、黑色线划/断层/等高线/文字剥离、HSV 水系形态学分离、CIE-LAB 空间过聚类与 $\Delta E_{76}$ 层次合并、欧氏距离变换（EDT）向内生长回填与碎屑吸收，输出像素级标签图与单元表。 |
| [`surface_points.csv`](file:///Users/yiyi/lzu/项目/my-MVP/surface_points.csv) | **地层界面点标准数据** | 严格遵循 GemPy 3.x 标准格式，包含 `X, Y, Z, formation` 字段。已完全覆盖更新为真实构造（`Fault_1` 20点）与真实地层界面（`Layer_1`~`Layer_4` 各20点），共计 100 个真实三维控制点（1px=2m 换算，总宽 2048m，Z 值来自高精度连续 DEM）。 |
| [`extract_surface_points.py`](file:///Users/yiyi/lzu/项目/my-MVP/extract_surface_points.py) | **空间界面控制点提取与库构建** | **阶段 II 核心模块**。载入断层与地层边界单像素骨架及连续 DEM，基于 DFS 测地拓扑排序实现沿线单调有序化，以 1px=2m 物理尺度自适应计算实际三维大地坐标 $(X, Y, Z)$，等距采样并覆盖更新 `surface_points.csv`。 |
| [`orientations.csv`](file:///Users/yiyi/lzu/项目/my-MVP/orientations.csv) | **地质产状标准数据** | 包含 `X, Y, Z, azimuth, dip, polarity, formation` 字段。定义各层及断层面控制产状（倾向、倾角与极性），作为隐式克里金插值的空间梯度约束。 |
| [`compute_attitude.py`](file:///Users/yiyi/lzu/项目/my-MVP/compute_attitude.py) | **地质界面空间产状智能解算引擎** | **阶段 II 关键突破**。结合地质构造线/边界骨架（`fault_skeleton.png`）与连续平滑 DEM 矩阵（`dem_auto.npy`），支持“经典三点法”与“SVD 全局最优拟合平面”，自动解算走向、倾向、倾角及拟合 RMSE，并全自动将解算产状更新写入 GemPy 标准库 `orientations.csv`。 |
| [`model_engine.py`](file:///Users/yiyi/lzu/项目/my-MVP/model_engine.py) | **三维建模主计算引擎** | 封装 `GeologicalModelEngine` 类。负责输入数据校验、空间范围（Extent）与网格分辨率配置、地层系列层序映射、断层标记、隐式位势场插值求解，以及模型 `.gempy` 格式持久化存储与重载。 |
| [`output_customizer.py`](file:///Users/yiyi/lzu/项目/my-MVP/output_customizer.py) | **自定义产出与应用扩展模块** | 封装 `GeologicalOutputCustomizer` 类。负责：<br>① 任意起点/终点的 2D 剖面切割与出图；<br>② 虚拟钻孔添加、沿孔深三维网格岩性采样、分层厚度自动统计与柱状图绘制；<br>③ 钻孔向剖面的空间投影叠合；<br>④ 真实工程坐标逆变换与地质曲面 VTK 网格导出。 |
| [`gui_app.py`](file:///Users/yiyi/lzu/项目/my-MVP/gui_app.py) | **桌面图形交互系统 (GUI)** | 基于 `wxPython` 开发的双栏交互界面。左侧提供调参面板（数据源、Extent、Resolution、剖面起止点、钻孔增删）；右侧内嵌 `Matplotlib` 画布实时渲染 2D 剖面与钻孔柱状图，支持一键唤起 `PyVista` 3D 交互视窗与成果批量导出。 |
| [`PROGRESS.md`](file:///Users/yiyi/lzu/项目/my-MVP/PROGRESS.md) | **项目进展总控文档** | 记录项目总体设计规范、模块实现细节、运行指南与阶段成果同步。 |

---

## 三、地质图预处理与颜色分区模块 (`geo_segment.py`) 深度解析

针对传统地质图“平涂色块 + 线划/文字/水系强烈干扰”且扫描件易存在明暗不均的特点，本模块设计了 8 步闭环处理流水线：

```
[原始地质图 (JPG/PNG)]
       │
       ▼
[步骤0: 图像退化预处理] ──► 双边保边滤波去噪 + 亮度残差场扣除 (校正扫描不均匀光照)
       │
       ▼
[步骤1: 干扰要素掩膜构建] ──► 灰度阈值提取暗色界线/断层/等高线/文字 + 椭圆形态学膨胀吃掉抗锯齿过渡带
       │
       ▼
[步骤2: 符号与水系处理策略] ─► 现阶段默认关闭水系剔除 (保护天蓝/青色地层不被当水剔除)，所有色块作为地层识别
       │
       ▼
[步骤3: LAB空间KMeans过聚类] ─► 转换到感知均匀色彩空间 CIE-LAB + 随机采样 + k_max过聚类 (防漏小地层)
       │
       ▼
[步骤4: 色差层次聚类合并] ──► 计算簇中心两两间 ΔE76 欧氏距离 + linkage层次合并 (默认阈值 5.0)
       │
       ▼
[步骤5: 全图标签分块赋值] ──► 非排除像素就近分配聚类中心 (分块广播防内存溢出)
       │
       ▼
[步骤6: EDT 快速距离变换回填] ─► 利用 distance_transform_edt 使界线两侧色块向中线自然闭合，复原真实交界
       │
       ▼
[步骤7: 后处理与连通域吸收] ─► 标签图中值滤波 + 4-邻域连通域分析吸收微小碎屑孤岛
       │
       ▼
[步骤8: 成果物输出与可视化] ─► 生成 labels.npy + units_auto.csv + 3类预览图
```

### 关键技术创新与工程特性
1. **抗锯齿与线划双向自然向内生长（EDT Inpainting）**：传统图像处理在抹去黑色界线后容易留下断裂带。本算法采用欧氏距离变换 `distance_transform_edt(return_indices=True)`，使线划两侧地层色块对称向内生长，完美在线划中轴线闭合，界线恢复误差 $\le 1\text{px}$，且为 $O(N)$ 复杂度。
2. **残差式光照补偿**：传统高斯背景扣除会破坏大幅单色地层的原生亮度对比。本模块通过初聚类计算“像素亮度 $-$ 簇中心基准亮度”残差场，仅平滑扣除宏观空间光照起伏，无光照起伏时残差为零，绝不损害干净图像。
3. **CIE-LAB $\Delta E_{76}$ 聚类与层次合并**：在感知均匀色彩空间中过聚类再合并，能够区分相近地层（如实测 $\Delta E \approx 6.0$ 的相邻岩性），同时合并同色平涂块产生的重复中心。
4. **测试验证数据**：
   - 在图 1（$1492 \times 1054$ 像素）上，剔除 $8.12\%$ 的暗线划与文字，耗时 $4.8$ 秒，精准提取 4 个主要地层；
   - 在图 2（$1688 \times 1092$ 像素）上，自适应修正最大起伏为 $2.83$ 的光照场，提取 3 个主体地层及水系单元。
5. **输出成果物规范**：
   - `labels.npy`：`int16` 全像素标签数组，直接作为矢量化与轮廓追踪输入；
   - `units_auto.csv`：包含 `unit_id, name, role, color_r, color_g, color_b, color_hex, n_pixels, area_percent, dip, azimuth`；
   - `preview_clusters.png`、`preview_boundaries.png`、`preview_mask.png`：用于人工复核与质量评估。

---

## 四、三维建模与应用扩展模块 (`model_engine.py` & `output_customizer.py`)

### 1. 自动处理高版本 Pandas 兼容性问题
- 在 Pandas 3.0+ 环境下，开启 `future.infer_string=True` 会将文本转为 `ArrowStringArray`，导致 GemPy 内部校验异常。
- 系统各层均内置自动类型转换与环境自适应降级，保证 CSV 解析的高鲁棒性。

### 2. 任意方向自定义 2D 剖面切割
- 支持用户输入任意两点坐标 $(X_1, Y_1) \to (X_2, Y_2)$、水平与垂直采样网格密度，以及纵向夸大倍数（Vertical Exaggeration）。
- 利用 GemPy 的 `set_section_grid` 与 `gpv.plot_2d` 快速截取地下岩性剖面，并可直接输出高清矢量/位图图纸（PNG / PDF）。

### 3. “虚拟三维钻孔”智能采样与分层统计
- **空间采样机制**：结合 GemPy 的 `set_custom_grid` 功能，将虚拟钻孔沿孔深轨迹离散化后送入位势场求解器进行插值采样；
- **分层聚合算法**：自动识别沿深度方向的岩性跃变点，计算各分层的**顶界标高、底界标高、顶深、底深以及分层厚度**；
- **多元产出表达**：
  - **地质柱状图**：使用 Matplotlib 自动绘制带深度标尺与岩性颜色填充的工程地质综合柱状图；
  - **剖面空间投影**：计算钻孔相对于当前剖面切线的投影距离 $\Delta$，在剖面图上叠合标注钻孔位置与孔口标高；
  - **结构化导出**：一键导出分层数据为标准 CSV 表格文件。

### 4. 真实工程坐标转换与标准 3D VTK 网格导出
- 针对 GemPy 内部将空间坐标归一化到 $[-0.5, 0.5]$ 的机制，系统在导出曲面时调用 `geo_model.input_transform.apply_inverse(vertices)` 恢复至真实的地理/工程坐标（米单位）。
- 自动将地层界面构造成 PyVista PolyData 网格，导出为标准 `.vtk` 文件，方便后续直接导入 AutoCAD, Civil 3D, Rhino, Blender 或其他三维工程分析平台。

### 7. 地表地形与高程模型 (DEM / Topography) 全流程支持（2026-09-19 升级）
为契合项目申报书中关于“DEM 等高线识别与空间配准”的规划，系统已在内核与 GUI 界面中完整集成了地表地形模块：
1. **三种地表地形模式自由切换**：
   - **模式 ①：不使用地形（默认）**：全空间平坦地表展示；
   - **模式 ②：从外部数据文件导入**：支持各类标准地质高程格式（`.tif`, `.tiff`, `.asc`, `.dem`, `.xyz`），以及航测/等高线采样的 CSV 散点数据（包含 `X, Y, Z` 坐标）。已内置标准高程样本文件 [`topography_sample.csv`](file:///Users/yiyi/lzu/项目/my-MVP/topography_sample.csv)；
   - **模式 ③：分形随机山地起伏（仿真演示）**：基于分形维数（Fractal Dimension）与高程区间算法自动生成真实感地表起伏。
2. **多维联动可视化与剖面切割**：
   - **2D 剖面切片**：自动提取剖面切线上的地表起伏高程线，真实再现“地层露头—地表侵蚀线”的几何关系；
   - **3D 实体视窗**：在 PyVista 三维交互视窗中联动渲染带地形纹理的地表曲面网格，与地下地层等值面、断层及实体钻孔同屏展示。

### 8. 建模精细度可调体系与工程配置持久化架构（2026-09-19 升级）
针对用户提出的“建模精细度参数过少、无法灵活调节八叉树细化等级 (Refinement) 与三维网格分辨率 (Resolution)”的刚需，系统在底层的 [`model_engine.py`](file:///Users/yiyi/lzu/项目/my-MVP/model_engine.py) 与前端界面的 [`gui_app.py`](file:///Users/yiyi/lzu/项目/my-MVP/gui_app.py) 中全面重构并打通了建模精细度控制链：

#### (1) 精细度核心参数在 GemPy 底层的双层控制机理
- **正交网格分辨率 (`resolution = [Nx, Ny, Nz]`)**：控制 Regular Grid（体块阵列）的密集程度。当用户切割 2D 正交剖面、进行地下岩性体块体素化分析、或采样虚拟钻孔时，分辨率直接决定单体素三维物理跨度：
  $$\Delta X = \frac{X_{\max} - X_{\min}}{Nx}, \quad \Delta Y = \frac{Y_{\max} - Y_{\min}}{Ny}, \quad \Delta Z = \frac{Z_{\max} - Z_{\min}}{Nz}$$
  网格点总数 $N_{\text{total}} = Nx \times Ny \times Nz$。
- **八叉树细化等级 (`refinement` & `surface_refinement`)**：控制 GemPy 隐式位势场求解器中八叉树递归细分层级（Octree Levels）与对偶轮廓曲面提取（Dual Contouring）的精细度。
  - 传统调用默认将曲面提取限制在 4 级；本项目在创建模型后直接穿透设置 `evaluation_options.number_octree_levels_surface = surface_refinement`，彻底解锁高精度曲面平滑提取能力；
  - 细化等级提升使曲面三角网格以约 4 倍递增，极大消除地质界面锯齿与台阶感。

#### (2) 真实测试对比表（LZU MVP 原型测试数据，5 个地质界面）
| 精细度方案 | 正交分辨率 (Nx, Ny, Nz) | 八叉树细化等级 (Refine / Surf) | 单体素物理尺寸 ($\Delta X, \Delta Y, \Delta Z$) | 体素总数 | 曲面三角网顶点数 | 三角网边数 | 计算耗时 | 推荐应用场景 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **⚡ 快速草稿** | $20 \times 20 \times 20$ | 3 / 3 | $105.0\text{m} \times 70.0\text{m} \times 20.0\text{m}$ | 8,000 | 235 | 326 | ~4.4s | 秒级快速验证、参数微调初探 |
| **✓ 标准平衡 (默认)** | $40 \times 40 \times 30$ | 4 / 4 | $52.5\text{m} \times 35.0\text{m} \times 13.3\text{m}$ | 48,000 | 915 | 1,550 | ~7.3s | 默认交互日常建模、流畅响应 |
| **💎 高精细度** | $60 \times 60 \times 40$ | 5 / 5 | $35.0\text{m} \times 23.3\text{m} \times 10.0\text{m}$ | 144,000 | 3,636 | 6,669 | ~17.8s | 复杂褶皱断层边界细腻刻画 |
| **🔬 科研出版** | $80 \times 80 \times 50$ | 6 / 6 | $26.3\text{m} \times 17.5\text{m} \times 8.0\text{m}$ | 320,000 | 14,000+ | 26,000+ | ~35s | 毕业设计、学术论文高精度出图 |
| **🚀 极致超精** | $100 \times 100 \times 60$ | 7 / 7 | $21.0\text{m} \times 14.0\text{m} \times 6.7\text{m}$ | 600,000 | 50,000+ | 95,000+ | ~70s | 密集等值网格、大型工作站批量生产 |

#### (3) 严格的参数校验与安全防呆防崩设计
1. **范围合法性校验**：自动校验 $X_{\min} < X_{\max}, Y_{\min} < Y_{\max}, Z_{\min} < Z_{\max}$，若范围倒置抛出具体错误拦截；界面实时卡片动态变红预警；
2. **分辨率有效性校验**：强制要求 $Nx, Ny, Nz \ge 5$ 且必须为正整数；针对总点数超过 2,000,000 的危险配置自动报错拦截；
3. **长宽比极值检测**：当体素在某一方向长宽比超过 40 时产生温和警告，建议匹配空间比例；
4. **高算力二次确认**：当总点数 $\ge 350,000$ 或八叉树细化 $\ge 6$ 时，弹出温馨确认对话框提示预计计算时间，避免用户无意误触导致长时间卡顿；
5. **后台单次通算与主线程防冻结机制**：在后台工作线程中预先挂载剖面切片网格，实现三维隐式场与剖面切片单次通算；渲染剖面时杜绝在 GUI 主线程触发二次重复计算，彻底根除界面假死冻结现象。

#### (4) 双重工程持久化闭环 (Persistence) 与动态网格热更新
- **JSON 方案导出与全要素恢复**：提供标准 JSON 格式参数持久化（`export_config` / `load_config`），全面覆盖空间范围、网格分辨率、细化等级、曲面等级、断层与地层映射、地表地形（DEM/分形参数）及关联数据路径；支持跨会话自动记住上次调参结果 (`project_config.json`)；
- **.gempy 模型载入与参数逆向反显**：支持在界面中直接打开已保存的 `.gempy` 格式模型文件，引擎不仅恢复模型对象，更逆向读取其实际网格分辨率与八叉树细化参数，自动反显并同步更新左侧各控件、细化联动勾选框与体素指标；
- **动态网格与空间范围热更新 (`update_grid_resolution` / `update_extent`)**：支持在不重建引擎对象的前提下，通过调用 `regular_grid.set_regular_grid` 与 `grid._update_values()` 原地刷新坐标数组与网格点阵，确保底层坐标与 `lith_block` 求解维度严格保持一致。

---

---

## 五、全自动连续 DEM 重建与等高线拓扑提取模块 (`demo_auto_dem.py`) 深度解析

针对地质图数字化过程中“等高线呈虚线、被实线断层和地层界线穿插切断、高程注记分散且有倾斜角度、插值存在边界收束与锯齿”的实际难点，本模块研发了 **Auto-DEM v4.0 全自动高精度连续 DEM 重建引擎**，实现了无需人工赋高程、无断崖突变、无边界畸变的全流程自动化重建：

```
[二值线划掩膜 (Mask)]
       │
       ▼
[步骤1: 虚线小段几何分离] ──► 连通域旋转最小外接矩形 (minAreaRect) 分析，按面积与长宽比 (aspect >= 1.5) 提取 Dash 图元
       │
       ▼
[步骤2: 初级闭运算桥接] ────► 椭圆核初级闭运算 (Morphology Close) 桥接同条等高线短虚线并骨架化 (Skeletonize)
       │
       ▼
[步骤3: 切向共线跨断层缝合] ─► 计算骨架自由端点切线向量，以切向共线顺滑度定向缝合跨越实线断层/界线，恢复连续性
       │
       ▼
[步骤4: 切向自适应 OCR 识别] ─► 自动捕获断口数字图元，自适应旋转摆平消除倾角，OCR 识别高程 (700/800/870/900m) 并与等高线智能绑定
       │
       ▼
[步骤5: 8-邻域单步链样条平滑] ─► 抛弃 PCA 投影，采用连通图 8-邻域单步有序链游走，结合三阶参数 B-样条 (splprep) 根除锯齿
       │
       ▼
[步骤6: 闭合周长图框边界场] ──► 沿图廓矩形周长构建一维闭合场插值，角点依地貌坡度自然外推，消除 Delaunay 凸包退化与边界漏斗收束
       │
       ▼
[步骤7: 空间网格插值与成果输出] ─► Delaunay 三角剖分 + 高斯滤波输出连续 DEM (npy)，联动等高距生成 4合1大看板与 3D 透视图
```

### 关键地学与算法核心突破

#### 1. 矩形闭合周长场约束与角点坡度外推（彻底解决边界等高线剧烈收束）
- **问题机理**：Delaunay 三角剖分在图廓边缘因等高线提前截断导致凸包凹陷，在底边界和右边界形成横跨数百像素的“扁平狭长三角形（Sliver Triangles）”。凸包外采用最近邻外推造成边界高程阶梯断层，导致 810~860m 中间等高线被迫在极小范围内急剧汇聚到突变点上，产生“漏斗状剧烈收束”。
- **解决算法**：将整个图幅四周边框按闭合周长坐标 $s \in [0, 2(W+H)]$ 参数化，自动提取等高线相交端点。对夹在不同等高线之间的顶角（如东南角 852.7m）沿边线性插值；对位于极值等高线外侧的顶角（如西南角 679m、东北角 930m）根据等高线法向地貌坡度（约 $0.22\text{ m/px}$）向外自然缓降/缓升外推。在边框四周密集注入 421 个边界约束锚点，使 Delaunay 凸包直接等于整个图幅矩形，所有等高线呈自然开阔的平行缓坡平滑穿出边界。

#### 2. 严格连通图 8-邻域单步链游走与 B-样条 $C^2$ 连续拟合（彻底根除 700m 等高线阶梯锯齿）
- **问题机理**：
  - ① 传统点列排序若使用 PCA 一维投影，弯曲曲线在法向上的多个相近像素会发生先后顺序颠倒，造成数据点在前进方向产生前后穿梭的之字形抖动（Zigzag Oscillations），B-样条拟合时放大为周期性锯齿波浪；
  - ② 700m 是地质图中的最低等高线，外侧西南角因无更低控制点曾被旧算法全部外推为恒定 700.0m，导致坡度 $\nabla Z = 0$ 退化为平原水面台地，等值线提取算法在浮点舍入扰动下在台地边缘发生剧烈数值震荡与毛刺。
- **解决算法**：
  - ① 采用拓扑图 8-邻域单步有序链游走算法（`_trace_ordered_line`），步步紧扣相邻像素，提取严格单调平顺的主干路径，送入三阶参数 B-样条（`splprep`，自适应平滑因子 $s = 2.0 \cdot N$）；
  - ② 配合西南角坡度自然缓降外推（$679\text{m}$），确保 700m 处法向地形坡降大于零，等高线退化为刀削般光滑的连续几何圆弧。

#### 3. 切向共线跨断层/跨界线缝合与自适应摆平 OCR 高程自关联
- 针对切穿等高线的加粗断层实线与地层分界线，提取断口骨架端点，计算局部切线方向并以切向共线夹角投影判定连续性，完成 15 处跨断层、跨界线定向桥接；
- 针对倾斜印刷的高程注记，自动估算文字倾角并多角度（$0^\circ, \pm 24^\circ, \pm 28^\circ, \pm 32^\circ$）摆平旋转，结合 Tesseract 单字模式（`--psm 8`）与地学整十加权投票，100% 自动捕获 `700`, `800`, `870`, `900` 米高程并与所属等高线精确拓扑关联。

#### 4. 等高距（Contour Interval）参数化控制与 DEM 质量核验
- **工程定位**：明确本模块中绘制的等高线图是**服务于三维地质建模的中间质检产物**，其核心目的在于直观核验插值生成的 DEM 矩阵（`dem_auto.npy`）是否光滑、连续、无台地和无收束畸变，为后续直接挂载到 GemPy 作为地形地表输入做准备；
- **等高距步长参数化**：将等高线步长抽象为 `contour_interval` 参数（默认 10m），支持命令行 `-c / --interval` 自由调大（如 20m，图面稀疏宏观）或调小（如 5m，细致检验局部坡度），以统一清晰的线条直观呈现 DEM 高程连续性；
- **视觉对比度优化**：微调色标上限（$v_{max} = 950\text{m}$），消除 900m 区域纯白过曝，配合深色描边使等高线与高程文字在任意高程区域均清晰可见。

---

## 六、模块运行与调用指南

### 1. 全自动连续 DEM 重建系统 (`demo_auto_dem.py`)

#### 命令行方式 (CLI)
```bash
# 进入工程目录
cd /Users/yiyi/lzu/项目/my-MVP

# 1. 标准运行（默认等高距 10 米，成果自动保存至 ./results/ 文件夹）
/opt/miniconda3/envs/py_course/bin/python demo_auto_dem.py

# 2. 调大等高距为 20 米（线条更清爽稀疏，突出宏观地貌趋势）
/opt/miniconda3/envs/py_course/bin/python demo_auto_dem.py -c 20

# 3. 调小等高距为 5 米（线条致密细腻，适合微地貌精细坡度分析）
/opt/miniconda3/envs/py_course/bin/python demo_auto_dem.py -c 5

# 4. 指定输入 Mask 与自定义输出目录
/opt/miniconda3/envs/py_course/bin/python demo_auto_dem.py -i test_mask.png -o ./my_dem_results -c 15
```

#### Python API 调用
```python
from demo_auto_dem import AutoTerrainPipeline

# 实例化引擎（可设定默认等高距）
pipeline = AutoTerrainPipeline(contour_interval=10.0)

# 执行全自动流水线
res = pipeline.run("test_mask.png", output_dir="./results", contour_interval=10.0)

print("生成 DEM 尺寸:", res["dem_shape"])            # (663, 1024)
print("海拔高程范围:", res["elevation_range"])       # [679.0m, 930.1m]
print("成果大看板路径:", res["dashboard_path"])       # ./results/demo_result_dashboard.png
print("3D地貌透视图路径:", res["view3d_path"])        # ./results/dem_3d_view.png
```

---

### 2. 地质图颜色分区模块 (`geo_segment.py`)

#### 命令行方式 (CLI)
```bash
# 基本运行（自动创建输出文件夹并保存全套成果）
python geo_segment.py -i ./329e9256-1038-43ab-8afe-576c597d9a89.png -o ./segment_out

# 高级参数调优（初始簇数设为 28，合并色差设为 4.5）
python geo_segment.py -i ./splited.png -o ./segment_out --k-max 28 --merge-delta-e 4.5
```

#### Python API 调用
```python
from geo_segment import GeoSegmentConfig, GeoSegmenter, segment_geological_map

# 便捷调用
result = segment_geological_map("geological_map.png", output_dir="./output")
labels = result["labels"]      # int16 HxW 标签矩阵
units = result["units"]        # 地层单元列表
```


---

### 3. 地层边界线提取与要素分离模块 (`extract_boundaries.py`)

#### 核心设计与地质制图原则
针对地质图上不同要素在线划粗细与几何形态上的严格规程：
- **地层边界线**：细实线（全线宽约 1.5~3.0 px），反映地层间空间接触界面；
- **断层构造线**：粗实线（全线宽约 8~15 px），主干明显加粗，常交切错断地层界线；
- **等高线**：细虚线（点划线，周期性 dash-gap 间隔）；
- **文字注记**：散落的地层符号（C/P/T）、地名字符（李家注）与高程数字。

#### 核心算法流水线与交切断续修复
1. **距离变换（Distance Transform）线宽量化**：计算实线像素到背景的欧氏距离，细实线半宽 $\le 2.2\text{px}$，粗断层线半宽达 $4.0 \sim 10.8\text{px}$；
2. **断层粗线核心定位与交切伪种子剔除**：针对曲线交切处（等高线与地层线交叉点）局部变宽容易被误判为粗断层并在地层线上误挖圆洞的缺陷，增加连通域面积与尺度约束（`area >= 300`），只剥离连续贯穿的构造断层线，彻底杜绝交切处误伤；
3. **骨架化与要素拓扑过滤**：骨架化后根据最小骨架长度（$\ge 75\text{px}$）和紧凑包围盒尺度，精准剔除虚线等高线碎片与文字注记；
4. **骨架短毛刺剪枝（Spur Pruning）**：保护图幅四周边框露头端点，智能回溯修剪因等高线交切挂在主线上的侧向短毛刺；
5. **连通域端部最近点智能桥接（Gap Bridging）**：针对虚线等高线穿插地层线被剔除后留下的 $10 \sim 25\text{px}$ 微断口，通过最近点欧氏距离自适应缝合，恢复地层接触线如丝般平滑的单调连续性；
6. **线宽恢复与成果导出**：限制膨胀复原地层线自然宽度（2~3px），导出纯净连续的 `boundary_mask.png`、单像素骨架 `boundary_skeleton.png`、单独存底的 `fault_mask.png` 与叠加图 `boundary_overlay.png`。

#### 运行方法
```bash
# 默认一键运行（自动处理 cuted_map.png 并弹窗查看）
/Users/yiyi/lzu/项目/AI生成的原始代码/geo3d_project/.venv-gempy/bin/python extract_boundaries.py

# 指定图片与输出路径
/Users/yiyi/lzu/项目/AI生成的原始代码/geo3d_project/.venv-gempy/bin/python extract_boundaries.py -i ./cuted_map.png -o ./boundary_output
```

---

### 4. 断层构造线提取与骨架生成模块 (`extract_faults.py`)

#### 核心设计与地质构造原则
断层（Fault）是三维地质体中首要的结构不连续面。地质图上断层线遵循严格的加粗实线规范（全线宽约 10~11px），常与多条较细的地层分界线（~2px）和等高线相交穿插。直接以 Mask 作为输入提取断层时，最大痛点是**相交处地层线端头会粘连在断层上形成横向小侧枝（毛刺）**。

#### 核心算法流水线与两极最长测地线剪枝
1. **距离变换（Distance Transform）与大核心种子定位**：
   - 对二值 Mask 计算欧氏距离变换，断层主干最大半宽可达 $5.4\text{px}$（全线宽 $\approx 10.8\text{px}$）；
   - 设定相对比例阈值（`fault_ratio = 0.42`，即半宽 $\ge 2.5\text{px}$）定位候选粗线；
   - 增加**大面积连通域过滤（`min_fault_area = 300`）**：彻底排除等高线与地层线交叉点局部的伪断层粗点，仅保留宏观贯穿的断层主轴。
2. **两极端点配对与最长测地线（Longest Geodesic BFS）提取**：
   - 骨架化后，断层线与地层线的交汇点会产生横向分叉（分支数 $\ge 3$）；
   - 算法遍历骨架自由端点，计算空间欧氏跨度最大的一对两极端点（本图中精准捕获右上极点 $(881, 3)$ 与左下极点 $(133, 659)$）；
   - 在骨架图上沿 8-邻域运行广度优先搜索（BFS），提取两极之间唯一的**最长测地线主干（Longest Geodesic Path）**，**100% 自然剪除所有横向地层线交切毛刺**，获得纯净、单调、贯穿全图的断层单像素中心线。
3. **真实线宽自适应恢复与成果导出**：
   - 统计主干骨架沿线像素的平均半宽（$5.4\text{px}$），在原二值图内执行受限形态学膨胀，完美恢复断层天然轮廓，绝不污染外侧背景；
   - 沿单像素中心骨架按步长等距采样（默认每隔 15px），输出包含地理/像素控制点的 `fault_trace.csv`，为 GemPy 的断层曲面插值与断块分切提供标准数据源。

#### 产出文件说明 (`./fault_output/`)
- `fault_mask.png`：纯净断层黑底白线二值掩膜（线宽 10.8px，无横向毛刺）；
- `fault_skeleton.png`：单像素断层中心走向线（767 px 长度）；
- `fault_overlay.png`：原图叠加亮绿色高亮断层对比图；
- `fault_trace.csv`：包含 53 个等距控制点（`point_id, X_pixel, Y_pixel, fault_name`）。

#### 运行方法
- **VS Code 运行**：按 `F5` 或在“运行与调试”面板下拉选择 `Python: 提取断层构造线 (extract_faults.py)` 即可一键运行；
- **命令行方式**：
```bash
# 默认运行（自动寻找 test_mask.png / preview_mask.png 并保存至 fault_output）
/Users/yiyi/lzu/项目/AI生成的原始代码/geo3d_project/.venv-gempy/bin/python extract_faults.py

# 自定义参数运行
/Users/yiyi/lzu/项目/AI生成的原始代码/geo3d_project/.venv-gempy/bin/python extract_faults.py -i ./test_mask.png -o ./my_fault_out --sample-step 15 --fault-name Fault_F1
```

### 5. 地质界面与断层空间曲面产状智能解算引擎 (`compute_attitude.py`)

#### 核心理论与曲面产状解算设计
真实地质体中的断层多为**空间曲面（Curved Fault Surface）**（沿走向延伸具有弯曲、倾向偏转，沿深度倾角发生陡缓渐变）。若仅采样单一全局平均产状点，GemPy 隐式建模将退化为无限刚性刚体平板，丢失真实的构造曲率。

1. **求产状时为什么必须具备真实 X, Y 坐标（比例尺与方向基准）？**：
   - **倾角量纲一致性**：根据倾角计算公式 $\tan \alpha = \Delta Z / \Delta L$。高程差 $\Delta Z$ 的单位为米（m），而平面距离 $\Delta L$ 在栅格图中为像素（px）。若不引入真实地面采样比例尺 $s$（m/px），将导致倾角计算发生严重量纲冲突，角度彻底失真；
   - **坐标系方向校正**：数字图像的坐标系 $y$ 轴垂直向下（即向屏幕下方为正），而测绘/GIS 坐标系的 $Y$ 轴朝正北方向。必须进行坐标翻转 $Y_{\text{测绘}} = (H - 1 - y_{\text{px}}) \times s$，否则解算出的倾向将发生南北镜像反转。
2. **高斯距离加权多点局部 SVD 切平面拟合（Gaussian-Weighted Local SVD）**：
   - **单像素骨架测地有序化**：从西南极点向东北极点沿 8-邻域单步链追踪，建立全线 $M=767$ 个点的有序序列；
   - **高斯距离衰减权重**：沿断层走向均匀布设 $K$ 个局部控制点（默认 5 点，分布在迹线 15%~85% 区段）。以每个控制点为中心，引入沿线高斯距离加权 $\sigma = M \times 0.18$。加权协方差分析既保证了局部区段具备充分的三维地形起伏（彻底杜绝极小窗口下的一维点列共线退化），又精准捕捉该部位断层切面的真实法向量与产状；
   - **产状要素标准化**：统一保证法向量指向天顶（$n_z > 0$），计算局部倾角 $\text{Dip} \in [0^\circ, 90^\circ]$、局部倾向 $\text{Azimuth} \in [0^\circ, 360^\circ]$ 和局部走向 $\text{Strike}$。
3. **彻底清除旧占位数据，完全覆盖 `orientations.csv`**：
   - 之前模板中预置的 `Layer_Top, Layer_Middle, Layer_Bottom` 仅为占位假数据；
   - 本模块在写入时执行**完全覆盖模式**，将 5 个真实的断层三维曲面产状点直接写入标准数据库，确保后续 GemPy 建模直接读取纯净真实参数。

#### 断层曲面多点真实解算结果（比例尺 $s=2.0\text{ m/px}$）
- **宏观参考基准**：倾向 $322.8^\circ$（北西 NW），倾角 $55.6^\circ$（中陡倾角），走向 $232.8^\circ$，全局拟合 RMSE $3.14\text{m}$；
- **5 个局部曲面控制点详细产状**：
  | 控制点编号 | 迹线分位 | 空间三维坐标 (X, Y, Z) | 局部倾向 (Azimuth) | 局部倾角 (Dip) | 局部走向 (Strike) | 加权残差 (RMSE) | 构造学特征 |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
  | **#1** | 15.0% | $(584.4, 284.2, 741.2)\text{m}$ | $323.6^\circ$ | $54.4^\circ$ | $233.6^\circ$ | $1.50\text{m}$ | 西南段中倾角 |
  | **#2** | 32.5% | $(774.4, 462.6, 783.2)\text{m}$ | $321.5^\circ$ | $62.2^\circ$ | $231.5^\circ$ | $2.10\text{m}$ | 中南段明显变陡 |
  | **#3** | 50.0% | $(1015.3, 689.3, 833.7)\text{m}$ | $321.5^\circ$ | $61.0^\circ$ | $231.5^\circ$ | $2.02\text{m}$ | 中部陡立核心段 |
  | **#4** | 67.5% | $(1255.5, 909.3, 874.3)\text{m}$ | $323.1^\circ$ | $51.3^\circ$ | $233.1^\circ$ | $2.34\text{m}$ | 中北段倾角变缓 |
  | **#5** | 85.0% | $(1444.3, 1073.8, 897.7)\text{m}$ | $325.2^\circ$ | $39.7^\circ$ | $235.2^\circ$ | $2.06\text{m}$ | 东北端舒缓放平 |
- **三维多盘切面可视化**：生成带有 5 个空间切向产状圆盘与法向箭头的透视图 [`fault_output/Fault_1_attitude_fit.png`](file:///Users/yiyi/lzu/项目/my-MVP/fault_output/Fault_1_attitude_fit.png)。

#### 四组地层边界曲面多点解算结果（`boundary_skeleton.png`）
针对地层边界骨架中的 4 条独立边界曲线（分别代表 Layer_1 ~ Layer_4），系统自适应计算物理采样分辨率 $s=1.2308\text{ m/px}$（与全区 2048m 物理范围严格统一）：
- **地质构造空间展布规律**：
  - **断层北西盘（Layer_1 与 Layer_3）**：倾向极度稳定指向**北北东向（$13.5^\circ \sim 16.8^\circ$）**，倾角为 $15^\circ \sim 27^\circ$ 的中缓倾角，走向近东西偏北（$285^\circ$）；
  - **断层南东盘（Layer_2 与 Layer_4）**：倾向稳定指向**南东向（$116.4^\circ \sim 143.5^\circ$）**，倾角为 $7^\circ \sim 11^\circ$ 的极缓倾角，走向北东（$26^\circ \sim 53^\circ$）；
  - 完美复现了断层两侧地层倾斜方向相反、被中陡倾角断层（Fault_1）错断切截的典型地质构造格局！
- **4 个地层单元详细产状表**：
  | 地层单元 | 骨架长度 | 全局宏观倾向/倾角 | 局部控制点 #1 (20%) | 局部控制点 #2 (50%) | 局部控制点 #3 (80%) | 拟合 RMSE | 构造盘区 |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
  | **Layer_1** | 1042 px | 倾向 15.5° / 倾角 16.9° | ∠18.3°∠15.6° (860.1m) | ∠18.1°∠15.6° (860.5m) | ∠14.9°∠16.8° (873.6m) | 1.27m ~ 1.64m | 断层西盘 (上部) |
  | **Layer_2** | 828 px | 倾向 138.5° / 倾角 9.4° | ∠11.0°∠143.5° (872.6m) | ∠8.3°∠135.4° (890.1m) | ∠9.1°∠137.6° (904.4m) | 0.50m ~ 1.42m | 断层东盘 (上部) |
  | **Layer_3** | 765 px | 倾向 16.2° / 倾角 23.3° | ∠19.2°∠13.5° (796.0m) | ∠24.9°∠15.7° (787.5m) | ∠27.1°∠15.5° (795.8m) | 0.48m ~ 1.99m | 断层西盘 (下部) |
  | **Layer_4** | 1145 px | 倾向 123.9° / 倾角 8.0° | ∠10.4°∠140.0° (798.1m) | ∠7.0°∠116.4° (833.0m) | ∠7.3°∠118.2° (864.7m) | 1.30m ~ 1.73m | 断层东盘 (下部) |
- **数据库已完整入库**：[`orientations.csv`](file:///Users/yiyi/lzu/项目/my-MVP/orientations.csv) 已合并保存 1 条断层（5点）与 4 组地层（12点），共计 17 个真实空间产状点；
- **三维透视成果图**：生成 4 地层三维空间切盘透视图 [`boundary_output/strata_attitude_fit.png`](file:///Users/yiyi/lzu/项目/my-MVP/boundary_output/strata_attitude_fit.png)。

#### 运行方法
```bash
# 1. 解算单条断层 (5个曲面控制点并覆盖写入 orientations.csv)
/opt/miniconda3/envs/py_course/bin/python compute_attitude.py \
    -i ./fault_output/fault_skeleton.png \
    --names Fault_1 -k 5 --no-retain

# 2. 解算4条地层边界并追加合并至 orientations.csv (每条3个控制点)
/opt/miniconda3/envs/py_course/bin/python compute_attitude.py \
    -i ./boundary_output/boundary_skeleton.png \
    --names Layer_1,Layer_2,Layer_3,Layer_4 -k 3
```

---

### 6. 桌面建模与可视化系统 (`gui_app.py`)
```bash
/opt/miniconda3/envs/py_course/bin/python gui_app.py
```
### 6. 桌面建模与可视化系统 (`gui_app.py`)
```bash
/opt/miniconda3/envs/py_course/bin/python gui_app.py
```
**操作步骤与精细度调优指南**：
1. **启动与精细度配置**：
   - 启动系统后，在左侧【建模参数】选项卡中，可直接在 **【精细度方案预设】** 下拉菜单中一键选取：
     - `⚡ 快速草稿 (20×20×20, 细化 3)`：秒级初算验证，快速校验产状与接触关系；
     - `✓ 标准平衡 (40×40×30, 细化 4)`：推荐日常交互，平衡计算速度与渲染效果；
     - `💎 高精细度 (60×60×40, 细化 5)`：地质界面光滑圆润，剖面网格细致；
     - `🔬 科研出版 (80×80×50, 细化 6)`：论文与申报材料高质量出图标准；
     - `🚀 极致超精 (100×100×60, 细化 7)`：大场景高精度等值面；
     - `⚙️ 自定义参数`：自由输入目标分辨率 $(Nx, Ny, Nz)$ 与细化等级；
   - **实时体素状态指示**：下方卡片实时显示每个体素的三维物理尺寸 $(\Delta X, \Delta Y, \Delta Z)$ 与总网格规模，并自动给出性能耗时评级；
   - **工程配置方案持久化**：点击【💾 保存配置方案...】或菜单栏【文件 -> 保存建模配置方案 (JSON)】将调参成果导出为 JSON 文件；点击【📂 载入配置方案...】随时复用；系统亦会在每次成功计算后自动记住当前参数；
2. **计算与安全预警**：
   - 点击底部 **【▶ 开始构建并计算模型】**；系统自动执行全维度参数校验；当总网格或细化等级极高时弹出贴心确认框；
3. **切换与多维分析**：
   - 切换至 **【剖面切割】** 输入起点/终点坐标切割 2D 剖面；
   - 切换至 **【虚拟钻孔】** 增添孔位并一键绘制岩性综合柱状图、导出分层 CSV；
   - 切换至 **【三维与导出】** 启动 PyVista 3D 旋转交互视窗，载入/保存 `.gempy` 完整模型，或导出真实坐标 VTK 地层曲面。

> [!NOTE]
> **macOS Cocoa 线程安全性适配**：Apple macOS AppKit 规定所有窗口创建与 OpenGL 渲染上下文（VTK `vtkCocoaRenderWindow`）必须在系统主线程（Main Thread 0）中执行。此前在子线程启动 `plotter.show()` 会直接触发 `SIGABRT`（Abort trap: 6）崩溃。现已在 `gui_app.py` 中对 `sys.platform == "darwin"` 增加主线程直接调度机制，确保在 Mac 上点击【启动交互式三维模型】时视窗能够平稳唤起并支持流畅旋转交互。

## 八、最新重大研发突破与向导式交互系统全面升级 (2026 年 9 月最新突破)

### 1. 8 步全流程分步向导交互系统 (`gui_app.py` v3.0)
系统已彻底重构为严密、直观、地学逻辑清晰的 8 步向导式图形交互流水线：
```
[步骤 1: 原始地质图输入] ──► 载入工区地质图并自适应换算物理比例尺 (1px = 2.0m)
       │
       ▼
[步骤 2: 黑白 Mask 提取] ──► 提取暗色综合线划/等高线/界线二值掩膜 (test_mask.png)
       │
       ▼
[步骤 3: 全自动连续 DEM] ──► 从 Mask 中智能分离虚线等高线，切向跨断层定向缝合、OCR 赋高程与三角剖分，完全前置自给自足
       │
       ▼
[步骤 4: 边界断层粗细分离] ──► 粗断层线 (~10px) 与细地层界线 (~2px) 骨架精准分离，滤除噪点
       │
       ▼
[步骤 5: 三维空间控制点] ──► 骨架测地有序化 + 结合 DEM 海拔提取真实空间迹线 (surface_points.csv)
       │
       ▼
[步骤 6: 地层人工交互合并] ──► 画布高亮对应连通段，将断层错断界面 (Layer_1+2 ➔ Boundary_C_P) 合并重构真实层序
       │
       ▼
[步骤 7: 产状解算与突变质检] ──► 全要素多点空间产状解算，倾向突变检测标红，集成一键地学先验校准与邻域平滑工具箱
       │
       ▼
[步骤 8: 三维建模与工程分析] ──► GemPy 隐式势场求解，2D 剖面任意切割，虚拟钻孔综合柱状图，PyVista 3D 自由交互
```

---

### 2. 地质产状解算下坡倾向方位角 180° 反转 Bug 彻底根治
- **数学与代码根因**：
  在 [`compute_attitude.py`](file:///Users/yiyi/lzu/项目/my-MVP/compute_attitude.py) 与 [`model_engine.py`](file:///Users/yiyi/lzu/项目/my-MVP/model_engine.py) 中，由空间切平面上法向量 $\vec{n} = (n_x, n_y, n_z)$（$n_z > 0$）换算地理方位角时，原代码误加了双重负号：
  $$\text{loc\_az} = \arctan2(-n_x, -n_y) \pmod{360^\circ}$$
  该公式指向的是平面的**最大仰角/上坡逆坡方向**，与真实地质倾向（**最大坡降下坡方向**）相差整整 180 度！
  导致在西北盘原本北高南低（南倾）的真实地层，硬生生被反转算成了北偏东（$15.6^\circ$）。
- **彻底根治方案**：
  全面移除多余负号，修改为严格的下坡倾向矢量方位角：
  $$\text{loc\_az} = \arctan2(n_x, n_y) \pmod{360^\circ}$$
  解算后西北盘产状直接恢复为正确的南偏西（$195.6^\circ \angle 11.3^\circ$），断层两侧反向撕裂彻底化解！

---

### 3. P 地层右下方（南东角）平切三角形外突畸变根因与地学根治
- **现象分析**：此前三维模型在南东边界处，P 地层外突出一大块尖锐的三角形“飞地”，造成模型严重失真；
- **几何与地学根因**：
  1. **单迹线 SVD 退化**：地表露头线在走向方向长达 2000 多米，但垂直走向在深度方向完全缺乏控制点。SVD 平面拟合发生病态退化，算出的倾角极其平缓（仅 $4.8^\circ$）；
  2. **山谷平切效应**：当 $4.8^\circ$ 的极缓地层面延伸至右下角低海拔冲沟谷底时，地层高程高于谷底地表高程，在等值面提取时以极大交角平切山谷，形成大面积三角形外突；
- **地学先验根治**：
  工区地质图严格遵循“北老南新”（泥盆系 D $\to$ 石炭系 C $\to$ 二叠系 P $\to$ 三叠系 T）单斜层序，地层整体必须为**南倾（约 $170^\circ \sim 190^\circ$）**且具有**中缓倾角（约 $18^\circ \sim 25^\circ$）**；
  经 GemPy 求解验证，引入南向 $20^\circ$ 先验产状后，P 地层曲面在 Y 轴的最小边界由 $-148\text{ m}$ 退缩至 $+362\text{ m}$，**三角形外突彻底消失，三维形态平整规整！**

---

### 4. 步骤 7 产状全要素展示与智能质检工具箱升级
- **全要素即时渲染**：
  打破原先仅显示断层产状的限制，步骤 7 表格与画布全量同步显示断层与所有地层（如 `Fault_1`、`Boundary_C_P`、`Boundary_D_C`）的全部产状控制点；
  - 断层点：深酒红走向线与倾向箭头；
  - 地层点：深蓝走向线与倾向箭头；
  - 异常点：自动打上黄色底框红字 `[异常]` 警告标牌；
- **异常产状一键修正工具箱**：
  - **`[🌐 一键校准地层南倾 (20°根治外突)]`**：一键将工区地层统一定向至地学标准南倾（$170^\circ \angle 20^\circ$），一键根治外突；
  - **`[🌐 应用至同地层全部点]`**：支持手工微调任一控制点后批量赋给同层全部点；
  - **`[🚀 修复异常突变 (对齐基准)]`**：以正常盘为基准自动翻转/平滑异常点；
  - **`[🔄 反转极性 (倾向翻转 180°)]`**、**`[📐 邻域平滑]`**、**`[⚖ 统一代表倾向]`**。

---

### 5. 质量保证与自动化回归测试
全套单元测试与回归套件持续 100% 全绿通过：
- [`test_guided_workflow.py`](file:///Users/yiyi/lzu/项目/my-MVP/test_guided_workflow.py)（9 项向导与突变测试）：通过（19.3s）
- [`test_comprehensive_regression.py`](file:///Users/yiyi/lzu/项目/my-MVP/test_comprehensive_regression.py)（4 组深层回归）：通过（32.5s）
- [`test_precision_and_persistence.py`](file:///Users/yiyi/lzu/项目/my-MVP/test_precision_and_persistence.py)（9 项精度测试）：通过（37.9s）


