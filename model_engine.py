"""
基于平面地质图的智能化三维构建 - 模型构建与计算主引擎 (model_engine.py)
----------------------------------------------------------------------
功能：
1. 读取地质界面点 (surface_points.csv) 与产状 (orientations.csv) 数据库；
2. 空间范围 (Extent)、正交网格分辨率 (Resolution [Nx, Ny, Nz]) 与八叉树细化等级 (Refinement)
   以及曲面细化等级 (Surface Refinement) 的严密校验、配置与动态调整；
3. 自动/自定义配置地层层序关系 (Series/Surfaces mapping) 与断层错断关系；
4. 调用 GemPy 隐式位势场算法求解连续地质场；
5. 提供模型精细度统计（体素尺寸、网格总数、三角网顶点/面片数）、
   JSON 工程配置方案导出/载入持久化，以及 .gempy 模型持久化与重载。
"""

import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# 解决 Pandas 3.0+ 将字符串推断为 ArrowStringArray 导致 GemPy 异常的问题
try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

import gempy as gp


class GeologicalModelEngine:
    """地质三维建模计算引擎"""

    def __init__(
        self,
        project_name: str = "Geological_MVP",
        extent: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        refinement: int = 4,
        surface_refinement: Optional[int] = None,
        octree_curvature_threshold: float = -1.0,
    ):
        """
        初始化建模引擎
        :param project_name: 项目名称
        :param extent: 空间范围 [xmin, xmax, ymin, ymax, zmin, zmax] (米)
        :param resolution: 正交网格分辨率 [nx, ny, nz]
        :param refinement: 八叉树细化等级 (通常 3~7)
        :param surface_refinement: 曲面提取细化等级 (通常同 refinement 或独立设置)
        :param octree_curvature_threshold: 曲率加密阈值 (-1.0 表示关闭)
        """
        self.project_name = project_name
        self.extent = extent or [0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0]
        self.resolution = resolution or [40, 40, 30]
        self.refinement = refinement
        self.surface_refinement = surface_refinement if surface_refinement is not None else refinement
        self.octree_curvature_threshold = octree_curvature_threshold

        # 参数合法性预检
        is_val, errs, _ = self.validate_parameters(
            extent=self.extent,
            resolution=self.resolution,
            refinement=self.refinement,
            surface_refinement=self.surface_refinement,
        )
        if not is_val:
            raise ValueError(f"初始化参数校验失败: {'; '.join(errs)}")

        self.surface_points_path: Optional[str] = None
        self.orientations_path: Optional[str] = None
        self.geo_model: Optional[Any] = None
        self.series_mapping: Optional[Dict[str, Any]] = None
        self.fault_series: Optional[List[str]] = None
        self.is_computed: bool = False
        self.last_compute_time: float = 0.0

        # 地形设置缓存
        self.topography_mode: str = "none"
        self.topography_filepath: Optional[str] = None
        self.topography_params: Optional[Dict[str, Any]] = None

    @staticmethod
    def validate_parameters(
        extent: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        refinement: Optional[int] = None,
        surface_refinement: Optional[int] = None,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        校验建模空间范围与精细度参数
        :param extent: [xmin, xmax, ymin, ymax, zmin, zmax]
        :param resolution: [nx, ny, nz]
        :param refinement: 八叉树细化等级 (1~8)
        :param surface_refinement: 曲面细化等级 (1~8)
        :return: (is_valid, errors, warnings)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. 范围校验 (Extent)
        if extent is not None:
            if not isinstance(extent, (list, tuple)) or len(extent) != 6:
                errors.append("空间范围 (Extent) 必须是包含 6 个数值的列表 [xmin, xmax, ymin, ymax, zmin, zmax]")
            else:
                try:
                    ext = [float(v) for v in extent]
                    if ext[0] >= ext[1]:
                        errors.append(f"X 轴范围非法: Xmin ({ext[0]}) 必须小于 Xmax ({ext[1]})")
                    if ext[2] >= ext[3]:
                        errors.append(f"Y 轴范围非法: Ymin ({ext[2]}) 必须小于 Ymax ({ext[3]})")
                    if ext[4] >= ext[5]:
                        errors.append(f"Z 轴范围非法: Zmin ({ext[4]}) 必须小于 Zmax ({ext[5]})")
                except (ValueError, TypeError):
                    errors.append("空间范围包含非法的非数值项")

        # 2. 正交网格分辨率校验 (Resolution)
        if resolution is not None:
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 3:
                errors.append("网格分辨率 (Resolution) 必须是包含 3 个正整数的列表 [nx, ny, nz]")
            else:
                try:
                    res = [int(v) for v in resolution]
                    if any(r <= 0 for r in res):
                        errors.append(f"网格分辨率各维度必须为大于 0 的正整数，当前输入为: {res}")
                    elif any(r < 5 for r in res):
                        errors.append(f"网格分辨率过低 (单向至少为 5)，当前输入为: {res}")
                    else:
                        total_voxels = res[0] * res[1] * res[2]
                        if total_voxels > 2000000:
                            errors.append(f"网格体素总数 ({total_voxels:,}) 超过安全上限 2,000,000，易造成内存溢出崩溃")
                        elif total_voxels > 500000:
                            warnings.append(f"网格体素总数达到 {total_voxels:,}，计算可能需要较长时间 (20~60 秒)")

                        # 极端长宽比与体素尺寸提示
                        if extent is not None and len(extent) == 6 and not errors:
                            dx = (extent[1] - extent[0]) / float(res[0])
                            dy = (extent[3] - extent[2]) / float(res[1])
                            dz = (extent[5] - extent[4]) / float(res[2])
                            min_d, max_d = min(dx, dy, dz), max(dx, dy, dz)
                            if min_d > 0 and (max_d / min_d) > 40:
                                warnings.append(
                                    f"网格体素单轴尺寸差异极大 (Δmax/Δmin = {max_d/min_d:.1f})，建议适度调整分辨率比例以贴合区域尺寸"
                                )
                except (ValueError, TypeError):
                    errors.append("网格分辨率必须全部为整数数值")

        # 3. 八叉树细化等级校验 (Refinement)
        if refinement is not None:
            try:
                ref = int(refinement)
                if ref < 1 or ref > 8:
                    errors.append(f"八叉树细化等级 (Refinement) 必须在 1~8 范围内 (工程推荐 3~7)，当前为: {ref}")
                elif ref < 3:
                    warnings.append(f"细化等级当前为 {ref}，提取出的地质曲面将较为粗糙并带有明显块状感")
                elif ref >= 7:
                    warnings.append(f"细化等级达到 {ref}，八叉树求值运算量巨大，可能耗时超过 30 秒")
            except (ValueError, TypeError):
                errors.append("细化等级 (Refinement) 必须为整数")

        # 4. 曲面提取细化等级校验 (Surface Refinement)
        if surface_refinement is not None:
            try:
                surf_ref = int(surface_refinement)
                if surf_ref < 1 or surf_ref > 8:
                    errors.append(f"曲面细化等级 (Surface Refinement) 必须在 1~8 范围内，当前为: {surf_ref}")
                elif refinement is not None:
                    ref = int(refinement)
                    if surf_ref > ref:
                        warnings.append(
                            f"曲面提取细化等级 ({surf_ref}) 高于体网格细化等级 ({ref})。"
                            f"受 GemPy 机制限制，实际生效值将被自动约束为不超过总体等级 ({ref})"
                        )
            except (ValueError, TypeError):
                errors.append("曲面细化等级必须为整数")

        return (len(errors) == 0, errors, warnings)

    def inspect_csv_files(
        self, surface_points_path: str, orientations_path: str
    ) -> Dict[str, Any]:
        """
        检查并分析输入 CSV 数据的地层与点位信息
        """
        if not os.path.exists(surface_points_path):
            raise FileNotFoundError(f"找不到界面点文件: {surface_points_path}")
        if not os.path.exists(orientations_path):
            raise FileNotFoundError(f"找不到产状文件: {orientations_path}")

        df_sp = pd.read_csv(surface_points_path, comment="#")
        df_or = pd.read_csv(orientations_path, comment="#")

        # 校验字段
        req_sp = {"X", "Y", "Z", "formation"}
        req_or = {"X", "Y", "Z", "formation"}
        if not req_sp.issubset(df_sp.columns):
            raise ValueError(f"界面点文件缺少必要字段: {req_sp - set(df_sp.columns)}")
        if not req_or.issubset(df_or.columns):
            raise ValueError(f"产状文件缺少必要字段: {req_or - set(df_or.columns)}")

        formations_sp = list(df_sp["formation"].dropna().unique())
        formations_or = list(df_or["formation"].dropna().unique())

        # 分离断层与常规地层 (名称含 fault/Fault/断层 视为断裂构造)
        fault_candidates = [
            f for f in formations_sp
            if any(k in f.lower() for k in ["fault", "断层"])
        ]
        strat_candidates = [
            f for f in formations_sp if f not in fault_candidates
        ]

        # 按平均高程从高到低对常规地层进行默认层序排序
        if strat_candidates:
            z_mean_order = (
                df_sp[df_sp["formation"].isin(strat_candidates)]
                .groupby("formation")["Z"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            z_mean_order = []

        # 空间范围统计
        x_min, x_max = float(min(df_sp["X"].min(), df_or["X"].min())), float(max(df_sp["X"].max(), df_or["X"].max()))
        y_min, y_max = float(min(df_sp["Y"].min(), df_or["Y"].min())), float(max(df_sp["Y"].max(), df_or["Y"].max()))
        z_min, z_max = float(min(df_sp["Z"].min(), df_or["Z"].min())), float(max(df_sp["Z"].max(), df_or["Z"].max()))

        # 留出 5%~10% 的工程余量并规整为整数
        dx = max((x_max - x_min) * 0.05, 50.0)
        dy = max((y_max - y_min) * 0.05, 50.0)
        dz = max((z_max - z_min) * 0.15, 50.0)

        suggested_extent = [
            float(np.floor((x_min - dx) / 50.0) * 50.0),
            float(np.ceil((x_max + dx) / 50.0) * 50.0),
            float(np.floor((y_min - dy) / 50.0) * 50.0),
            float(np.ceil((y_max + dy) / 50.0) * 50.0),
            float(np.floor((z_min - dz) / 50.0) * 50.0),
            float(np.ceil((z_max + dz) / 50.0) * 50.0),
        ]

        return {
            "surface_points_count": len(df_sp),
            "orientations_count": len(df_or),
            "formations_in_points": formations_sp,
            "formations_in_orientations": formations_or,
            "faults": fault_candidates,
            "stratigraphy": strat_candidates,
            "inferred_stratigraphy_order": z_mean_order,
            "actual_data_bounds": {
                "X": (x_min, x_max),
                "Y": (y_min, y_max),
                "Z": (z_min, z_max),
            },
            "suggested_extent": suggested_extent,
        }

    def initialize_model(
        self,
        surface_points_path: str,
        orientations_path: str,
        series_mapping: Optional[Dict[str, Any]] = None,
        fault_series: Optional[List[str]] = None,
        extent: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        refinement: Optional[int] = None,
        surface_refinement: Optional[int] = None,
        octree_curvature_threshold: Optional[float] = None,
        topography_mode: str = "none",
        topography_filepath: Optional[str] = None,
        topography_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        配置并创建 GemPy 地质模型
        :param surface_points_path: 界面控制点 CSV 路径
        :param orientations_path: 产状 CSV 路径
        :param series_mapping: 地层/断层系列映射字典
        :param fault_series: 断层系列名称列表
        :param extent: 空间范围 [xmin, xmax, ymin, ymax, zmin, zmax]
        :param resolution: 正交体块网格分辨率 [nx, ny, nz]
        :param refinement: 八叉树细化等级 (3~7)
        :param surface_refinement: 曲面提取细化等级 (3~7)
        :param octree_curvature_threshold: 曲率细化阈值
        :param topography_mode: 地形模式 ("none", "file", "random")
        :param topography_filepath: 地形文件路径
        :param topography_params: 地形参数
        """
        self.surface_points_path = surface_points_path
        self.orientations_path = orientations_path
        if extent is not None:
            self.extent = [float(x) for x in extent]
        if resolution is not None:
            self.resolution = [int(r) for r in resolution]
        if refinement is not None:
            self.refinement = int(refinement)
        if surface_refinement is not None:
            self.surface_refinement = int(surface_refinement)
        elif refinement is not None:
            self.surface_refinement = int(refinement)
        if octree_curvature_threshold is not None:
            self.octree_curvature_threshold = float(octree_curvature_threshold)

        # 全面校验
        is_val, errs, warnings = self.validate_parameters(
            extent=self.extent,
            resolution=self.resolution,
            refinement=self.refinement,
            surface_refinement=self.surface_refinement,
        )
        if not is_val:
            raise ValueError(f"建模参数校验不通过: {'; '.join(errs)}")

        # 检查数据
        info = self.inspect_csv_files(surface_points_path, orientations_path)
        strat_order = info["inferred_stratigraphy_order"]
        fault_names = info["faults"]
        sp_formations = set(info["formations_in_points"])
        or_formations = set(info["formations_in_orientations"])

        # 如果没有传入层序映射，则智能构建默认映射 (自动分离断层与地层)
        if series_mapping is None:
            self.series_mapping = {}
            if fault_names:
                # 每个断层作为一个独立的断层系列或组合
                self.series_mapping["Fault_Series"] = tuple(fault_names) if len(fault_names) > 1 else fault_names[0]
            if strat_order:
                self.series_mapping["Strat_Series"] = tuple(strat_order) if len(strat_order) > 1 else strat_order[0]
        else:
            # 清洗 series_mapping，确保其中的地层均真实存在于 surface_points 中
            sanitized_mapping = {}
            for s_name, elements in series_mapping.items():
                if isinstance(elements, (list, tuple)):
                    valid_elems = [e for e in elements if e in sp_formations]
                    if valid_elems:
                        sanitized_mapping[s_name] = tuple(valid_elems) if len(valid_elems) > 1 else valid_elems[0]
                elif isinstance(elements, str):
                    if elements in sp_formations:
                        sanitized_mapping[s_name] = elements
            self.series_mapping = sanitized_mapping

        # 自动判定断层系列
        if fault_series is not None:
            self.fault_series = [f for f in fault_series if f in self.series_mapping]
        else:
            self.fault_series = ["Fault_Series"] if "Fault_Series" in self.series_mapping else []

        # 保护性对齐与自动补全：GemPy 严苛要求每个 Stack/Series 必须同时包含 surface_points 和 orientations，缺一不可
        effective_or_path = orientations_path
        tmp_or_file = None
        extra_or_forms = [f for f in or_formations if f not in sp_formations]
        missing_or_forms = [f for f in sp_formations if f not in or_formations]

        if extra_or_forms or missing_or_forms:
            df_sp = pd.read_csv(surface_points_path, comment="#")
            df_or = pd.read_csv(orientations_path, comment="#") if os.path.exists(orientations_path) else pd.DataFrame(columns=["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"])

            # 过滤掉 surface_points 中不存在的孤立产状记录
            if len(df_or) > 0 and "formation" in df_or.columns:
                filtered_or = df_or[df_or["formation"].isin(sp_formations)].copy()
            else:
                filtered_or = pd.DataFrame(columns=["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"])

            default_az = float(df_or["azimuth"].median()) if ("azimuth" in df_or.columns and len(df_or) > 0) else 160.0
            default_dip = float(df_or["dip"].median()) if ("dip" in df_or.columns and len(df_or) > 0) else 15.0

            # 针对 orientations 中缺失的地层，自动基于其界面控制点通过平面拟合补全有效产状
            append_rows = []
            for form in missing_or_forms:
                pts = df_sp[df_sp["formation"] == form]
                if len(pts) >= 3:
                    xyz = pts[["X", "Y", "Z"]].to_numpy(dtype=float)
                    centroid = np.mean(xyz, axis=0)
                    centered = xyz - centroid
                    try:
                        _u, _s, vt = np.linalg.svd(centered)
                        n = vt[2]
                        if n[2] < 0:
                            n = -n
                        n /= (np.linalg.norm(n) + 1e-8)
                        dip_calc = float(np.degrees(np.arccos(np.clip(n[2], 0.0, 1.0))))
                        az_calc = float(np.degrees(np.arctan2(n[0], n[1])) % 360.0)
                    except Exception:
                        az_calc, dip_calc = default_az, default_dip
                    cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
                    append_rows.append({
                        "X": round(cx, 1), "Y": round(cy, 1), "Z": round(cz, 1),
                        "azimuth": round(az_calc, 1), "dip": round(dip_calc, 1),
                        "polarity": 1.0, "formation": form,
                    })
                elif len(pts) > 0:
                    cx = float(pts["X"].mean())
                    cy = float(pts["Y"].mean())
                    cz = float(pts["Z"].mean())
                    append_rows.append({
                        "X": round(cx, 1), "Y": round(cy, 1), "Z": round(cz, 1),
                        "azimuth": round(default_az, 1), "dip": round(default_dip, 1),
                        "polarity": 1.0, "formation": form,
                    })

            if append_rows:
                filtered_or = pd.concat([filtered_or, pd.DataFrame(append_rows)], ignore_index=True)

            tmp_or_fd, tmp_or_path = tempfile.mkstemp(prefix="sanitized_or_", suffix=".csv")
            os.close(tmp_or_fd)
            filtered_or.to_csv(tmp_or_path, index=False, float_format="%.1f")
            effective_or_path = tmp_or_path
            tmp_or_file = tmp_or_path

        try:
            # 创建 ImporterHelper
            importer = gp.data.ImporterHelper(
                path_to_surface_points=surface_points_path,
                path_to_orientations=effective_or_path,
                pandas_reader_kwargs={"comment": "#"},
            )

            # 构建 GeoModel (注意：传入 refinement 会初始化八叉树细化等级)
            self.geo_model = gp.create_geomodel(
                project_name=self.project_name,
                extent=self.extent,
                resolution=self.resolution,
                refinement=self.refinement,
                importer_helper=importer,
            )
        finally:
            if tmp_or_file and os.path.exists(tmp_or_file):
                try:
                    os.remove(tmp_or_file)
                except Exception:
                    pass

        # 明确同步曲面提取等级与八叉树细化等级，克服底层默认限制
        if hasattr(self.geo_model, "interpolation_options") and hasattr(
            self.geo_model.interpolation_options, "evaluation_options"
        ):
            eval_opts = self.geo_model.interpolation_options.evaluation_options
            if self.surface_refinement is not None:
                eval_opts.number_octree_levels_surface = int(self.surface_refinement)
            if self.octree_curvature_threshold is not None:
                eval_opts.octree_curvature_threshold = float(self.octree_curvature_threshold)

        # 映射堆叠层序与地层
        gp.map_stack_to_surfaces(
            gempy_model=self.geo_model, mapping_object=self.series_mapping
        )

        # 设置断层系列
        if self.fault_series:
            gp.set_is_fault(self.geo_model, self.fault_series)

        # 地形数据载入与配置
        self.topography_mode = topography_mode
        self.topography_filepath = topography_filepath
        self.topography_params = topography_params or {}

        if topography_mode == "random":
            dz = self.topography_params.get(
                "d_z",
                [self.extent[4] + (self.extent[5] - self.extent[4]) * 0.65, self.extent[5]],
            )
            fd = self.topography_params.get("fractal_dimension", 1.2)
            res = self.topography_params.get("topography_resolution", [50, 50])
            gp.set_topography_from_random(
                grid=self.geo_model.grid,
                fractal_dimension=float(fd),
                d_z=np.array(dz),
                topography_resolution=np.array(res),
            )
        elif topography_mode == "file" and topography_filepath and os.path.exists(topography_filepath):
            if topography_filepath.lower().endswith(".csv"):
                df_t = pd.read_csv(topography_filepath, comment="#")
                cols = {c.upper(): c for c in df_t.columns}
                if {"X", "Y", "Z"}.issubset(cols.keys()):
                    xyz = df_t[[cols["X"], cols["Y"], cols["Z"]]].to_numpy(dtype=float)
                    gp.set_topography_from_arrays(self.geo_model.grid, xyz)
                else:
                    raise ValueError(f"地形 CSV 文件缺少 X, Y, Z 坐标列: {list(df_t.columns)}")
            elif topography_filepath.lower().endswith(".npy"):
                # 直接解析 DEM 标高矩阵 (如 demo_auto_dem 生成的 dem_auto.npy)
                dem = np.load(topography_filepath)
                h, w = dem.shape
                scale_x = (self.extent[1] - self.extent[0]) / float(w)
                scale_y = (self.extent[3] - self.extent[2]) / float(h)
                # 均匀抽样保证点阵密集度与计算速度的最佳平衡 (约 120x120 网格)
                step_y = max(1, h // 120)
                step_x = max(1, w // 120)
                ys = np.arange(0, h, step_y)
                xs = np.arange(0, w, step_x)
                mesh_y, mesh_x = np.meshgrid(ys, xs, indexing="ij")
                X = self.extent[0] + mesh_x.ravel() * scale_x
                Y = self.extent[3] - mesh_y.ravel() * scale_y
                Z = dem[mesh_y.ravel(), mesh_x.ravel()]
                xyz = np.column_stack([X, Y, Z])
                gp.set_topography_from_arrays(self.geo_model.grid, xyz)
            else:
                gp.set_topography_from_file(self.geo_model.grid, topography_filepath)

        self.is_computed = False
        return self.geo_model

    def update_grid_resolution(self, resolution: List[int]) -> List[int]:
        """
        动态更新现有模型的正交网格分辨率并标记待重算
        :param resolution: [nx, ny, nz]
        """
        is_val, errs, _ = self.validate_parameters(extent=self.extent, resolution=resolution)
        if not is_val:
            raise ValueError(f"网格分辨率非法: {'; '.join(errs)}")

        self.resolution = [int(r) for r in resolution]
        if self.geo_model is not None and hasattr(self.geo_model, "grid"):
            if hasattr(self.geo_model.grid, "regular_grid") and self.geo_model.grid.regular_grid is not None:
                self.geo_model.grid.regular_grid.set_regular_grid(
                    extent=self.geo_model.grid.regular_grid.extent,
                    resolution=self.resolution,
                )
                if hasattr(self.geo_model.grid, "_update_values"):
                    self.geo_model.grid._update_values()
            self.is_computed = False
        return self.resolution

    def update_extent(self, extent: List[float]) -> List[float]:
        """
        动态更新现有模型的空间范围并标记待重算
        :param extent: [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        is_val, errs, _ = self.validate_parameters(extent=extent, resolution=self.resolution)
        if not is_val:
            raise ValueError(f"空间范围非法: {'; '.join(errs)}")

        self.extent = [float(x) for x in extent]
        if self.geo_model is not None and hasattr(self.geo_model, "grid"):
            if hasattr(self.geo_model.grid, "regular_grid") and self.geo_model.grid.regular_grid is not None:
                self.geo_model.grid.regular_grid.set_regular_grid(
                    extent=self.extent,
                    resolution=self.resolution,
                )
                if hasattr(self.geo_model.grid, "_update_values"):
                    self.geo_model.grid._update_values()
            self.is_computed = False
        return self.extent

    def update_refinement(
        self, refinement: int, surface_refinement: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        动态更新现有模型的八叉树细化等级及曲面细化等级并标记待重算
        :param refinement: 八叉树等级 (3~7)
        :param surface_refinement: 曲面提取等级 (3~7)
        """
        surf = surface_refinement if surface_refinement is not None else refinement
        is_val, errs, _ = self.validate_parameters(
            refinement=refinement, surface_refinement=surf
        )
        if not is_val:
            raise ValueError(f"细化等级非法: {'; '.join(errs)}")

        self.refinement = int(refinement)
        self.surface_refinement = int(surf)

        if self.geo_model is not None and hasattr(self.geo_model, "interpolation_options"):
            eval_opts = self.geo_model.interpolation_options.evaluation_options
            eval_opts.number_octree_levels = self.refinement
            eval_opts.number_octree_levels_surface = self.surface_refinement
            self.is_computed = False
        return (self.refinement, self.surface_refinement)

    def compute(self, add_random_topography: bool = False) -> Dict[str, Any]:
        """
        计算地质隐式位势场模型并返回详尽的计算指标与精细度参数
        """
        if self.geo_model is None:
            raise RuntimeError("地质模型尚未初始化，请先调用 initialize_model()")

        # 可选增加地形起伏
        if add_random_topography:
            z_mid = (self.extent[4] + self.extent[5]) / 2.0
            z_max = self.extent[5]
            gp.set_topography_from_random(
                grid=self.geo_model.grid,
                fractal_dimension=1.2,
                d_z=np.array([z_mid, z_max]),
                topography_resolution=np.array([50, 50]),
            )

        start_t = time.time()
        gp.compute_model(self.geo_model)
        self.last_compute_time = time.time() - start_t
        self.is_computed = True

        lith_shape = (
            self.geo_model.solutions.raw_arrays.lith_block.shape
            if hasattr(self.geo_model.solutions.raw_arrays, "lith_block")
            else None
        )
        meshes = getattr(self.geo_model.solutions, "dc_meshes", [])
        mesh_count = len(meshes) if meshes else 0
        total_verts = sum(len(m.vertices) for m in meshes) if meshes else 0
        total_edges = sum(len(m.edges) for m in meshes) if meshes else 0

        # 体素尺寸计算 (单位: 米)
        dx = (self.extent[1] - self.extent[0]) / float(self.resolution[0])
        dy = (self.extent[3] - self.extent[2]) / float(self.resolution[1])
        dz = (self.extent[5] - self.extent[4]) / float(self.resolution[2])
        total_voxels = int(self.resolution[0] * self.resolution[1] * self.resolution[2])

        return {
            "success": True,
            "compute_time_seconds": round(self.last_compute_time, 3),
            "resolution": list(self.resolution),
            "total_voxels": total_voxels,
            "voxel_size": [round(dx, 2), round(dy, 2), round(dz, 2)],
            "refinement": self.refinement,
            "surface_refinement": self.surface_refinement,
            "lith_block_shape": lith_shape,
            "surfaces_mesh_count": mesh_count,
            "total_mesh_vertices": total_verts,
            "total_mesh_edges": total_edges,
            "structural_elements": [
                e.name for e in self.geo_model.structural_frame.structural_elements
            ],
        }

    def export_config(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        将当前建模工程全部参数（范围、分辨率、细化等级、数据路径等）导出为标准 JSON 配置文件
        :param filepath: 目标保存路径 (.json)
        :param extra_meta: 附加元数据（如预设名称、备注等）
        :return: 保存文件的绝对路径
        """
        config = {
            "project_name": self.project_name,
            "extent": list(self.extent),
            "resolution": list(self.resolution),
            "refinement": int(self.refinement),
            "surface_refinement": int(self.surface_refinement),
            "octree_curvature_threshold": float(self.octree_curvature_threshold),
            "surface_points_path": self.surface_points_path,
            "orientations_path": self.orientations_path,
            "series_mapping": self.series_mapping,
            "fault_series": self.fault_series,
            "topography": {
                "mode": self.topography_mode,
                "filepath": self.topography_filepath,
                "params": self.topography_params,
            },
            "meta": extra_meta or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return os.path.abspath(filepath)

    def load_config(self, filepath: str) -> Dict[str, Any]:
        """
        从 JSON 配置文件加载建模参数并更新引擎配置
        :param filepath: JSON 配置文件路径
        :return: 读取的配置字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "extent" in config:
            self.extent = [float(x) for x in config["extent"]]
        if "resolution" in config:
            self.resolution = [int(r) for r in config["resolution"]]
        if "refinement" in config:
            self.refinement = int(config["refinement"])
        if "surface_refinement" in config:
            self.surface_refinement = int(config["surface_refinement"])
        elif "refinement" in config:
            self.surface_refinement = int(config["refinement"])
        if "octree_curvature_threshold" in config:
            self.octree_curvature_threshold = float(config["octree_curvature_threshold"])
        if "project_name" in config:
            self.project_name = config["project_name"]
        if "surface_points_path" in config:
            self.surface_points_path = config["surface_points_path"]
        if "orientations_path" in config:
            self.orientations_path = config["orientations_path"]
        if "series_mapping" in config:
            self.series_mapping = config["series_mapping"]
        if "fault_series" in config:
            self.fault_series = config["fault_series"]
        if "topography" in config and isinstance(config["topography"], dict):
            topo = config["topography"]
            self.topography_mode = topo.get("mode", "none")
            self.topography_filepath = topo.get("filepath", None)
            self.topography_params = topo.get("params", None)

        # 校验载入参数
        is_val, errs, _ = self.validate_parameters(
            extent=self.extent,
            resolution=self.resolution,
            refinement=self.refinement,
            surface_refinement=self.surface_refinement,
        )
        if not is_val:
            raise ValueError(f"载入的配置参数非法: {'; '.join(errs)}")

        return config

    def save_model(self, output_path: str) -> str:
        """保存模型至 .gempy 格式二进制文件"""
        if not self.is_computed or self.geo_model is None:
            raise RuntimeError("模型尚未计算，无法保存")
        gp.save_model(self.geo_model, path=output_path)
        return output_path

    def load_model(self, model_path: str) -> Any:
        """从 .gempy 文件加载已有模型并同步引擎状态"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        self.geo_model = gp.load_model(model_path)
        self.is_computed = True

        # 自动将已保存模型的 Extent、Resolution 与 Refinement 反显同步到引擎变量中
        try:
            if hasattr(self.geo_model.grid, "regular_grid") and self.geo_model.grid.regular_grid is not None:
                self.resolution = [int(r) for r in self.geo_model.grid.regular_grid.resolution]
                self.extent = [float(e) for e in self.geo_model.grid.regular_grid.extent]
            if hasattr(self.geo_model, "interpolation_options") and hasattr(
                self.geo_model.interpolation_options, "evaluation_options"
            ):
                eval_opts = self.geo_model.interpolation_options.evaluation_options
                self.refinement = int(getattr(eval_opts, "number_octree_levels", self.refinement))
                self.surface_refinement = int(
                    getattr(eval_opts, "number_octree_levels_surface", self.refinement)
                )
        except Exception:
            pass

        return self.geo_model

    def get_summary(self) -> Dict[str, Any]:
        """获取当前模型摘要信息与精细度指标"""
        if self.geo_model is None:
            return {"status": "Not initialized"}

        elements = [
            {"name": e.name, "id": int(e.id), "color": str(e.color)}
            for e in self.geo_model.structural_frame.structural_elements
        ]
        dx = (self.extent[1] - self.extent[0]) / float(self.resolution[0])
        dy = (self.extent[3] - self.extent[2]) / float(self.resolution[1])
        dz = (self.extent[5] - self.extent[4]) / float(self.resolution[2])
        total_voxels = int(self.resolution[0] * self.resolution[1] * self.resolution[2])

        meshes = getattr(self.geo_model.solutions, "dc_meshes", [])
        total_verts = sum(len(m.vertices) for m in meshes) if meshes else 0
        total_edges = sum(len(m.edges) for m in meshes) if meshes else 0

        return {
            "project_name": self.project_name,
            "extent": self.extent,
            "resolution": self.resolution,
            "total_voxels": total_voxels,
            "voxel_size": [round(dx, 2), round(dy, 2), round(dz, 2)],
            "refinement": self.refinement,
            "surface_refinement": self.surface_refinement,
            "is_computed": self.is_computed,
            "last_compute_time": self.last_compute_time,
            "elements": elements,
            "surfaces_mesh_count": len(meshes) if meshes else 0,
            "total_mesh_vertices": total_verts,
            "total_mesh_edges": total_edges,
            "active_grids": str(self.geo_model.grid.active_grids),
        }


if __name__ == "__main__":
    print("===== [GemPy 建模主引擎精细度与参数测试] =====")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sp_file = os.path.join(cur_dir, "surface_points.csv")
    or_file = os.path.join(cur_dir, "orientations.csv")

    engine = GeologicalModelEngine(
        project_name="LZU_MVP_Demo",
        resolution=[40, 40, 30],
        refinement=4,
        surface_refinement=4,
    )
    print("1. 正在解析 CSV 数据...")
    info = engine.inspect_csv_files(sp_file, or_file)
    print(f"   识别地层: {info['inferred_stratigraphy_order']}")
    print(f"   点位总数: {info['surface_points_count']} 点, {info['orientations_count']} 产状")

    print("2. 正在初始化模型与层序关系 (Resolution=[40, 40, 30], Refinement=4)...")
    engine.initialize_model(sp_file, or_file)

    print("3. 开始执行隐式三维地质场计算...")
    res = engine.compute()
    print(f"   计算完成! 耗时: {res['compute_time_seconds']}s")
    print(f"   正交网格: {res['resolution']}, 体素总数: {res['total_voxels']:,}, 体素尺寸: {res['voxel_size']}m")
    print(f"   八叉树细化: 等级={res['refinement']}, 曲面等级={res['surface_refinement']}")
    print(f"   曲面三角网: {res['surfaces_mesh_count']} 个界面, 共 {res['total_mesh_vertices']} 顶点, {res['total_mesh_edges']} 边")

    # 测试配置导出与重载
    tmp_cfg = os.path.join(cur_dir, "scratch", "test_config_export.json")
    saved_cfg = engine.export_config(tmp_cfg, extra_meta={"preset": "标准平衡"})
    print(f"4. 测试配置导出成功: {saved_cfg}")
    loaded_cfg = engine.load_config(tmp_cfg)
    print(f"   测试配置读取成功: Resolution={loaded_cfg['resolution']}, Refinement={loaded_cfg['refinement']}")

    print("===== 测试完成，引擎运行正常! =====")
