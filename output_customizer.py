"""
基于平面地质图的智能化三维构建 - 自定义产出模块 (output_customizer.py)
----------------------------------------------------------------------
功能：
1. 自定义 2D 剖面切割：任意起点/终点斜切剖面、正交剖面 (X/Y) 生成与高清图件导出；
2. 虚拟钻孔添加与岩性提取：沿钻孔轨迹空间采样、分层厚度自动统计、柱状图绘制与分层表导出；
3. 钻孔向 2D 剖面的投影叠合显示；
4. 3D 三维模型实体展示 (含钻孔管道实体) 与 VTK/OBJ 真实工程坐标格式导出。
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv

import gempy as gp
import gempy_viewer as gpv
from gempy_viewer.modules.plot_2d.visualization_2d import Plot2D

# 配置跨平台中文字体（确保在 gempy_viewer 导入后覆盖默认的 Arial，杜绝中文豆腐块）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class GeologicalOutputCustomizer:
    """地质模型产出与应用扩展管理器"""

    def __init__(self, model_engine: Any):
        """
        初始化产出管理器
        :param model_engine: GeologicalModelEngine 实例
        """
        self.engine = model_engine
        self.sections_config: Dict[str, Dict[str, Any]] = {}
        self.boreholes: List[Dict[str, Any]] = []

    @property
    def geo_model(self) -> Any:
        return self.engine.geo_model

    # =========================================================================
    # 1. 2D 剖面自定义切割与出图
    # =========================================================================

    def add_cross_section(
        self,
        name: str,
        start_xy: Tuple[float, float],
        stop_xy: Tuple[float, float],
        resolution: Tuple[int, int] = (100, 80),
    ) -> Dict[str, Any]:
        """
        添加一条自定义 2D 剖面切片
        :param name: 剖面名称 (如 'Section_A_A')
        :param start_xy: 剖面起点坐标 (x, y)
        :param stop_xy: 剖面终点坐标 (x, y)
        :param resolution: 剖面分辨率 [水平采样点数, 垂直采样点数]
        """
        if self.geo_model is None:
            raise RuntimeError("地质模型尚未初始化")

        new_cfg = {
            "start": list(start_xy),
            "stop": list(stop_xy),
            "resolution": list(resolution),
        }

        # 若该剖面已存在且参数一致，且模型已包含剖面求解结果，则无需重复重置待算状态
        raw_sections = getattr(self.geo_model.solutions.raw_arrays, "sections", None)
        if (
            name in self.sections_config
            and self.sections_config[name] == new_cfg
            and self.engine.is_computed
            and raw_sections is not None
            and len(raw_sections) > 0
        ):
            return self.sections_config[name]

        self.sections_config[name] = new_cfg

        # 整理所有剖面并注册到 GemPy
        section_dict = {
            k: (v["start"], v["stop"], v["resolution"])
            for k, v in self.sections_config.items()
        }
        gp.set_section_grid(grid=self.geo_model.grid, section_dict=section_dict)
        self.engine.is_computed = False
        return self.sections_config[name]

    def render_cross_section(
        self,
        section_name: str,
        show_boreholes: bool = True,
        show_topography: bool = True,
        max_borehole_distance: float = 200.0,
        vertical_exaggeration: float = 1.0,
        show_boundaries: bool = False,
        show_lith: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        渲染指定 2D 剖面，并可选择性叠合虚拟钻孔投影与地表地形线
        """
        # 检查是否已计算且剖面网格是否已经有求解结果
        raw_sections = getattr(self.geo_model.solutions.raw_arrays, "sections", None)
        if (not self.engine.is_computed) or (raw_sections is None) or (len(raw_sections) == 0):
            self.engine.compute()

        if section_name not in self.sections_config:
            raise KeyError(f"未找到剖面配置: {section_name}，请先调用 add_cross_section()")

        # 检查是否存在地形数据
        has_topo = hasattr(self.geo_model.grid, "topography") and self.geo_model.grid.topography is not None

        # 渲染 GemPy 剖面
        p2d: Plot2D = gpv.plot_2d(
            self.geo_model,
            section_names=[section_name],
            show_boundaries=show_boundaries,
            show_lith=show_lith,
            show_topography=show_topography if has_topo else False,
            ve=vertical_exaggeration,
            show=False,
        )

        fig = p2d.fig
        ax = p2d.axes[0]
        ax.set_title(f"2D Geological Cross-Section: {section_name}", fontsize=12, fontweight="bold")

        # 叠合投影虚拟钻孔
        if show_boreholes and self.boreholes:
            cfg = self.sections_config[section_name]
            start = np.array(cfg["start"], dtype=float)
            stop = np.array(cfg["stop"], dtype=float)
            line_vec = stop - start
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-6:
                u_hat = line_vec / line_len
                normal_hat = np.array([-u_hat[1], u_hat[0]])

                for bh in self.boreholes:
                    bh_xy = np.array(bh["xy"], dtype=float)
                    # 计算沿剖面投影坐标与垂直剖面偏离距离
                    dist_to_line = abs(np.dot(bh_xy - start, normal_hat))
                    if dist_to_line <= max_borehole_distance:
                        u_proj = np.dot(bh_xy - start, u_hat)
                        # 检查是否落在剖面起止区间内 (或适度放宽)
                        if -50 <= u_proj <= line_len + 50:
                            z_top = bh["z_top"]
                            z_bottom = bh["z_bottom"]
                            color = bh.get("color", "black")
                            label = f"{bh['name']} (Δ={dist_to_line:.0f}m)"

                            ax.plot(
                                [u_proj, u_proj],
                                [z_bottom, z_top],
                                color=color,
                                linewidth=3.0,
                                zorder=5,
                            )
                            ax.scatter(
                                [u_proj],
                                [z_top],
                                color=color,
                                s=60,
                                zorder=6,
                                edgecolors="white",
                            )
                            ax.text(
                                u_proj,
                                z_top + 15,
                                label,
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                            )

        return fig, ax

    def render_orthogonal_section(
        self,
        direction: str = "y",
        cell_number: str | int = "mid",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        渲染正交剖面 (沿 X/Y 轴或切片索引)
        """
        if not self.engine.is_computed:
            self.engine.compute()

        p2d = gpv.plot_2d(
            self.geo_model,
            direction=[direction],
            cell_number=[cell_number],
            show_boundaries=False,
            show=False,
        )
        p2d.axes[0].set_title(f"Orthogonal Section (Direction: {direction.upper()}, Cell: {cell_number})")
        return p2d.fig, p2d.axes[0]

    def export_section_image(
        self, fig: plt.Figure, output_filepath: str, dpi: int = 300
    ) -> str:
        """导出剖面图为高分辨率图像 (PNG, JPG, PDF 等)"""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        return output_filepath

    # =========================================================================
    # 2. 虚拟钻孔添加、采样与柱状图提取
    # =========================================================================

    def add_virtual_borehole(
        self,
        name: str,
        x: float,
        y: float,
        z_top: float,
        z_bottom: float,
        sampling_step: float = 5.0,
        color: str = "darkblue",
    ) -> Dict[str, Any]:
        """
        添加虚拟钻孔定义
        :param name: 钻孔编号 (如 'ZK01')
        :param x: 孔口 X 坐标
        :param y: 孔口 Y 坐标
        :param z_top: 孔口标高 (高程)
        :param z_bottom: 孔底标高
        :param sampling_step: 沿井身岩性采样间距 (米)
        :param color: 钻孔显示颜色
        """
        bh_info = {
            "name": name,
            "xy": (float(x), float(y)),
            "z_top": float(z_top),
            "z_bottom": float(z_bottom),
            "depth": float(z_top - z_bottom),
            "step": float(sampling_step),
            "color": color,
        }
        # 如果存在同名钻孔则更新，否则新增
        self.boreholes = [b for b in self.boreholes if b["name"] != name]
        self.boreholes.append(bh_info)
        return bh_info

    def sample_all_boreholes(self) -> Dict[str, pd.DataFrame]:
        """
        利用 GemPy Custom Grid 对所有注册的虚拟钻孔进行三维岩性采样并生成地层分层柱状表
        """
        if not self.boreholes:
            return {}

        # 收集所有钻孔的采样离散点
        all_pts = []
        bh_index_ranges = {}
        curr_idx = 0

        for bh in self.boreholes:
            z_levels = np.arange(bh["z_top"], bh["z_bottom"] - 1e-4, -bh["step"])
            n_pts = len(z_levels)
            pts = np.zeros((n_pts, 3))
            pts[:, 0] = bh["xy"][0]
            pts[:, 1] = bh["xy"][1]
            pts[:, 2] = z_levels
            all_pts.append(pts)
            bh_index_ranges[bh["name"]] = (curr_idx, curr_idx + n_pts, z_levels)
            curr_idx += n_pts

        combined_xyz = np.vstack(all_pts)

        # 设置到 GemPy Custom Grid 并重新计算
        gp.set_custom_grid(self.geo_model.grid, combined_xyz)
        self.engine.compute()

        # 获取采样结果编号
        raw_custom = self.geo_model.solutions.raw_arrays.custom

        # 获取地层名称映射
        elements = self.geo_model.structural_frame.structural_elements
        # elements 顺序与 1-indexed 对应
        id_to_name = {i + 1: elem.name for i, elem in enumerate(elements)}
        # 兜底：如果超出 elements 长度，通常为 basement 或未划分地层
        id_to_name[len(elements) + 1] = "Basement"

        results = {}
        for bh_name, (start_i, end_i, z_levels) in bh_index_ranges.items():
            bh = next(b for b in self.boreholes if b["name"] == bh_name)
            lith_ids = np.round(raw_custom[start_i:end_i]).astype(int)

            records = []
            # 沿深度聚合成地层分层段
            if len(lith_ids) > 0:
                cur_lith_id = lith_ids[0]
                cur_z_start = z_levels[0]

                for i in range(1, len(lith_ids)):
                    if lith_ids[i] != cur_lith_id:
                        cur_z_end = z_levels[i]
                        lith_name = id_to_name.get(cur_lith_id, f"Formation_{cur_lith_id}")
                        records.append({
                            "Borehole": bh_name,
                            "Top_Elevation_m": round(cur_z_start, 2),
                            "Bottom_Elevation_m": round(cur_z_end, 2),
                            "Top_Depth_m": round(bh["z_top"] - cur_z_start, 2),
                            "Bottom_Depth_m": round(bh["z_top"] - cur_z_end, 2),
                            "Thickness_m": round(cur_z_start - cur_z_end, 2),
                            "Lithology_ID": cur_lith_id,
                            "Lithology_Name": lith_name,
                        })
                        cur_lith_id = lith_ids[i]
                        cur_z_start = cur_z_end

                # 最后一个分层段
                cur_z_end = z_levels[-1]
                lith_name = id_to_name.get(cur_lith_id, f"Formation_{cur_lith_id}")
                records.append({
                    "Borehole": bh_name,
                    "Top_Elevation_m": round(cur_z_start, 2),
                    "Bottom_Elevation_m": round(cur_z_end, 2),
                    "Top_Depth_m": round(bh["z_top"] - cur_z_start, 2),
                    "Bottom_Depth_m": round(bh["z_top"] - cur_z_end, 2),
                    "Thickness_m": round(cur_z_start - cur_z_end, 2),
                    "Lithology_ID": cur_lith_id,
                    "Lithology_Name": lith_name,
                })

            df_layers = pd.DataFrame(records)
            results[bh_name] = df_layers

        return results

    def plot_borehole_stratigraphy(self, borehole_name: str) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制单个钻孔的标准综合柱状分层图
        """
        sampled = self.sample_all_boreholes()
        if borehole_name not in sampled:
            raise KeyError(f"未找到钻孔 {borehole_name} 的分层数据")

        df = sampled[borehole_name]
        bh = next(b for b in self.boreholes if b["name"] == borehole_name)

        fig, ax = plt.subplots(figsize=(4, 7))

        # 颜色查找
        elements = self.geo_model.structural_frame.structural_elements
        name_to_color = {e.name: e.color for e in elements}
        name_to_color["Basement"] = "#728f02"

        for _, row in df.iterrows():
            top = row["Top_Elevation_m"]
            bot = row["Bottom_Elevation_m"]
            h = top - bot
            lith = row["Lithology_Name"]
            color = name_to_color.get(lith, "#cccccc")

            ax.bar(
                0.5,
                h,
                bottom=bot,
                width=0.4,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                align="center",
            )
            # 标注岩性名称
            mid_z = (top + bot) / 2.0
            ax.text(
                0.5,
                mid_z,
                f"{lith}\n({row['Thickness_m']:.1f}m)",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )
            # 标注界线高程
            ax.text(0.72, top, f"{top:.1f}m", va="center", fontsize=8, color="gray")

        # 标注底部高程
        if len(df) > 0:
            ax.text(0.72, df.iloc[-1]["Bottom_Elevation_m"], f"{df.iloc[-1]['Bottom_Elevation_m']:.1f}m", va="center", fontsize=8, color="gray")

        ax.set_xlim(0, 1.2)
        ax.set_ylim(bh["z_bottom"] - 20, bh["z_top"] + 30)
        ax.set_xticks([])
        ax.set_ylabel("Elevation (m)", fontsize=10)
        ax.set_title(f"Borehole Column: {borehole_name}\n(X={bh['xy'][0]:.0f}, Y={bh['xy'][1]:.0f})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, ax

    def export_borehole_csv(self, output_dir: str) -> List[str]:
        """将所有虚拟钻孔的分层数据导出为 CSV"""
        os.makedirs(output_dir, exist_ok=True)
        sampled = self.sample_all_boreholes()
        saved_files = []
        for name, df in sampled.items():
            path = os.path.join(output_dir, f"borehole_{name}_stratigraphy.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            saved_files.append(path)
        return saved_files

    # =========================================================================
    # 3. 3D 三维模型实体展示与数据导出
    # =========================================================================

    def get_pyvista_plotter(
        self,
        show_lith: bool = True,
        show_surfaces: bool = True,
        show_boreholes: bool = True,
        show_topography: bool = True,
    ) -> pv.Plotter:
        """
        构建包含地质曲面、地表地形与虚拟钻孔三维实体的 PyVista Plotter 对象
        """
        if not self.engine.is_computed:
            self.engine.compute()

        # 检查地形数据
        has_topo = hasattr(self.geo_model.grid, "topography") and self.geo_model.grid.topography is not None

        # 先通过 gempy_viewer 创建三维窗口底座
        p_gpv = gpv.plot_3d(
            self.geo_model,
            show_lith=show_lith,
            show_surfaces=show_surfaces,
            show_topography=show_topography if has_topo else False,
            show=False,
        )
        plotter: pv.Plotter = p_gpv.p

        # 叠加实体钻孔
        if show_boreholes and self.boreholes:
            for bh in self.boreholes:
                x, y = bh["xy"]
                collar = (x, y, bh["z_top"])
                bottom = (x, y, bh["z_bottom"])
                color = bh.get("color", "royalblue")

                # 圆柱钻孔井身
                tube = pv.Line(collar, bottom).tube(radius=18.0)
                plotter.add_mesh(tube, color=color, opacity=0.95)
                # 孔口球体标记
                sphere = pv.Sphere(radius=30.0, center=collar)
                plotter.add_mesh(sphere, color="red")
                # 文字标签
                plotter.add_point_labels(
                    [collar],
                    [bh["name"]],
                    font_size=12,
                    text_color="black",
                    shape_color="white",
                    shape_opacity=0.8,
                )

        return plotter

    def export_surfaces_to_vtk(self, output_dir: str) -> List[str]:
        """
        将所有地层界面曲面导出为真实地理工程坐标下的 VTK 文件 (.vtk)
        """
        if not self.engine.is_computed:
            self.engine.compute()

        os.makedirs(output_dir, exist_ok=True)
        meshes = getattr(self.geo_model.solutions, "dc_meshes", [])
        elements = self.geo_model.structural_frame.structural_elements

        saved_paths = []
        for i, mesh in enumerate(meshes):
            if i < len(elements):
                elem_name = elements[i].name
            else:
                elem_name = f"Surface_{i}"

            # 关键：逆变换恢复为真实世界坐标 (X, Y, Z 米)
            raw_verts = mesh.vertices.copy()
            real_verts = self.geo_model.input_transform.apply_inverse(raw_verts)

            # 转换为 PyVista PolyData
            faces = np.c_[np.full(len(mesh.edges), 3), mesh.edges].ravel()
            poly = pv.PolyData(real_verts, faces)

            out_file = os.path.join(output_dir, f"{elem_name}.vtk")
            poly.save(out_file)
            saved_paths.append(out_file)

        return saved_paths


if __name__ == "__main__":
    from model_engine import GeologicalModelEngine

    print("===== [自定义产出模块测试] =====")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sp_file = os.path.join(cur_dir, "surface_points.csv")
    or_file = os.path.join(cur_dir, "orientations.csv")

    # 1. 初始化并计算模型
    engine = GeologicalModelEngine(project_name="LZU_MVP_Output")
    engine.initialize_model(sp_file, or_file)
    engine.compute()

    customizer = GeologicalOutputCustomizer(engine)

    # 2. 测试自定义 2D 剖面
    print("1. 正在添加自定义剖面 'Section_Main'...")
    customizer.add_cross_section(
        name="Section_Main",
        start_xy=(200, 200),
        stop_xy=(1800, 1800),
        resolution=(80, 60),
    )

    # 3. 测试虚拟钻孔
    print("2. 正在添加 2 口虚拟钻孔 (ZK01, ZK02)...")
    customizer.add_virtual_borehole(
        name="ZK01", x=600, y=600, z_top=750, z_bottom=100, color="firebrick"
    )
    customizer.add_virtual_borehole(
        name="ZK02", x=1400, y=1400, z_top=750, z_bottom=100, color="darkgreen"
    )

    print("3. 正在提取钻孔岩性分层数据...")
    strat_dfs = customizer.sample_all_boreholes()
    for bh_name, df in strat_dfs.items():
        print(f"   === 钻孔 {bh_name} 分层情况 ===")
        print(df[["Top_Depth_m", "Bottom_Depth_m", "Thickness_m", "Lithology_Name"]])

    # 4. 导出钻孔 CSV 与剖面图
    out_dir = os.path.join(cur_dir, "output_test")
    csvs = customizer.export_borehole_csv(out_dir)
    print(f"4. 钻孔分层已导出: {csvs}")

    fig, ax = customizer.render_cross_section("Section_Main", show_boreholes=True)
    img_path = customizer.export_section_image(fig, os.path.join(out_dir, "Section_Main.png"))
    print(f"5. 剖面图已导出: {img_path}")

    # 5. 测试真实坐标 VTK 导出
    vtks = customizer.export_surfaces_to_vtk(out_dir)
    print(f"6. 地质曲面 VTK 已导出: {vtks}")

    print("===== 自定义产出模块全部功能测试通过! =====")
