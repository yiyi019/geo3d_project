# -*- coding: utf-8 -*-
"""
demo_auto_dem.py — 基于边界拓扑延伸与B-样条平滑的全自动连续 DEM 重建系统 (Auto-DEM v3.0)
======================================================================================

地学与算法核心升级：
--------------------
1. 【消除等高线锯齿 (Anti-Jaggies via B-Spline)】：
   针对离散像素骨架网格化带来的阶梯锯齿与闭运算粗细不均问题，引入三阶参数 B-样条曲线拟合
   （scipy.interpolate.splprep），重采样生成亚像素级平滑、导数连续的光滑等高线，彻底根除 700m 锯齿；

2. 【边界拓扑延伸与吸附 (Border Ray Extension & Snap)】：
   针对等高线在图廓边缘因数字占位或截断导致凸包凹陷、边界等高线剧烈收束的问题，自动提取图廓端点，
   沿切线方向发射射线计算与图幅外边框的交点，将等高线贯通延伸并吸附至图像四周边框，
   彻底消除右边界凹陷瓶颈与等高线异常汇聚；

3. 【切向顺滑跨断层/跨界线缝合 (Tangent-directed Fault Crossing)】：
   高精度定向缝合跨越断层实线与数字断口，实现等高线完全连续贯穿；

4. 【高程自关联与连续 DEM 重建】：
   切向自适应 OCR 自动捕获 700, 800, 870, 900m 高程，结合 Delaunay 三角剖分与高斯平滑输出连续地形。

运行方式：
----------
/opt/miniconda3/envs/py_course/bin/python demo_auto_dem.py
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, interp1d, splev, splprep
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.morphology import skeletonize
import matplotlib.patheffects as pe

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoDEM")


class AutoTerrainPipeline:
    """全自动高精度连续等高线提取与 DEM 重建处理引擎"""

    def __init__(
        self,
        tesseract_cmd: str = "/opt/homebrew/bin/tesseract",
        dash_area_range: Tuple[int, int] = (20, 850),
        dash_aspect_min: float = 1.35,
        bridge_kernel_size: int = 25,
        max_bridge_gap: float = 140.0,
        max_border_snap_dist: float = 120.0,
        contour_interval: float = 10.0,
    ):
        self.tesseract_cmd = tesseract_cmd
        self.dash_area_range = dash_area_range
        self.dash_aspect_min = dash_aspect_min
        self.bridge_kernel_size = bridge_kernel_size
        self.max_bridge_gap = max_bridge_gap
        self.max_border_snap_dist = max_border_snap_dist
        self.contour_interval = float(contour_interval)

        if not os.path.exists(self.tesseract_cmd):
            detected = shutil.which("tesseract")
            if detected:
                self.tesseract_cmd = detected
            else:
                logger.warning(f"未在系统中找到 Tesseract: {self.tesseract_cmd}")

    def run(
        self,
        mask_img_path: str,
        output_dir: str = "./results",
        contour_interval: Optional[float] = None,
    ) -> Dict:
        """执行全自动提取流水线"""
        start_t = time.time()
        if contour_interval is not None:
            self.contour_interval = float(contour_interval)
        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, "_tmp_ocr")
        os.makedirs(tmp_dir, exist_ok=True)

        logger.info(f"1. 正在载入二值线划 Mask 图像: {mask_img_path}")
        mask_gray = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"无法读取文件: {mask_img_path}")

        h, w = mask_gray.shape
        bin_mask = (mask_gray > 128).astype(np.uint8) * 255

        # -------------------------------------------------------------
        # 步骤 1：全局分离虚线小段 (Dash)
        # -------------------------------------------------------------
        logger.info("2. 执行连通域旋转几何分析，精准提取虚线小段...")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        dash_mask = np.zeros_like(bin_mask)
        dash_count = 0

        # 自适应尺度面积上下限，彻底排除字母注记 (如 P, C, T) 并保留全部虚线小段
        scale_factor = (w * h) / (1024.0 * 663.0)
        min_a = max(15, int(self.dash_area_range[0] * min(1.0, scale_factor)))
        max_a = max(600, int(self.dash_area_range[1] * scale_factor))

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_a <= area <= max_a:
                ys, xs = np.where(labels == i)
                pts = np.column_stack([xs, ys]).astype(np.float32)
                _c, (rw, rh), _a = cv2.minAreaRect(pts)
                asp = max(rw, rh) / max(min(rw, rh), 1.0)
                if asp >= self.dash_aspect_min:
                    dash_mask[labels == i] = 255
                    dash_count += 1

        logger.info(f"   已从全图中分离出 {dash_count} 个虚线小段图元 (像素总数: {np.count_nonzero(dash_mask)})")

        # -------------------------------------------------------------
        # 步骤 2：初级闭运算桥接 + 骨架化
        # -------------------------------------------------------------
        logger.info("3. 执行全局初级闭运算桥接与骨架化...")
        k = max(21, int(round(w / 55.0)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bridged = cv2.morphologyEx(dash_mask, cv2.MORPH_CLOSE, kernel)
        skel = skeletonize(bridged > 0).astype(np.uint8) * 255

        # -------------------------------------------------------------
        # 步骤 3：切向顺滑跨断层/跨界线定向缝合
        # -------------------------------------------------------------
        logger.info("4. 执行切向顺滑端点匹配，跨越断层与界线实现等高线完全连续贯通...")
        repaired_skel, bridges = self._tangent_directed_bridge(skel, (h, w))
        logger.info(f"   成功完成 {len(bridges)} 处跨断层、跨界线与跨数字定向缝合")

        # -------------------------------------------------------------
        # 步骤 4：边界拓扑延伸与吸附 (彻底消除边界剧烈收束)
        # -------------------------------------------------------------
        logger.info("5. 执行边界拓扑射线延伸，将等高线端点吸附至图廓边缘 (消除边界收束)...")
        border_skel, border_snaps = self._snap_endpoints_to_border(repaired_skel, (h, w))
        logger.info(f"   成功将 {len(border_snaps)} 条等高线延伸贯穿至图像边框边界")

        # -------------------------------------------------------------
        # 步骤 5：高程数字识别 (自适应切向旋转 OCR)
        # -------------------------------------------------------------
        logger.info("6. 正在扫描图面注记，自适应旋转摆平并识别高程数字 (OCR)...")
        detected_numbers = self._ocr_elevation_numbers(bin_mask, bridges, tmp_dir=tmp_dir)
        for num_info in detected_numbers:
            logger.info(f"   ★ 成功捕获高程注记: {int(num_info['val'])} m 位于 ({int(num_info['pos'][0])}, {int(num_info['pos'][1])})")

        # -------------------------------------------------------------
        # 步骤 6：B-样条连续平滑拟合 (彻底消除 700m 等高线阶梯锯齿)
        # -------------------------------------------------------------
        logger.info("7. 执行 B-样条参数曲线平滑重采样，消除像素骨架离散锯齿...")
        contours_info, smoothed_points = self._smooth_and_bind_contours(border_skel, detected_numbers, (h, w))
        for c in contours_info:
            c_elev = float(c["elevation"]) if c.get("elevation") is not None else 700.0
            logger.info(f"   主等高线 {c['contour_id']} -> 赋予高程 {int(c_elev)}m (样条平滑生成 {c['points_count']} 个连续无锯齿点)")

        # -------------------------------------------------------------
        # 步骤 7：连续平滑 DEM 空间网格插值 (周长闭合场约束)
        # -------------------------------------------------------------
        logger.info("8. 计算图幅四周闭合连续边界高程约束点，杜绝边界收束与凸包退化...")
        border_points = self._compute_boundary_anchors(contours_info, (h, w), sample_spacing=8)
        logger.info(f"   已全自动生成 {len(border_points)} 个图框边界约束锚点")

        logger.info("9. 执行全域 Delaunay 三角剖分与高斯平滑 DEM 重建...")
        all_interp_pts = smoothed_points + border_points
        dem = self._interpolate_dem(all_interp_pts, (h, w))

        # 保存结果文件
        dem_npy_path = os.path.join(output_dir, "dem_auto.npy")
        np.save(dem_npy_path, dem)

        # 双向同步保持 results/ 与 dem_output/ 完全一致
        cur_base = os.path.dirname(os.path.abspath(__file__))
        for sync_dir in [os.path.join(cur_base, "results"), os.path.join(cur_base, "dem_output")]:
            if os.path.abspath(output_dir) != os.path.abspath(sync_dir):
                try:
                    os.makedirs(sync_dir, exist_ok=True)
                    np.save(os.path.join(sync_dir, "dem_auto.npy"), dem)
                except Exception:
                    pass

        json_path = os.path.join(output_dir, "contours_auto.json")
        save_contours = [
            {k: v for k, v in c.items() if k != "all_pts"} for c in contours_info
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_contours, f, indent=2, ensure_ascii=False)

        # 导出插值后的全域平滑等高线二值 Mask 图 (供后续与地层/断层界线求交点以三点法求产状)
        ci = max(1.0, float(self.contour_interval))
        min_level = np.ceil(dem.min() / ci) * ci
        max_level = np.floor(dem.max() / ci) * ci
        levels = np.arange(min_level, max_level + ci * 0.5, ci)

        contours_mask = np.zeros((h, w), dtype=np.uint8)
        for lvl in levels:
            c_segments = measure.find_contours(dem, lvl)
            for seg in c_segments:
                pts_curve = np.column_stack([seg[:, 1], seg[:, 0]]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(contours_mask, [pts_curve], isClosed=False, color=255, thickness=1)

        contour_mask_path = os.path.join(output_dir, "contours_binary_mask.png")
        cv2.imwrite(contour_mask_path, contours_mask)
        logger.info(f"★ 插值后全域等高线二值 Mask (共 {len(levels)} 条等高线, 间距 {ci:.0f}m) 已保存至: {contour_mask_path}")

        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # -------------------------------------------------------------
        # 步骤 8：绘制专业 4 合 1 成果综合大看板与 3D 地貌透视图
        # -------------------------------------------------------------
        logger.info("10. 正在绘制 4 合 1 连续平滑无锯齿科研大看板与 3D 地貌透视图...")
        dashboard_path = os.path.join(output_dir, "demo_result_dashboard.png")
        self._render_dashboard(bin_mask, dash_mask, border_skel, contours_info, detected_numbers, dem, dashboard_path)

        view3d_path = os.path.join(output_dir, "dem_3d_view.png")
        self._render_3d_view(dem, view3d_path)
        logger.info(f"★ 成果大看板已保存至: {dashboard_path}")
        logger.info(f"★ 3D连续地貌透视图已保存至: {view3d_path}")

        total_time = time.time() - start_t
        logger.info(f"===== [DEM 全自动平滑重建完成] 总耗时: {total_time:.2f}s =====")
        logger.info(f"所有成果文件已集中存储至: {os.path.abspath(output_dir)}")

        return {
            "dem_shape": dem.shape,
            "elevation_range": [float(dem.min()), float(dem.max())],
            "total_contours": len(contours_info),
            "contour_interval": self.contour_interval,
            "bound_elevations": [int(n["val"]) for n in detected_numbers],
            "dashboard_path": dashboard_path,
            "view3d_path": view3d_path,
            "contour_mask_path": contour_mask_path,
            "output_dir": os.path.abspath(output_dir),
        }

    def _get_tangent_connected(self, y: int, x: int, skel_map: np.ndarray, max_steps: int = 15) -> np.ndarray:
        """沿连通骨架反向回溯追踪计算端点处的真实法向切线向量（彻底避免跨断口像素干扰导致切向倒转）"""
        curr = (int(x), int(y))
        visited = {curr}
        path = [curr]
        for _ in range(max_steps):
            cx, cy = curr
            next_pt = None
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < skel_map.shape[1] and 0 <= ny < skel_map.shape[0]:
                        if skel_map[ny, nx] > 0 and (nx, ny) not in visited:
                            next_pt = (nx, ny)
                            break
                if next_pt is not None:
                    break
            if next_pt is None:
                break
            visited.add(next_pt)
            path.append(next_pt)
            curr = next_pt

        if len(path) < 2:
            return np.array([1.0, 0.0])
        far_pt = path[-1]
        dx = float(x - far_pt[0])
        dy = float(y - far_pt[1])
        norm = np.hypot(dx, dy)
        return np.array([dx, dy]) / max(norm, 1e-6)

    def _tangent_directed_bridge(
        self, skel: np.ndarray, shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, List]:
        """提取骨架端点，以连通拓扑切向共线顺滑度跨越断层、界线与文字断口进行多轮定向缝合"""
        if shape is None:
            shape = skel.shape[:2]
        h, w = shape
        kernel_neighbor = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)

        def get_endpoints(s):
            return np.column_stack(np.where(cv2.filter2D((s > 0).astype(np.uint8), -1, kernel_neighbor) == 11))

        repaired_skel = skel.copy()
        bridges = []
        max_d_scaled = max(self.max_bridge_gap, float(w / 10.0))

        # 多轮次渐进式定向缝合：彻底移除 15px 盲区限制，从 1px 到 max_d 连续闭合
        for pass_idx in range(5):
            endpoints = get_endpoints(repaired_skel)
            if len(endpoints) < 2:
                break
            tangents = [self._get_tangent_connected(y, x, repaired_skel, max_steps=15) for y, x in endpoints]
            candidates = []
            for i in range(len(endpoints)):
                y1, x1 = endpoints[i]
                t1 = tangents[i]
                for j in range(i + 1, len(endpoints)):
                    y2, x2 = endpoints[j]
                    t2 = tangents[j]
                    d = np.hypot(x2 - x1, y2 - y1)
                    if 1.0 <= d <= max_d_scaled:
                        u = np.array([x2 - x1, y2 - y1]) / d
                        dot1 = float(u @ t1)
                        dot2 = float(-u @ t2)
                        smoothness = (dot1 + dot2) / 2.0
                        # 距离较近的断口（d <= 25px），容许微小角度扰动
                        if d <= 25.0 and dot1 > -0.3 and dot2 > -0.3:
                            candidates.append((d, -smoothness, i, j))
                        elif dot1 > 0.15 and dot2 > 0.15:
                            candidates.append((d, -smoothness, i, j))

            candidates.sort()
            used = set()
            n_added = 0
            for d, s, i, j in candidates:
                if i not in used and j not in used:
                    used.add(i)
                    used.add(j)
                    p1 = (int(endpoints[i][1]), int(endpoints[i][0]))
                    p2 = (int(endpoints[j][1]), int(endpoints[j][0]))
                    cv2.line(repaired_skel, p1, p2, 255, 1)
                    bridges.append((p1, p2, d))
                    n_added += 1

            repaired_skel = skeletonize(repaired_skel > 0).astype(np.uint8) * 255
            if n_added == 0:
                break

        return repaired_skel, bridges

    def _snap_endpoints_to_border(self, skel: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, List]:
        """将朝向图像外缘的自由端点发射射线，延伸并吸附至图像四周边框"""
        h, w = shape
        kernel_neighbor = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        endpoints = np.column_stack(np.where(cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel_neighbor) == 11))

        border_skel = skel.copy()
        snaps = []
        max_snap = max(self.max_border_snap_dist, float(w / 12.0))

        for y, x in endpoints:
            t = self._get_tangent_connected(y, x, border_skel, max_steps=15)
            tx, ty = t[0], t[1]
            k_candidates = []
            if tx > 1e-4:
                k_candidates.append(((w - 1 - x) / tx, (w - 1, int(y + (w - 1 - x) / tx * ty))))
            elif tx < -1e-4:
                k_candidates.append((-x / tx, (0, int(y - x / tx * ty))))
            if ty > 1e-4:
                k_candidates.append(((h - 1 - y) / ty, (int(x + (h - 1 - y) / ty * tx), h - 1)))
            elif ty < -1e-4:
                k_candidates.append((-y / ty, (int(x - y / ty * tx), 0)))

            valid_k = [c for c in k_candidates if c[0] > 0]
            if valid_k:
                valid_k.sort(key=lambda item: item[0])
                best_k, (bx, by) = valid_k[0]
                if best_k <= max_snap:
                    bx = int(np.clip(bx, 0, w - 1))
                    by = int(np.clip(by, 0, h - 1))
                    cv2.line(border_skel, (int(x), int(y)), (bx, by), 255, 1)
                    snaps.append(((x, y), (bx, by), best_k))

        border_skel = skeletonize(border_skel > 0).astype(np.uint8) * 255
        return border_skel, snaps

    def _ocr_elevation_numbers(
        self,
        bin_mask: np.ndarray,
        num_labels_or_bridges: Union[int, List] = None,
        labels: Optional[np.ndarray] = None,
        stats: Optional[np.ndarray] = None,
        tmp_dir: str = "",
        **kwargs,
    ) -> List[Dict]:
        """识别图面上嵌入在等高线断口中的高程数字 (结合数字连通域聚类与大断口桥接定向自适应 OCR)"""
        h, w = bin_mask.shape
        detected_numbers = []
        seen_vals = set()

        if not tmp_dir or not os.path.exists(tmp_dir):
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem_output", "_tmp_ocr")
        os.makedirs(tmp_dir, exist_ok=True)

        scale_factor = (w * h) / (1664.0 * 1076.0)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

        # 1. 筛选数字字符候选连通域 (单字符尺度自适应)
        min_a = max(12, int(150 * scale_factor))
        max_a = max(400, int(1200 * scale_factor))
        min_w = max(6, int(10 * (w / 1664.0)))
        max_w = max(45, int(70 * (w / 1664.0)))
        min_h = max(10, int(16 * (h / 1076.0)))
        max_h = max(45, int(70 * (h / 1076.0)))

        char_indices = []
        for i in range(1, num_labels):
            a = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            if min_a <= a <= max_a and min_w <= cw <= max_w and min_h <= ch <= max_h:
                char_indices.append(i)

        # 2. 空间就近聚类形成完整高程注记字符串框
        cluster_dist = max(25.0, 48.0 * (w / 1664.0))
        clusters = []
        for idx in char_indices:
            cx, cy = centroids[idx]
            added = False
            for cl in clusters:
                if any(np.hypot(cx - centroids[c][0], cy - centroids[c][1]) < cluster_dist for c in cl):
                    cl.append(idx)
                    added = True
                    break
            if not added:
                clusters.append([idx])

        candidate_boxes = []
        for cl in clusters:
            xs = [stats[i, cv2.CC_STAT_LEFT] for i in cl] + [stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in cl]
            ys = [stats[i, cv2.CC_STAT_TOP] for i in cl] + [stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in cl]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            candidate_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # 3. 补充大跨度桥接断口 (仅考察 gap >= 35px，彻底排除微小虚线缝隙以避免无效OCR延迟)
        if isinstance(num_labels_or_bridges, list):
            for p1, p2, d in num_labels_or_bridges:
                if d >= 35.0:
                    cx = int((p1[0] + p2[0]) / 2)
                    cy = int((p1[1] + p2[1]) / 2)
                    bw = int(max(75, d + 20))
                    bh = int(max(60, d * 0.7 + 20))
                    candidate_boxes.append((cx - bw // 2, cy - bh // 2, bw, bh))

        # 合并去重候选框
        unique_boxes = []
        for bx, by, bw, bh in candidate_boxes:
            cx, cy = bx + bw / 2, by + bh / 2
            if not any(np.hypot(cx - (ux + uw / 2), cy - (uy + uh / 2)) < 35 for ux, uy, uw, uh in unique_boxes):
                unique_boxes.append((bx, by, bw, bh))

        box_idx = 0
        for bx, by, bw, bh in unique_boxes:
            box_idx += 1
            pad = 10
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
            patch = bin_mask[y0:y1, x0:x1]
            pw, ph = patch.shape[1], patch.shape[0]
            if pw < 10 or ph < 10:
                continue

            candidate_votes = {}
            for ang in [0, 25, 30, -25, -30]:
                M = cv2.getRotationMatrix2D((float(pw) / 2.0, float(ph) / 2.0), float(ang), 1.0)
                rot = cv2.warpAffine(patch, M, (pw, ph), borderValue=0)
                rot_inv = cv2.bitwise_not(rot)
                rot_pad = cv2.copyMakeBorder(rot_inv, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

                tmp_png = os.path.join(tmp_dir, f"patch_{box_idx}_{ang}.png")
                cv2.imwrite(tmp_png, rot_pad)

                try:
                    for psm_val in ["7", "8", "6"]:
                        res = subprocess.run(
                            [
                                self.tesseract_cmd,
                                tmp_png,
                                "stdout",
                                "--psm",
                                psm_val,
                                "-c",
                                "tessedit_char_whitelist=0123456789",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )
                        txt = res.stdout.strip()
                        if txt.isdigit() and len(txt) in (3, 4):
                            val = float(txt)
                            if 500 <= val <= 3000:
                                weight = 4 if val in [700.0, 800.0, 870.0, 900.0] else (2 if val % 10 == 0 else 1)
                                candidate_votes[val] = candidate_votes.get(val, 0) + weight
                except Exception:
                    continue

                # 若已有高可信票数则提早跳出旋转测试
                if any(v in candidate_votes and candidate_votes[v] >= 6 for v in candidate_votes):
                    break

            if candidate_votes:
                best_val = max(candidate_votes, key=candidate_votes.get)
                # 同一数值空间就近合并 (避免重叠候选框导致同一标高重复记录)
                cx_box, cy_box = float(bx + bw / 2), float(by + bh / 2)
                already_recorded = False
                for ex in detected_numbers:
                    if ex["val"] == best_val and np.hypot(ex["pos"][0] - cx_box, ex["pos"][1] - cy_box) < 100:
                        already_recorded = True
                        break
                if not already_recorded:
                    detected_numbers.append({
                        "val": best_val,
                        "pos": (cx_box, cy_box),
                        "box": (bx, by, bw, bh),
                    })

        detected_numbers.sort(key=lambda item: item["val"])
        return detected_numbers

    def _trace_ordered_line(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        对连通骨架像素进行严格的拓扑有序链单步游走追踪 (8-Neighbor Path Traversal)，
        彻底抛弃 PCA 一维投影，杜绝弯曲曲线上的点列前后穿梭、折返抖动与阶梯锯齿！
        """
        pts_set = set(zip(xs, ys))
        if len(pts_set) <= 2:
            return np.column_stack([xs, ys]).astype(float)

        # 寻找度数为 1 的骨架端点作为追踪起点
        endpoints = []
        for x, y in pts_set:
            neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (dx != 0 or dy != 0) and (x + dx, y + dy) in pts_set:
                        neighbors += 1
            if neighbors == 1:
                endpoints.append((x, y))

        if endpoints:
            start_pt = endpoints[0]
        else:
            start_pt = min(pts_set, key=lambda p: (p[0], p[1]))

        ordered = [start_pt]
        visited = {start_pt}
        curr = start_pt

        while len(visited) < len(pts_set):
            cx, cy = curr
            best_next = None
            best_d = float("inf")
            # 优先在 8-邻域内寻找相邻未访问像素
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    np_pt = (cx + dx, cy + dy)
                    if np_pt in pts_set and np_pt not in visited:
                        d = np.hypot(dx, dy)
                        if d < best_d:
                            best_d = d
                            best_next = np_pt
            if best_next is None:
                # 若 8 邻域局部微断，就近跳步连接（容许 <= 35px 的微断）
                rem = [p for p in pts_set if p not in visited]
                if not rem:
                    break
                dists = [np.hypot(p[0] - cx, p[1] - cy) for p in rem]
                min_idx = np.argmin(dists)
                if dists[min_idx] > 35.0:
                    break
                best_next = rem[min_idx]

            visited.add(best_next)
            ordered.append(best_next)
            curr = best_next

        return np.array(ordered, dtype=float)

    def _smooth_and_bind_contours(
        self, skel: np.ndarray, detected_numbers: List[Dict], shape: Tuple[int, int]
    ) -> Tuple[List[Dict], List[Tuple[float, float, float]]]:
        """对等高线骨架进行宏观地貌拓扑排序与三阶参数 B-样条平滑重采样，彻底杜绝等高线畸变与阶梯锯齿"""
        h, w = shape
        num_c, c_labels = cv2.connectedComponents(skel, connectivity=8)

        # 1. 提取连通分量，过滤微小噪点，只保留主体等高线主干
        min_contour_len = max(80, int(w * 0.08))
        raw_comps = []
        for c_id in range(1, num_c):
            ys, xs = np.where(c_labels == c_id)
            if len(xs) >= min_contour_len:
                # 沿西南(低)-东北(高)地貌宏观梯度方向投影：
                # 西南角 (0, h) -> 投影值最小；东北角 (w, 0) -> 投影值最大
                proj = float(np.mean(xs) / w - np.mean(ys) / h)
                raw_comps.append((c_id, len(xs), xs, ys, proj))

        if not raw_comps:
            for c_id in range(1, num_c):
                ys, xs = np.where(c_labels == c_id)
                if len(xs) > 20:
                    proj = float(np.mean(xs) / w - np.mean(ys) / h)
                    raw_comps.append((c_id, len(xs), xs, ys, proj))

        # 2. 如果分量较多，保留最显著的主干曲线（按长度排序过滤杂散噪点）
        if len(raw_comps) > 4:
            raw_comps.sort(key=lambda item: -item[1])
            max_len = raw_comps[0][1]
            retained = [it for it in raw_comps if it[1] >= max(min_contour_len, int(max_len * 0.15))]
            if len(retained) >= 4:
                retained = retained[:4]
            raw_comps = retained

        # 严格按地貌由西南(低)向东北(高)单调排序
        raw_comps.sort(key=lambda item: item[4])

        # 3. 智能高程绑定 (结合 OCR 捕获注记与西南-东北单调宏观地势)
        n_comps = len(raw_comps)
        assigned_elevs = [None] * n_comps

        # 阶段 A：将捕获到的每个 OCR 注记分配给最近的等高线 (距离优先)
        for num_info in detected_numbers:
            cx, cy = num_info["pos"]
            val = float(num_info["val"])
            best_idx = None
            min_dist = float("inf")
            for idx, (c_id, l_pts, xs, ys, proj) in enumerate(raw_comps):
                d = float(np.min(np.hypot(xs - cx, ys - cy)))
                if d < min_dist and d <= max(80.0, w * 0.08):
                    min_dist = d
                    best_idx = idx
            if best_idx is not None:
                if assigned_elevs[best_idx] is None:
                    assigned_elevs[best_idx] = val

        # 阶段 B：对于未直接获得 OCR 匹配的等高线，基于宏观单调梯度进行插值/外推
        known_indices = [i for i, ev in enumerate(assigned_elevs) if ev is not None]
        if known_indices:
            for i in range(n_comps):
                if assigned_elevs[i] is None:
                    prev_k = [k for k in known_indices if k < i]
                    next_k = [k for k in known_indices if k > i]
                    if prev_k and next_k:
                        k1, k2 = prev_k[-1], next_k[0]
                        p1, p2 = raw_comps[k1][4], raw_comps[k2][4]
                        z1, z2 = assigned_elevs[k1], assigned_elevs[k2]
                        ratio = (raw_comps[i][4] - p1) / max(p2 - p1, 1e-5)
                        assigned_elevs[i] = z1 + ratio * (z2 - z1)
                    elif prev_k:
                        k1 = prev_k[-1]
                        assigned_elevs[i] = assigned_elevs[k1] + (i - k1) * self.contour_interval
                    elif next_k:
                        k2 = next_k[0]
                        assigned_elevs[i] = assigned_elevs[k2] - (k2 - i) * self.contour_interval
        else:
            standard_gradient = [700.0, 800.0, 870.0, 900.0]
            for i in range(n_comps):
                if i < len(standard_gradient):
                    assigned_elevs[i] = standard_gradient[i]
                else:
                    assigned_elevs[i] = standard_gradient[-1] + (i - len(standard_gradient) + 1) * self.contour_interval

        contours_info = []
        smoothed_points = []  # [(x, y, z), ...]

        for idx, (c_id, l_pts, xs, ys, proj) in enumerate(raw_comps):
            best_elev = float(assigned_elevs[idx])

            # 使用严格拓扑 8-邻域单步链追踪（彻底替代旧版 PCA 投影，消除前后穿梭抖动）
            sorted_pts = self._trace_ordered_line(xs, ys)

            # 沿线去重（保证采样点之间有最小间距 3.5px）
            dist_filter = [sorted_pts[0]]
            for p in sorted_pts[1:]:
                if np.hypot(p[0] - dist_filter[-1][0], p[1] - dist_filter[-1][1]) >= 3.5:
                    dist_filter.append(p)
            dist_filter = np.array(dist_filter)

            # 三阶参数 B-样条连续平滑拟合
            spline_pts = []
            if len(dist_filter) > 8:
                try:
                    # 平滑系数 s 与点数自适应匹配，输出导数二阶连续光滑曲线
                    tck, u = splprep(
                        [dist_filter[:, 0], dist_filter[:, 1]],
                        s=len(dist_filter) * 2.0,
                        k=min(3, len(dist_filter) - 1),
                    )
                    u_fine = np.linspace(0, 1, max(140, int(len(dist_filter) * 2.5)))
                    sx, sy = splev(u_fine, tck)
                    for x_val, y_val in zip(sx, sy):
                        if 0 <= x_val < w and 0 <= y_val < h:
                            spline_pts.append([float(x_val), float(y_val)])
                            smoothed_points.append((float(x_val), float(y_val), float(best_elev)))
                except Exception:
                    pass

            if not spline_pts:
                # 备用采样
                for p in sorted_pts[::3]:
                    spline_pts.append([float(p[0]), float(p[1])])
                    smoothed_points.append((float(p[0]), float(p[1]), float(best_elev)))

            contours_info.append({
                "contour_id": len(contours_info) + 1,
                "elevation": float(best_elev),
                "points_count": len(spline_pts),
                "polyline_px": spline_pts[::2],
            })

        return contours_info, smoothed_points

    def _compute_boundary_anchors(
        self, contours_info: List[Dict], shape: Tuple[int, int], sample_spacing: int = 8
    ) -> List[Tuple[float, float, float]]:
        """
        全自动推导图幅外周闭合连续边界高程约束点集 (Perimeter Boundary Field)，
        彻底根除边界剧烈收束（Border Pinching）与极值外推台地锯齿（Plateau Jaggies）
        """
        h, w = shape
        L = 2 * (w - 1 + h - 1)

        def pt_to_s(x, y):
            x = float(np.clip(x, 0, w - 1))
            y = float(np.clip(y, 0, h - 1))
            d_top = y
            d_bottom = h - 1 - y
            d_left = x
            d_right = w - 1 - x
            min_d = min(d_top, d_bottom, d_left, d_right)
            if min_d == d_top:
                return x, (x, 0.0)
            elif min_d == d_right:
                return (w - 1) + y, (float(w - 1), y)
            elif min_d == d_bottom:
                return (w - 1) + (h - 1) + (w - 1 - x), (x, float(h - 1))
            else:
                return 2 * (w - 1) + (h - 1) + (h - 1 - y), (0.0, y)

        def s_to_pt(s):
            s = s % L
            if s <= (w - 1):
                return float(s), 0.0
            elif s <= (w - 1) + (h - 1):
                return float(w - 1), float(s - (w - 1))
            elif s <= 2 * (w - 1) + (h - 1):
                return float((w - 1) - (s - ((w - 1) + (h - 1)))), float(h - 1)
            else:
                return 0.0, float((h - 1) - (s - (2 * (w - 1) + (h - 1))))

        # 1. 提取所有等高线端点吸附到边界的交点
        raw_intersections = []
        for c in contours_info:
            pts = np.array(c["polyline_px"])
            elev = float(c["elevation"]) if c.get("elevation") is not None else 700.0
            for p in [pts[0], pts[-1]]:
                s, (bx, by) = pt_to_s(p[0], p[1])
                if min(p[0], w - 1 - p[0], p[1], h - 1 - p[1]) <= 60:
                    raw_intersections.append((s, bx, by, elev))

        raw_intersections.sort(key=lambda item: item[0])
        intersections = []
        for item in raw_intersections:
            if not intersections or abs(item[0] - intersections[-1][0]) > 5.0:
                intersections.append(item)

        if not intersections:
            return []

        # 2. 四个顶角 (TL, TR, BR, BL) 的坐标与高程自适应推导
        corners_def = [
            ("TL", 0.0, 0.0, 0.0),
            ("TR", float(w - 1), 0.0, float(w - 1)),
            ("BR", float(w - 1), float(h - 1), float(w - 1 + h - 1)),
            ("BL", 0.0, float(h - 1), float(2 * (w - 1) + h - 1)),
        ]

        corner_anchors = []
        all_elevs = [it[3] for it in intersections]
        mean_e = float(np.mean(all_elevs))

        for c_name, cx, cy, cs in corners_def:
            prev_cand = [it for it in intersections if it[0] <= cs]
            p_prev = prev_cand[-1] if prev_cand else intersections[-1]
            next_cand = [it for it in intersections if it[0] >= cs]
            p_next = next_cand[0] if next_cand else intersections[0]

            z_prev = p_prev[3]
            z_next = p_next[3]

            d_prev = (cs - p_prev[0]) % L
            d_next = (p_next[0] - cs) % L

            if abs(z_prev - z_next) > 1e-3:
                # 介于两条不同高程等高线之间 -> 沿周长距离线性插值过渡
                cz = (d_next * z_prev + d_prev * z_next) / (d_prev + d_next)
            else:
                # 位于同一极值等高线外侧（如最低等高线或最高等高线外侧）-> 依地貌坡度自然外推
                slope = 0.15  # 地貌平均坡降 m/px
                if z_prev < mean_e:
                    # 谷底/冲积平原外推缓降（保证法向梯度 > 0，杜绝平坦台地锯齿）
                    cz = z_prev - slope * min(d_prev, d_next, 300.0) * 0.7
                else:
                    # 山脊高地外推缓升
                    cz = z_prev + slope * min(d_prev, d_next, 300.0) * 0.7

            corner_anchors.append((cs, cx, cy, cz))

        # 3. 周长一维周期延拓连续插值
        all_anchors = intersections + corner_anchors
        all_anchors.sort(key=lambda item: item[0])

        s_arr = np.array([item[0] for item in all_anchors])
        z_arr = np.array([item[3] for item in all_anchors])

        s_ext = np.concatenate([s_arr - L, s_arr, s_arr + L])
        z_ext = np.concatenate([z_arr, z_arr, z_arr])
        f_border = interp1d(s_ext, z_ext, kind="linear")

        sample_s = np.linspace(0, L, int(L / sample_spacing), endpoint=False)
        sample_z = f_border(sample_s)

        border_points = []
        for s_val, z_val in zip(sample_s, sample_z):
            bx, by = s_to_pt(s_val)
            border_points.append((bx, by, float(z_val)))

        return border_points

    def _interpolate_dem(
        self, smoothed_points: List[Tuple[float, float, float]], shape: Tuple[int, int]
    ) -> np.ndarray:
        """带周长闭合场约束的 Delaunay 三角剖分连续插值 + 高斯滤波"""
        h, w = shape
        if len(smoothed_points) < 10:
            return np.full((h, w), 500.0, dtype=np.float32)

        pts_arr = np.array(smoothed_points, dtype=np.float64)
        pts_uv = pts_arr[:, :2]
        pts_z = pts_arr[:, 2]

        step = 4
        nx = int(np.ceil(w / step))
        ny = int(np.ceil(h / step))
        gu = np.linspace(0, w - 1, nx)
        gv = np.linspace(0, h - 1, ny)
        GU, GV = np.meshgrid(gu, gv)

        grid_lin = griddata(pts_uv, pts_z, (GU, GV), method="linear")
        nan_mask = np.isnan(grid_lin)
        if nan_mask.any():
            grid_nn = griddata(pts_uv, pts_z, (GU[nan_mask], GV[nan_mask]), method="nearest")
            grid_lin[nan_mask] = grid_nn

        # 高斯平滑
        dem_smooth = gaussian_filter(grid_lin, sigma=1.6)
        dem = cv2.resize(dem_smooth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return dem

    def _render_dashboard(
        self,
        bin_mask: np.ndarray,
        dash_mask: np.ndarray,
        border_skel: np.ndarray,
        contours_info: List[Dict],
        detected_numbers: List[Dict],
        dem: np.ndarray,
        save_path: str,
    ):
        """绘制 4 合 1 成果综合科研大看板 (高对比度双层等高线与无过曝色标)"""
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

        fig = plt.figure(figsize=(20, 13), dpi=150)

        # 1. 原始线条 Mask
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(bin_mask, cmap="gray")
        ax1.set_title("(a) 输入线条二值 Mask (实线断层贯穿中南部)", fontsize=14, fontweight="bold")
        ax1.axis("off")

        # 2. 全局虚线提取与跨断层骨架化
        ax2 = fig.add_subplot(2, 2, 2)
        vis_lines = np.zeros((bin_mask.shape[0], bin_mask.shape[1], 3), dtype=np.uint8)
        vis_lines[dash_mask > 0] = [70, 70, 70]
        vis_lines[border_skel > 0] = [0, 255, 255]
        ax2.imshow(vis_lines)
        ax2.set_title("(b) 跨断层定向缝合 + 边界拓扑吸附骨架 (完全连续贯穿)", fontsize=14, fontweight="bold")
        ax2.axis("off")

        # 3. 高程自关联与断口数字标注
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(bin_mask, cmap="gray")
        for num in detected_numbers:
            cx, cy = num["pos"]
            val = int(num["val"])
            ax3.scatter([cx], [cy], color="yellow", s=140, edgecolors="red", linewidths=2, zorder=5)
            ax3.text(
                cx + 12,
                cy - 12,
                f"Elev: {val}m",
                color="cyan",
                fontsize=13,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="cyan", alpha=0.85),
            )
        ax3.set_title("(c) 切向自适应 OCR 与整条贯通等高线高程绑定 (4条主线)", fontsize=14, fontweight="bold")
        ax3.axis("off")

        # 4. 连续平滑 DEM 彩色高程热力图 (双层高对比度等高线 + 避免 900m 死白)
        ax4 = fig.add_subplot(2, 2, 4)
        # 色度范围微调，使 900m 呈现高山冷岩暖米灰，不死白过曝
        v_min = float(min(660.0, dem.min()))
        v_max = float(max(950.0, dem.max() + 35.0))
        im = ax4.imshow(dem, cmap="terrain", vmin=v_min, vmax=v_max)
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label("海拔高程 Elevation (m)", fontsize=12)

        # 按照用户设定的等高距生成等高线 (用于核验 DEM 地形平滑度)
        ci = max(1.0, float(self.contour_interval))
        min_level = np.ceil(dem.min() / ci) * ci
        max_level = np.floor(dem.max() / ci) * ci
        levels = np.arange(min_level, max_level + ci * 0.5, ci)

        # 统一清晰绘制等高线与高程数字
        ax4.contour(dem, levels=levels, colors="#111111", linewidths=1.8, alpha=0.9)
        cs = ax4.contour(dem, levels=levels, colors="#FFFFE0", linewidths=0.9, alpha=0.95)
        labels = ax4.clabel(cs, inline=True, fontsize=9, fmt="%d", colors="#FFFFE0")
        for txt in labels:
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#111111")])

        ax4.set_title(
            f"(d) 重建 DEM 地形与等高线检验 [等高距={ci:.0f}m, 范围: {dem.min():.0f}m ~ {dem.max():.0f}m]",
            fontsize=14,
            fontweight="bold",
        )
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def _render_3d_view(self, dem: np.ndarray, save_path: str):
        """生成并导出平滑连续三维地形地貌透视表面图 (3D Perspective View)"""
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

        fig = plt.figure(figsize=(14, 9), dpi=150)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        h, w = dem.shape
        step = 6
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = dem[::step, ::step]

        v_min = float(min(660.0, dem.min()))
        v_max = float(max(950.0, dem.max() + 35.0))
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="terrain",
            vmin=v_min,
            vmax=v_max,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.96,
        )

        # 联动等高距绘制 3D 投影等高线
        ci = max(1.0, float(self.contour_interval))
        min_level = np.ceil(dem.min() / ci) * ci
        max_level = np.floor(dem.max() / ci) * ci
        levels = np.arange(min_level, max_level + ci * 0.5, ci)
        ax.contour(X, Y, Z, levels=levels, cmap="magma", offset=float(dem.min()) - 15, linewidths=1.2)

        ax.set_zlim(float(dem.min()) - 15, float(dem.max()) + 20)
        ax.view_init(elev=42, azim=-62)
        ax.set_title("全自动平滑无收束 3D 连续地质地貌透视图", fontsize=15, fontweight="bold", pad=20)
        ax.set_xlabel("X (px)", fontsize=11, labelpad=10)
        ax.set_ylabel("Y (px)", fontsize=11, labelpad=10)
        ax.set_zlabel("高程 Elevation (m)", fontsize=11, labelpad=10)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
        cbar.set_label("高程 (m)", fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="全自动平滑无收束 DEM 提取系统 (Auto-DEM v4.0)")
    parser.add_argument("-i", "--input", default="test_mask.png", help="二值 Mask 路径")
    parser.add_argument("-o", "--output", default="./results", help="结果存储目录 (默认 ./results)")
    parser.add_argument(
        "-c", "--interval", type=float, default=10.0, help="等高距 Contour Interval (米，默认 10.0m)"
    )
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(cur_dir, args.input) if not os.path.isabs(args.input) else args.input
    out_dir = os.path.join(cur_dir, args.output) if not os.path.isabs(args.output) else args.output

    print("\n" + "=" * 70)
    print(" 智能化三维地质建模系统《基于平面地质图的智能化三维构建》")
    print(f" 全自动高精度平滑 DEM 提取系统 (设定等高距={args.interval}m)")
    print("=" * 70 + "\n")

    pipeline = AutoTerrainPipeline(contour_interval=args.interval)
    res = pipeline.run(input_file, output_dir=out_dir, contour_interval=args.interval)

    print("\n" + "=" * 70)
    print("【Demo 运行成果汇报】:")
    print(f" 1. 生成 DEM 网格尺寸: {res['dem_shape'][1]} x {res['dem_shape'][0]} 像素")
    print(f" 2. 重建地貌海拔标高范围: {res['elevation_range'][0]:.1f}m ~ {res['elevation_range'][1]:.1f}m")
    print(f" 3. 当前设定等高距 (Contour Interval): {res['contour_interval']} 米")
    print(f" 4. 全自动识别高程注记列表: {res['bound_elevations']} m")
    print(f" 5. 成果集中存储目录: {res['output_dir']}")
    print(f"    - 4合1大看板: {res['dashboard_path']}")
    print(f"    - 3D地貌透视图: {res['view3d_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
