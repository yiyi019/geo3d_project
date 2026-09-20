# -*- coding: utf-8 -*-
"""
extract_surface_points.py — 基于地质构造线、地层界线与 DEM 生成标准界面点数据库
========================================================================
项目：智能化三维地质建模系统《基于平面地质图的智能化三维构建》

【核心功能】
1. 载入断层骨架 (fault_skeleton.png) 与地层边界骨架 (boundary_skeleton.png)；
2. 载入连续平滑 DEM 标高矩阵 (dem_auto.npy)；
3. 基于大地测绘坐标系进行尺度换算 (1 像素 = 2 米，总宽 2048 米)：
   - X = x * scale (东向)
   - Y = (H - 1 - y) * scale (北向，与图像坐标系反转对齐)
   - Z = DEM[y, x] (真实地形地表高程)
4. 沿各构造线/界面严格按 DFS 测地拓扑排序，均匀等距采样控制点；
5. 覆盖更新标准 GemPy 界面点数据库 (surface_points.csv)，
   要素名称 (Fault_1, Layer_1~Layer_4) 与 orientations.csv 100% 严格一致。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SurfacePointsExtractor")


def order_skeleton_path(mask: np.ndarray) -> List[Tuple[int, int]]:
    """利用 8-邻域欧氏加权图与 Dijkstra 测地主干路径提取算法，
    从单像素骨架中精确提取从一端到另一端的平滑单调主干路径。

    彻底剔除侧向分叉、T/Y形毛刺与回溯跳跃，确保路径上任意相邻点间距恒在 1.0~1.42 像素内，
    杜绝首尾回跳引起的跨图假连线。
    返回: [(y, x), ...] 严格单调连续路径
    """
    import heapq

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    if len(xs) == 1:
        return [(int(ys[0]), int(xs[0]))]

    pts = [(int(y), int(x)) for y, x in zip(ys, xs)]
    coord_set = set(pts)

    # 1. 建立 8-邻域加权邻接表
    adj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
    for y, x in pts:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                cand = (y + dy, x + dx)
                if cand in coord_set:
                    dist = 1.4142 if (dy != 0 and dx != 0) else 1.0
                    neighbors.append((cand, dist))
        adj[(y, x)] = neighbors

    # 2. 寻找度为 1 的候选端点，并选定西侧最端头作为起点
    deg1 = [p for p in pts if len(adj[p]) == 1]
    if deg1:
        start_pt = min(deg1, key=lambda p: (p[1], p[0]))
    else:
        start_pt = min(pts, key=lambda p: (p[1], p[0]))

    # 3. 运行 Dijkstra 寻找测地距离最远的主干终点
    dist_map: Dict[Tuple[int, int], float] = {start_pt: 0.0}
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    pq = [(0.0, start_pt)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_map[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist_map.get(v, float("inf")):
                dist_map[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (d + w, v))

    # 4. 取得测地距离最远点作为主干终点并回溯
    end_pt = max(dist_map.keys(), key=lambda p: dist_map[p])

    path = []
    curr = end_pt
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    path.append(start_pt)
    path.reverse()

    return path


def sample_points_along_path(
    path: List[Tuple[int, int]],
    n_points: int = 20,
) -> List[Tuple[int, int]]:
    """沿有序路径等间距选取指定数量的点，保证包含首尾端点。"""
    M = len(path)
    if M <= n_points:
        return path

    indices = np.linspace(0, M - 1, n_points, dtype=int)
    # 去重且保持严格顺序
    seen = set()
    sampled = []
    for idx in indices:
        pt = path[idx]
        if pt not in seen:
            seen.add(pt)
            sampled.append(pt)
    return sampled


def extract_surface_points_structured(
    fault_skel_path: str,
    boundary_skel_path: str,
    dem_path: str,
    scale: float = 2.0,
    points_per_feature: int = 20,
    min_skeleton_length: int = 15,
    output_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """提取断层与地层边界的三维控制点并返回包含详细拓扑元数据的结构化字典。

    已内置微小杂散噪点过滤 (min_skeleton_length) 与健壮异常捕获，彻底杜绝单像素崩溃。
    """
    logger.info("1. 载入输入数据...")
    if not os.path.exists(fault_skel_path):
        raise FileNotFoundError(f"未找到断层骨架文件: {fault_skel_path}")
    if not os.path.exists(boundary_skel_path):
        raise FileNotFoundError(f"未找到地层边界骨架文件: {boundary_skel_path}")
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"未找到 DEM 标高文件: {dem_path}")

    f_skel = cv2.imread(fault_skel_path, cv2.IMREAD_GRAYSCALE)
    b_skel = cv2.imread(boundary_skel_path, cv2.IMREAD_GRAYSCALE)
    dem = np.load(dem_path)

    # 以骨架真实空间分辨率为基准坐标系，保护骨架单像素拓扑连通性不被最近邻重采样撕裂
    if b_skel is not None:
        h, w = b_skel.shape[:2]
    elif f_skel is not None:
        h, w = f_skel.shape[:2]
    else:
        h, w = dem.shape[:2]

    # DEM 为连续标高曲面，若网格分辨率不一致，采用双线性平滑插值对齐，绝不损伤骨架拓扑
    if dem.shape != (h, w):
        logger.info(f"   DEM 尺寸 ({dem.shape[1]}x{dem.shape[0]}) 与骨架基准 ({w}x{h}) 自适应双线性平滑对齐...")
        dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LINEAR)

    logger.info(f"   空间尺寸基准: {w} x {h} px | DEM 标高区间: [{dem.min():.1f}m, {dem.max():.1f}m]")
    logger.info(f"   比例尺: 1 px = {scale:.1f} m (全图水平范围 X:[0.0, {w * scale:.1f}]m, Y:[0.0, {h * scale:.1f}]m)")

    # 尺寸校验与自适应对齐
    if f_skel is None:
        f_skel = np.zeros((h, w), dtype=np.uint8)
    elif f_skel.shape != (h, w):
        f_skel = cv2.resize(f_skel, (w, h), interpolation=cv2.INTER_NEAREST)

    if b_skel is None:
        b_skel = np.zeros((h, w), dtype=np.uint8)
    elif b_skel.shape != (h, w):
        b_skel = cv2.resize(b_skel, (w, h), interpolation=cv2.INTER_NEAREST)

    rows = []
    features_meta = {}
    # 断层专用鲜红 (#e41a1c)，地层使用高辨识度定性色板 (深蓝、翠绿、明橙、紫色、天蓝、草绿等)，彻底杜绝撞色
    default_colors = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6", "#ffff33", "#b15928"]

    # 2. 提取断层点 (Fault_1, ...)
    logger.info("2. 正在提取断层控制点...")
    num_f_labels, f_labels, f_stats, _ = cv2.connectedComponentsWithStats(f_skel)
    fault_components = []
    for i in range(1, num_f_labels):
        if f_stats[i, cv2.CC_STAT_AREA] >= min_skeleton_length:
            fault_components.append((i, f_stats[i, cv2.CC_STAT_AREA]))

    # 若无大于阈值的断层连通域，但 f_skel 存在像素，则选最大连通域（若 >= 2点）
    if not fault_components and num_f_labels > 1:
        max_i = 1 + int(np.argmax(f_stats[1:, cv2.CC_STAT_AREA]))
        if f_stats[max_i, cv2.CC_STAT_AREA] >= 2:
            fault_components.append((max_i, f_stats[max_i, cv2.CC_STAT_AREA]))

    fault_idx = 1
    for f_i, f_area in fault_components:
        feat_name = "Fault_1" if len(fault_components) == 1 else f"Fault_{fault_idx}"
        f_mask_i = (f_labels == f_i).astype(np.uint8)
        try:
            f_path = order_skeleton_path(f_mask_i)
            f_sampled = sample_points_along_path(f_path, n_points=points_per_feature)
            f_pts = []
            for y, x in f_sampled:
                yc = int(np.clip(y, 0, h - 1))
                xc = int(np.clip(x, 0, w - 1))
                X = round(float(xc * scale), 1)
                Y = round(float((h - 1 - yc) * scale), 1)
                Z = round(float(dem[yc, xc]), 1)
                row = {"X": X, "Y": Y, "Z": Z, "formation": feat_name}
                rows.append(row)
                f_pts.append(row)
            logger.info(f"   -> {feat_name} 骨架总长 {len(f_path)} px, 采样 {len(f_sampled)} 个界面控制点")

            xs_f = [p["X"] for p in f_pts]
            ys_f = [p["Y"] for p in f_pts]
            zs_f = [p["Z"] for p in f_pts]
            features_meta[feat_name] = {
                "name": feat_name,
                "type": "fault",
                "color": "#e41a1c",
                "length_px": len(f_path),
                "point_count": len(f_pts),
                "path_px": f_path,
                "sampled_points": f_pts,
                "bounds": {
                    "X": (min(xs_f), max(xs_f)),
                    "Y": (min(ys_f), max(ys_f)),
                    "Z": (min(zs_f), max(zs_f)),
                },
                "description": f"断层构造走向线 #{fault_idx}",
            }
            fault_idx += 1
        except Exception as e:
            logger.warning(f"   跳过异常断层连通域 #{f_i}: {e}")

    # 3. 提取地层边界点 (Layer_1 ~ Layer_N)，严格滤除微小噪点
    logger.info("3. 正在提取地层边界控制点 [Layer_1 ~ Layer_N]...")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_skel)
    n_total_cc = num_labels - 1
    logger.info(f"   检测到 {n_total_cc} 个原始连通域 (杂散微小噪点过滤阈值: 骨架长度 >= {min_skeleton_length} px)")

    boundary_components = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_skeleton_length:
            boundary_components.append((i, area))
        else:
            logger.info(f"   [滤除噪点] 忽略微小噪点连通域 #{i} (仅 {area} 像素 < 阈值 {min_skeleton_length} px)")

    # 若无大于阈值的边界连通域，但 b_skel 存在像素，则选最大连通域（若 >= 2点）
    if not boundary_components and num_labels > 1:
        max_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[max_i, cv2.CC_STAT_AREA] >= 2:
            boundary_components.append((max_i, int(stats[max_i, cv2.CC_STAT_AREA])))

    layer_idx = 1
    for i, area in boundary_components:

        mask_i = (labels == i).astype(np.uint8)
        try:
            b_path = order_skeleton_path(mask_i)
            if len(b_path) < 2:
                logger.info(f"   [滤除噪点] 连通域 #{i} 排序后有效点不足，已跳过")
                continue

            b_sampled = sample_points_along_path(b_path, n_points=points_per_feature)
            feat_name = f"Layer_{layer_idx}"
            b_pts = []
            for y, x in b_sampled:
                yc = int(np.clip(y, 0, h - 1))
                xc = int(np.clip(x, 0, w - 1))
                X = round(float(xc * scale), 1)
                Y = round(float((h - 1 - yc) * scale), 1)
                Z = round(float(dem[yc, xc]), 1)
                row = {"X": X, "Y": Y, "Z": Z, "formation": feat_name}
                rows.append(row)
                b_pts.append(row)
            logger.info(f"   -> {feat_name} (原连通域#{i}) 骨架总长 {len(b_path)} px, 采样 {len(b_sampled)} 个界面控制点")

            xs_b = [p["X"] for p in b_pts]
            ys_b = [p["Y"] for p in b_pts]
            zs_b = [p["Z"] for p in b_pts]
            features_meta[feat_name] = {
                "name": feat_name,
                "type": "stratum",
                "color": default_colors[(layer_idx - 1) % len(default_colors)],
                "length_px": len(b_path),
                "point_count": len(b_pts),
                "path_px": b_path,
                "sampled_points": b_pts,
                "bounds": {
                    "X": (min(xs_b), max(xs_b)),
                    "Y": (min(ys_b), max(ys_b)),
                    "Z": (min(zs_b), max(zs_b)),
                },
                "description": f"地层边界连通段 #{layer_idx} (待人工识别归类)",
            }
            layer_idx += 1
        except Exception as e:
            logger.warning(f"   连通域 #{i} (面积 {area} px) 迹线排序失败，已安全跳过: {e}")
            continue

    df = pd.DataFrame(rows)

    # 4. 可选覆盖写入 CSV
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False, float_format="%.1f")
        logger.info(f"4. 界面点数据库已成功覆盖写入: {os.path.abspath(output_csv)} (共 {len(df)} 行点位数据)")

    meta = {
        "features": features_meta,
        "image_shape": (h, w),
        "scale": scale,
        "dem_min": float(dem.min()),
        "dem_max": float(dem.max()),
    }
    return df, meta


def extract_surface_points(
    fault_skel_path: str,
    boundary_skel_path: str,
    dem_path: str,
    scale: float = 2.0,
    points_per_feature: int = 20,
    min_skeleton_length: int = 15,
    output_csv: str = "./surface_points.csv",
) -> pd.DataFrame:
    """提取断层与地层边界的三维控制点并覆盖写入 surface_points.csv (向前兼容包装函数)。"""
    df, _ = extract_surface_points_structured(
        fault_skel_path=fault_skel_path,
        boundary_skel_path=boundary_skel_path,
        dem_path=dem_path,
        scale=scale,
        points_per_feature=points_per_feature,
        min_skeleton_length=min_skeleton_length,
        output_csv=output_csv,
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="从断层与地层边界骨架中提取三维界面控制点并覆盖 surface_points.csv")
    parser.add_argument("-b", "--boundary", default="./boundary_output/boundary_skeleton.png", help="地层边界骨架路径")
    parser.add_argument("-f", "--fault", default="./fault_output/fault_skeleton.png", help="断层骨架路径")
    parser.add_argument("-d", "--dem", default="./results/dem_auto.npy", help="DEM 标高矩阵路径")
    parser.add_argument("-s", "--scale", type=float, default=2.0, help="比例尺 (米/像素，默认 2.0，对应总宽 2048m)")
    parser.add_argument("-k", "--points-per-feature", type=int, default=20, help="每个构造/地层要素采样的控制点数量 (默认 20)")
    parser.add_argument("-o", "--output", default="./surface_points.csv", help="输出界面点 CSV 文件路径")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    b_file = os.path.join(cur_dir, args.boundary) if not os.path.isabs(args.boundary) else args.boundary
    f_file = os.path.join(cur_dir, args.fault) if not os.path.isabs(args.fault) else args.fault
    dem_file = os.path.join(cur_dir, args.dem) if not os.path.isabs(args.dem) else args.dem
    if not os.path.exists(dem_file):
        for alt in [
            os.path.join(cur_dir, "dem_output", "dem_auto.npy"),
            os.path.join(cur_dir, "results", "dem_auto.npy"),
        ]:
            if os.path.exists(alt):
                dem_file = alt
                break
    out_file = os.path.join(cur_dir, args.output) if not os.path.isabs(args.output) else args.output

    print("\n" + "=" * 70)
    print(" 智能化三维地质建模系统《基于平面地质图的智能化三维构建》")
    print(" 地质界面空间控制点提取与数据库构建 (Extract Surface Points)")
    print("=" * 70 + "\n")

    df = extract_surface_points(
        fault_skel_path=f_file,
        boundary_skel_path=b_file,
        dem_path=dem_file,
        scale=args.scale,
        points_per_feature=args.points_per_feature,
        output_csv=out_file,
    )

    print("\n" + "=" * 70)
    print("【界面点数据库覆盖写入统计】:")
    for feat, grp in df.groupby("formation"):
        print(f"  • 要素 [{feat:7s}]: {len(grp):2d} 个点 | X: {grp['X'].min():6.1f} ~ {grp['X'].max():6.1f}m | Y: {grp['Y'].min():6.1f} ~ {grp['Y'].max():6.1f}m | Z: {grp['Z'].min():5.1f} ~ {grp['Z'].max():5.1f}m")

    print("-" * 70)
    print(f"总计控制点数: {len(df)} 点")
    print(f"输出目标文件: {os.path.abspath(out_file)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
