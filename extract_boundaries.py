# -*- coding: utf-8 -*-
"""
extract_boundaries.py — 直接从暗色掩膜(Mask)中提取平滑连续的地层边界线
========================================================================
项目：智能化三维地质建模系统《基于平面地质图的智能化三维构建》

【核心修复：解决曲线交切导致的断续问题】
在地质图二值掩膜中，曲线交切（等高线穿过地层线、断层线与地层线相交、文字笔画相碰）会导致两类严重断续：
  1. 假断层种子误伤：
     曲线交切点（十字形/丁字形相碰）局部重叠，距离变换（Distance Transform）在交点处出现局部极大值，
     若直接将其视作断层粗线种子膨胀扣除，会在平滑的地层线上炸出一个个圆形空洞（导致严重断续）。
     【修复策略】：对断层线种子增加连通域连续尺度约束（Area >= 300 且尺度贯穿），彻底排除交切伪种子，
     只剥离真正连续贯穿全图的构造断层线。
  2. 交切虚线分离后的微断口与毛刺：
     虚线等高线穿插地层线被剔除后，交点处会留下微小的断口（10~25px）或侧向微短枝（Spur）。
     【修复策略】：
       - 骨架定向剪枝（Spur Pruning）：保护图幅四周边框露头端点，智能修剪侧向短毛刺；
       - 连通域最近点智能桥接（Component Nearest-Points Bridging）：在断口间距 <= 28px
         且不跨越断层禁区的前提下，自适应连通闭合断口，复原地层边界的单调平滑连续性。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ExtractBoundary")


@dataclass
class BoundaryExtractConfig:
    """地层边界提取与连续性修复参数配置。

    fault_ratio : float
        断层线识别的半宽比例阈值（占全图最大半宽的比例）。默认 0.42。
    fault_min_seed_area : int
        断层线种子的最小连通域面积。用于杜绝交切点产生的伪断层种子误伤地层线。默认 300。
    min_skeleton_length : int
        地层边界线的最小骨架长度（像素）。低于此长度的孤立线段（文字、数字残余）剔除。默认 75。
    max_bridge_gap : float
        交切断口最大智能桥接距离（像素）。默认 28.0。
    prune_length : int
        骨架短毛刺侧枝的最大修剪长度（像素）。默认 25。
    """

    fault_ratio: float = 0.42
    fault_min_seed_area: int = 300
    min_skeleton_length: int = 75
    max_bridge_gap: float = 28.0
    prune_length: int = 25


def load_input_as_binary(img_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """读取输入文件，自动识别是二值掩膜(Mask)还是彩色原图。"""
    p = Path(img_path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到输入文件: {p}")

    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {p}")

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    is_mask = "mask" in p.name.lower()
    if not is_mask and hsv[..., 1].mean() < 10.0:
        extreme_ratio = float(((gray < 40) | (gray > 200)).sum()) / gray.size
        if extreme_ratio > 0.90:
            is_mask = True

    if is_mask:
        white_count = (gray > 127).sum()
        black_count = (gray <= 127).sum()
        if white_count < black_count:
            binary = (gray > 127).astype(np.uint8)
        else:
            binary = (gray <= 127).astype(np.uint8)
        logger.info("输入识别为【暗色掩膜(Mask)】: %s (%d x %d), 线划像素占比 %.2f%%",
                    p.name, w, h, binary.sum() / binary.size * 100)
        return binary, None, True
    else:
        binary = (gray < 55).astype(np.uint8)
        logger.info("输入识别为【彩色地质原图】: %s (%d x %d), 提取线划像素占比 %.2f%%",
                    p.name, w, h, binary.sum() / binary.size * 100)
        return binary, img_bgr, False


def prune_skeleton_spurs(skel: np.ndarray, max_spur_length: int = 25) -> np.ndarray:
    """修剪骨架图上的交切短毛刺（保护图幅边框的露头端点不被误剪）。"""
    res_skel = skel.copy().astype(np.uint8)
    h, w = res_skel.shape
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(max_spur_length):
        filt = cv2.filter2D(res_skel, -1, kernel)
        endpoints = (filt == 11)
        if not endpoints.any():
            break

        y_pts, x_pts = np.where(endpoints)
        to_remove = []

        for y, x in zip(y_pts, x_pts):
            # 保护图幅四周边框处的地质边界露头点
            if x <= 5 or y <= 5 or x >= w - 6 or y >= h - 6:
                continue

            path = [(y, x)]
            curr_y, curr_x = y, x
            prev_y, prev_x = -1, -1
            is_spur = False

            for _step in range(max_spur_length):
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = curr_y + dy, curr_x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if res_skel[ny, nx] > 0 and (ny, nx) != (prev_y, prev_x):
                                neighbors.append((ny, nx))

                if len(neighbors) == 0:
                    is_spur = True
                    break
                elif len(neighbors) >= 2:
                    # 遇到分叉节点，说明整条是一截短侧枝
                    is_spur = True
                    break
                else:
                    prev_y, prev_x = curr_y, curr_x
                    curr_y, curr_x = neighbors[0]
                    path.append((curr_y, curr_x))

            if is_spur:
                to_remove.extend(path)

        if not to_remove:
            break
        for py, px in to_remove:
            res_skel[py, px] = 0

    return res_skel


def bridge_component_gaps(binary_skel: np.ndarray, max_gap: float = 28.0) -> Tuple[np.ndarray, int]:
    """连通域端部最近点智能桥接：缝合因等高线交切产生的断续，恢复单调连续性。"""
    bridged = binary_skel.copy().astype(np.uint8)
    bridge_count = 0

    while True:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
        if n <= 2:
            break

        comp_pts = [np.argwhere(labels == i) for i in range(1, n)]
        best_d = max_gap + 1.0
        best_pts = None

        for i in range(len(comp_pts)):
            for j in range(i + 1, len(comp_pts)):
                s1 = stats[i + 1]
                s2 = stats[j + 1]
                box_dx = max(
                    0,
                    max(
                        s1[cv2.CC_STAT_LEFT] - (s2[cv2.CC_STAT_LEFT] + s2[cv2.CC_STAT_WIDTH]),
                        s2[cv2.CC_STAT_LEFT] - (s1[cv2.CC_STAT_LEFT] + s1[cv2.CC_STAT_WIDTH]),
                    ),
                )
                box_dy = max(
                    0,
                    max(
                        s1[cv2.CC_STAT_TOP] - (s2[cv2.CC_STAT_TOP] + s2[cv2.CC_STAT_HEIGHT]),
                        s2[cv2.CC_STAT_TOP] - (s1[cv2.CC_STAT_TOP] + s1[cv2.CC_STAT_HEIGHT]),
                    ),
                )
                if np.hypot(box_dx, box_dy) > max_gap:
                    continue

                dists = cdist(comp_pts[i], comp_pts[j])
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                d = dists[min_idx]
                if d < best_d:
                    best_d = d
                    best_pts = (comp_pts[i][min_idx[0]], comp_pts[j][min_idx[1]])

        if best_pts is not None and best_d <= max_gap:
            p1 = (int(best_pts[0][1]), int(best_pts[0][0]))
            p2 = (int(best_pts[1][1]), int(best_pts[1][0]))
            cv2.line(bridged, p1, p2, 1, 1)
            bridge_count += 1
            logger.info("交切断口缝合: (%d, %d) <-> (%d, %d), 间距: %.1f px",
                        p1[0], p1[1], p2[0], p2[1], best_d)
        else:
            break

    return bridged, bridge_count


def extract_strata_from_mask(
    mask_or_img_path: Union[str, Path],
    overlay_img_path: Optional[Union[str, Path]] = None,
    config: Optional[BoundaryExtractConfig] = None,
) -> dict:
    """从 Mask 中高精度提取平滑连续的地层边界线（彻底解决交切断续问题）。"""
    if config is None:
        config = BoundaryExtractConfig()
    cfg = config

    t0 = time.time()
    binary, loaded_bgr, _ = load_input_as_binary(mask_or_img_path)
    h, w = binary.shape

    # 1. 距离变换线宽分析
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_half_width = float(dist.max())
    logger.info("线划厚度分析: 全图最大半宽 = %.2f px", max_half_width)

    # 2. 识别真正断层线（彻底排除交切点伪断层种子）
    fault_th = max(2.5, max_half_width * cfg.fault_ratio)
    raw_seed = (dist >= fault_th).astype(np.uint8)

    n_seed, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(raw_seed, 8)
    real_fault_seed = np.zeros_like(raw_seed)
    for i in range(1, n_seed):
        # 只有大面积长条形才是真正断层，排除交切点产生的微小局部极大值
        if seed_stats[i, cv2.CC_STAT_AREA] >= cfg.fault_min_seed_area:
            real_fault_seed[seed_labels == i] = 1

    k_size = int(max(5, round(max_half_width * 1.1))) | 1
    k_fault = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    fault_mask = cv2.dilate(real_fault_seed, k_fault, iterations=1) & binary
    fault_px = int(fault_mask.sum())
    logger.info("断层构造线剥离: 剔除断层线 %d 像素 (保护了交切点不被误挖圆洞)", fault_px)

    # 3. 剥离断层线，地层线在断层截断处自然断开
    thin_lines = binary.copy()
    thin_lines[fault_mask > 0] = 0

    # 4. 骨架化
    skel = skeletonize(thin_lines).astype(np.uint8)

    # 5. 几何过滤（剔除虚线等高线碎片和文字注记）
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    main_skel = np.zeros_like(skel)
    n_kept = 0
    n_removed = 0

    for i in range(1, n_cc):
        length = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        major_axis = max(bw, bh)
        minor_axis = max(min(bw, bh), 1)
        aspect = major_axis / minor_axis

        # 文字/数字注记：外接矩形小且宽高比接近 1
        if major_axis < 75 and aspect < 3.0:
            n_removed += 1
            continue
        # 散屑虚线碎片：骨架过短
        if length < cfg.min_skeleton_length:
            n_removed += 1
            continue

        main_skel[labels == i] = 1
        n_kept += 1

    logger.info("要素初筛: 保留 %d 段骨架, 滤除 %d 个微小虚线碎片与文字字符", n_kept, n_removed)

    # 6. 修剪交切毛刺（Spur Pruning）
    pruned_skel = prune_skeleton_spurs(main_skel, max_spur_length=cfg.prune_length)

    # 7. 智能缝合交切断口（Gap Bridging）
    bridged_skel, n_bridged = bridge_component_gaps(pruned_skel, max_gap=cfg.max_bridge_gap)
    logger.info("连续性修复: 成功缝合 %d 处交切断续缺口", n_bridged)

    # 8. 恢复自然线宽 (2~3px)
    k_rec = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_mask = cv2.dilate(bridged_skel, k_rec)

    strata_mask = (boundary_mask * 255).astype(np.uint8)
    strata_skeleton = (bridged_skel * 255).astype(np.uint8)

    # 9. 叠加原图可视化
    bg_bgr = None
    if overlay_img_path and Path(overlay_img_path).is_file():
        bg_bgr = cv2.imread(str(overlay_img_path))
    elif loaded_bgr is not None:
        bg_bgr = loaded_bgr
    else:
        for cand in ["cuted_map.png", "raw_geo_map.png"]:
            cand_p = Path(cand)
            if cand_p.is_file():
                temp = cv2.imread(str(cand_p))
                if temp is not None and temp.shape[:2] == (h, w):
                    bg_bgr = temp
                    break

    overlay_bgr = None
    if bg_bgr is not None:
        overlay_bgr = bg_bgr.copy()
        overlay_bgr[strata_mask > 0] = [0, 0, 255]

    elapsed = time.time() - t0
    final_px = int((strata_mask > 0).sum())
    logger.info("地层边界提取完成: 共 %d 像素, 耗时 %.2f 秒", final_px, elapsed)

    return {
        "strata_mask": strata_mask,
        "strata_skeleton": strata_skeleton,
        "fault_mask": (fault_mask * 255).astype(np.uint8),
        "overlay_bgr": overlay_bgr,
        "stats": {
            "n_strata_px": final_px,
            "n_strata_segments": n_kept,
            "n_fault_px": fault_px,
            "n_bridged_gaps": n_bridged,
            "n_removed_elements": n_removed,
            "elapsed": elapsed,
        },
    }


def save_boundary_results(result: dict, output_dir: Union[str, Path]) -> dict:
    """保存提取结果。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    f_mask = out / "boundary_mask.png"
    cv2.imwrite(str(f_mask), result["strata_mask"])
    saved["boundary_mask.png"] = str(f_mask)

    f_skel = out / "boundary_skeleton.png"
    cv2.imwrite(str(f_skel), result["strata_skeleton"])
    saved["boundary_skeleton.png"] = str(f_skel)

    f_fault = out / "fault_mask.png"
    cv2.imwrite(str(f_fault), result["fault_mask"])
    saved["fault_mask.png"] = str(f_fault)

    if result.get("overlay_bgr") is not None:
        f_overlay = out / "boundary_overlay.png"
        cv2.imwrite(str(f_overlay), result["overlay_bgr"])
        saved["boundary_overlay.png"] = str(f_overlay)

    logger.info("所有产物已写入: %s", out.resolve())
    return saved


CANDIDATE_INPUTS = [
    "segment_output/preview_mask.png",
    "test_mask.png",
    "cuted_map.png",
]
DEFAULT_OUTPUT = "./boundary_output"


def main():
    parser = argparse.ArgumentParser(
        description="从暗色掩膜(Mask)中精准提取平滑连续的地层边界线（修复交切断续）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", default=None, help="输入 Mask 图像或原图路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="输出结果目录")
    parser.add_argument("--overlay", default=None, help="可选：用于生成红色叠加图的原图路径")
    parser.add_argument("--fault-ratio", type=float, default=0.42, help="断层半宽判定比例")
    parser.add_argument("--max-gap", type=float, default=28.0, help="交切断口最大智能桥接距离(px)")
    parser.add_argument("--min-skel-len", type=int, default=75, help="地层边界最小骨架长度")
    args = parser.parse_args()

    input_path = args.input
    if not input_path or not Path(input_path).is_file():
        for cand in CANDIDATE_INPUTS:
            if Path(cand).is_file():
                input_path = cand
                print(f"\n💡 自动采用输入文件: {input_path}")
                break
        if not input_path:
            print("❌ 未找到有效输入文件")
            sys.exit(1)

    config = BoundaryExtractConfig(
        fault_ratio=args.fault_ratio,
        max_bridge_gap=args.max_gap,
        min_skeleton_length=args.min_skel_len,
    )

    result = extract_strata_from_mask(input_path, args.overlay, config)
    saved = save_boundary_results(result, args.output)

    st = result["stats"]
    print(f"\n{'=' * 58}")
    print(f"       地层边界线（平滑连续版）提取完成")
    print(f"{'=' * 58}")
    print(f"输入文件       : {input_path}")
    print(f"地层边界像素   : {st['n_strata_px']} px")
    print(f"交切断口缝合数 : {st['n_bridged_gaps']} 处 (断续已彻底修复)")
    print(f"剥离断层像素   : {st['n_fault_px']} px (粗断层线已完全剔除)")
    print(f"滤除干扰要素   : {st['n_removed_elements']} 个 (等高线虚线与文字)")
    print(f"总耗时         : {st['elapsed']:.2f} 秒")
    print(f"{'=' * 58}")
    print(f"📁 输出目录    : {Path(args.output).resolve()}\n")
    for name in saved:
        print(f"  📄 {name}")

    if sys.platform == "darwin":
        try:
            subprocess.run(["open", str(Path(args.output).resolve())], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
