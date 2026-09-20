# -*- coding: utf-8 -*-
"""
extract_faults.py — 从地质图暗色掩膜(Mask)中提取断层构造线与中心骨架
=======================================================================
项目：智能化三维地质建模系统《基于平面地质图的智能化三维构建》

【地质背景与技术定位】
断层（Fault）是三维地质建模中的一阶构造不连续面：
  1. 在二维地质图上，断层线通常采用加粗实线（线宽 8~15px）绘制，明显粗于普通地层界线（1.5~3px）；
  2. 断层线常跨越并切断不同时代的地层色块，与地层线、等高线形成复杂的 T 型或十字型交切；
  3. 本模块采用与地层线分离完全对称的互补机制，从暗色 Mask 中精准提取断层构造线，
     剥离所有交切地层线、等高线虚线与文字注记，输出纯净的断层 Mask、单像素中心骨架及断层迹线坐标表。

【核心算法流程】
  1. 距离变换（Distance Transform）线宽分析：
     - 利用断层加粗特征定位全图最粗骨干（半宽 >= 0.42 * 全图最大半宽）；
  2. 构造主干最大测地线提取（Longest Geodesic Path）：
     - 针对断层与地层线交切处向外伸出的横向侧枝，通过全图骨架两极主干追踪，
       只保留贯穿全图的宏观断层中心走向线，自然剔除所有交切毛刺；
  3. 真实线宽自适应恢复：
     - 测量断层线沿骨架的法向真实半宽，限制在原二值图内受限膨胀恢复；
  4. 成果导出：
     - fault_mask.png     : 纯净断层掩膜（黑底白线）
     - fault_skeleton.png : 单像素中心骨架（用于断层曲面控制点生成）
     - fault_overlay.png  : 原图叠加亮绿色断层对比图
     - fault_trace.csv    : 断层迹线点序表（X, Y, fault_name）
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.morphology import skeletonize

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ExtractFault")


@dataclass
class FaultExtractConfig:
    """断层提取参数配置。

    fault_ratio : float
        断层线识别的半宽比例阈值（占全图最大半宽的比例）。默认 0.42。
    min_fault_area : int
        断层线核心种子的最小连通域面积，用于排除交切点局部极大值。默认 300。
    sample_step : int
        导出断层迹线 CSV 时的采样步长（每隔多少个像素取一个点）。默认 15。
    fault_name : str
        导出的断层名称标签。默认 "Fault_1"。
    """

    fault_ratio: float = 0.42
    min_fault_area: int = 300
    sample_step: int = 15
    fault_name: str = "Fault_1"


def load_mask_as_binary(img_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读取掩膜图像并转化为标准二值矩阵（1=线划, 0=背景）。"""
    p = Path(img_path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到输入文件: {p}")

    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像: {p}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    is_mask = "mask" in p.name.lower() or hsv[..., 1].mean() < 10.0

    if is_mask:
        white_count = (gray > 127).sum()
        black_count = (gray <= 127).sum()
        binary = (gray > 127).astype(np.uint8) if white_count < black_count else (gray <= 127).astype(np.uint8)
        logger.info("输入载入为【暗色掩膜(Mask)】: %s (%d x %d)", p.name, binary.shape[1], binary.shape[0])
        return binary, None
    else:
        binary = (gray < 55).astype(np.uint8)
        logger.info("输入载入为【彩色地质原图】: %s (%d x %d)", p.name, binary.shape[1], binary.shape[0])
        return binary, bgr


def extract_fault_mainline(
    binary: np.ndarray,
    config: FaultExtractConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """提取纯净的断层单像素中心线和断层掩膜。

    返回:
        fault_mask     : uint8 HxW (255=断层线, 0=背景)
        fault_skeleton : uint8 HxW (单像素中心线)
        mean_half_width: 断层线平均半宽 (px)
    """
    h, w = binary.shape

    # 1. 2px Padding 防边缘断开
    pad_bin = np.pad(binary, 2, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(pad_bin, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    fault_th = max(2.5, max_d * config.fault_ratio)

    fault_seed = (dist >= fault_th).astype(np.uint8)

    # 2. 提取大面积核心种子（排除交切伪种子）
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(fault_seed, 8)
    if n_cc <= 1:
        raise ValueError("未检测到足够粗壮的断层线特征，请检查输入 Mask 或调低 fault_ratio")

    # 找到最大面积的连通域
    top_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    core = (labels == top_idx).astype(np.uint8)
    logger.info("断层核心种子提取: 核心连通面积 %d px, 跨度 %d x %d",
                stats[top_idx, cv2.CC_STAT_AREA],
                stats[top_idx, cv2.CC_STAT_WIDTH],
                stats[top_idx, cv2.CC_STAT_HEIGHT])

    # 3. 骨架化
    skel = skeletonize(core).astype(np.uint8)

    # 4. 寻找两极端点并提取最长测地线主干（自动剥离横向地层线交切毛刺）
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filt = cv2.filter2D(skel, -1, kernel)
    ey, ex = np.where((filt == 11) & skel)
    pts = list(zip(ex, ey))

    if len(pts) < 2:
        main_path = list(zip(*np.where(skel)[::-1]))
    else:
        max_dist = 0.0
        best_pair = (pts[0], pts[1])
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
                if d > max_dist:
                    max_dist = d
                    best_pair = (pts[i], pts[j])

        coord_set = set(zip(*np.where(skel)[::-1]))
        q = deque([[best_pair[0]]])
        visited = {best_pair[0]}
        main_path = None

        while q:
            path = q.popleft()
            curr = path[-1]
            if curr == best_pair[1]:
                main_path = path
                break
            cx, cy = curr
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in coord_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append(path + [(nx, ny)])

        if main_path is None:
            main_path = list(coord_set)

    # 5. 去除 padding 并映射回原尺寸
    skel_unpad = np.zeros((h, w), dtype=np.uint8)
    for px, py in main_path:
        ux, uy = px - 2, py - 2
        if 0 <= ux < w and 0 <= uy < h:
            skel_unpad[uy, ux] = 1

    # 6. 计算实际线宽并受限膨胀恢复
    orig_dist = dist[2:-2, 2:-2]
    mean_hw = float(orig_dist[skel_unpad > 0].mean())
    k_sz = int(round(mean_hw * 0.95))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_sz + 1, 2 * k_sz + 1))
    fault_mask = cv2.dilate(skel_unpad, k) & binary

    logger.info("断层主干提取完成: 骨架长度 %d px, 平均半宽 %.2f px (全线宽约 %.1f px)",
                len(main_path), mean_hw, 2 * mean_hw)

    return (fault_mask * 255).astype(np.uint8), (skel_unpad * 255).astype(np.uint8), mean_hw


def export_fault_trace_csv(
    fault_skeleton: np.ndarray,
    csv_path: Union[str, Path],
    step: int = 15,
    fault_name: str = "Fault_1",
) -> int:
    """沿断层单像素骨架有序采样，导出控制点迹线表。"""
    pts = np.argwhere(fault_skeleton > 0)
    if len(pts) == 0:
        return 0

    # 按主坐标排序（y 坐标递增排序，即从上至下）
    sorted_pts = [tuple(int(c) for c in pt) for pt in sorted(pts, key=lambda p: (p[0], p[1]))]
    sampled = sorted_pts[::step]
    if len(sorted_pts) > 0 and sorted_pts[-1] != sampled[-1]:
        sampled.append(sorted_pts[-1])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "X_pixel", "Y_pixel", "fault_name"])
        for idx, pt in enumerate(sampled):
            # pt 为 (y, x)
            writer.writerow([idx + 1, pt[1], pt[0], fault_name])

    logger.info("断层迹线控制点已写入: %s (共采样 %d 点)", csv_path, len(sampled))
    return len(sampled)


def extract_fault_from_mask(
    mask_path: Union[str, Path],
    overlay_img_path: Optional[Union[str, Path]] = None,
    config: Optional[FaultExtractConfig] = None,
) -> dict:
    """主提取流水线接口。"""
    if config is None:
        config = FaultExtractConfig()

    t0 = time.time()
    binary, loaded_bgr = load_mask_as_binary(mask_path)
    h, w = binary.shape

    fault_mask, fault_skel, mean_hw = extract_fault_mainline(binary, config)

    # 叠加图
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
                if temp is not None:
                    if temp.shape[:2] == (h, w):
                        bg_bgr = temp
                    else:
                        bg_bgr = cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)
                    break

    overlay_bgr = None
    if bg_bgr is not None:
        overlay_bgr = bg_bgr.copy()
        overlay_bgr[fault_mask > 0] = [0, 255, 0] # 亮绿色高亮标注断层

    elapsed = time.time() - t0
    return {
        "fault_mask": fault_mask,
        "fault_skeleton": fault_skel,
        "overlay_bgr": overlay_bgr,
        "mean_half_width": mean_hw,
        "elapsed": elapsed,
    }


def save_fault_results(result: dict, output_dir: Union[str, Path], config: FaultExtractConfig) -> dict:
    """保存断层提取成果物。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    # 1. 掩膜
    f_mask = out / "fault_mask.png"
    cv2.imwrite(str(f_mask), result["fault_mask"])
    saved["fault_mask.png"] = str(f_mask)

    # 2. 骨架
    f_skel = out / "fault_skeleton.png"
    cv2.imwrite(str(f_skel), result["fault_skeleton"])
    saved["fault_skeleton.png"] = str(f_skel)

    # 3. 叠加图
    if result.get("overlay_bgr") is not None:
        f_overlay = out / "fault_overlay.png"
        cv2.imwrite(str(f_overlay), result["overlay_bgr"])
        saved["fault_overlay.png"] = str(f_overlay)

    # 4. CSV 迹线控制点
    f_csv = out / "fault_trace.csv"
    n_pts = export_fault_trace_csv(
        result["fault_skeleton"],
        f_csv,
        step=config.sample_step,
        fault_name=config.fault_name,
    )
    saved["fault_trace.csv"] = str(f_csv)

    return saved


CANDIDATE_INPUTS = [
    "test_mask.png",
    "segment_output/preview_mask.png",
    "cuted_map.png",
]
DEFAULT_OUTPUT = "./fault_output"


def main():
    parser = argparse.ArgumentParser(
        description="从地质图暗色掩膜(Mask)中提取纯净断层构造线与中心骨架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", default=None, help="输入 Mask 或地质图路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="输出结果目录")
    parser.add_argument("--overlay", default=None, help="可选：原图背景叠加图路径")
    parser.add_argument("--sample-step", type=int, default=15, help="断层迹线控制点采样步长(px)")
    parser.add_argument("--fault-name", default="Fault_1", help="断层名称")
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

    cfg = FaultExtractConfig(
        sample_step=args.sample_step,
        fault_name=args.fault_name,
    )

    result = extract_fault_from_mask(input_path, args.overlay, cfg)
    saved = save_fault_results(result, args.output, cfg)

    print(f"\n{'=' * 58}")
    print(f"       断层构造线精准提取与迹线构建完成")
    print(f"{'=' * 58}")
    print(f"输入文件       : {input_path}")
    print(f"断层平均线宽   : {result['mean_half_width'] * 2:.1f} px")
    print(f"断层骨架像素   : {(result['fault_skeleton'] > 0).sum()} px")
    print(f"提取耗时       : {result['elapsed']:.2f} 秒")
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
