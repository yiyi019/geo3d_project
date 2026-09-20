# -*- coding: utf-8 -*-
"""
geo_segment.py — 基于平面地质图的颜色分区与预处理程序
======================================================
基于平面地质图的智能化三维构建原型系统



【模块任务】
将输入的二维 JPG/PNG 栅格地质图进行图像预处理（保边滤波、去噪、光照校正）与干扰要素剥离
（黑色界线、粗断层线、等高线、文字代号、细小河流），并在感知均匀色彩空间（CIE-LAB）中
对地层颜色进行高精度聚类分割与层次合并，最后利用欧氏距离变换（EDT）使地质界线自然闭合，
输出干净的像素级地层标签图（labels.npy）、结构化地层单元表（units_auto.csv）及多维度可视化预览图。

【核心算法流程】
1. 扫描退化预处理：
   - 保边滤波（双边滤波 Bilateral Filter），消除图像噪点与 JPEG 压缩块效应，同时保护地质界线锐度；
   - 自适应强噪声检测（estimate_sigma），必要时启动非局部均值去噪（fastNlMeansDenoisingColored）；
   - 残差式不均匀光照校正，利用色彩残差平滑场消除纸质地质图扫描时的光照不均，不破坏平涂对比度。
2. 干扰要素排除掩膜构建：
   - 灰度阈值提取黑色地层分界线、深灰等高线、断层线与文字标注，结合椭圆形态学膨胀消除抗锯齿过渡边缘；
   - HSV 空间识别蓝色水系域，通过形态学开运算区分线状河流（排除回填）与面状湖泊水体（独立聚类）。
3. LAB 空间 KMeans 过聚类与层次合并：
   - CIE-LAB 感知均匀空间中欧氏距离对应人眼色差 ΔE76；
   - 采用大 K 值过聚类（k_max）避免遗漏细小地层单元；
   - 基于 ΔE76 距离矩阵进行层次聚类（linkage + fcluster），将过聚类的重复中心合并。
4. 全图标签快速赋值与 EDT 双向向内生长回填：
   - 分块计算所有非排除像素的最近中心分配，控制内存开销；
   - 利用欧氏距离变换（distance_transform_edt）将线划和文字覆盖区用最近地层标签回填，两侧地层
     向线划中心自然闭合，精准复原地层界线。
5. 后处理与小碎片清理：
   - 标签图中值滤波消除杂色孤立点；
   - 4-邻域连通域分析，将低于面积阈值的小碎屑自动并入周围主地层。
6. 结果导出与可视化：
   - labels.npy（全像素单元标签矩阵）；
   - units_auto.csv（包含单元编号、代表色 RGB/HEX、像素数、面积占比、角色推断）；
   - preview_clusters.png（地层填色重构图）；
   - preview_boundaries.png（原图叠加 1px 红色提取边界图）；
   - preview_mask.png（干扰排除掩膜图）。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GeoSegment")

# 蓝色水系判定的 HSV 默认范围（OpenCV 规范：H∈[0, 180], S∈[0, 255], V∈[0, 255]）
_WATER_H_MIN, _WATER_H_MAX = 95, 135
_WATER_S_MIN = 60
_WATER_V_MIN = 120


@dataclass
class GeoSegmentConfig:
    """地质图颜色分割与预处理参数配置。

    参数说明：
    ----------
    mode : str
        分割模式：'auto' 为全自动无监督聚类；'legend' 为基于图例颜色库的匹配分割。
    k_max : int
        过聚类初始簇数 K。应明显大于地质图中的实际地层单元数（例如预期 8-15 个地层，取 20-30）。
    merge_delta_e : float
        CIE-LAB 空间 ΔE76 簇中心合并色差阈值。低于此值的重复/相近中心将被合并。
        通常 4.0~6.0 既可合并重复中心，又可保留相近地层。
    dark_gray_max : int
        暗色线划/文字的灰度上限（0-255）。低于此灰度的像素视为线划、断层或文字。
    dark_dilate : int
        暗色掩膜的膨胀半径（像素）。用于吃掉线划抗锯齿（anti-aliasing）过渡带。
    smooth_median : int
        后处理标签图中值滤波核尺寸（必须为奇数，<=1 表示关闭）。用于消除离散椒盐噪点。
    min_region_px : int
        小连通域吸收阈值（像素面积）。小于此面积的细小孤岛和文字碎屑将被邻近主地层吞并。
    sample_max : int
        KMeans 聚类抽样上限。控制大图聚类的内存与计算耗时。
    seed : int
        随机种子，确保抽样与 KMeans 聚类结果完全可复现。
    legend_colors : list of tuple, optional
        legend 模式下的图例颜色定义列表：[("地层名", (R, G, B)), ...]
    illum_correct : bool
        是否启用残差式不均匀光照校正（对手机拍摄或旧图扫描光照不均特别有效）。
    denoise : bool
        是否启用双边滤波保边去噪（抑制噪点与压缩伪影）。
    """

    mode: str = "auto"
    k_max: int = 24
    merge_delta_e: float = 5.0
    dark_gray_max: int = 110
    dark_dilate: int = 2
    smooth_median: int = 5
    min_region_px: int = 150
    sample_max: int = 300_000
    seed: int = 42
    legend_colors: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None
    illum_correct: bool = True
    denoise: bool = True
    detect_water: bool = False    # 目前阶段暂不考虑水系分离，避免将奥陶/志留等蓝色地层误判为水体或被细河流抹除


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------


def _is_water_color(rgb: Union[List[int], Tuple[int, int, int], np.ndarray]) -> bool:
    """判断给定的 RGB 颜色是否属于蓝色水系范围。"""
    c = np.uint8([[list(rgb)]])
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[0, 0]
    return (
        _WATER_H_MIN <= int(hsv[0]) <= _WATER_H_MAX
        and int(hsv[1]) > _WATER_S_MIN
        and int(hsv[2]) > _WATER_V_MIN
    )


def _lab_to_rgb255(lab_center: np.ndarray) -> List[int]:
    """将单个 CIE-LAB 簇中心转换为 0~255 的整数 RGB 列表。"""
    rgb = lab2rgb(np.asarray(lab_center, dtype=np.float64).reshape(1, 1, 3))
    return [int(round(float(v) * 255.0)) for v in np.clip(rgb.reshape(3), 0.0, 1.0)]


def _rgb_to_hex(rgb: List[int]) -> str:
    """将 RGB 列表转换为十六进制颜色码（例如 #RRGGBB）。"""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _fill_from_nearest(labels: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """利用欧氏距离变换（EDT），将 hole_mask=True 的空洞/排除像素用最近的有效像素标签快速回填。

    算法优势：
    相比逐像素搜索或传统形态学膨胀回填，distance_transform_edt 可一次性返回每个空洞像素在二维网格中
    距离最近的有效像素坐标索引，通过花式索引实现 O(N) 极速回填，且能保证边界在线划中心精准交汇。
    """
    if not hole_mask.any() or hole_mask.all():
        return labels
    indices = distance_transform_edt(
        hole_mask, return_distances=False, return_indices=True
    )
    return labels[tuple(indices)]


def _assign_nearest_center(
    lab_flat: np.ndarray,
    keep_idx: np.ndarray,
    centers: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """对非排除掩膜像素分块计算到各 LAB 中心的欧氏距离，并赋予最近中心的标签。

    分块处理避免了大尺寸图像直接计算 (N, k) 全距离矩阵导致的内存暴涨。
    """
    labels_flat = np.zeros(lab_flat.shape[0], dtype=np.int16)
    c = centers.astype(np.float32)[None, :, :]  # (1, k, 3)

    for start in range(0, keep_idx.size, chunk_size):
        sel = keep_idx[start : start + chunk_size]
        part = lab_flat[sel][:, None, :]  # (chunk, 1, 3)
        dists_sq = ((part - c) ** 2).sum(axis=2)  # (chunk, k)
        labels_flat[sel] = dists_sq.argmin(axis=1).astype(np.int16)

    return labels_flat


# ----------------------------------------------------------------------
# 核心处理类与主函数
# ----------------------------------------------------------------------


class GeoSegmenter:
    """地质图颜色分割与预处理执行器。"""

    def __init__(self, config: Optional[GeoSegmentConfig] = None):
        self.config = config or GeoSegmentConfig()

    def process(self, img_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """执行地质图预处理与颜色分区主流程。

        参数
        ----
        img_input : str | Path | np.ndarray
            图像文件路径，或已载入的 RGB/BGR 图像矩阵（uint8）。

        返回
        ----
        dict:
            - "labels": np.ndarray (int16 HxW)，全像素地层单元标签矩阵
            - "units": list of dict，提取出的地层单元属性信息列表
            - "exclude_mask": np.ndarray (uint8 HxW)，排除并已回填的掩膜（1=线划/文字/河流/碎屑）
            - "water_mask": np.ndarray (uint8 HxW)，识别出的蓝色水系域掩膜
            - "cleaned_rgb": np.ndarray (uint8 HxWx3)，预处理校正后的 RGB 图像
            - "elapsed_time": float，处理总耗时（秒）
        """
        start_time = time.time()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # 1. 加载并规范化图像为 RGB 格式
        if isinstance(img_input, (str, Path)):
            p = Path(img_input)
            if not p.is_file():
                raise FileNotFoundError(f"未找到地质图文件: {p}")
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"无法使用 OpenCV 读取图像: {p}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.ndim == 3 and img_input.shape[2] >= 3:
                img = img_input[..., :3].copy()
            else:
                raise ValueError(f"无效的图像数组维度: {img_input.shape}")
        else:
            raise TypeError("img_input 必须是文件路径字符串、Path 对象或 numpy 数组")

        img = np.ascontiguousarray(img, dtype=np.uint8)
        h, w = img.shape[:2]
        logger.info("地质图载入成功，图像尺寸: %d x %d 像素", w, h)

        # --------------------------------------------------------------
        # 步骤 0：图像退化预处理（去噪 + 残差光照校正）
        # --------------------------------------------------------------
        if cfg.denoise:
            logger.info("步骤 0.1: 执行保边去噪滤波...")
            try:
                from skimage.restoration import estimate_sigma

                sigma_est = float(np.mean(estimate_sigma(img, channel_axis=-1)))
                if sigma_est > 8.0:
                    logger.info("检测到强高斯噪声 (sigma=%.2f)，启用非局部均值滤波", sigma_est)
                    img = cv2.fastNlMeansDenoisingColored(
                        img,
                        None,
                        h=int(min(3 * sigma_est, 30)),
                        hColor=10,
                        templateWindowSize=7,
                        searchWindowSize=21,
                    )
            except Exception as e:
                logger.debug("skimage 噪声估计跳过: %s", e)

            img = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=5)

        if cfg.illum_correct:
            logger.info("步骤 0.2: 执行残差式不均匀光照校正...")
            from dataclasses import replace

            # 首次粗聚类获取色彩基准
            first_cfg = replace(
                cfg,
                illum_correct=False,
                denoise=False,
                merge_delta_e=max(cfg.merge_delta_e, 14.0),
            )
            first_segmenter = GeoSegmenter(first_cfg)
            first_res = first_segmenter.process(img)

            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            lum = lab_img[:, :, 0]

            max_uid = max(u["unit_id"] for u in first_res["units"])
            center_l = np.zeros(max_uid + 1, dtype=np.float32)
            for u in first_res["units"]:
                c = np.uint8([[list(u["color"])]])
                center_l[u["unit_id"]] = float(
                    cv2.cvtColor(c, cv2.COLOR_RGB2LAB)[0, 0, 0]
                )

            resid = lum - center_l[np.clip(first_res["labels"], 0, max_uid)]
            valid = (first_res["exclude_mask"] == 0).astype(np.float32)
            sigma_spatial = min(h, w) / 8.0

            # 空间加权高斯平滑残差场
            num = cv2.GaussianBlur(resid * valid, (0, 0), sigmaX=sigma_spatial)
            den = cv2.GaussianBlur(valid, (0, 0), sigmaX=sigma_spatial) + 1e-6
            bg_field = num / den

            if float(np.abs(bg_field).max()) > 2.0:
                logger.info("光照场起伏显著 (max=%.2f)，扣除光照残差", np.abs(bg_field).max())
                lab_img[:, :, 0] = np.clip(lum - bg_field, 0, 255)
                img = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # --------------------------------------------------------------
        # 步骤 1：暗色线划、断层与注记文字提取
        # --------------------------------------------------------------
        logger.info("步骤 1: 提取黑色界线、断层线与文字标注掩膜...")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dark_mask = (gray < cfg.dark_gray_max).astype(np.uint8)

        if cfg.dark_dilate > 0:
            kernel_size = 2 * cfg.dark_dilate + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dark_mask = cv2.dilate(dark_mask, kernel)

        # --------------------------------------------------------------
        # 步骤 2：蓝色水系提取与形态学河流/湖泊分离（当前阶段默认跳过）
        # --------------------------------------------------------------
        if cfg.detect_water:
            logger.info("步骤 2: 提取蓝色水系并区分线状河流与湖泊水体...")
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            water_mask = (
                (hsv[..., 0] >= _WATER_H_MIN)
                & (hsv[..., 0] <= _WATER_H_MAX)
                & (hsv[..., 1] > _WATER_S_MIN)
                & (hsv[..., 2] > _WATER_V_MIN)
            ).astype(np.uint8)

            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            lake_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k5)
            river_mask = cv2.subtract(water_mask, lake_mask)

            if cfg.dark_dilate > 0:
                river_mask = cv2.dilate(river_mask, kernel)
                river_mask = cv2.subtract(river_mask, lake_mask)

            exclude_mask = (dark_mask > 0) | (river_mask > 0)
        else:
            logger.info("步骤 2: 暂不考虑水系分离（保护天蓝/青色等蓝色系地层不被误剔除）")
            water_mask = np.zeros((h, w), dtype=np.uint8)
            exclude_mask = (dark_mask > 0)

        exclude_percent = float(exclude_mask.sum()) / (h * w) * 100.0
        logger.info("干扰掩膜构建完成，共排除线划/文字等像素占比: %.2f%%", exclude_percent)

        # --------------------------------------------------------------
        # 步骤 3：CIE-LAB 色彩空间转换与 KMeans 过聚类
        # --------------------------------------------------------------
        logger.info("步骤 3: 转换至 CIE-LAB 空间并执行 KMeans 过聚类 (K=%d)...", cfg.k_max)
        lab_flat = rgb2lab(img.astype(np.float64) / 255.0).reshape(-1, 3).astype(np.float32)
        keep_idx = np.flatnonzero(~exclude_mask.ravel())

        if keep_idx.size == 0:
            raise ValueError("所有像素均被掩膜排除，请检查 dark_gray_max 参数！")

        if keep_idx.size > cfg.sample_max:
            sample_idx = rng.choice(keep_idx, size=cfg.sample_max, replace=False)
        else:
            sample_idx = keep_idx

        sample_data = lab_flat[sample_idx]
        actual_k = int(min(cfg.k_max, sample_data.shape[0]))

        km = KMeans(n_clusters=actual_k, n_init=4, random_state=cfg.seed)
        with warnings.catch_warnings():
            from sklearn.exceptions import ConvergenceWarning

            warnings.simplefilter("ignore", ConvergenceWarning)
            km.fit(sample_data)

        centers = km.cluster_centers_.astype(np.float64)
        center_counts = np.bincount(km.labels_, minlength=actual_k).astype(np.float64)

        # --------------------------------------------------------------
        # 步骤 4：基于 ΔE76 的层次聚类中心合并
        # --------------------------------------------------------------
        logger.info("步骤 4: 计算 ΔE76 距离矩阵并执行层次合并 (阈值=%.1f)...", cfg.merge_delta_e)
        if actual_k > 1:
            dist_matrix = pdist(centers)
            linkage_matrix = linkage(dist_matrix, method="average")
            cluster_groups = (
                fcluster(linkage_matrix, t=cfg.merge_delta_e, criterion="distance") - 1
            )
        else:
            cluster_groups = np.zeros(1, dtype=int)

        num_merged = int(cluster_groups.max()) + 1
        merged_centers = np.zeros((num_merged, 3), dtype=np.float64)

        for g in range(num_merged):
            sel = cluster_groups == g
            weight_sum = center_counts[sel].sum()
            weights = center_counts[sel] / weight_sum if weight_sum > 0 else None
            merged_centers[g] = np.average(centers[sel], axis=0, weights=weights)

        logger.info("中心合并完成: 初始 %d 簇 -> 合并后 %d 簇", actual_k, num_merged)

        # --------------------------------------------------------------
        # 步骤 5：全图分配与 EDT 双向向内生长回填
        # --------------------------------------------------------------
        logger.info("步骤 5: 分块分配全图像素标签...")
        labels_flat = _assign_nearest_center(lab_flat, keep_idx, merged_centers)
        labels = labels_flat.reshape(h, w)

        logger.info("步骤 6: 欧氏距离变换 (EDT) 界线与文字回填...")
        labels = _fill_from_nearest(labels, exclude_mask)

        # --------------------------------------------------------------
        # 步骤 7：中值滤波与小连通域清理
        # --------------------------------------------------------------
        if cfg.smooth_median and cfg.smooth_median > 1:
            ksz = cfg.smooth_median | 1
            labels = cv2.medianBlur(labels.astype(np.uint8), ksz).astype(np.int16)

        if cfg.min_region_px > 0:
            logger.info("步骤 7: 清理面积 < %d 像素的小连通碎屑...", cfg.min_region_px)
            small_mask = np.zeros((h, w), dtype=bool)
            for lab_val in np.unique(labels):
                _, comp, stats, _ = cv2.connectedComponentsWithStats(
                    (labels == lab_val).astype(np.uint8), connectivity=4
                )
                areas = stats[1:, cv2.CC_STAT_AREA]
                bad_ids = np.flatnonzero(areas < cfg.min_region_px) + 1
                if bad_ids.size > 0:
                    small_mask |= np.isin(comp, bad_ids)

            if small_mask.any() and not small_mask.all():
                labels = _fill_from_nearest(labels, small_mask)
                exclude_mask = (exclude_mask | small_mask).astype(np.uint8)

        # --------------------------------------------------------------
        # 步骤 8：构建结构化地层单元表与重编号
        # --------------------------------------------------------------
        total_pixels = h * w
        units: List[Dict[str, Any]] = []

        if cfg.mode == "legend" and cfg.legend_colors:
            logger.info("步骤 8: 按照给定图例颜色进行映射...")
            legend_names = [str(nm) for nm, _c in cfg.legend_colors]
            legend_rgb = np.array([list(c) for _nm, c in cfg.legend_colors], dtype=np.uint8)
            legend_lab = rgb2lab(legend_rgb.reshape(1, -1, 3).astype(np.float64) / 255.0).reshape(-1, 3)

            delta_e_matrix = np.linalg.norm(
                merged_centers[:, None, :] - legend_lab[None, :, :], axis=2
            )
            to_legend = delta_e_matrix.argmin(axis=1).astype(np.int16)
            labels = to_legend[labels]

            counts = np.bincount(labels.ravel(), minlength=len(legend_names))
            for li, name in enumerate(legend_names):
                px_count = int(counts[li])
                if px_count == 0:
                    continue
                color = [int(v) for v in legend_rgb[li]]
                role = "water" if (cfg.detect_water and _is_water_color(color)) else "strata"
                units.append(
                    {
                        "unit_id": int(li),
                        "name": name,
                        "color": color,
                        "hex": _rgb_to_hex(color),
                        "n_px": px_count,
                        "percent": round(px_count / total_pixels * 100.0, 2),
                        "role": role,
                    }
                )
        else:
            logger.info("步骤 8: 自动模式，按像素面积降序分配 unit_id (0..M-1)...")
            unique_vals, counts = np.unique(labels, return_counts=True)
            sort_order = np.argsort(-counts)

            lut = np.full(num_merged, -1, dtype=np.int16)
            for new_id, idx in enumerate(sort_order):
                old_val = int(unique_vals[idx])
                lut[old_val] = new_id
                color = _lab_to_rgb255(merged_centers[old_val])
                px_count = int(counts[idx])
                role = "water" if (cfg.detect_water and _is_water_color(color)) else "strata"
                units.append(
                    {
                        "unit_id": int(new_id),
                        "name": f"strata_{new_id}",
                        "color": color,
                        "hex": _rgb_to_hex(color),
                        "n_px": px_count,
                        "percent": round(px_count / total_pixels * 100.0, 2),
                        "role": role,
                    }
                )
            labels = lut[labels]

        elapsed = time.time() - start_time
        logger.info("分割完成！共提取 %d 个地层单元，用时: %.2f 秒", len(units), elapsed)

        return {
            "labels": labels.astype(np.int16),
            "units": units,
            "exclude_mask": exclude_mask.astype(np.uint8),
            "water_mask": water_mask.astype(np.uint8),
            "cleaned_rgb": img,
            "elapsed_time": elapsed,
        }


# ----------------------------------------------------------------------
# 导出与可视化保存函数
# ----------------------------------------------------------------------


def save_segment_results(
    result: Dict[str, Any],
    output_dir: Union[str, Path],
    original_rgb: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """将分割结果输出到指定目录中。

    写出的文件：
    1. labels.npy            : int16 HxW 地层标签矩阵
    2. units_auto.csv        : 结构化地质单元表
    3. preview_clusters.png  : 代表色填色预览图
    4. preview_boundaries.png: 原图叠加红色地层界线图（需传 original_rgb 或使用 cleaned_rgb）
    5. preview_mask.png      : 干扰要素排除掩膜预览图
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(result["labels"]).astype(np.int16)
    saved_files: Dict[str, str] = {}

    # 1. 保存 labels.npy
    lbl_file = out_path / "labels.npy"
    np.save(lbl_file, labels)
    saved_files["labels.npy"] = str(lbl_file)

    # 2. 保存 units_auto.csv
    rows = []
    for u in result["units"]:
        rows.append(
            {
                "unit_id": u["unit_id"],
                "name": u.get("name", f"strata_{u['unit_id']}"),
                "role": u["role"],
                "color_r": u["color"][0],
                "color_g": u["color"][1],
                "color_b": u["color"][2],
                "color_hex": u["hex"],
                "n_pixels": u["n_px"],
                "area_percent": u["percent"],
                "dip": "",
                "azimuth": "",
            }
        )
    csv_file = out_path / "units_auto.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False, encoding="utf-8-sig")
    saved_files["units_auto.csv"] = str(csv_file)

    # 3. 保存 preview_clusters.png
    max_id = int(labels.max())
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for u in result["units"]:
        palette[u["unit_id"]] = u["color"]
    cluster_preview = palette[np.clip(labels, 0, max_id)]
    cluster_file = out_path / "preview_clusters.png"
    cv2.imwrite(str(cluster_file), cv2.cvtColor(cluster_preview, cv2.COLOR_RGB2BGR))
    saved_files["preview_clusters.png"] = str(cluster_file)

    # 4. 保存 preview_boundaries.png
    bg_img = original_rgb if original_rgb is not None else result.get("cleaned_rgb")
    if bg_img is not None:
        overlay = np.ascontiguousarray(bg_img[..., :3], dtype=np.uint8).copy()
        boundary_mask = np.zeros(labels.shape, dtype=bool)
        boundary_mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
        boundary_mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        # 用高亮亮红色标注边界 (RGB: 255, 0, 0)
        overlay[boundary_mask] = (255, 0, 0)
        boundary_file = out_path / "preview_boundaries.png"
        cv2.imwrite(str(boundary_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved_files["preview_boundaries.png"] = str(boundary_file)

    # 5. 保存 preview_mask.png
    mask = result.get("exclude_mask")
    if mask is not None:
        mask_vis = (mask * 255).astype(np.uint8)
        mask_file = out_path / "preview_mask.png"
        cv2.imwrite(str(mask_file), mask_vis)
        saved_files["preview_mask.png"] = str(mask_file)

    logger.info("所有产物已写入目录: %s", out_path.resolve())
    return saved_files


def segment_geological_map(
    img_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[GeoSegmentConfig] = None,
) -> Dict[str, Any]:
    """便捷调用接口：输入地质图路径，完成分割并可选自动保存产物。"""
    segmenter = GeoSegmenter(config)
    res = segmenter.process(img_path)
    if output_dir is not None:
        res["saved_files"] = save_segment_results(res, output_dir)
    return res


# ----------------------------------------------------------------------
# 命令行 CLI 入口
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# VS Code 直接点击运行时的默认配置（您也可以直接在此修改图片路径）
# ----------------------------------------------------------------------
DEFAULT_INPUT_IMAGE = "cuted_map.png"       # 默认测试地质图
DEFAULT_OUTPUT_DIR = "./segment_output"  # 默认成果保存目录


def main():
    import subprocess

    parser = argparse.ArgumentParser(
        description="平面地质图颜色分区与图像预处理工具 (项目团队)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", default=None, help="输入的栅格地质图文件路径 (JPG/PNG)，留空则自动检测"
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, help="输出成果物保存目录"
    )
    parser.add_argument(
        "--k-max", type=int, default=24, help="初始 KMeans 过聚类最大簇数"
    )
    parser.add_argument(
        "--merge-delta-e",
        type=float,
        default=5.0,
        help="CIE-LAB 空间中心合并色差阈值 ΔE76",
    )
    parser.add_argument(
        "--dark-gray-max",
        type=int,
        default=110,
        help="暗色线划/文字灰度过滤阈值 (0-255)",
    )
    parser.add_argument(
        "--dark-dilate",
        type=int,
        default=2,
        help="暗色线划抗锯齿膨胀核半径 (像素)",
    )
    parser.add_argument(
        "--min-region-px",
        type=int,
        default=150,
        help="微小碎屑连通域过滤面积阈值",
    )
    parser.add_argument(
        "--no-illum", action="store_true", help="禁用残差式不均匀光照校正"
    )
    parser.add_argument(
        "--no-denoise", action="store_true", help="禁用保边滤波去噪"
    )
    parser.add_argument(
        "--detect-water", action="store_true", help="启用蓝色水系/河流分离（默认关闭，避免将蓝色地层误判为水）"
    )

    args = parser.parse_args()

    # 如果用户在 VS Code 中直接点“运行”按钮（未传入 -i 参数），或者指定的文件不存在
    input_path = args.input
    if not input_path or not Path(input_path).is_file():
        if input_path and not Path(input_path).is_file():
            print(f"\n⚠️ 提示：指定的文件 '{input_path}' 不存在，已自动切换为可用地质图。")
        # 1. 优先使用 DEFAULT_INPUT_IMAGE
        if Path(DEFAULT_INPUT_IMAGE).is_file():
            input_path = DEFAULT_INPUT_IMAGE
            print(f"💡 [VS Code 运行模式] 使用默认地质图: {input_path}")
        else:
            # 2. 自动在当前工作区扫描可用的地质图
            candidates = [
                p for p in Path(".").glob("*.*")
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
                and not p.name.startswith("preview")
                and not p.name.startswith("Section")
            ]
            if candidates:
                input_path = str(candidates[0])
                print(f"💡 [VS Code 运行模式] 自动选择当前目录图片: {input_path}")
            else:
                print("❌ 错误：当前目录下未找到任何 PNG/JPG 图片！请将地质图放入本目录或在脚本中指定路径。")
                sys.exit(1)
        print("   如需更换测试图片，可直接在 geo_segment.py 文件第 615 行修改 DEFAULT_INPUT_IMAGE，或使用命令行参数 -i。\n")

    config = GeoSegmentConfig(
        k_max=args.k_max,
        merge_delta_e=args.merge_delta_e,
        dark_gray_max=args.dark_gray_max,
        dark_dilate=args.dark_dilate,
        min_region_px=args.min_region_px,
        illum_correct=not args.no_illum,
        denoise=not args.no_denoise,
        detect_water=args.detect_water,
    )

    logger.info("正在处理地质图: %s", input_path)
    result = segment_geological_map(
        img_path=input_path, output_dir=args.output, config=config
    )

    print("\n" + "=" * 54)
    print("              地质图分区处理完成摘要")
    print("=" * 54)
    print(f"输入图像     : {Path(input_path).resolve()}")
    print(f"总耗时       : {result['elapsed_time']:.2f} 秒")
    print(f"提取单元数   : {len(result['units'])} 个")
    print("-" * 54)
    print(f"{'单元ID':<8}{'名称':<12}{'角色':<10}{'颜色(HEX)':<12}{'面积占比(%)':<10}")
    print("-" * 54)
    for u in result["units"]:
        print(
            f"{u['unit_id']:<8}{u['name']:<12}{u['role']:<10}{u['hex']:<12}{u['percent']:<10.2f}"
        )
    print("=" * 54)
    out_dir_abs = Path(args.output).resolve()
    print(f"📁 结果已保存在目录: {out_dir_abs}\n")

    # 在 macOS 系统中自动打开结果文件夹，方便用户在访达中直接查看
    if sys.platform == "darwin":
        try:
            subprocess.run(["open", str(out_dir_abs)], check=False)
            print("🚀 已自动在访达 (Finder) 中为您打开成果文件夹！\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
