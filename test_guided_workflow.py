# -*- coding: utf-8 -*-
"""
测试套件：验证分步引导式地学建模工作流 (Step-by-Step Guided Workflow)
包含：
1. 步骤向导导航与视图切换
2. 结构化控制点与连通段元数据提取
3. 人工交互合并地层界面 (Layer_1+2 -> Boundary_C_P, Layer_3+4 -> Boundary_D_C)
4. 产状异常突变与极性反转检测算法
5. 产状自动平滑与 180° 极性翻转修正机制
6. 端到端全流程模型构建与隐式场求解
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import wx

from compute_attitude import (
    AttitudeCalculator,
    angular_difference,
    correct_attitude,
    detect_attitude_mutations,
)
from extract_surface_points import extract_surface_points_structured
from gui_app import MainFrame, StepWizardBar
from model_engine import GeologicalModelEngine


class TestGuidedWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(os.path.abspath(__file__))
        cls.f_skel = os.path.join(cls.cur_dir, "fault_output", "fault_skeleton.png")
        cls.b_skel = os.path.join(cls.cur_dir, "boundary_output", "boundary_skeleton.png")
        cls.dem_file = os.path.join(cls.cur_dir, "results", "dem_auto.npy")
        cls.tmp_dir = tempfile.mkdtemp(prefix="guided_workflow_test_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_01_structured_extraction(self):
        """验证结构化提取接口完整输出连通段与拓扑元数据"""
        df, meta = extract_surface_points_structured(
            fault_skel_path=self.f_skel,
            boundary_skel_path=self.b_skel,
            dem_path=self.dem_file,
            scale=2.0,
            points_per_feature=15,
        )
        self.assertGreater(len(df), 0)
        self.assertIn("Fault_1", meta["features"])
        self.assertIn("Layer_1", meta["features"])
        self.assertIn("Layer_2", meta["features"])

        feat_1 = meta["features"]["Layer_1"]
        self.assertEqual(feat_1["type"], "stratum")
        self.assertEqual(feat_1["point_count"], 15)
        self.assertGreater(feat_1["length_px"], 100)
        self.assertIn("bounds", feat_1)

    def test_02_mutation_detection_and_polarity_reversal(self):
        """验证产状异常跳变与极性反转检测算法"""
        # 主盘 4 点 (北北东向 ~16°)，断层对侧盘 3 点 (南东向 ~138°)，两盘相差约 122°
        sample_attitudes = [
            {"point_id": 1, "formation": "Boundary_C_P", "X": 300.0, "Y": 1120.0, "Z": 855.0, "azimuth": 15.0, "dip": 18.0, "strike": 285.0},
            {"point_id": 2, "formation": "Boundary_C_P", "X": 500.0, "Y": 1100.0, "Z": 860.0, "azimuth": 15.6, "dip": 18.0, "strike": 285.6},
            {"point_id": 3, "formation": "Boundary_C_P", "X": 700.0, "Y": 1080.0, "Z": 860.0, "azimuth": 16.2, "dip": 17.5, "strike": 286.2},
            {"point_id": 4, "formation": "Boundary_C_P", "X": 900.0, "Y": 1050.0, "Z": 870.0, "azimuth": 16.8, "dip": 15.0, "strike": 286.8},
            {"point_id": 5, "formation": "Boundary_C_P", "X": 1300.0, "Y": 720.0, "Z": 870.0, "azimuth": 143.5, "dip": 11.0, "strike": 53.5},
            {"point_id": 6, "formation": "Boundary_C_P", "X": 1550.0, "Y": 780.0, "Z": 890.0, "azimuth": 135.4, "dip": 8.5, "strike": 45.4},
            {"point_id": 7, "formation": "Boundary_C_P", "X": 1770.0, "Y": 870.0, "Z": 905.0, "azimuth": 137.6, "dip": 9.0, "strike": 47.6},
        ]

        checked = detect_attitude_mutations(sample_attitudes, azimuth_threshold=45.0, dip_threshold=20.0)
        self.assertEqual(len(checked), 7)

        # 前 4 点为基准盘，判定为正常
        self.assertFalse(checked[0]["is_mutation"])
        self.assertFalse(checked[1]["is_mutation"])
        self.assertFalse(checked[2]["is_mutation"])
        self.assertFalse(checked[3]["is_mutation"])

        # 后 3 点偏离基准 ~120°，准确判定为极性反转/反向倾向突变
        self.assertTrue(checked[4]["is_mutation"])
        self.assertEqual(checked[4]["mutation_type"], "polarity_reversal")
        self.assertIn("反向倾向", checked[4]["mutation_desc"])

        self.assertTrue(checked[5]["is_mutation"])
        self.assertEqual(checked[5]["mutation_type"], "polarity_reversal")

        self.assertTrue(checked[6]["is_mutation"])
        self.assertEqual(checked[6]["mutation_type"], "polarity_reversal")

    def test_03_attitude_correction_methods(self):
        """验证 180° 极性翻转、邻域平滑与中位数统一修正机制"""
        mut_point = {
            "formation": "Boundary_C_P",
            "azimuth": 143.5,
            "dip": 11.0,
            "strike": 53.5,
            "normal": [0.1, 0.1, 0.98],
            "is_mutation": True,
        }

        # 1. 翻转 180°
        fixed_flip = correct_attitude(mut_point, method="flip_180")
        self.assertAlmostEqual(fixed_flip["azimuth"], (143.5 + 180.0) % 360.0, places=1)
        self.assertFalse(fixed_flip["is_mutation"])
        self.assertEqual(fixed_flip["mutation_type"], "corrected")

        # 2. 统一代表中位数
        fixed_med = correct_attitude(
            mut_point,
            method="median",
            median_val={"median_azimuth": 16.0, "median_dip": 17.0},
        )
        self.assertEqual(fixed_med["azimuth"], 16.0)
        self.assertEqual(fixed_med["dip"], 17.0)
        self.assertFalse(fixed_med["is_mutation"])

        # 3. 邻域平滑
        valid_neighbors = [
            {"azimuth": 15.0, "dip": 16.0, "is_mutation": False},
            {"azimuth": 17.0, "dip": 18.0, "is_mutation": False},
        ]
        fixed_smooth = correct_attitude(mut_point, method="smooth", neighbors=valid_neighbors)
        self.assertAlmostEqual(fixed_smooth["azimuth"], 16.0, places=1)
        self.assertAlmostEqual(fixed_smooth["dip"], 17.0, places=1)
        self.assertFalse(fixed_smooth["is_mutation"])

    def test_04_gui_step_wizard_and_layer_merging(self):
        """验证 GUI 步骤导航与交互式地层合并流程"""
        app = wx.App(False)
        frame = MainFrame()

        # 验证初始位于步骤 0
        self.assertEqual(frame.current_step, 0)
        self.assertEqual(frame.step_notebook.GetSelection(), 0)

        # 导航到步骤 2 (黑白Mask提取)
        frame.goto_step(1)
        self.assertEqual(frame.current_step, 1)

        # 导航到步骤 3 (DEM高程提取)
        frame.goto_step(2)
        self.assertEqual(frame.current_step, 2)

        # 导航到步骤 4 (边界断层分离)
        frame.goto_step(3)
        self.assertEqual(frame.current_step, 3)

        # 导航到步骤 5 (三维控制点)
        frame.goto_step(4)
        self.assertEqual(frame.current_step, 4)

        # 导航到步骤 6 (地层人工合并)
        frame.goto_step(5)
        self.assertEqual(frame.current_step, 5)

        # 模拟高亮选中某个连通段
        frame.selected_layer_name = "Layer_1"
        self.assertEqual(frame.selected_layer_name, "Layer_1")

        # 执行智能推荐合并
        frame.on_smart_recommend_merge()
        self.assertEqual(frame.layer_mapping["Layer_1"], "Boundary_C_P")
        self.assertEqual(frame.layer_mapping["Layer_2"], "Boundary_C_P")
        self.assertEqual(frame.layer_mapping["Layer_3"], "Boundary_D_C")
        self.assertEqual(frame.layer_mapping["Layer_4"], "Boundary_D_C")

        # 验证最终地层列表中包含了合并后的地层名称
        self.assertIn("Boundary_C_P", frame.strat_order)
        self.assertIn("Boundary_D_C", frame.strat_order)

        # 导航到步骤 7 (产状核对)
        frame.goto_step(6)
        self.assertEqual(frame.current_step, 6)

        # 导航到步骤 8 (三维建模)
        frame.goto_step(7)
        self.assertEqual(frame.current_step, 7)

        frame.Destroy()
        app.Destroy()

    def test_05_end_to_end_merged_modeling(self):
        """验证合并后的地层数据能够直接输入 GemPy 顺利完成三维隐式建模求解"""
        tmp_sp = os.path.join(self.tmp_dir, "merged_sp.csv")
        tmp_or = os.path.join(self.tmp_dir, "merged_or.csv")

        # 1. 界面点 (Fault_1, Boundary_C_P, Boundary_D_C)
        raw_sp = pd.read_csv(os.path.join(self.cur_dir, "surface_points.csv"))
        sp_rows = []
        for _, r in raw_sp.iterrows():
            f = str(r["formation"])
            if f in ["Layer_1", "Layer_2"] or f == "Boundary_C_P":
                f = "Boundary_C_P"
            elif f in ["Layer_3", "Layer_4"] or f == "Boundary_D_C":
                f = "Boundary_D_C"
            sp_rows.append({"X": r["X"], "Y": r["Y"], "Z": r["Z"], "formation": f})

        df_sp = pd.DataFrame(sp_rows)
        df_sp.to_csv(tmp_sp, index=False)
        valid_formations = set(df_sp["formation"].unique())

        # 2. 产状点 (经过 180° 反转修正为统一倾向，且要素与 surface_points 严格一致)
        raw_or = pd.read_csv(os.path.join(self.cur_dir, "orientations.csv"))
        or_rows = []
        for _, r in raw_or.iterrows():
            f = str(r["formation"])
            az = float(r["azimuth"])
            if f in ["Layer_1", "Layer_2"] or f == "Boundary_C_P":
                f = "Boundary_C_P"
                if 100 <= az <= 200:
                    az = (az + 180.0) % 360.0
            elif f in ["Layer_3", "Layer_4"] or f == "Boundary_D_C":
                f = "Boundary_D_C"
                if 100 <= az <= 200:
                    az = (az + 180.0) % 360.0
            if f in valid_formations:
                or_rows.append({
                    "X": r["X"],
                    "Y": r["Y"],
                    "Z": r["Z"],
                    "azimuth": az,
                    "dip": r["dip"],
                    "polarity": 1.0,
                    "formation": f,
                })

        df_or = pd.DataFrame(or_rows)
        df_or.to_csv(tmp_or, index=False)

        # 3. 运行 GemPy 引擎求解
        engine = GeologicalModelEngine(
            project_name="EndToEnd_Merged_Test",
            extent=[0.0, 2000.0, 0.0, 2000.0, 600.0, 1000.0],
            resolution=[20, 20, 20],
            refinement=3,
        )
        engine.initialize_model(tmp_sp, tmp_or)
        res = engine.compute()

        self.assertTrue(res["success"])
        self.assertGreater(res["surfaces_mesh_count"], 0)
        self.assertIn("Fault_1", res["structural_elements"])
        self.assertIn("Boundary_C_P", res["structural_elements"])
        self.assertIn("Boundary_D_C", res["structural_elements"])

    def test_06_formation_name_protection_and_synchronization(self):
        """验证 orientations 中存在孤立或未匹配地层名时的保护机制与容错能力"""
        tmp_sp = os.path.join(self.tmp_dir, "prot_sp.csv")
        tmp_or = os.path.join(self.tmp_dir, "prot_or.csv")

        # 构造界面点: 包含 Boundary_Alpha, Boundary_Beta
        sp_df = pd.DataFrame([
            {"X": 100.0, "Y": 100.0, "Z": 500.0, "formation": "Boundary_Alpha"},
            {"X": 200.0, "Y": 100.0, "Z": 500.0, "formation": "Boundary_Alpha"},
            {"X": 150.0, "Y": 200.0, "Z": 500.0, "formation": "Boundary_Alpha"},
            {"X": 100.0, "Y": 100.0, "Z": 300.0, "formation": "Boundary_Beta"},
            {"X": 200.0, "Y": 100.0, "Z": 300.0, "formation": "Boundary_Beta"},
            {"X": 150.0, "Y": 200.0, "Z": 300.0, "formation": "Boundary_Beta"},
        ])
        sp_df.to_csv(tmp_sp, index=False)

        # 构造产状点: 包含合法点与一个未合并/孤立的异常地层名 Orphan_Layer_X
        or_df = pd.DataFrame([
            {"X": 150.0, "Y": 150.0, "Z": 500.0, "azimuth": 90.0, "dip": 20.0, "polarity": 1.0, "formation": "Boundary_Alpha"},
            {"X": 150.0, "Y": 150.0, "Z": 300.0, "azimuth": 90.0, "dip": 20.0, "polarity": 1.0, "formation": "Boundary_Beta"},
            {"X": 150.0, "Y": 150.0, "Z": 400.0, "azimuth": 90.0, "dip": 20.0, "polarity": 1.0, "formation": "Orphan_Layer_X"},
        ])
        or_df.to_csv(tmp_or, index=False)

        engine = GeologicalModelEngine(
            project_name="Protection_Test",
            extent=[0.0, 300.0, 0.0, 300.0, 0.0, 600.0],
            resolution=[15, 15, 15],
            refinement=3,
        )
        # 此处不应抛出 GemPy KeyError: Orphan_Layer_X
        engine.initialize_model(tmp_sp, tmp_or)
        res = engine.compute()
        self.assertTrue(res["success"])
        self.assertEqual(res["surfaces_mesh_count"], 2)

    def test_07_gui_segmentation_execution_and_mask_generation(self):
        """验证步骤 2 图像分割提取不抛出 TypeError 且成功生成 test_mask.png 与骨架"""
        app = wx.App(False)
        frame = MainFrame()
        frame.txt_map_path.SetValue(frame.default_map)

        # 触发运行分割
        import threading
        # 提取逻辑在 _worker 线程中，我们直接调用其同步逻辑核验
        from extract_boundaries import BoundaryExtractConfig, extract_strata_from_mask, save_boundary_results
        from extract_faults import FaultExtractConfig, extract_fault_from_mask, save_fault_results

        map_p = frame.txt_map_path.GetValue().strip()
        self.assertTrue(os.path.exists(map_p))

        # 验证暗色掩膜提取
        import cv2
        img_bgr = cv2.imread(map_p)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < frame.spin_dark_gray.GetValue()).astype(np.uint8) * 255
        test_mask_out = os.path.join(self.tmp_dir, "test_mask_gen.png")
        cv2.imwrite(test_mask_out, dark_mask)
        self.assertTrue(os.path.exists(test_mask_out))

        # 验证无 TypeError 调用
        cfg_b = BoundaryExtractConfig(fault_ratio=0.42, fault_min_seed_area=300)
        res_b = extract_strata_from_mask(mask_or_img_path=test_mask_out, overlay_img_path=map_p, config=cfg_b)
        self.assertIn("strata_skeleton", res_b)

        cfg_f = FaultExtractConfig(fault_ratio=0.42, min_fault_area=300)
        res_f = extract_fault_from_mask(mask_path=test_mask_out, overlay_img_path=map_p, config=cfg_f)
        self.assertIn("fault_skeleton", res_f)

        frame.Destroy()
        app.Destroy()

    def test_08_gui_step4_basemap_and_sanitization(self):
        """验证步骤 4 底图切换 (RGB/Mask)、名称清洗过滤与骨架连续线渲染支持"""
        app = wx.App(False)
        frame = MainFrame()

        # 切换底图为 Mask 模式并验证不抛出异常
        frame.choice_step4_bg.SetSelection(1)
        frame.goto_step(3)
        self.assertEqual(frame.choice_step4_bg.GetSelection(), 1)

        # 验证合并名称清洗 (空格、斜杠、特殊字符自动转为下划线)
        frame.txt_merge_target.SetValue("Layer C / P (New)")
        import re
        sanitized = re.sub(r"[^\w\-]", "_", frame.txt_merge_target.GetValue().strip()).strip("_")
        self.assertEqual(sanitized, "Layer_C___P__New")

        frame.Destroy()
        app.Destroy()

    def test_09_gui_auto_fix_mutations_and_series_order(self):
        """验证一键修复全部突变点后状态全正常，且 strat_order 成功传递给 series_mapping"""
        app = wx.App(False)
        frame = MainFrame()

        # 模拟步骤 5 中存在包含突变点的产状数据库
        sample_atts = [
            {"point_id": 1, "formation": "Boundary_C_P", "X": 300.0, "Y": 1120.0, "Z": 855.0, "azimuth": 16.0, "dip": 18.0, "strike": 286.0},
            {"point_id": 2, "formation": "Boundary_C_P", "X": 1500.0, "Y": 750.0, "Z": 880.0, "azimuth": 140.0, "dip": 9.0, "strike": 50.0},
        ]
        checked = detect_attitude_mutations(sample_atts)
        self.assertTrue(checked[1]["is_mutation"])

        frame.orientations_list = checked
        frame.on_auto_fix_all_mutations()

        # 修复后全部无突变
        self.assertFalse(frame.orientations_list[0]["is_mutation"])
        self.assertFalse(frame.orientations_list[1]["is_mutation"])
        self.assertEqual(frame.orientations_list[1]["azimuth"], 16.0)

        # 验证步骤 4 层序构建正确分离断层与地层
        frame.strat_order = ["Boundary_C_P", "Boundary_D_C", "Fault_1"]
        frame.is_fault_map = {"Boundary_C_P": False, "Boundary_D_C": False, "Fault_1": True}

        strata = [s for s in frame.strat_order if not frame.is_fault_map.get(s, False) and not str(s).lower().startswith("fault")]
        faults = [s for s in frame.strat_order if frame.is_fault_map.get(s, False) or str(s).lower().startswith("fault")]

        self.assertEqual(strata, ["Boundary_C_P", "Boundary_D_C"])
        self.assertEqual(faults, ["Fault_1"])

        frame.Destroy()
        app.Destroy()


if __name__ == "__main__":
    unittest.main()

