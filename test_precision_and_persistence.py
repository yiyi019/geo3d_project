"""
测试套件：验证三维地质建模精细度控制 (Resolution & Refinement)、参数校验、动态更新与持久化体系
"""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from model_engine import GeologicalModelEngine


class TestPrecisionAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(os.path.abspath(__file__))
        cls.sp_file = os.path.join(cls.cur_dir, "surface_points.csv")
        cls.or_file = os.path.join(cls.cur_dir, "orientations.csv")
        cls.tmp_dir = tempfile.mkdtemp(prefix="gempy_test_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_01_parameter_validation_success(self):
        """测试合法参数的校验通过情况"""
        extent = [0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0]
        resolution = [40, 40, 30]
        refinement = 4
        surface_refinement = 4

        is_val, errs, warns = GeologicalModelEngine.validate_parameters(
            extent=extent,
            resolution=resolution,
            refinement=refinement,
            surface_refinement=surface_refinement,
        )
        self.assertTrue(is_val)
        self.assertEqual(len(errs), 0)

    def test_02_parameter_validation_errors(self):
        """测试非法参数的拦截与错误返回"""
        # 1. 范围颠倒
        is_val, errs, _ = GeologicalModelEngine.validate_parameters(
            extent=[2000.0, 1000.0, 0.0, 2000.0, 0.0, 800.0]
        )
        self.assertFalse(is_val)
        self.assertTrue(any("X 轴范围非法" in e for e in errs))

        # 2. 分辨率非正整数或过低
        is_val, errs, _ = GeologicalModelEngine.validate_parameters(
            resolution=[0, 30, 30]
        )
        self.assertFalse(is_val)
        self.assertTrue(any("正整数" in e for e in errs))

        is_val, errs, _ = GeologicalModelEngine.validate_parameters(
            resolution=[3, 30, 30]
        )
        self.assertFalse(is_val)
        self.assertTrue(any("单向至少为 5" in e for e in errs))

        # 3. 分辨率总点数超限 (> 200万)
        is_val, errs, _ = GeologicalModelEngine.validate_parameters(
            resolution=[200, 200, 100]  # 4,000,000
        )
        self.assertFalse(is_val)
        self.assertTrue(any("上限" in e for e in errs))

        # 4. 细化等级超出 1~8 范围
        is_val, errs, _ = GeologicalModelEngine.validate_parameters(refinement=0)
        self.assertFalse(is_val)
        is_val, errs, _ = GeologicalModelEngine.validate_parameters(refinement=9)
        self.assertFalse(is_val)

        # 5. 曲面细化高于总体细化产生预警
        is_val, errs, warns = GeologicalModelEngine.validate_parameters(
            refinement=4, surface_refinement=6
        )
        self.assertTrue(is_val)
        self.assertTrue(any("高于体网格细化等级" in w for w in warns))

    def test_03_engine_model_compute_metrics(self):
        """测试引擎实际计算并检验指标完整性 (体素尺寸、总数、曲面点面数)"""
        engine = GeologicalModelEngine(
            project_name="Test_Metrics",
            extent=[0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0],
            resolution=[30, 30, 20],
            refinement=3,
            surface_refinement=3,
        )
        engine.initialize_model(self.sp_file, self.or_file)
        res = engine.compute()

        self.assertTrue(res["success"])
        self.assertEqual(res["resolution"], [30, 30, 20])
        self.assertEqual(res["total_voxels"], 30 * 30 * 20)
        # dx = 2000/30 = 66.67, dy = 2000/30 = 66.67, dz = 800/20 = 40.0
        self.assertAlmostEqual(res["voxel_size"][0], 66.67, places=1)
        self.assertAlmostEqual(res["voxel_size"][2], 40.0, places=1)
        self.assertEqual(res["refinement"], 3)
        self.assertEqual(res["surface_refinement"], 3)
        self.assertGreater(res["surfaces_mesh_count"], 0)
        self.assertGreater(res["total_mesh_vertices"], 0)
        self.assertGreater(res["total_mesh_edges"], 0)

        # 验证 summary 字典
        summary = engine.get_summary()
        self.assertEqual(summary["total_voxels"], 18000)
        self.assertEqual(summary["refinement"], 3)
        self.assertEqual(summary["surfaces_mesh_count"], res["surfaces_mesh_count"])

    def test_04_dynamic_updates(self):
        """测试动态修改分辨率与细化等级及空间范围，并确认网格底层数组同步"""
        engine = GeologicalModelEngine(
            project_name="Test_Dyn",
            extent=[0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0],
            resolution=[25, 25, 20],
            refinement=3,
        )
        engine.initialize_model(self.sp_file, self.or_file)
        engine.compute()
        init_verts = sum(len(m.vertices) for m in engine.geo_model.solutions.dc_meshes)

        # 动态提高网格分辨率，验证底层 regular_grid.values 与 grid.values 同步更新
        engine.update_grid_resolution([35, 35, 25])
        self.assertFalse(engine.is_computed)
        expected_points = 35 * 35 * 25
        self.assertEqual(engine.geo_model.grid.regular_grid.values.shape, (expected_points, 3))
        self.assertEqual(engine.geo_model.grid.values.shape, (expected_points, 3))

        res2 = engine.compute()
        self.assertEqual(res2["resolution"], [35, 35, 25])
        self.assertEqual(res2["total_voxels"], expected_points)
        self.assertEqual(res2["lith_block_shape"], (expected_points,))

        # 动态更新空间范围 Extent
        new_extent = [100.0, 2100.0, 100.0, 2100.0, 200.0, 900.0]
        engine.update_extent(new_extent)
        self.assertFalse(engine.is_computed)
        self.assertEqual(list(engine.geo_model.grid.regular_grid.extent), new_extent)

        # 动态提高八叉树等级 (3 -> 4)
        engine.update_refinement(refinement=4, surface_refinement=4)
        res3 = engine.compute()
        new_verts = res3["total_mesh_vertices"]
        self.assertEqual(res3["refinement"], 4)
        self.assertGreater(new_verts, init_verts)

    def test_05_json_config_persistence(self):
        """测试 JSON 工程配置文件导出与全面恢复 (含数据路径与地形设置)"""
        cfg_path = os.path.join(self.tmp_dir, "test_proj.json")
        engine = GeologicalModelEngine(
            project_name="Export_Test",
            extent=[100.0, 2100.0, 200.0, 1800.0, 500.0, 1100.0],
            resolution=[50, 50, 40],
            refinement=5,
            surface_refinement=5,
        )
        engine.surface_points_path = self.sp_file
        engine.orientations_path = self.or_file
        engine.topography_mode = "random"
        engine.topography_params = {"d_z": [800.0, 1050.0]}

        exported = engine.export_config(cfg_path, extra_meta={"preset_name": "高精细度"})
        self.assertTrue(os.path.exists(exported))

        # 校验写入的内容
        with open(exported, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["project_name"], "Export_Test")
        self.assertEqual(data["resolution"], [50, 50, 40])
        self.assertEqual(data["refinement"], 5)
        self.assertEqual(data["surface_refinement"], 5)
        self.assertEqual(data["meta"]["preset_name"], "高精细度")
        self.assertEqual(data["surface_points_path"], self.sp_file)
        self.assertEqual(data["orientations_path"], self.or_file)
        self.assertEqual(data["topography"]["mode"], "random")

        # 使用新引擎载入并验证全字段恢复
        new_engine = GeologicalModelEngine()
        loaded = new_engine.load_config(exported)
        self.assertEqual(new_engine.project_name, "Export_Test")
        self.assertEqual(new_engine.resolution, [50, 50, 40])
        self.assertEqual(new_engine.refinement, 5)
        self.assertEqual(new_engine.surface_refinement, 5)
        self.assertEqual(new_engine.extent, [100.0, 2100.0, 200.0, 1800.0, 500.0, 1100.0])
        self.assertEqual(new_engine.surface_points_path, self.sp_file)
        self.assertEqual(new_engine.orientations_path, self.or_file)
        self.assertEqual(new_engine.topography_mode, "random")
        self.assertEqual(new_engine.topography_params, {"d_z": [800.0, 1050.0]})

    def test_06_gempy_model_save_and_load_sync(self):
        """测试 .gempy 格式保存与载入后的参数状态同步"""
        model_path = os.path.join(self.tmp_dir, "test_model.gempy")
        engine = GeologicalModelEngine(
            project_name="SaveLoad_Test",
            extent=[0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0],
            resolution=[30, 30, 25],
            refinement=3,
        )
        engine.initialize_model(self.sp_file, self.or_file)
        engine.compute()
        engine.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        # 新建一个无参引擎并载入
        loader_engine = GeologicalModelEngine()
        loader_engine.load_model(model_path)

        self.assertTrue(loader_engine.is_computed)
        self.assertEqual(loader_engine.resolution, [30, 30, 25])
        self.assertEqual(loader_engine.extent, [0.0, 2000.0, 0.0, 2000.0, 0.0, 800.0])
        self.assertEqual(loader_engine.refinement, 3)

    def test_07_gui_components_and_presets(self):
        """测试 GUI 界面类与精细度预设选择逻辑及细化联动切换"""
        import wx
        from gui_app import MainFrame

        app = wx.App(False)
        frame = MainFrame()

        # 验证预设选项与默认选中
        self.assertEqual(frame.choice_preset.GetSelection(), 1)  # 默认选中标准平衡
        self.assertEqual(frame.txt_nx.GetValue(), "40")
        self.assertEqual(frame.txt_ny.GetValue(), "40")
        self.assertEqual(frame.txt_nz.GetValue(), "30")
        self.assertEqual(frame.spin_refine.GetValue(), 4)
        self.assertEqual(frame.spin_surf_refine.GetValue(), 4)
        self.assertTrue(frame.chk_sync_refine.IsChecked())

        # 测试切换为快速草稿预设 (index 0)
        frame.choice_preset.SetSelection(0)
        frame.on_preset_selected()
        self.assertEqual(frame.txt_nx.GetValue(), "20")
        self.assertEqual(frame.txt_ny.GetValue(), "20")
        self.assertEqual(frame.txt_nz.GetValue(), "20")
        self.assertEqual(frame.spin_refine.GetValue(), 3)
        self.assertEqual(frame.spin_surf_refine.GetValue(), 3)
        self.assertTrue(frame.chk_sync_refine.IsChecked())

        # 测试独立修改曲面细化等级，联动同步复选框自动取消勾选
        frame.spin_surf_refine.SetValue(5)
        frame.on_surf_refine_changed()
        self.assertFalse(frame.chk_sync_refine.IsChecked())

        # 测试手动修改数值，预设自动降级为自定义
        frame.txt_nx.SetValue("47")
        frame.on_resolution_input_changed()
        self.assertEqual(frame.choice_preset.GetSelection(), len(frame.preset_options) - 1)

        # 测试恢复默认
        frame.on_reset_default_precision()
        self.assertEqual(frame.txt_nx.GetValue(), "40")
        self.assertEqual(frame.spin_refine.GetValue(), 4)
        self.assertTrue(frame.chk_sync_refine.IsChecked())

        frame.Destroy()
        app.Destroy()

    def test_08_cross_section_single_pass_no_recompute(self):
        """测试预先挂载剖面后 compute，add_cross_section 不重复置位 is_computed=False"""
        import gempy as gp
        from output_customizer import GeologicalOutputCustomizer

        engine = GeologicalModelEngine(
            resolution=[20, 20, 20], refinement=3, surface_refinement=3
        )
        engine.initialize_model(self.sp_file, self.or_file)
        customizer = GeologicalOutputCustomizer(engine)

        sec_name = "Sec_Bench"
        sec_start = (200.0, 200.0)
        sec_stop = (1800.0, 1800.0)
        sec_res = (50, 40)

        # 模拟后台线程预先挂载剖面
        customizer.sections_config[sec_name] = {
            "start": list(sec_start),
            "stop": list(sec_stop),
            "resolution": list(sec_res),
        }
        gp.set_section_grid(grid=engine.geo_model.grid, section_dict={
            sec_name: (list(sec_start), list(sec_stop), list(sec_res))
        })

        # 单次计算
        engine.compute()
        self.assertTrue(engine.is_computed)
        self.assertIsNotNone(engine.geo_model.solutions.raw_arrays.sections)

        # 验证调用 add_cross_section 不会抹掉 is_computed 状态
        customizer.add_cross_section(sec_name, sec_start, sec_stop, sec_res)
        self.assertTrue(engine.is_computed, "相同剖面不应破坏已计算状态，避免主线程二次重算卡死")

    def test_09_gui_validation_error_feedback(self):
        """测试 GUI 输入非法范围与低分辨率时的错误提示显示"""
        import wx
        from gui_app import MainFrame

        app = wx.App(False)
        frame = MainFrame()

        # 1. 测试范围反转触发错误提示
        frame.txt_xmin.SetValue("2000")
        frame.txt_xmax.SetValue("1000")
        frame.on_resolution_or_extent_changed()
        self.assertIn("范围参数非法", frame.lbl_voxel_info.GetLabel())

        # 2. 测试分辨率过低触发错误提示
        frame.txt_xmin.SetValue("0")
        frame.txt_xmax.SetValue("2000")
        frame.txt_nx.SetValue("3")
        frame.on_resolution_or_extent_changed()
        self.assertIn("网格分辨率过低", frame.lbl_voxel_info.GetLabel())

        frame.Destroy()
        app.Destroy()


if __name__ == "__main__":
    unittest.main()
