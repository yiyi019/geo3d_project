# -*- coding: utf-8 -*-
"""
基于平面地质图的智能化三维构建 - 桌面交互图形界面 (gui_app.py)
----------------------------------------------------------------------
智能化三维地质建模系统 MVP 原型系统

【核心工作流重构：分步引导式全流程地学建模系统】
步骤 1: 🗺️ 原始地质图输入与空间尺度标定
步骤 2: 🎭 图像分割与地质界线/断层 Mask 提取与展示
步骤 3: 📍 三维界面控制点 (Surface Points) 与各连通段 Layer 提取与可视化
步骤 4: 🧩 人工交互合并地层界面 (点击连通段在原图高亮显示、错断地层合并、层序编排)
步骤 5: 🧭 产状智能解算与异常突变检测/人工核对 (倾向跳变检测、极性180°反转、平滑修正)
步骤 6: 🏛️ GemPy 三维地质体建模与成果展示 (网格调优、剖面切割、虚拟钻孔、PyVista 3D、VTK导出)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import wx
import wx.lib.agw.genericmessagedialog as gmd

# 配置 Matplotlib 跨平台中文字体支持函数（确保被任何三方库覆盖后可随时恢复）
def setup_chinese_font():
    """配置 Matplotlib 跨平台中文字体支持，防止被三方库（如 gempy_viewer）覆盖导致中文豆腐块"""
    cn_fonts = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = cn_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

import gempy as gp
import gempy_viewer as gpv

# 确保在 gempy_viewer 导入后再次强制应用中文字体
setup_chinese_font()

from compute_attitude import (
    AttitudeCalculator,
    angular_difference,
    correct_attitude,
    detect_attitude_mutations,
)
from extract_boundaries import (
    BoundaryExtractConfig,
    extract_strata_from_mask,
    save_boundary_results,
)
from extract_faults import FaultExtractConfig, extract_fault_from_mask, save_fault_results
from extract_surface_points import extract_surface_points_structured
from model_engine import GeologicalModelEngine
from output_customizer import GeologicalOutputCustomizer


class StepWizardBar(wx.Panel):
    """顶部步骤向导指示条 (全流程 8 大闭环步骤)"""

    STEP_NAMES = [
        "① 原始地质图",
        "② 黑白Mask提取",
        "③ DEM高程提取",
        "④ 边界断层分离",
        "⑤ 三维控制点",
        "⑥ 地层人工合并",
        "⑦ 产状解算核对",
        "⑧ 三维建模展示",
    ]

    def __init__(self, parent, on_step_changed=None):
        super().__init__(parent)
        self.on_step_changed = on_step_changed
        self.active_step = 0
        self.buttons: List[wx.Button] = []
        self._init_ui()

    def _init_ui(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        for i, name in enumerate(self.STEP_NAMES):
            btn = wx.Button(self, label=name, size=(-1, 32))
            btn.Bind(wx.EVT_BUTTON, lambda e, idx=i: self._on_btn_click(idx))
            self.buttons.append(btn)
            sizer.Add(btn, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(sizer)
        self.refresh_styles()

    def _on_btn_click(self, idx: int):
        self.set_active_step(idx)
        if self.on_step_changed:
            self.on_step_changed(idx)

    def set_active_step(self, idx: int):
        self.active_step = max(0, min(idx, len(self.STEP_NAMES) - 1))
        self.refresh_styles()

    def refresh_styles(self):
        for i, btn in enumerate(self.buttons):
            if i == self.active_step:
                btn.SetBackgroundColour(wx.Colour(30, 100, 200))
                btn.SetForegroundColour(wx.WHITE)
                btn.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            elif i < self.active_step:
                btn.SetBackgroundColour(wx.Colour(220, 240, 220))
                btn.SetForegroundColour(wx.Colour(20, 120, 40))
                btn.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            else:
                btn.SetBackgroundColour(wx.Colour(245, 245, 245))
                btn.SetForegroundColour(wx.Colour(80, 80, 80))
                btn.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.Layout()


class MainFrame(wx.Frame):
    """智能化三维地质建模主程序窗口 (分步交互式工作流)"""

    def __init__(self, parent=None, title="智能化三维地质建模系统 (MVP) - 智能化三维地质建模系统"):
        super().__init__(parent, title=title, size=(1320, 880))

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine = GeologicalModelEngine(project_name="LZU_Geology_MVP")
        self.customizer = GeologicalOutputCustomizer(self.engine)

        # 默认数据路径
        self.default_map = (
            os.path.join(self.cur_dir, "cuted_map.png")
            if os.path.exists(os.path.join(self.cur_dir, "cuted_map.png"))
            else os.path.join(self.cur_dir, "raw_geo_map.png")
        )
        self.default_dem = (
            os.path.join(self.cur_dir, "dem_output", "dem_auto.npy")
            if os.path.exists(os.path.join(self.cur_dir, "dem_output", "dem_auto.npy"))
            else (
                os.path.join(self.cur_dir, "results", "dem_auto.npy")
                if os.path.exists(os.path.join(self.cur_dir, "results", "dem_auto.npy"))
                else os.path.join(self.cur_dir, "topography_sample.csv")
            )
        )
        self.default_sp = os.path.join(self.cur_dir, "surface_points.csv")
        self.default_or = os.path.join(self.cur_dir, "orientations.csv")

        # 工作流状态变量
        self.current_step = 0
        self.scale_gsd = 2.0  # 米/像素
        self.features_meta: Dict[str, Any] = {}
        self.layer_mapping: Dict[str, str] = {}  # 连通段原始名称 -> 合并后地层名称
        self.strat_order: List[str] = []  # 最终地层层序
        self.is_fault_map: Dict[str, bool] = {}  # 要素是否为断层
        self.selected_layer_name: Optional[str] = None  # 步骤 4 当前高亮显示的连通段
        self.orientations_list: List[Dict[str, Any]] = []  # 步骤 5 产状数据列表
        self.selected_att_idx: Optional[int] = None  # 步骤 5 当前选中查看/编辑的产状点
        self.surface_points_df: Optional[pd.DataFrame] = None

        # 预设精细度方案定义
        self.preset_options = [
            ("⚡ 快速草稿 (20×20×20, 细化 3) - 秒级初算验证", [20, 20, 20], 3, 3),
            ("✓ 标准平衡 (40×40×30, 细化 4) - 推荐日常建模", [40, 40, 30], 4, 4),
            ("💎 高精细度 (60×60×40, 细化 5) - 细腻平滑曲面", [60, 60, 40], 5, 5),
            ("🔬 科研出版 (80×80×50, 6) - 论文高清出图", [80, 80, 50], 6, 6),
            ("🚀 极致超精 (100×100×60, 细化 7) - 密集等值网格", [100, 100, 60], 7, 7),
            ("⚙️ 自定义参数 (自由调节)", None, None, None),
        ]

        # 初始化界面组件
        self._init_menu_bar()
        self._init_ui()
        self.Centre()
        self.Bind(wx.EVT_CLOSE, self.on_close_window)

        # 启动后就绪加载
        wx.CallAfter(self._initial_setup)

    def on_close_window(self, event):
        try:
            plt.close("all")
        except Exception:
            pass
        self.Destroy()

    def _init_menu_bar(self):
        menubar = wx.MenuBar()

        # 文件菜单
        file_menu = wx.Menu()
        item_calc = file_menu.Append(wx.ID_ANY, "重新计算三维模型\tCtrl+R", "重新加载数据并计算三维模型")
        file_menu.AppendSeparator()
        item_save_cfg = file_menu.Append(wx.ID_ANY, "保存建模配置方案 (JSON)...", "导出当前空间范围与网格精细度配置")
        item_load_cfg = file_menu.Append(wx.ID_ANY, "载入建模配置方案 (JSON)...", "从外部 JSON 载入配置方案")
        file_menu.AppendSeparator()
        item_load_gempy = file_menu.Append(wx.ID_ANY, "载入已有 GemPy 模型 (.gempy)...", "加载已有三维模型文件并同步界面参数")
        item_save_gempy = file_menu.Append(wx.ID_ANY, "保存 GemPy 模型 (.gempy)", "保存隐式模型数据")
        item_export_vtk = file_menu.Append(wx.ID_ANY, "导出所有地层曲面 (VTK)...", "导出为真实坐标 VTK 文件")
        file_menu.AppendSeparator()
        item_exit = file_menu.Append(wx.ID_EXIT, "退出\tCtrl+Q", "退出应用程序")

        # 视图与步骤跳转菜单
        step_menu = wx.Menu()
        for idx, name in enumerate(StepWizardBar.STEP_NAMES):
            itm = step_menu.Append(wx.ID_ANY, f"跳转至 {name}\tCtrl+{idx+1}")
            self.Bind(wx.EVT_MENU, lambda e, s=idx: self.goto_step(s), itm)

        # 帮助菜单
        help_menu = wx.Menu()
        item_about = help_menu.Append(wx.ID_ABOUT, "关于本系统", "查看项目背景与作者信息")

        menubar.Append(file_menu, "文件 (&F)")
        menubar.Append(step_menu, "步骤向导 (&S)")
        menubar.Append(help_menu, "帮助 (&H)")
        self.SetMenuBar(menubar)

        # 绑定通用菜单事件
        self.Bind(wx.EVT_MENU, lambda e: self.on_compute_model(), item_calc)
        self.Bind(wx.EVT_MENU, lambda e: self.on_save_config_profile(), item_save_cfg)
        self.Bind(wx.EVT_MENU, lambda e: self.on_load_config_profile(), item_load_cfg)
        self.Bind(wx.EVT_MENU, lambda e: self.on_load_gempy_model(), item_load_gempy)
        self.Bind(wx.EVT_MENU, lambda e: self.on_save_gempy_model(), item_save_gempy)
        self.Bind(wx.EVT_MENU, lambda e: self.on_export_vtk(), item_export_vtk)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), item_exit)
        self.Bind(wx.EVT_MENU, self.on_show_about, item_about)

    def _init_ui(self):
        # 整体分割窗口：左侧参数配置与步骤向导，右侧多维视图与日志
        self.splitter = wx.SplitterWindow(self, style=wx.SP_3D | wx.SP_LIVE_UPDATE)
        self.left_panel = wx.Panel(self.splitter)
        self.right_panel = wx.Panel(self.splitter)

        self._build_left_panel()
        self._build_right_panel()

        self.splitter.SplitVertically(self.left_panel, self.right_panel, 500)
        self.splitter.SetMinimumPaneSize(380)

        # 状态栏
        self.status_bar = self.CreateStatusBar(2)
        self.status_bar.SetStatusWidths([-2, -1])
        self.status_bar.SetStatusText("系统准备就绪 - 步骤 1: 原始地质图输入", 0)
        self.status_bar.SetStatusText("GemPy 3.x 内核", 1)

    def _build_left_panel(self):
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # 顶部步骤向导指示条
        self.step_bar = StepWizardBar(self.left_panel, on_step_changed=self.goto_step)
        left_sizer.Add(self.step_bar, 0, wx.EXPAND | wx.ALL, 5)

        # 8个步骤对应的内容容器 Notebook
        self.step_notebook = wx.Notebook(self.left_panel)

        self.pnl_step1 = wx.Panel(self.step_notebook)
        self._build_step1_panel(self.pnl_step1)
        self.step_notebook.AddPage(self.pnl_step1, "1. 原始地质图")

        self.pnl_step2 = wx.Panel(self.step_notebook)
        self._build_step2_panel(self.pnl_step2)
        self.step_notebook.AddPage(self.pnl_step2, "2. 黑白Mask提取")

        self.pnl_step3 = wx.Panel(self.step_notebook)
        self._build_step3_panel(self.pnl_step3)
        self.step_notebook.AddPage(self.pnl_step3, "3. DEM高程提取")

        self.pnl_step4 = wx.Panel(self.step_notebook)
        self._build_step4_panel(self.pnl_step4)
        self.step_notebook.AddPage(self.pnl_step4, "4. 边界断层分离")

        self.pnl_step5 = wx.Panel(self.step_notebook)
        self._build_step5_panel(self.pnl_step5)
        self.step_notebook.AddPage(self.pnl_step5, "5. 三维控制点")

        self.pnl_step6 = wx.Panel(self.step_notebook)
        self._build_step6_panel(self.pnl_step6)
        self.step_notebook.AddPage(self.pnl_step6, "6. 地层人工合并")

        self.pnl_step7 = wx.Panel(self.step_notebook)
        self._build_step7_panel(self.pnl_step7)
        self.step_notebook.AddPage(self.pnl_step7, "7. 产状解算核对")

        self.pnl_step8 = wx.Panel(self.step_notebook)
        self._build_step8_panel(self.pnl_step8)
        self.step_notebook.AddPage(self.pnl_step8, "8. 三维建模展示")

        self.step_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self._on_notebook_page_changed)
        left_sizer.Add(self.step_notebook, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # 底部导航按钮栏 (上一步 / 下一步)
        h_nav = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_prev_step = wx.Button(self.left_panel, label="◀ 上一步", size=(-1, 36))
        self.btn_prev_step.Bind(wx.EVT_BUTTON, lambda e: self.on_prev_step())
        self.btn_next_step = wx.Button(self.left_panel, label="下一步: 黑白Mask提取 ▶", size=(-1, 36))
        self.btn_next_step.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_next_step.SetBackgroundColour(wx.Colour(46, 139, 87))
        self.btn_next_step.SetForegroundColour(wx.WHITE)
        self.btn_next_step.Bind(wx.EVT_BUTTON, lambda e: self.on_next_step())

        h_nav.Add(self.btn_prev_step, 1, wx.RIGHT, 4)
        h_nav.Add(self.btn_next_step, 2)
        left_sizer.Add(h_nav, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.left_panel.SetSizer(left_sizer)

    # -------------------------------------------------------------------------
    # 步骤 1: 原始地质图输入与尺度标定
    # -------------------------------------------------------------------------
    def _build_step1_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_input = wx.StaticBox(parent, label="原始平面地质图输入")
        sb_sizer = wx.StaticBoxSizer(sb_input, wx.VERTICAL)

        # 原始地质图选择
        sb_sizer.Add(wx.StaticText(parent, label="原始平面地质图 (PNG/JPG/TIF):"), 0, wx.TOP | wx.LEFT, 4)
        h_m = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_map_path = wx.TextCtrl(parent, value=self.default_map)
        btn_m = wx.Button(parent, label="浏览...")
        btn_m.Bind(
            wx.EVT_BUTTON,
            lambda e: self._choose_file(self.txt_map_path, "选择原始地质图", "地质图图片 (*.png;*.jpg;*.tif)|*.png;*.jpg;*.tif|所有文件 (*.*)|*.*"),
        )
        self.txt_map_path.Bind(wx.EVT_TEXT, self._on_map_or_dem_path_changed)
        h_m.Add(self.txt_map_path, 1, wx.EXPAND | wx.RIGHT, 4)
        h_m.Add(btn_m, 0)
        sb_sizer.Add(h_m, 0, wx.EXPAND | wx.ALL, 4)

        # 比例尺标定
        h_s = wx.BoxSizer(wx.HORIZONTAL)
        lbl_scale = wx.StaticText(parent, label="空间测绘比例尺:")
        self.txt_scale_input = wx.TextCtrl(parent, value=f"{self.scale_gsd:.2f}", size=(70, -1))
        lbl_unit = wx.StaticText(parent, label="米 / 像素 (GSD)")
        self.txt_scale_input.Bind(wx.EVT_TEXT, self._on_scale_changed)
        h_s.Add(lbl_scale, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        h_s.Add(self.txt_scale_input, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        h_s.Add(lbl_unit, 0, wx.ALIGN_CENTER_VERTICAL)
        sb_sizer.Add(h_s, 0, wx.ALL, 6)

        # DEM 高程模型状态（已完全内嵌为步骤 ③ 自动生成，隐藏手动路径项保留变量）
        self.txt_dem_path = wx.TextCtrl(parent, value=self.default_dem)
        self.txt_dem_path.Hide()

        lbl_dem_info = wx.StaticText(
            parent,
            label="💡 DEM 高程模型说明：\n本系统已实现 DEM 从二值黑白 Mask 全自动提取与连续重建，前置于地质边界分离之前（步骤 ③），完全无需外部外接 DEM 数据！",
        )
        lbl_dem_info.SetForegroundColour(wx.Colour(0, 102, 153))
        lbl_dem_info.Wrap(420)
        sb_sizer.Add(lbl_dem_info, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(sb_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 图像信息展示面板
        sb_info = wx.StaticBox(parent, label="地质图空间范围快速体检")
        sb_info_sizer = wx.StaticBoxSizer(sb_info, wx.VERTICAL)
        self.lbl_map_inspect = wx.StaticText(
            parent,
            label="正在读取图像属性与空间测绘尺度...",
        )
        sb_info_sizer.Add(self.lbl_map_inspect, 1, wx.EXPAND | wx.ALL, 6)
        sizer.Add(sb_info_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 提示文案
        txt_tip = wx.StaticText(
            parent,
            label="【工作流指引】\n全流程 8 步分步闭环：① 输入地质图 ➔ ② 黑白Mask提取 ➔ ③ DEM高程自动重建 ➔ ④ 界线断层骨架分离 ➔ ⑤ 三维控制点提取 ➔ ⑥ 地层交互合并 ➔ ⑦ 产状质检 ➔ ⑧ 三维建模展示。",
        )
        txt_tip.SetForegroundColour(wx.Colour(90, 90, 90))
        txt_tip.Wrap(440)
        sizer.Add(txt_tip, 0, wx.ALL, 8)

        btn_run_s1 = wx.Button(parent, label="▶ 开始提取黑白综合掩膜 (进入步骤 2)", size=(-1, 38))
        btn_run_s1.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_run_s1.SetBackgroundColour(wx.Colour(30, 100, 200))
        btn_run_s1.SetForegroundColour(wx.WHITE)
        btn_run_s1.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(1))
        sizer.Add(btn_run_s1, 0, wx.EXPAND | wx.ALL, 8)

        parent.SetSizer(sizer)

    def _on_map_or_dem_path_changed(self, event=None):
        self._inspect_map_attributes()
        if self.current_step == 0:
            self._render_current_step_canvas()

    def _on_scale_changed(self, event=None):
        try:
            val = float(self.txt_scale_input.GetValue().strip())
            if val > 0:
                self.scale_gsd = val
                self._inspect_map_attributes()
                if self.current_step == 0:
                    self._render_current_step_canvas()
        except Exception:
            pass

    def _inspect_map_attributes(self):
        map_p = self.txt_map_path.GetValue().strip()
        dem_p = self.txt_dem_path.GetValue().strip()
        text_lines = []

        if os.path.exists(map_p):
            try:
                img = cv2.imread(map_p)
                h, w = img.shape[:2]
                text_lines.append(f"• 地质图尺寸: {w} × {h} 像素")
                text_lines.append(f"• 物理跨度: X[0.0, {w * self.scale_gsd:.1f}]m | Y[0.0, {h * self.scale_gsd:.1f}]m")
            except Exception as e:
                text_lines.append(f"• 地质图读取异常: {e}")
        else:
            text_lines.append("• 地质图: 待选择合法文件")

        if os.path.exists(dem_p):
            try:
                if dem_p.endswith(".npy"):
                    dem = np.load(dem_p)
                    text_lines.append(f"• 当前 DEM 状态: {dem.shape[1]} × {dem.shape[0]} 网格 (标高: {dem.min():.1f}m ~ {dem.max():.1f}m)")
                else:
                    text_lines.append(f"• 当前 DEM 状态: {os.path.basename(dem_p)}")
            except Exception as e:
                text_lines.append(f"• DEM 状态提示: {e}")
        else:
            text_lines.append("• DEM 状态: 待步骤 ③ 基于黑白Mask自动提取")

        self.lbl_map_inspect.SetLabel("\n".join(text_lines))
        self.pnl_step1.Layout()

    # -------------------------------------------------------------------------
    # 步骤 2: 黑白 Mask 提取（综合掩膜/等高线/界线）
    # -------------------------------------------------------------------------
    def _build_step2_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_mask = wx.StaticBox(parent, label="黑白综合掩膜提取参数 (Dark Mask)")
        sb_mask_sizer = wx.StaticBoxSizer(sb_mask, wx.VERTICAL)

        g_mask = wx.FlexGridSizer(1, 4, 6, 6)
        g_mask.AddGrowableCol(1)
        g_mask.AddGrowableCol(3)

        self.spin_dark_gray = wx.SpinCtrl(parent, min=20, max=150, initial=65)
        self.spin_dark_dilate = wx.SpinCtrl(parent, min=0, max=4, initial=0)
        self.spin_dark_dilate.SetToolTip("抗锯齿膨胀半径：设为0可保留等高线数字注记最锐利清晰的笔画，最利于步骤 ③ DEM 自动识别")

        g_mask.Add(wx.StaticText(parent, label="暗色灰度阈值:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_mask.Add(self.spin_dark_gray, 1, wx.EXPAND)
        g_mask.Add(wx.StaticText(parent, label="抗锯齿膨胀半径:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_mask.Add(self.spin_dark_dilate, 1, wx.EXPAND)
        sb_mask_sizer.Add(g_mask, 0, wx.EXPAND | wx.ALL, 4)

        btn_run_mask = wx.Button(parent, label="⚡ 运行/重新提取黑白综合掩膜 (test_mask.png)", size=(-1, 35))
        btn_run_mask.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_run_mask.Bind(wx.EVT_BUTTON, self.on_run_mask_extraction)
        sb_mask_sizer.Add(btn_run_mask, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(sb_mask_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 画布显示切换控制
        sb_view = wx.StaticBox(parent, label="右侧成果产物视图切换")
        sb_view_sizer = wx.StaticBoxSizer(sb_view, wx.VERTICAL)

        self.step2_view_options = [
            "① 原始彩色地质图 (Raw Map)",
            "② 暗色综合黑白掩膜 (test_mask.png)",
            "③ 掩膜与原图叠加对比图 (Overlay)",
        ]
        self.choice_step2_view = wx.Choice(parent, choices=self.step2_view_options)
        self.choice_step2_view.SetSelection(1)  # 默认显示暗色综合黑白掩膜
        self.choice_step2_view.Bind(wx.EVT_CHOICE, lambda e: self._render_current_step_canvas())
        sb_view_sizer.Add(self.choice_step2_view, 0, wx.EXPAND | wx.ALL, 6)

        self.lbl_mask_stat = wx.StaticText(
            parent,
            label="✓ 就绪: 点击上方按钮提取暗色黑白综合掩膜 (包含等高线/界线/断层)",
        )
        sb_view_sizer.Add(self.lbl_mask_stat, 0, wx.ALL, 6)
        sizer.Add(sb_view_sizer, 0, wx.EXPAND | wx.ALL, 5)

        btn_to_s3 = wx.Button(parent, label="▶ 下一步: DEM 高程自动提取 (进入步骤 3)", size=(-1, 38))
        btn_to_s3.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s3.SetBackgroundColour(wx.Colour(30, 100, 200))
        btn_to_s3.SetForegroundColour(wx.WHITE)
        btn_to_s3.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(2))
        sizer.Add(btn_to_s3, 0, wx.EXPAND | wx.ALL, 8)

        parent.SetSizer(sizer)

    def on_run_mask_extraction(self, event=None):
        """步骤 2: 从原始地质图提取二值综合黑白掩膜 (test_mask.png)"""
        map_path = self.txt_map_path.GetValue().strip()
        if not os.path.exists(map_path):
            wx.MessageBox("未找到原始地质图，请先在步骤 1 指定有效文件！", "提示", wx.OK | wx.ICON_WARNING)
            return

        self.log("\n>>> 开始提取黑白综合掩膜 (test_mask.png)...")
        self.status_bar.SetStatusText("正在提取黑白综合掩膜...", 0)

        def _worker():
            try:
                dark_gray_th = self.spin_dark_gray.GetValue()
                dilate_r = self.spin_dark_dilate.GetValue()

                img_bgr = cv2.imread(map_path)
                if img_bgr is None:
                    raise ValueError(f"无法读取地质图文件: {map_path}")
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                dark_mask = (gray < dark_gray_th).astype(np.uint8) * 255
                if dilate_r > 0:
                    k_size = 2 * dilate_r + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    dark_mask = cv2.dilate(dark_mask, kernel)

                mask_file = os.path.join(self.cur_dir, "test_mask.png")
                cv2.imwrite(mask_file, dark_mask)
                self.log(f"✓ 已提取暗色综合掩膜: {mask_file} (灰度<{dark_gray_th}, 膨胀={dilate_r})")

                def _done():
                    self.status_bar.SetStatusText("✓ 黑白综合掩膜提取成功", 0)
                    if hasattr(self, "lbl_mask_stat"):
                        h, w = dark_mask.shape
                        n_px = int(np.count_nonzero(dark_mask))
                        self.lbl_mask_stat.SetLabel(f"✓ 掩膜已生成: {w}×{h} px | 线划像素: {n_px} ({n_px/(w*h)*100:.1f}%)")
                    self._render_current_step_canvas()
                    wx.MessageBox(f"黑白综合掩膜提取完成！\n保存至: {mask_file}", "成功", wx.OK | wx.ICON_INFORMATION)

                wx.CallAfter(_done)
            except Exception as e:
                self.log(f"✖ 掩膜提取失败: {e}")

                def _err():
                    wx.MessageBox(f"黑白掩膜提取失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
                    self.status_bar.SetStatusText("掩膜提取失败", 0)

                wx.CallAfter(_err)

        threading.Thread(target=_worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # 步骤 3: DEM 高程模型自动提取 (完全由 Mask 自动生成，前置于边界分离)
    # -------------------------------------------------------------------------
    def _build_step3_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_dem = wx.StaticBox(parent, label="DEM 连续高程模型全自动提取 (AutoTerrainPipeline)")
        sb_dem_sizer = wx.StaticBoxSizer(sb_dem, wx.VERTICAL)

        txt_intro = wx.StaticText(
            parent,
            label="【前置全自动 DEM 流程】\n直接从步骤 ② 提取的黑白 Mask 中智能分离虚线等高线，通过切向跨断层定向缝合、OCR/自适应赋高程并结合闭合周长场约束三角剖分插值，全自动生成连续平滑高程模型，完全无需外部外接 DEM 数据。",
        )
        txt_intro.SetForegroundColour(wx.Colour(60, 60, 60))
        txt_intro.Wrap(440)
        sb_dem_sizer.Add(txt_intro, 0, wx.ALL, 6)

        h_ci = wx.BoxSizer(wx.HORIZONTAL)
        h_ci.Add(wx.StaticText(parent, label="基准等高距 (Contour Interval):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.spin_contour_interval = wx.SpinCtrl(parent, min=1, max=100, initial=10)
        self.spin_contour_interval.Bind(wx.EVT_SPINCTRL, lambda e: self._render_current_step_canvas())
        self.spin_contour_interval.Bind(wx.EVT_TEXT, lambda e: self._render_current_step_canvas())
        h_ci.Add(self.spin_contour_interval, 1, wx.EXPAND | wx.RIGHT, 6)
        h_ci.Add(wx.StaticText(parent, label="米 (m)"), 0, wx.ALIGN_CENTER_VERTICAL)
        sb_dem_sizer.Add(h_ci, 0, wx.EXPAND | wx.ALL, 6)

        btn_run_dem = wx.Button(parent, label="⚡ 运行全自动连续 DEM 重建 (从黑白 Mask)", size=(-1, 35))
        btn_run_dem.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_run_dem.SetBackgroundColour(wx.Colour(46, 139, 87))
        btn_run_dem.SetForegroundColour(wx.WHITE)
        btn_run_dem.Bind(wx.EVT_BUTTON, self.on_auto_generate_dem)
        sb_dem_sizer.Add(btn_run_dem, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(sb_dem_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 画布显示切换控制
        sb_view = wx.StaticBox(parent, label="右侧 DEM 成果视图切换")
        sb_view_sizer = wx.StaticBoxSizer(sb_view, wx.VERTICAL)

        self.step3_view_options = [
            "① 连续平滑 DEM 彩色高程热力图 (含等高线)",
            "② 全自动 3D 连续地貌透视图 (dem_3d_view.png)",
            "③ DEM 成果 4合1 科研大看板 (demo_result_dashboard.png)",
        ]
        self.choice_step3_view = wx.Choice(parent, choices=self.step3_view_options)
        self.choice_step3_view.SetSelection(0)
        self.choice_step3_view.Bind(wx.EVT_CHOICE, lambda e: self._render_current_step_canvas())
        sb_view_sizer.Add(self.choice_step3_view, 0, wx.EXPAND | wx.ALL, 6)

        self.lbl_dem_stat = wx.StaticText(
            parent,
            label="✓ DEM 状态: 点击上方按钮即可自动由黑白 Mask 重建连续 DEM",
        )
        sb_view_sizer.Add(self.lbl_dem_stat, 0, wx.ALL, 6)
        sizer.Add(sb_view_sizer, 0, wx.EXPAND | wx.ALL, 5)

        btn_to_s4 = wx.Button(parent, label="▶ 下一步: 地质边界与断层分离 (进入步骤 4)", size=(-1, 38))
        btn_to_s4.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s4.SetBackgroundColour(wx.Colour(30, 100, 200))
        btn_to_s4.SetForegroundColour(wx.WHITE)
        btn_to_s4.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(3))
        sizer.Add(btn_to_s4, 0, wx.EXPAND | wx.ALL, 8)

        parent.SetSizer(sizer)

    def on_auto_generate_dem(self, event=None):
        """利用 AutoTerrainPipeline 从黑白 Mask 全自动生成连续 DEM 高程网格"""
        map_p = self.txt_map_path.GetValue().strip()
        mask_p = os.path.join(self.cur_dir, "test_mask.png")
        if not os.path.exists(mask_p):
            if os.path.exists(map_p):
                img = cv2.imread(map_p)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    d_mask = (gray < 65).astype(np.uint8) * 255
                    cv2.imwrite(mask_p, d_mask)
            if not os.path.exists(mask_p):
                wx.MessageBox("未找到有效掩膜文件或地质图，请先在步骤 2 提取黑白 Mask！", "提示", wx.OK | wx.ICON_WARNING)
                return

        ci_val = float(self.spin_contour_interval.GetValue()) if hasattr(self, "spin_contour_interval") else 10.0
        self.log(f"\n>>> 开始从黑白 Mask 执行全自动连续 DEM 重建 (设定等高距 {ci_val}m)...")
        self.status_bar.SetStatusText("正在生成连续 DEM 高程网格...", 0)

        def _worker():
            try:
                from demo_auto_dem import AutoTerrainPipeline
                pipeline = AutoTerrainPipeline(contour_interval=ci_val)
                out_d = os.path.join(self.cur_dir, "dem_output")
                res = pipeline.run(mask_p, output_dir=out_d, contour_interval=ci_val)
                dem_file = os.path.join(out_d, "dem_auto.npy")

                # 同时同步一份到 results/ 保证向前完全兼容
                res_d = os.path.join(self.cur_dir, "results")
                os.makedirs(res_d, exist_ok=True)
                if os.path.exists(dem_file):
                    import shutil
                    shutil.copy2(dem_file, os.path.join(res_d, "dem_auto.npy"))

                def _done():
                    self.txt_dem_path.SetValue(dem_file)
                    self.log(f"✓ DEM 连续高程矩阵已生成: {dem_file} (高程区间: {res['elevation_range'][0]:.1f}m ~ {res['elevation_range'][1]:.1f}m)")
                    self.status_bar.SetStatusText("✓ DEM 自动提取成功", 0)
                    if hasattr(self, "lbl_dem_stat"):
                        self.lbl_dem_stat.SetLabel(f"✓ DEM 尺寸: {res['dem_shape'][1]}×{res['dem_shape'][0]} 网格 | 标高: {res['elevation_range'][0]:.1f}m ~ {res['elevation_range'][1]:.1f}m (高差 {res['elevation_range'][1]-res['elevation_range'][0]:.1f}m)")
                    self._inspect_map_attributes()
                    self._render_current_step_canvas()
                    wx.MessageBox(f"全自动连续 DEM 生成成功!\n已保存至: {dem_file}\n标高范围: {res['elevation_range'][0]:.1f}m ~ {res['elevation_range'][1]:.1f}m", "成功", wx.OK | wx.ICON_INFORMATION)

                wx.CallAfter(_done)
            except Exception as e:
                self.log(f"✖ DEM 自动生成失败: {e}")

                def _err():
                    self.status_bar.SetStatusText("DEM 生成失败", 0)
                    wx.MessageBox(f"DEM 自动生成失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

                wx.CallAfter(_err)

        threading.Thread(target=_worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # 步骤 4: 地质界线与断层线粗细分离提取
    # -------------------------------------------------------------------------
    def _build_step4_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_seg = wx.StaticBox(parent, label="地质界线与断层线粗细特征分离参数")
        sb_seg_sizer = wx.StaticBoxSizer(sb_seg, wx.VERTICAL)

        g_seg = wx.FlexGridSizer(1, 4, 6, 6)
        g_seg.AddGrowableCol(1)
        g_seg.AddGrowableCol(3)

        self.txt_fault_ratio = wx.TextCtrl(parent, value="0.42")
        self.txt_min_fault_area = wx.TextCtrl(parent, value="300")

        g_seg.Add(wx.StaticText(parent, label="断层粗线阈比:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_seg.Add(self.txt_fault_ratio, 1, wx.EXPAND)
        g_seg.Add(wx.StaticText(parent, label="断层最小面积:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_seg.Add(self.txt_min_fault_area, 1, wx.EXPAND)
        sb_seg_sizer.Add(g_seg, 0, wx.EXPAND | wx.ALL, 4)

        lbl_hint_ratio = wx.StaticText(parent, label="💡 粗线阈比: 粗断层/细界线分离敏感度，推荐 0.35 ~ 0.50 (默认 0.42，越高要求线条越粗)")
        lbl_hint_ratio.SetForegroundColour(wx.Colour(100, 100, 100))
        lbl_hint_ratio.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sb_seg_sizer.Add(lbl_hint_ratio, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        lbl_hint_area = wx.StaticText(parent, label="💡 最小面积: 断层连通像素下限，推荐 200 ~ 600 px (默认 300，滤除碎短虚线/字斑)")
        lbl_hint_area.SetForegroundColour(wx.Colour(100, 100, 100))
        lbl_hint_area.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sb_seg_sizer.Add(lbl_hint_area, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        btn_run_seg = wx.Button(parent, label="⚡ 运行/重新分离提取地质界线与断层骨架", size=(-1, 35))
        btn_run_seg.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_run_seg.Bind(wx.EVT_BUTTON, self.on_run_boundary_fault_separation)
        sb_seg_sizer.Add(btn_run_seg, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(sb_seg_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 画布显示切换控制
        sb_view = wx.StaticBox(parent, label="右侧成果产物视图切换")
        sb_view_sizer = wx.StaticBoxSizer(sb_view, wx.VERTICAL)

        self.step4_view_options = [
            "① 细地层边界纯净骨架 (boundary_skeleton.png)",
            "② 粗断层走向中心骨架 (fault_skeleton.png)",
            "③ 粗细分离综合叠加对比图 (Overlay)",
        ]
        self.choice_step4_view = wx.Choice(parent, choices=self.step4_view_options)
        self.choice_step4_view.SetSelection(2)  # 默认显示综合对比图
        self.choice_step4_view.Bind(wx.EVT_CHOICE, lambda e: self._render_current_step_canvas())
        sb_view_sizer.Add(self.choice_step4_view, 0, wx.EXPAND | wx.ALL, 6)

        self.lbl_bf_stat = wx.StaticText(
            parent,
            label="✓ 已就绪: 检测到预提取骨架成果 (boundary_skeleton.png, fault_skeleton.png)",
        )
        sb_view_sizer.Add(self.lbl_bf_stat, 0, wx.ALL, 6)
        sizer.Add(sb_view_sizer, 0, wx.EXPAND | wx.ALL, 5)

        btn_to_s5 = wx.Button(parent, label="▶ 下一步: 提取三维控制点与连通段 (进入步骤 5)", size=(-1, 38))
        btn_to_s5.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s5.SetBackgroundColour(wx.Colour(30, 100, 200))
        btn_to_s5.SetForegroundColour(wx.WHITE)
        btn_to_s5.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(4))
        sizer.Add(btn_to_s5, 0, wx.EXPAND | wx.ALL, 8)

        parent.SetSizer(sizer)

    def on_run_boundary_fault_separation(self, event=None):
        """步骤 4: 基于黑白 Mask 分离地质界线与断层骨架"""
        map_path = self.txt_map_path.GetValue().strip()
        mask_file = os.path.join(self.cur_dir, "test_mask.png")
        if not os.path.exists(mask_file):
            if os.path.exists(map_path):
                img_bgr = cv2.imread(map_path)
                if img_bgr is not None:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(mask_file, (gray < 65).astype(np.uint8) * 255)
            if not os.path.exists(mask_file):
                wx.MessageBox("未找到有效掩膜文件，请先在步骤 2 提取黑白 Mask！", "提示", wx.OK | wx.ICON_WARNING)
                return

        self.log("\n>>> 开始执行地质界线与断层线粗细特征分离提取...")
        self.status_bar.SetStatusText("正在进行图像分割与骨架提取...", 0)

        def _worker():
            try:
                fault_ratio_val = float(self.txt_fault_ratio.GetValue())
                min_fault_area_val = int(self.txt_min_fault_area.GetValue())

                # 1. 提取地层边界
                cfg_b = BoundaryExtractConfig(
                    fault_ratio=fault_ratio_val,
                    fault_min_seed_area=min_fault_area_val,
                )
                res_b = extract_strata_from_mask(
                    mask_or_img_path=mask_file,
                    overlay_img_path=map_path if os.path.exists(map_path) else None,
                    config=cfg_b,
                )
                b_dir = os.path.join(self.cur_dir, "boundary_output")
                save_boundary_results(res_b, b_dir)

                # 2. 提取断层
                cfg_f = FaultExtractConfig(
                    fault_ratio=fault_ratio_val,
                    min_fault_area=min_fault_area_val,
                )
                res_f = extract_fault_from_mask(
                    mask_path=mask_file,
                    overlay_img_path=map_path if os.path.exists(map_path) else None,
                    config=cfg_f,
                )
                f_dir = os.path.join(self.cur_dir, "fault_output")
                save_fault_results(res_f, f_dir, config=cfg_f)

                self.log("✓ 地质界线与断层线骨架提取完毕并已保存。")

                def _done():
                    self.status_bar.SetStatusText("✓ 地质界线与断层骨架分离成功", 0)
                    if hasattr(self, "lbl_bf_stat"):
                        self.lbl_bf_stat.SetLabel("✓ 最新骨架成果已生成 (boundary_skeleton.png, fault_skeleton.png)")
                    self._render_current_step_canvas()
                    wx.MessageBox("地质界线与断层骨架分离提取完成！", "成功", wx.OK | wx.ICON_INFORMATION)

                wx.CallAfter(_done)
            except Exception as e:
                self.log(f"✖ 分割提取失败: {e}")

                def _err():
                    wx.MessageBox(f"图像分割提取失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
                    self.status_bar.SetStatusText("分割提取失败", 0)

                wx.CallAfter(_err)

        threading.Thread(target=_worker, daemon=True).start()

    on_run_segmentation_and_masks = on_run_boundary_fault_separation  # 兼容别名

    # -------------------------------------------------------------------------
    # 步骤 5: 控制点与连通段要素提取 (结合步骤 ③ DEM 标高)
    # -------------------------------------------------------------------------
    def _build_step5_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_samp = wx.StaticBox(parent, label="三维控制点采样与微小噪点过滤设置")
        sb_samp_sizer = wx.StaticBoxSizer(sb_samp, wx.VERTICAL)

        g_p = wx.FlexGridSizer(1, 4, 6, 6)
        g_p.AddGrowableCol(1)
        g_p.AddGrowableCol(3)

        self.spin_pts_per_feature = wx.SpinCtrl(parent, min=5, max=100, initial=20)
        self.spin_min_skel_len = wx.SpinCtrl(parent, min=2, max=100, initial=15)
        self.spin_min_skel_len.SetToolTip("自动过滤骨架长度低于此阈值的杂散孤立噪点，彻底防止单像素噪点误判引发崩溃")

        g_p.Add(wx.StaticText(parent, label="单条线控制点数:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_p.Add(self.spin_pts_per_feature, 1, wx.EXPAND)
        g_p.Add(wx.StaticText(parent, label="最小骨架阈值 (px):"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_p.Add(self.spin_min_skel_len, 1, wx.EXPAND)
        sb_samp_sizer.Add(g_p, 0, wx.EXPAND | wx.ALL, 4)

        btn_extract = wx.Button(parent, label="⚡ 重新提取控制点 (结合步骤 ③ DEM 标高)", size=(-1, 35))
        btn_extract.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_extract.Bind(wx.EVT_BUTTON, self.on_extract_surface_points_step)
        sb_samp_sizer.Add(btn_extract, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(sb_samp_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 提取连通段要素列表
        sb_feat = wx.StaticBox(parent, label="已提取独立构造线与地层连通段 (Raw Layers)")
        sb_feat_sizer = wx.StaticBoxSizer(sb_feat, wx.VERTICAL)

        self.list_step5_features = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_step5_features.InsertColumn(0, "要素代号", width=85)
        self.list_step5_features.InsertColumn(1, "类型", width=70)
        self.list_step5_features.InsertColumn(2, "骨架长 (px)", width=85)
        self.list_step5_features.InsertColumn(3, "控制点数", width=70)
        self.list_step5_features.InsertColumn(4, "X 跨度 (m)", width=95)
        self.list_step5_features.InsertColumn(5, "Y 跨度 (m)", width=95)
        self.list_step5_features.InsertColumn(6, "标高 Z (m)", width=95)
        self.list_step3_features = self.list_step5_features  # 兼容别名

        sb_feat_sizer.Add(self.list_step5_features, 1, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_feat_sizer, 1, wx.EXPAND | wx.ALL, 5)

        txt_info = wx.StaticText(
            parent,
            label="说明：程序结合骨架与步骤 ③ 生成的高程模型提取真实三维空间控制点 (X, Y, Z)，微小噪点已自动过滤。请点击下方按钮进入【地层人工合并】。",
        )
        txt_info.SetForegroundColour(wx.Colour(80, 80, 80))
        txt_info.Wrap(440)
        sizer.Add(txt_info, 0, wx.ALL, 6)

        btn_to_s6 = wx.Button(parent, label="▶ 下一步: 人工交互合并地层界面 (进入步骤 6)", size=(-1, 38))
        btn_to_s6.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s6.SetBackgroundColour(wx.Colour(46, 139, 87))
        btn_to_s6.SetForegroundColour(wx.WHITE)
        btn_to_s6.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(5))
        sizer.Add(btn_to_s6, 0, wx.EXPAND | wx.ALL, 8)

        parent.SetSizer(sizer)

    def on_extract_surface_points_step(self, event=None):
        """步骤 5: 提取控制点与连通段并更新列表 (内置噪点过滤与防崩保护)"""
        b_skel = os.path.join(self.cur_dir, "boundary_output", "boundary_skeleton.png")
        f_skel = os.path.join(self.cur_dir, "fault_output", "fault_skeleton.png")
        dem_file = self.txt_dem_path.GetValue().strip()

        # 智能自适应寻找生成的 DEM
        if not os.path.exists(dem_file):
            for cand in [
                os.path.join(self.cur_dir, "dem_output", "dem_auto.npy"),
                os.path.join(self.cur_dir, "results", "dem_auto.npy"),
            ]:
                if os.path.exists(cand):
                    dem_file = cand
                    self.txt_dem_path.SetValue(cand)
                    break

        if not os.path.exists(b_skel) or not os.path.exists(f_skel):
            wx.MessageBox("未找到骨架文件，请先在步骤 4 执行骨架分离提取！", "提示", wx.OK | wx.ICON_WARNING)
            return
        if not os.path.exists(dem_file):
            wx.MessageBox("未找到 DEM 文件，请先在步骤 3 自动提取 DEM 高程！", "提示", wx.OK | wx.ICON_WARNING)
            return

        self.log("\n>>> 开始提取三维空间控制点与拓扑有序连通段...")
        n_pts = self.spin_pts_per_feature.GetValue()
        min_len = self.spin_min_skel_len.GetValue() if hasattr(self, "spin_min_skel_len") else 15

        try:
            df, meta = extract_surface_points_structured(
                fault_skel_path=f_skel,
                boundary_skel_path=b_skel,
                dem_path=dem_file,
                scale=self.scale_gsd,
                points_per_feature=n_pts,
                min_skeleton_length=min_len,
                output_csv=self.default_sp,
            )
            self.surface_points_df = df
            self.features_meta = meta["features"]

            # 初始化连通段映射与层序
            self._sync_features_to_ui()
            self.log(f"✓ 成功提取 {len(self.features_meta)} 个有效要素，共计 {len(df)} 个界面三维控制点 (已过滤杂散微小噪点)。")
            self.status_bar.SetStatusText(f"已提取 {len(self.features_meta)} 个构造与地层要素", 0)
            self._render_current_step_canvas()
        except Exception as e:
            self.log(f"✖ 提取控制点失败: {e}")
            wx.MessageBox(f"提取控制点失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

    def _sync_features_to_ui(self):
        """将提取的要素元数据同步至步骤 5 列表与步骤 6 交互合并列表"""
        target_list = getattr(self, "list_step5_features", getattr(self, "list_step3_features", None))
        if target_list:
            target_list.DeleteAllItems()
        self.list_raw_layers.DeleteAllItems()

        # 默认初始化映射并补充新提取要素
        for name, feat in self.features_meta.items():
            if name not in self.layer_mapping:
                self.layer_mapping[name] = name
            self.is_fault_map[name] = (feat["type"] == "fault")

        # 刷新步骤 5 列表
        if target_list:
            for idx, (name, feat) in enumerate(self.features_meta.items()):
                b = feat["bounds"]
                row = target_list.InsertItem(idx, name)
                target_list.SetItem(row, 1, "断层" if feat["type"] == "fault" else "地层界面")
                target_list.SetItem(row, 2, f"{feat['length_px']}")
                target_list.SetItem(row, 3, f"{feat['point_count']}")
                target_list.SetItem(row, 4, f"{b['X'][0]:.0f}~{b['X'][1]:.0f}")
                target_list.SetItem(row, 5, f"{b['Y'][0]:.0f}~{b['Y'][1]:.0f}")
                target_list.SetItem(row, 6, f"{b['Z'][0]:.0f}~{b['Z'][1]:.0f}")

        # 刷新步骤 6 列表
        self._refresh_step4_tables()

    # -------------------------------------------------------------------------
    # 步骤 6: 人工交互合并地层界面 (核心亮点)
    # -------------------------------------------------------------------------
    def _build_step6_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # 提示指南
        lbl_hint = wx.StaticText(
            parent,
            label="💡 点击下方连通段可在右侧原图实时高亮定位边界！\n请将断层两侧错开的同一地质界面勾选合并为真实地层。",
        )
        lbl_hint.SetForegroundColour(wx.Colour(0, 102, 204))
        lbl_hint.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        sizer.Add(lbl_hint, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        # 底图展示模式切换 (原图 vs 二值 Mask 图)
        h_bg = wx.BoxSizer(wx.HORIZONTAL)
        h_bg.Add(wx.StaticText(parent, label="右侧底图视图:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.choice_step6_bg = wx.Choice(
            parent,
            choices=["① 原始彩色地质图 (RGB)", "② 黑白界线掩膜 (Mask)"],
        )
        self.choice_step6_bg.SetSelection(0)
        self.choice_step6_bg.Bind(wx.EVT_CHOICE, lambda e: self._render_current_step_canvas())
        self.choice_step4_bg = self.choice_step6_bg  # 兼容别名
        h_bg.Add(self.choice_step6_bg, 1, wx.EXPAND)
        sizer.Add(h_bg, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 4)

        # 原始连通段列表 (点击可实时高亮)
        sb_raw = wx.StaticBox(parent, label="① 原始连通段要素列表 (点击即在右侧高亮查看)")
        sb_raw_sizer = wx.StaticBoxSizer(sb_raw, wx.VERTICAL)

        self.list_raw_layers = wx.ListCtrl(parent, style=wx.LC_REPORT)
        self.list_raw_layers.InsertColumn(0, "要素代号", width=80)
        self.list_raw_layers.InsertColumn(1, "类型", width=65)
        self.list_raw_layers.InsertColumn(2, "长度 (px)", width=75)
        self.list_raw_layers.InsertColumn(3, "点数", width=55)
        self.list_raw_layers.InsertColumn(4, "当前合并归属 / 目标地层", width=160)
        self.list_raw_layers.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_raw_layer_item_selected)

        sb_raw_sizer.Add(self.list_raw_layers, 1, wx.EXPAND | wx.ALL, 4)

        # 操作工具条
        h_op = wx.BoxSizer(wx.HORIZONTAL)
        h_op.Add(wx.StaticText(parent, label="合并后名称:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.txt_merge_target = wx.TextCtrl(parent, value="Boundary_C_P", size=(120, -1))
        h_op.Add(self.txt_merge_target, 1, wx.EXPAND | wx.RIGHT, 4)

        btn_merge = wx.Button(parent, label="🔗 合并选中要素")
        btn_merge.Bind(wx.EVT_BUTTON, self.on_merge_selected_layers)
        btn_split = wx.Button(parent, label="↩ 拆分/恢复初始")
        btn_split.Bind(wx.EVT_BUTTON, self.on_split_reset_layers)
        h_op.Add(btn_merge, 0, wx.RIGHT, 4)
        h_op.Add(btn_split, 0)
        sb_raw_sizer.Add(h_op, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # 智能推荐合并按钮
        btn_smart_rec = wx.Button(parent, label="⚡ 智能推荐合并 (Layer_1+2 ➔ Boundary_C_P, Layer_3+4 ➔ Boundary_D_C)")
        btn_smart_rec.SetForegroundColour(wx.Colour(20, 100, 40))
        btn_smart_rec.Bind(wx.EVT_BUTTON, self.on_smart_recommend_merge)
        sb_raw_sizer.Add(btn_smart_rec, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(sb_raw_sizer, 1, wx.EXPAND | wx.ALL, 4)

        # 最终地层列表与层序管理
        sb_final = wx.StaticBox(parent, label="② 最终地层序列与年代层序编排 (从新到老)")
        sb_final_sizer = wx.StaticBoxSizer(sb_final, wx.VERTICAL)

        self.list_final_layers = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_final_layers.InsertColumn(0, "层序", width=45)
        self.list_final_layers.InsertColumn(1, "地层/断层名称", width=120)
        self.list_final_layers.InsertColumn(2, "包含原始连通段", width=140)
        self.list_final_layers.InsertColumn(3, "要素属性", width=80)
        self.list_final_layers.InsertColumn(4, "总控制点", width=70)
        sb_final_sizer.Add(self.list_final_layers, 1, wx.EXPAND | wx.ALL, 4)

        h_order = wx.BoxSizer(wx.HORIZONTAL)
        btn_up = wx.Button(parent, label="▲ 上移 (较新)")
        btn_up.Bind(wx.EVT_BUTTON, lambda e: self._move_strat_order(-1))
        btn_down = wx.Button(parent, label="▼ 下移 (较老)")
        btn_down.Bind(wx.EVT_BUTTON, lambda e: self._move_strat_order(1))
        btn_save_sp = wx.Button(parent, label="💾 保存并更新 surface_points.csv")
        btn_save_sp.Bind(wx.EVT_BUTTON, self.on_save_merged_surface_points)

        h_order.Add(btn_up, 1, wx.RIGHT, 4)
        h_order.Add(btn_down, 1, wx.RIGHT, 4)
        h_order.Add(btn_save_sp, 2)
        sb_final_sizer.Add(h_order, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(sb_final_sizer, 1, wx.EXPAND | wx.ALL, 4)

        btn_to_s7 = wx.Button(parent, label="▶ 下一步: 产状智能解算与突变核对 (进入步骤 7)", size=(-1, 38))
        btn_to_s7.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s7.SetBackgroundColour(wx.Colour(30, 100, 200))
        btn_to_s7.SetForegroundColour(wx.WHITE)
        btn_to_s7.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(6))
        sizer.Add(btn_to_s7, 0, wx.EXPAND | wx.ALL, 6)

        parent.SetSizer(sizer)

    def _on_raw_layer_item_selected(self, event):
        """点击列表项时在右侧原图/Mask 上高亮对应边界"""
        idx = event.GetIndex()
        item_name = self.list_raw_layers.GetItemText(idx)
        self.selected_layer_name = item_name
        self.log(f"高亮选中山体/断层界线: {item_name}")
        self.status_bar.SetStatusText(f"当前高亮选中界面: {item_name}", 0)
        self._render_current_step_canvas()

    def on_merge_selected_layers(self, event=None):
        """将选中的多个连通段合并为一个地层"""
        selected_raw_names = []
        item = -1
        while True:
            item = self.list_raw_layers.GetNextItem(item, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)
            if item == -1:
                break
            selected_raw_names.append(self.list_raw_layers.GetItemText(item))

        if len(selected_raw_names) < 1:
            wx.MessageBox("请先在上方列表中选中至少一个需要合并或重命名的要素！", "提示", wx.OK | wx.ICON_INFORMATION)
            return

        raw_target = self.txt_merge_target.GetValue().strip()
        import re
        target_name = re.sub(r"[^\w\-]", "_", raw_target).strip("_")
        if not target_name:
            wx.MessageBox("请输入合并后的有效名称（支持字母、数字与下划线）！", "提示", wx.OK | wx.ICON_WARNING)
            return

        is_any_fault = any(self.features_meta.get(name, {}).get("type") == "fault" for name in selected_raw_names)
        is_any_stratum = any(self.features_meta.get(name, {}).get("type") == "stratum" for name in selected_raw_names)
        if is_any_fault and is_any_stratum:
            ret = wx.MessageBox(
                f"注意：您正在将断层构造与沉积地层混合合并为同一要素 [{target_name}]。\n"
                "在 GemPy 中断层与地层具有不同的构造性质，通常建议分别归类。\n是否确认合并？",
                "构造性质提示",
                wx.YES_NO | wx.ICON_QUESTION,
            )
            if ret != wx.YES:
                return

        for name in selected_raw_names:
            self.layer_mapping[name] = target_name
            if self.features_meta.get(name, {}).get("type") == "fault":
                self.is_fault_map[target_name] = True

        self.log(f"✓ 已将要素 {selected_raw_names} 人工合并归类为: [{target_name}]")
        self._refresh_step4_tables()
        self.on_save_merged_surface_points(show_msg=False)
        self._render_current_step_canvas()

    def on_split_reset_layers(self, event=None):
        """恢复拆分为初始独立连通段"""
        for name in self.features_meta.keys():
            self.layer_mapping[name] = name
        self.log("↺ 已恢复为原始独立连通段划分。")
        self._refresh_step4_tables()
        self.on_save_merged_surface_points(show_msg=False)
        self._render_current_step_canvas()

    def on_smart_recommend_merge(self, event=None):
        """一键智能推荐合并断层错断地层"""
        # 针对当前地质图的标准格局: Layer_1(西上) + Layer_2(东上) -> Boundary_C_P; Layer_3(西下) + Layer_4(东下) -> Boundary_D_C
        if "Layer_1" in self.features_meta and "Layer_2" in self.features_meta:
            self.layer_mapping["Layer_1"] = "Boundary_C_P"
            self.layer_mapping["Layer_2"] = "Boundary_C_P"
        if "Layer_3" in self.features_meta and "Layer_4" in self.features_meta:
            self.layer_mapping["Layer_3"] = "Boundary_D_C"
            self.layer_mapping["Layer_4"] = "Boundary_D_C"
        if "Fault_1" in self.features_meta:
            self.layer_mapping["Fault_1"] = "Fault_1"

        self.log("⚡ 已应用智能推荐合并: Layer_1+2 ➔ Boundary_C_P, Layer_3+4 ➔ Boundary_D_C")
        self._refresh_step4_tables()
        self.on_save_merged_surface_points(show_msg=False)
        self._render_current_step_canvas()

    def _refresh_step4_tables(self):
        """刷新步骤 4 的两张表格"""
        # 1. 原始连通段列表
        self.list_raw_layers.DeleteAllItems()
        for idx, (name, feat) in enumerate(self.features_meta.items()):
            row = self.list_raw_layers.InsertItem(idx, name)
            self.list_raw_layers.SetItem(row, 1, "断层" if feat["type"] == "fault" else "地层界面")
            self.list_raw_layers.SetItem(row, 2, f"{feat['length_px']}")
            self.list_raw_layers.SetItem(row, 3, f"{feat['point_count']}")
            self.list_raw_layers.SetItem(row, 4, self.layer_mapping.get(name, name))

        # 2. 最终地层序列
        merged_groups = {}
        for raw_name, target in self.layer_mapping.items():
            if target not in merged_groups:
                merged_groups[target] = []
            merged_groups[target].append(raw_name)

        # 维护 strat_order 保持稳定
        existing = [s for s in self.strat_order if s in merged_groups]
        new_items = [k for k in merged_groups if k not in existing]
        self.strat_order = existing + new_items

        self.list_final_layers.DeleteAllItems()
        for idx, final_name in enumerate(self.strat_order):
            raws = merged_groups.get(final_name, [])
            is_fault = any(self.features_meta.get(r, {}).get("type") == "fault" for r in raws)
            tot_pts = sum(self.features_meta.get(r, {}).get("point_count", 0) for r in raws)

            row = self.list_final_layers.InsertItem(idx, f"#{idx+1}")
            self.list_final_layers.SetItem(row, 1, final_name)
            self.list_final_layers.SetItem(row, 2, ", ".join(raws))
            self.list_final_layers.SetItem(row, 3, "断层构造" if is_fault else "沉积地层")
            self.list_final_layers.SetItem(row, 4, f"{tot_pts}")

    def _move_strat_order(self, delta: int):
        """调整层序上下顺序"""
        sel = self.list_final_layers.GetFirstSelected()
        if sel < 0:
            return
        new_idx = sel + delta
        if 0 <= new_idx < len(self.strat_order):
            item = self.strat_order.pop(sel)
            self.strat_order.insert(new_idx, item)
            self._refresh_step4_tables()
            self.list_final_layers.Select(new_idx)

    def on_save_merged_surface_points(self, show_msg: bool = True):
        """将合并后的地层要素重新打包并覆盖 surface_points.csv，并联动同步更新 orientations.csv 与内部产状数据"""
        if not self.features_meta:
            return

        all_rows = []
        for raw_name, feat in self.features_meta.items():
            final_name = self.layer_mapping.get(raw_name, raw_name)
            for pt in feat["sampled_points"]:
                all_rows.append({
                    "X": pt["X"],
                    "Y": pt["Y"],
                    "Z": pt["Z"],
                    "formation": final_name,
                })

        df = pd.DataFrame(all_rows)
        self.surface_points_df = df
        df.to_csv(self.default_sp, index=False, float_format="%.1f")
        self.log(f"✓ 合并地层控制点库已覆盖更新: {self.default_sp} (共 {len(df)} 行数据)")

        valid_formations = set(df["formation"].unique())

        # 联动同步更新 orientations.csv
        target_or_files = [self.default_or]
        if hasattr(self, "txt_or"):
            custom_or = self.txt_or.GetValue().strip()
            if custom_or and custom_or not in target_or_files:
                target_or_files.append(custom_or)

        for or_path in target_or_files:
            if os.path.exists(or_path):
                try:
                    df_or = pd.read_csv(or_path, comment="#")
                    if "formation" in df_or.columns:
                        df_or["formation"] = df_or["formation"].astype(str).map(
                            lambda f: self.layer_mapping.get(f, f)
                        )
                        df_or = df_or[df_or["formation"].isin(valid_formations)]
                        
                    if len(df_or) > 0:
                        df_or.to_csv(or_path, index=False, float_format="%.1f")
                except Exception as e:
                    self.log(f"同步更新 orientations.csv 警告: {e}")

        # 联动触发全量产状解算与内存/表格同步，确保所有要素（断层与地层）产状即时就绪
        try:
            self.on_compute_attitudes_step(silent=True)
            self.on_save_orientations_csv(show_msg=False)
        except Exception as e:
            self.log(f"合并后自动同步产状提示: {e}")

        if show_msg:
            wx.MessageBox(f"已成功将合并地层数据更新保存至:\n{self.default_sp}\n共 {len(df)} 个界面控制点", "保存成功", wx.OK | wx.ICON_INFORMATION)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 步骤 7: 产状计算与异常突变检测/人工核对 (核心亮点)
    # -------------------------------------------------------------------------
    def _build_step7_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_param = wx.StaticBox(parent, label="产状解算与突变跳变阈值控制")
        sb_param_sizer = wx.StaticBoxSizer(sb_param, wx.VERTICAL)

        g_p = wx.FlexGridSizer(2, 4, 6, 6)
        g_p.AddGrowableCol(1)
        g_p.AddGrowableCol(3)

        self.spin_att_num_pts = wx.SpinCtrl(parent, min=2, max=10, initial=3)
        self.spin_att_az_threshold = wx.SpinCtrl(parent, min=15, max=90, initial=45)
        self.spin_att_dip_threshold = wx.SpinCtrl(parent, min=10, max=60, initial=20)

        g_p.Add(wx.StaticText(parent, label="局部采样点数:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_p.Add(self.spin_att_num_pts, 1, wx.EXPAND)
        g_p.Add(wx.StaticText(parent, label="倾向突变阈值:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_p.Add(self.spin_att_az_threshold, 1, wx.EXPAND)

        g_p.Add(wx.StaticText(parent, label="倾角突变阈值:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_p.Add(self.spin_att_dip_threshold, 1, wx.EXPAND)
        btn_recalc = wx.Button(parent, label="⚡ 重新解算产状与突变检测")
        btn_recalc.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_recalc.Bind(wx.EVT_BUTTON, self.on_compute_attitudes_step)
        g_p.Add(btn_recalc, 0)
        g_p.Add(wx.StaticText(parent, label=""), 0)

        sb_param_sizer.Add(g_p, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_param_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # 产状数据列表 (标红显示突变异常点)
        sb_list = wx.StaticBox(parent, label="产状控制点数据库 (⚠️ 红色标注突变异常)")
        sb_list_sizer = wx.StaticBoxSizer(sb_list, wx.VERTICAL)

        self.list_attitudes = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_attitudes.InsertColumn(0, "序号", width=45)
        self.list_attitudes.InsertColumn(1, "所属地层", width=105)
        self.list_attitudes.InsertColumn(2, "X (m)", width=65)
        self.list_attitudes.InsertColumn(3, "Y (m)", width=65)
        self.list_attitudes.InsertColumn(4, "Z (m)", width=60)
        self.list_attitudes.InsertColumn(5, "倾向 (°)", width=65)
        self.list_attitudes.InsertColumn(6, "倾角 (°)", width=60)
        self.list_attitudes.InsertColumn(7, "走向 (°)", width=65)
        self.list_attitudes.InsertColumn(8, "质检诊断状态", width=170)
        self.list_attitudes.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_att_item_selected)

        sb_list_sizer.Add(self.list_attitudes, 1, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_list_sizer, 1, wx.EXPAND | wx.ALL, 4)

        # 异常修正工具箱
        sb_corr = wx.StaticBox(parent, label="人工核对与异常修正工具箱")
        sb_corr_sizer = wx.StaticBoxSizer(sb_corr, wx.VERTICAL)

        h_c1 = wx.BoxSizer(wx.HORIZONTAL)
        btn_flip = wx.Button(parent, label="🔄 反转极性 (倾向翻转 180°)")
        btn_flip.SetToolTip("针对法向量符号二义性或反向倾向点，将其旋转 180 度修正为同向倾向")
        btn_flip.Bind(wx.EVT_BUTTON, lambda e: self.on_correct_selected_attitude("flip_180"))

        btn_smooth = wx.Button(parent, label="📐 邻域平滑")
        btn_smooth.SetToolTip("采用沿线邻域有效点的向量加权平均修正当前点")
        btn_smooth.Bind(wx.EVT_BUTTON, lambda e: self.on_correct_selected_attitude("smooth"))

        btn_med = wx.Button(parent, label="⚖ 统一代表倾向")
        btn_med.SetToolTip("将当前点对齐为该地层的稳健代表产状")
        btn_med.Bind(wx.EVT_BUTTON, lambda e: self.on_correct_selected_attitude("median"))

        h_c1.Add(btn_flip, 1, wx.RIGHT, 4)
        h_c1.Add(btn_smooth, 1, wx.RIGHT, 4)
        h_c1.Add(btn_med, 1)
        sb_corr_sizer.Add(h_c1, 0, wx.EXPAND | wx.ALL, 4)

        # 手动输入与批量应用
        h_c2 = wx.BoxSizer(wx.HORIZONTAL)
        h_c2.Add(wx.StaticText(parent, label="手工指定: 倾向"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.txt_manual_az = wx.TextCtrl(parent, value="", size=(50, -1))
        h_c2.Add(self.txt_manual_az, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        h_c2.Add(wx.StaticText(parent, label="倾角"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.txt_manual_dip = wx.TextCtrl(parent, value="", size=(45, -1))
        h_c2.Add(self.txt_manual_dip, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        btn_apply_manual = wx.Button(parent, label="✓ 应用当前点")
        btn_apply_manual.Bind(wx.EVT_BUTTON, self.on_apply_manual_attitude)
        h_c2.Add(btn_apply_manual, 0, wx.RIGHT, 4)
        btn_apply_formation = wx.Button(parent, label="🌐 应用至同地层全部点")
        btn_apply_formation.Bind(wx.EVT_BUTTON, self.on_apply_formation_attitude)
        h_c2.Add(btn_apply_formation, 0)
        sb_corr_sizer.Add(h_c2, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # 智能一键修复按钮
        h_c3 = wx.BoxSizer(wx.HORIZONTAL)
        btn_fix_all = wx.Button(parent, label="🚀 修复异常突变 (对齐基准)", size=(-1, 30))
        btn_fix_all.Bind(wx.EVT_BUTTON, self.on_auto_fix_all_mutations)
        h_c3.Add(btn_fix_all, 1, wx.RIGHT, 4)

        btn_prior = wx.Button(parent, label="🌐 一键校准地层南倾 (20°根治外突)", size=(-1, 30))
        btn_prior.SetForegroundColour(wx.Colour(20, 100, 40))
        btn_prior.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_prior.Bind(wx.EVT_BUTTON, self.on_align_strat_prior)
        h_c3.Add(btn_prior, 1)
        sb_corr_sizer.Add(h_c3, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # 保存更新 orientations.csv
        btn_save_or = wx.Button(parent, label="💾 确认无误并更新保存 orientations.csv")
        btn_save_or.Bind(wx.EVT_BUTTON, self.on_save_orientations_csv)
        sb_corr_sizer.Add(btn_save_or, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(sb_corr_sizer, 0, wx.EXPAND | wx.ALL, 4)

        btn_to_s8 = wx.Button(parent, label="▶ 下一步: 进入三维建模与剖面分析 (进入步骤 8)", size=(-1, 38))
        btn_to_s8.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_to_s8.SetBackgroundColour(wx.Colour(46, 139, 87))
        btn_to_s8.SetForegroundColour(wx.WHITE)
        btn_to_s8.Bind(wx.EVT_BUTTON, lambda e: self.goto_step(7))
        sizer.Add(btn_to_s8, 0, wx.EXPAND | wx.ALL, 6)

        parent.SetSizer(sizer)

    def on_compute_attitudes_step(self, event=None, silent: bool = False):
        """计算断层与全部地层曲面产状并执行倾向突变与平缓倾角检测"""
        dem_file = self.txt_dem_path.GetValue().strip()
        if not os.path.exists(dem_file):
            for cand in [
                os.path.join(self.cur_dir, "dem_output", "dem_auto.npy"),
                os.path.join(self.cur_dir, "results", "dem_auto.npy"),
            ]:
                if os.path.exists(cand):
                    dem_file = cand
                    self.txt_dem_path.SetValue(cand)
                    break

        if not os.path.exists(dem_file):
            if not silent:
                wx.MessageBox("未找到 DEM 标高文件，请先在步骤 3 自动提取 DEM！", "提示", wx.OK | wx.ICON_WARNING)
            return

        calc = AttitudeCalculator(scale=self.scale_gsd)
        num_pts = self.spin_att_num_pts.GetValue() if hasattr(self, "spin_att_num_pts") else 3

        self.log("\n>>> 开始断层与全部地层界面多点产状拟合与倾向突变质检...")
        all_attitudes = []

        # 优先按每个物理连通段分别解算产状，再赋予合并后的地层归属
        # 从而准确捕获断层错断两盘之间的产状突变与极性反转
        if self.features_meta:
            for raw_name, feat in self.features_meta.items():
                form_name = self.layer_mapping.get(raw_name, raw_name)
                pts = feat.get("sampled_points", [])
                if len(pts) < 3:
                    continue
                pts_xyz = np.array([[p["X"], p["Y"], p["Z"]] for p in pts])
                try:
                    res = calc.calculate_points_attitude(pts_xyz, feature_name=form_name, num_points=num_pts)
                    for att in res["local_attitudes"]:
                        att["raw_feature"] = raw_name
                        all_attitudes.append(att)
                except Exception as e:
                    self.log(f"要素 [{raw_name} -> {form_name}] 产状计算跳过: {e}")
        else:
            if self.surface_points_df is None or len(self.surface_points_df) == 0:
                if os.path.exists(self.default_sp):
                    self.surface_points_df = pd.read_csv(self.default_sp)

            if self.surface_points_df is None or len(self.surface_points_df) == 0:
                if not silent:
                    wx.MessageBox("控制点数据库为空，请先在步骤 5 提取控制点！", "提示", wx.OK | wx.ICON_WARNING)
                return

            for form_name, grp in self.surface_points_df.groupby("formation"):
                pts_xyz = grp[["X", "Y", "Z"]].to_numpy()
                if len(pts_xyz) < 3:
                    continue
                try:
                    res = calc.calculate_points_attitude(pts_xyz, feature_name=form_name, num_points=num_pts)
                    all_attitudes.extend(res["local_attitudes"])
                except Exception as e:
                    self.log(f"要素 [{form_name}] 产状计算跳过: {e}")

        # 运行倾向跳变与平缓倾角质检检测
        az_thresh = float(self.spin_att_az_threshold.GetValue()) if hasattr(self, "spin_att_az_threshold") else 45.0
        dip_thresh = float(self.spin_att_dip_threshold.GetValue()) if hasattr(self, "spin_att_dip_threshold") else 20.0
        checked_atts = detect_attitude_mutations(
            all_attitudes,
            azimuth_threshold=az_thresh,
            dip_threshold=dip_thresh,
        )
        self.orientations_list = checked_atts
        self._refresh_step5_table()

        mut_count = sum(1 for a in checked_atts if a.get("is_mutation", False))
        forms_found = set(a.get("formation") for a in checked_atts)
        self.log(f"✓ 已全量解算 {len(checked_atts)} 个产状控制点，覆盖要素: {list(forms_found)}")
        if mut_count > 0:
            self.log(f"⚠️ 质检预警: 检测到 {mut_count} 处反向极性或极缓倾角点 (已标红提示)。")
            self.log("💡 建议: 可点击【🚀 一键智能修复全部】将地层产状纠正为南向稳定中缓倾角(20°)，消除三维平切外突！")
            self.status_bar.SetStatusText(f"⚠️ 预警: 检测到 {mut_count} 个产状突变/极缓点待核对修复", 0)
        else:
            self.log("✓ 质检合格: 所有控制点倾向与倾角均平滑合理，无异常突变。")
            self.status_bar.SetStatusText("✓ 产状质检合格: 全要素平滑正常", 0)

        self._render_current_step_canvas()

    def _refresh_step5_table(self):
        """刷新步骤 7 产状表格"""
        self.list_attitudes.DeleteAllItems()
        for idx, att in enumerate(self.orientations_list):
            row = self.list_attitudes.InsertItem(idx, f"#{idx+1}")
            form_display = str(att.get("formation", ""))
            if att.get("raw_feature") and att["raw_feature"] != form_display:
                form_display = f"{form_display} ({att['raw_feature']})"
            self.list_attitudes.SetItem(row, 1, form_display)
            self.list_attitudes.SetItem(row, 2, f"{att.get('X', 0.0):.1f}")
            self.list_attitudes.SetItem(row, 3, f"{att.get('Y', 0.0):.1f}")
            self.list_attitudes.SetItem(row, 4, f"{att.get('Z', 0.0):.1f}")
            self.list_attitudes.SetItem(row, 5, f"{att.get('azimuth', 0.0):.1f}")
            self.list_attitudes.SetItem(row, 6, f"{att.get('dip', 0.0):.1f}")
            self.list_attitudes.SetItem(row, 7, f"{att.get('strike', 0.0):.1f}")

            is_mut = att.get("is_mutation", False)
            desc = att.get("mutation_desc", "正常")
            self.list_attitudes.SetItem(row, 8, desc)

            if is_mut:
                self.list_attitudes.SetItemTextColour(row, wx.Colour(200, 0, 0))
                self.list_attitudes.SetItemBackgroundColour(row, wx.Colour(255, 235, 235))
            else:
                self.list_attitudes.SetItemTextColour(row, wx.Colour(0, 0, 0))
                self.list_attitudes.SetItemBackgroundColour(row, wx.WHITE)

    def _refresh_step7_table(self):
        """兼容步骤 7 别名"""
        self._refresh_step5_table()

    def _on_att_item_selected(self, event):
        idx = event.GetIndex()
        if 0 <= idx < len(self.orientations_list):
            self.selected_att_idx = idx
            att = self.orientations_list[idx]
            self.txt_manual_az.SetValue(f"{att['azimuth']:.1f}")
            self.txt_manual_dip.SetValue(f"{att['dip']:.1f}")
            self._render_current_step_canvas()

    def on_correct_selected_attitude(self, method: str):
        """修正当前选中的产状点"""
        sel = self.list_attitudes.GetFirstSelected()
        if sel < 0 or sel >= len(self.orientations_list):
            wx.MessageBox("请先在列表中选中一个需要修正的产状点！", "提示", wx.OK | wx.ICON_INFORMATION)
            return

        cur = self.orientations_list[sel]
        form = cur["formation"]
        same_form_pts = [a for a in self.orientations_list if a["formation"] == form]

        corrected = correct_attitude(
            cur,
            method=method,
            neighbors=same_form_pts,
            median_val={"median_azimuth": cur.get("median_azimuth", cur["azimuth"]), "median_dip": cur.get("median_dip", cur["dip"])},
        )
        self.orientations_list[sel] = corrected

        # 重新质检
        az_thresh = float(self.spin_att_az_threshold.GetValue())
        dip_thresh = float(self.spin_att_dip_threshold.GetValue())
        self.orientations_list = detect_attitude_mutations(self.orientations_list, az_thresh, dip_thresh)
        self._refresh_step5_table()
        self.list_attitudes.Select(sel)
        self.log(f"✓ 已对点 #{sel+1} ({form}) 执行修正: {corrected.get('mutation_desc', '')}")
        self._render_current_step_canvas()

    def on_apply_manual_attitude(self, event=None):
        """手动输入并应用产状数值至当前选中点"""
        sel = self.list_attitudes.GetFirstSelected()
        if sel < 0 or sel >= len(self.orientations_list):
            wx.MessageBox("请先在列表中选中要编辑的产状点！", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        try:
            az = float(self.txt_manual_az.GetValue().strip()) % 360.0
            dip = float(self.txt_manual_dip.GetValue().strip())
            dip = max(0.0, min(90.0, dip))
        except Exception:
            wx.MessageBox("请输入合法的倾向 (0~360°) 与倾角 (0~90°) 数值！", "输入错误", wx.OK | wx.ICON_ERROR)
            return

        self.orientations_list[sel]["azimuth"] = round(az, 1)
        self.orientations_list[sel]["dip"] = round(dip, 1)
        self.orientations_list[sel]["strike"] = round((az - 90.0) % 360.0, 1)
        self.orientations_list[sel]["is_mutation"] = False
        self.orientations_list[sel]["mutation_type"] = "manual"
        self.orientations_list[sel]["mutation_desc"] = "人工指定核准"

        az_thresh = float(self.spin_att_az_threshold.GetValue())
        dip_thresh = float(self.spin_att_dip_threshold.GetValue())
        self.orientations_list = detect_attitude_mutations(self.orientations_list, az_thresh, dip_thresh)
        self._refresh_step5_table()
        self.list_attitudes.Select(sel)
        self.log(f"✓ 人工核准点 #{sel+1} 产状: 倾向 {az:.1f}° | 倾角 {dip:.1f}°")
        self._render_current_step_canvas()

    def on_apply_formation_attitude(self, event=None):
        """将手工输入的倾向与倾角批量应用至当前选中要素/地层的全部控制点"""
        sel = self.list_attitudes.GetFirstSelected()
        if sel < 0 or sel >= len(self.orientations_list):
            wx.MessageBox("请先在列表中选中要批量设置的目标要素/地层！", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        try:
            az = float(self.txt_manual_az.GetValue().strip()) % 360.0
            dip = float(self.txt_manual_dip.GetValue().strip())
            dip = max(0.0, min(90.0, dip))
        except Exception:
            wx.MessageBox("请输入合法的倾向 (0~360°) 与倾角 (0~90°) 数值！", "输入错误", wx.OK | wx.ICON_ERROR)
            return

        target_form = self.orientations_list[sel]["formation"]
        applied_cnt = 0
        for att in self.orientations_list:
            if att["formation"] == target_form:
                att["azimuth"] = round(az, 1)
                att["dip"] = round(dip, 1)
                att["strike"] = round((az - 90.0) % 360.0, 1)
                att["is_mutation"] = False
                att["mutation_type"] = "manual"
                att["mutation_desc"] = "人工批量指定"
                applied_cnt += 1

        az_thresh = float(self.spin_att_az_threshold.GetValue())
        dip_thresh = float(self.spin_att_dip_threshold.GetValue())
        self.orientations_list = detect_attitude_mutations(self.orientations_list, az_thresh, dip_thresh)
        self._refresh_step5_table()
        self.list_attitudes.Select(sel)
        self.log(f"✓ 已将要素 [{target_form}] 的全部 {applied_cnt} 个控制点批量指定为: 倾向 {az:.1f}° | 倾角 {dip:.1f}°")
        self._render_current_step_canvas()

    def on_auto_fix_all_mutations(self, event=None):
        """一键智能修复所有异常突变点 (对齐同地层代表基准产状)"""
        fixed_count = 0
        for i, att in enumerate(self.orientations_list):
            if att.get("is_mutation", False):
                self.orientations_list[i] = correct_attitude(
                    att,
                    method="median",
                    median_val={
                        "median_azimuth": att.get("median_azimuth", att["azimuth"]),
                        "median_dip": att.get("median_dip", att["dip"]),
                    },
                )
                fixed_count += 1

        az_thresh = float(self.spin_att_az_threshold.GetValue()) if hasattr(self, "spin_att_az_threshold") else 45.0
        dip_thresh = float(self.spin_att_dip_threshold.GetValue()) if hasattr(self, "spin_att_dip_threshold") else 20.0
        self.orientations_list = detect_attitude_mutations(self.orientations_list, az_thresh, dip_thresh)
        self._refresh_step5_table()
        self.log(f"🚀 一键智能修正完成: 已将 {fixed_count} 处异常突变点对齐为同地层代表基准产状！")
        self._render_current_step_canvas()

    def on_align_strat_prior(self, event=None):
        """一键依地层层序校准地层产状 (南向倾斜 165°，倾角 20°，根除三维外突畸变)"""
        count = 0
        for att in self.orientations_list:
            form = str(att.get("formation", ""))
            if not any(k in form.lower() for k in ["fault", "断层"]):
                att["azimuth"] = 165.0
                att["dip"] = 20.0
                att["strike"] = 75.0
                att["is_mutation"] = False
                att["mutation_type"] = "manual"
                att["mutation_desc"] = "已校准层序南向产状(165°∠20°)"
                count += 1

        az_thresh = float(self.spin_att_az_threshold.GetValue()) if hasattr(self, "spin_att_az_threshold") else 45.0
        dip_thresh = float(self.spin_att_dip_threshold.GetValue()) if hasattr(self, "spin_att_dip_threshold") else 20.0
        self.orientations_list = detect_attitude_mutations(self.orientations_list, az_thresh, dip_thresh)
        self._refresh_step5_table()
        self.log(f"🌐 已将全部 {count} 个地层控制点校准为工区合理南向产状 (倾向 165°，倾角 20°)，有效根治三维模型平切外突！")
        self._render_current_step_canvas()

    def on_save_orientations_csv(self, event=None, show_msg: bool = True):
        """保存产状数据至 orientations.csv"""
        if not self.orientations_list:
            if show_msg:
                wx.MessageBox("暂无产状数据，无法保存！", "提示", wx.OK | wx.ICON_WARNING)
            return

        df = pd.DataFrame([
            {
                "X": round(float(a["X"]), 1),
                "Y": round(float(a["Y"]), 1),
                "Z": round(float(a["Z"]), 1),
                "azimuth": round(float(a["azimuth"]), 1),
                "dip": round(float(a["dip"]), 1),
                "polarity": 1.0,
                "formation": a["formation"],
            }
            for a in self.orientations_list
        ])
        df.to_csv(self.default_or, index=False, float_format="%.1f")
        self.log(f"✓ 产状数据库已成功同步写入: {self.default_or} (共 {len(df)} 行)")
        if show_msg:
            wx.MessageBox(f"产状数据库已成功同步保存至:\n{self.default_or}\n共 {len(df)} 个产状控制点", "保存成功", wx.OK | wx.ICON_INFORMATION)

    # -------------------------------------------------------------------------
    # 步骤 8: GemPy 三维地质体建模与成果展示 (承接原有高级建模、剖面、钻孔)
    # -------------------------------------------------------------------------
    def _build_step8_panel(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # 嵌套 Notebook 承载高级建模、剖面切割、虚拟钻孔与 3D 导出
        self.modeling_sub_notebook = wx.Notebook(parent)

        # Tab 1: 建模参数配置
        tab_params = wx.Panel(self.modeling_sub_notebook)
        self._build_tab_params(tab_params)
        self.modeling_sub_notebook.AddPage(tab_params, "建模参数")

        # Tab 2: 自定义剖面切割
        tab_sections = wx.Panel(self.modeling_sub_notebook)
        self._build_tab_sections(tab_sections)
        self.modeling_sub_notebook.AddPage(tab_sections, "剖面切割")

        # Tab 3: 虚拟钻孔管理
        tab_boreholes = wx.Panel(self.modeling_sub_notebook)
        self._build_tab_boreholes(tab_boreholes)
        self.modeling_sub_notebook.AddPage(tab_boreholes, "虚拟钻孔")

        # Tab 4: 三维展示与导出
        tab_export = wx.Panel(self.modeling_sub_notebook)
        self._build_tab_export(tab_export)
        self.modeling_sub_notebook.AddPage(tab_export, "三维与导出")

        sizer.Add(self.modeling_sub_notebook, 1, wx.EXPAND | wx.ALL, 4)

        # 底部大按钮：开始计算
        self.btn_calc = wx.Button(parent, label="▶ 开始构建并计算模型", size=(-1, 40))
        self.btn_calc.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_calc.SetBackgroundColour(wx.Colour(46, 139, 87))
        self.btn_calc.SetForegroundColour(wx.WHITE)
        self.btn_calc.Bind(wx.EVT_BUTTON, lambda e: self.on_compute_model())
        sizer.Add(self.btn_calc, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        parent.SetSizer(sizer)

    def _build_tab_params(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # 数据文件通道
        sb_data = wx.StaticBox(parent, label="数据输入通道 (CSV)")
        sb_data_sizer = wx.StaticBoxSizer(sb_data, wx.VERTICAL)

        lbl_sp = wx.StaticText(parent, label="接触面/界面点数据 (surface_points.csv):")
        sb_data_sizer.Add(lbl_sp, 0, wx.TOP | wx.LEFT, 4)
        h_sp = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_sp = wx.TextCtrl(parent, value=self.default_sp)
        btn_sp = wx.Button(parent, label="浏览...")
        btn_sp.Bind(wx.EVT_BUTTON, lambda e: self._choose_file(self.txt_sp, "选择界面点 CSV", "*.csv"))
        h_sp.Add(self.txt_sp, 1, wx.EXPAND | wx.RIGHT, 4)
        h_sp.Add(btn_sp, 0)
        sb_data_sizer.Add(h_sp, 0, wx.EXPAND | wx.ALL, 4)

        lbl_or = wx.StaticText(parent, label="产状数据 (orientations.csv):")
        sb_data_sizer.Add(lbl_or, 0, wx.TOP | wx.LEFT, 4)
        h_or = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_or = wx.TextCtrl(parent, value=self.default_or)
        btn_or = wx.Button(parent, label="浏览...")
        btn_or.Bind(wx.EVT_BUTTON, lambda e: self._choose_file(self.txt_or, "选择产状 CSV", "*.csv"))
        h_or.Add(self.txt_or, 1, wx.EXPAND | wx.RIGHT, 4)
        h_or.Add(btn_or, 0)
        sb_data_sizer.Add(h_or, 0, wx.EXPAND | wx.ALL, 4)

        sizer.Add(sb_data_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # 空间范围 Extent
        sb_ext = wx.StaticBox(parent, label="空间范围 Extent [米]")
        sb_ext_sizer = wx.StaticBoxSizer(sb_ext, wx.VERTICAL)

        btn_auto_ext = wx.Button(parent, label="📐 根据输入数据自动计算推荐 Extent 范围")
        btn_auto_ext.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_auto_ext.Bind(wx.EVT_BUTTON, lambda e: self.on_auto_detect_extent(show_msg=True))
        sb_ext_sizer.Add(btn_auto_ext, 0, wx.EXPAND | wx.ALL, 4)

        grid_ext = wx.FlexGridSizer(3, 4, 4, 4)
        grid_ext.AddGrowableCol(1)
        grid_ext.AddGrowableCol(3)

        self.txt_xmin = wx.TextCtrl(parent, value="0.0")
        self.txt_xmax = wx.TextCtrl(parent, value="3350.0")
        self.txt_ymin = wx.TextCtrl(parent, value="0.0")
        self.txt_ymax = wx.TextCtrl(parent, value="2200.0")
        self.txt_zmin = wx.TextCtrl(parent, value="500.0")
        self.txt_zmax = wx.TextCtrl(parent, value="1000.0")

        for ctrl in [self.txt_xmin, self.txt_xmax, self.txt_ymin, self.txt_ymax, self.txt_zmin, self.txt_zmax]:
            ctrl.Bind(wx.EVT_TEXT, self.on_resolution_or_extent_changed)

        grid_ext.Add(wx.StaticText(parent, label="X Min:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_xmin, 1, wx.EXPAND)
        grid_ext.Add(wx.StaticText(parent, label="X Max:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_xmax, 1, wx.EXPAND)

        grid_ext.Add(wx.StaticText(parent, label="Y Min:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_ymin, 1, wx.EXPAND)
        grid_ext.Add(wx.StaticText(parent, label="Y Max:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_ymax, 1, wx.EXPAND)

        grid_ext.Add(wx.StaticText(parent, label="Z Min:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_zmin, 1, wx.EXPAND)
        grid_ext.Add(wx.StaticText(parent, label="Z Max:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_ext.Add(self.txt_zmax, 1, wx.EXPAND)

        sb_ext_sizer.Add(grid_ext, 1, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_ext_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # 离散网格与建模精细度控制
        sb_grid = wx.StaticBox(parent, label="离散网格与精细度控制 (Resolution & Refinement)")
        sb_grid_sizer = wx.StaticBoxSizer(sb_grid, wx.VERTICAL)

        h_preset = wx.BoxSizer(wx.HORIZONTAL)
        lbl_preset = wx.StaticText(parent, label="精细度预设:")
        lbl_preset.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.choice_preset = wx.Choice(parent, choices=[opt[0] for opt in self.preset_options])
        self.choice_preset.SetSelection(1)
        self.choice_preset.Bind(wx.EVT_CHOICE, self.on_preset_selected)
        h_preset.Add(lbl_preset, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        h_preset.Add(self.choice_preset, 1, wx.EXPAND)
        sb_grid_sizer.Add(h_preset, 0, wx.EXPAND | wx.ALL, 4)

        grid_res = wx.FlexGridSizer(3, 4, 4, 4)
        grid_res.AddGrowableCol(1)
        grid_res.AddGrowableCol(3)

        self.txt_nx = wx.TextCtrl(parent, value="40")
        self.txt_ny = wx.TextCtrl(parent, value="40")
        self.txt_nz = wx.TextCtrl(parent, value="30")
        self.spin_refine = wx.SpinCtrl(parent, min=1, max=8, initial=4)
        self.spin_surf_refine = wx.SpinCtrl(parent, min=1, max=8, initial=4)

        for ctrl in [self.txt_nx, self.txt_ny, self.txt_nz]:
            ctrl.Bind(wx.EVT_TEXT, self.on_resolution_input_changed)

        self.spin_refine.Bind(wx.EVT_SPINCTRL, self.on_refine_changed)
        self.spin_surf_refine.Bind(wx.EVT_SPINCTRL, self.on_surf_refine_changed)

        grid_res.Add(wx.StaticText(parent, label="网格 Nx:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_res.Add(self.txt_nx, 1, wx.EXPAND)
        grid_res.Add(wx.StaticText(parent, label="网格 Ny:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_res.Add(self.txt_ny, 1, wx.EXPAND)

        grid_res.Add(wx.StaticText(parent, label="网格 Nz:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_res.Add(self.txt_nz, 1, wx.EXPAND)
        grid_res.Add(wx.StaticText(parent, label="体网格细化:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_res.Add(self.spin_refine, 1, wx.EXPAND)

        grid_res.Add(wx.StaticText(parent, label="曲面细化:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_res.Add(self.spin_surf_refine, 1, wx.EXPAND)

        self.chk_sync_refine = wx.CheckBox(parent, label="曲面等级随体网格同步")
        self.chk_sync_refine.SetValue(True)
        self.chk_sync_refine.Bind(wx.EVT_CHECKBOX, self.on_toggle_sync_refine)
        grid_res.Add(self.chk_sync_refine, 0, wx.ALIGN_CENTER_VERTICAL)

        sb_grid_sizer.Add(grid_res, 0, wx.EXPAND | wx.ALL, 4)

        # 体素尺寸评估
        self.pnl_voxel_metric = wx.Panel(parent)
        self.pnl_voxel_metric.SetBackgroundColour(wx.Colour(246, 248, 252))
        box_metric = wx.BoxSizer(wx.VERTICAL)
        self.lbl_voxel_info = wx.StaticText(self.pnl_voxel_metric, label="📐 体素尺寸: 正在分析...\n📊 网格总数: 正在分析...")
        box_metric.Add(self.lbl_voxel_info, 1, wx.ALL | wx.EXPAND, 6)
        self.pnl_voxel_metric.SetSizer(box_metric)
        sb_grid_sizer.Add(self.pnl_voxel_metric, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # 配置方案持久化
        h_cfg_bar = wx.BoxSizer(wx.HORIZONTAL)
        btn_save_cfg = wx.Button(parent, label="💾 保存配置方案...")
        btn_save_cfg.Bind(wx.EVT_BUTTON, lambda e: self.on_save_config_profile())
        btn_load_cfg = wx.Button(parent, label="📂 载入配置方案...")
        btn_load_cfg.Bind(wx.EVT_BUTTON, lambda e: self.on_load_config_profile())
        btn_reset_cfg = wx.Button(parent, label="↺ 恢复默认")
        btn_reset_cfg.Bind(wx.EVT_BUTTON, lambda e: self.on_reset_default_precision())

        h_cfg_bar.Add(btn_save_cfg, 1, wx.RIGHT, 4)
        h_cfg_bar.Add(btn_load_cfg, 1, wx.RIGHT, 4)
        h_cfg_bar.Add(btn_reset_cfg, 0)
        sb_grid_sizer.Add(h_cfg_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(sb_grid_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # 地表地形与高程模型
        sb_topo = wx.StaticBox(parent, label="地表地形与高程模型 (Topography)")
        sb_topo_sizer = wx.StaticBoxSizer(sb_topo, wx.VERTICAL)

        self.choice_topo_mode = wx.Choice(
            parent,
            choices=["① 不使用地形 (全空间平坦)", "② 外部 DEM/高程文件导入", "③ 分形随机起伏仿真"],
        )
        self.choice_topo_mode.SetSelection(1 if os.path.exists(self.default_dem) else 0)
        self.choice_topo_mode.Bind(wx.EVT_CHOICE, self.on_topo_mode_changed)
        sb_topo_sizer.Add(self.choice_topo_mode, 0, wx.EXPAND | wx.ALL, 4)

        self.pnl_topo_file = wx.Panel(parent)
        h_tf = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_topo_path = wx.TextCtrl(self.pnl_topo_file, value=self.default_dem)
        btn_tb = wx.Button(self.pnl_topo_file, label="浏览...")
        btn_tb.Bind(wx.EVT_BUTTON, lambda e: self._choose_file(self.txt_topo_path, "选择高程文件", "*.npy;*.csv;*.tif"))
        h_tf.Add(self.txt_topo_path, 1, wx.EXPAND | wx.RIGHT, 4)
        h_tf.Add(btn_tb, 0)
        self.pnl_topo_file.SetSizer(h_tf)
        sb_topo_sizer.Add(self.pnl_topo_file, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        self.pnl_topo_random = wx.Panel(parent)
        grid_tr = wx.FlexGridSizer(1, 4, 4, 4)
        grid_tr.AddGrowableCol(1)
        grid_tr.AddGrowableCol(3)
        self.txt_topo_zmin = wx.TextCtrl(self.pnl_topo_random, value="880.0")
        self.txt_topo_zmax = wx.TextCtrl(self.pnl_topo_random, value="980.0")
        grid_tr.Add(wx.StaticText(self.pnl_topo_random, label="Zmin:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_tr.Add(self.txt_topo_zmin, 1, wx.EXPAND)
        grid_tr.Add(wx.StaticText(self.pnl_topo_random, label="Zmax:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_tr.Add(self.txt_topo_zmax, 1, wx.EXPAND)
        self.pnl_topo_random.SetSizer(grid_tr)
        sb_topo_sizer.Add(self.pnl_topo_random, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(sb_topo_sizer, 0, wx.EXPAND | wx.ALL, 4)
        parent.SetSizer(sizer)
        self.on_topo_mode_changed()

    def on_topo_mode_changed(self, event=None):
        sel = self.choice_topo_mode.GetSelection()
        if sel == 0:
            self.pnl_topo_file.Hide()
            self.pnl_topo_random.Hide()
        elif sel == 1:
            self.pnl_topo_file.Show()
            self.pnl_topo_random.Hide()
        elif sel == 2:
            self.pnl_topo_file.Hide()
            self.pnl_topo_random.Show()
        self.pnl_step6.Layout()

    def _build_tab_sections(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sb = wx.StaticBox(parent, label="自定义剖面切割参数")
        sb_sizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        grid = wx.FlexGridSizer(5, 2, 6, 6)
        grid.AddGrowableCol(1)

        self.txt_sec_name = wx.TextCtrl(parent, value="Section_A_A")
        self.txt_sec_start = wx.TextCtrl(parent, value="200, 200")
        self.txt_sec_stop = wx.TextCtrl(parent, value="1800, 1800")
        self.txt_sec_res = wx.TextCtrl(parent, value="100, 80")
        self.txt_sec_ve = wx.TextCtrl(parent, value="1.0")

        grid.Add(wx.StaticText(parent, label="剖面名称:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_sec_name, 1, wx.EXPAND)
        grid.Add(wx.StaticText(parent, label="起点 X, Y (米):"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_sec_start, 1, wx.EXPAND)
        grid.Add(wx.StaticText(parent, label="终点 X, Y (米):"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_sec_stop, 1, wx.EXPAND)
        grid.Add(wx.StaticText(parent, label="采样点 (水平, 垂直):"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_sec_res, 1, wx.EXPAND)
        grid.Add(wx.StaticText(parent, label="纵向夸大倍数 (VE):"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_sec_ve, 1, wx.EXPAND)
        sb_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 4)

        self.chk_sec_bh = wx.CheckBox(parent, label="在剖面上投影虚拟钻孔 (投影容差 200m)")
        self.chk_sec_bh.SetValue(True)
        sb_sizer.Add(self.chk_sec_bh, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)

        self.chk_sec_topo = wx.CheckBox(parent, label="在剖面上绘制地表地形轮廓线 (Topography)")
        self.chk_sec_topo.SetValue(True)
        sb_sizer.Add(self.chk_sec_topo, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(sb_sizer, 0, wx.EXPAND | wx.ALL, 4)

        h_btn = wx.BoxSizer(wx.HORIZONTAL)
        btn_cut = wx.Button(parent, label="✂ 切割并在右侧显示剖面")
        btn_cut.Bind(wx.EVT_BUTTON, self.on_cut_section)
        btn_export_sec = wx.Button(parent, label="💾 导出剖面图片...")
        btn_export_sec.Bind(wx.EVT_BUTTON, self.on_export_section_image)
        h_btn.Add(btn_cut, 1, wx.RIGHT, 4)
        h_btn.Add(btn_export_sec, 1)
        sizer.Add(h_btn, 0, wx.EXPAND | wx.ALL, 4)

        sb_preset = wx.StaticBox(parent, label="快速正交切片")
        sb_preset_sizer = wx.StaticBoxSizer(sb_preset, wx.HORIZONTAL)
        btn_px = wx.Button(parent, label="沿着 X 轴切片 (中线)")
        btn_py = wx.Button(parent, label="沿着 Y 轴切片 (中线)")
        btn_px.Bind(wx.EVT_BUTTON, lambda e: self.on_quick_orthogonal("x"))
        btn_py.Bind(wx.EVT_BUTTON, lambda e: self.on_quick_orthogonal("y"))
        sb_preset_sizer.Add(btn_px, 1, wx.RIGHT, 4)
        sb_preset_sizer.Add(btn_py, 1)
        sizer.Add(sb_preset_sizer, 0, wx.EXPAND | wx.ALL, 4)

        parent.SetSizer(sizer)

    def _build_tab_boreholes(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sb_list = wx.StaticBox(parent, label="虚拟钻孔列表")
        sb_list_sizer = wx.StaticBoxSizer(sb_list, wx.VERTICAL)

        self.list_bh = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_bh.InsertColumn(0, "孔号", width=60)
        self.list_bh.InsertColumn(1, "X (m)", width=65)
        self.list_bh.InsertColumn(2, "Y (m)", width=65)
        self.list_bh.InsertColumn(3, "孔口标高", width=70)
        self.list_bh.InsertColumn(4, "孔底标高", width=70)
        self.list_bh.InsertColumn(5, "孔深 (m)", width=65)
        sb_list_sizer.Add(self.list_bh, 1, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_list_sizer, 1, wx.EXPAND | wx.ALL, 4)

        sb_add = wx.StaticBox(parent, label="添加 / 更新虚拟钻孔")
        sb_add_sizer = wx.StaticBoxSizer(sb_add, wx.VERTICAL)

        g_bh = wx.FlexGridSizer(3, 4, 4, 4)
        g_bh.AddGrowableCol(1)
        g_bh.AddGrowableCol(3)

        self.txt_bh_name = wx.TextCtrl(parent, value="ZK03")
        self.txt_bh_x = wx.TextCtrl(parent, value="1000")
        self.txt_bh_y = wx.TextCtrl(parent, value="1000")
        self.txt_bh_ztop = wx.TextCtrl(parent, value="750")
        self.txt_bh_zbot = wx.TextCtrl(parent, value="100")

        g_bh.Add(wx.StaticText(parent, label="孔号:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_bh.Add(self.txt_bh_name, 1, wx.EXPAND)
        g_bh.Add(wx.StaticText(parent, label="孔口 X:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_bh.Add(self.txt_bh_x, 1, wx.EXPAND)
        g_bh.Add(wx.StaticText(parent, label="孔口 Y:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_bh.Add(self.txt_bh_y, 1, wx.EXPAND)
        g_bh.Add(wx.StaticText(parent, label="孔口标高:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_bh.Add(self.txt_bh_ztop, 1, wx.EXPAND)
        g_bh.Add(wx.StaticText(parent, label="孔底标高:"), 0, wx.ALIGN_CENTER_VERTICAL)
        g_bh.Add(self.txt_bh_zbot, 1, wx.EXPAND)
        sb_add_sizer.Add(g_bh, 0, wx.EXPAND | wx.ALL, 4)

        h_bh_btns = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(parent, label="➕ 添加/更新钻孔")
        btn_add.Bind(wx.EVT_BUTTON, self.on_add_borehole)
        btn_del = wx.Button(parent, label="✖ 删除选中钻孔")
        btn_del.Bind(wx.EVT_BUTTON, self.on_delete_borehole)
        h_bh_btns.Add(btn_add, 1, wx.RIGHT, 4)
        h_bh_btns.Add(btn_del, 1)
        sb_add_sizer.Add(h_bh_btns, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_add_sizer, 0, wx.EXPAND | wx.ALL, 4)

        h_sample = wx.BoxSizer(wx.HORIZONTAL)
        btn_view_col = wx.Button(parent, label="📊 绘制选中钻孔综合柱状图")
        btn_view_col.Bind(wx.EVT_BUTTON, self.on_plot_borehole_stratigraphy)
        btn_exp_csv = wx.Button(parent, label="📥 导出所有钻孔柱状 CSV")
        btn_exp_csv.Bind(wx.EVT_BUTTON, self.on_export_borehole_csv)
        h_sample.Add(btn_view_col, 1, wx.RIGHT, 4)
        h_sample.Add(btn_exp_csv, 1)
        sizer.Add(h_sample, 0, wx.EXPAND | wx.ALL, 4)

        parent.SetSizer(sizer)

    def _build_tab_export(self, parent):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sb_3d = wx.StaticBox(parent, label="三维交互与可视化")
        sb_3d_sizer = wx.StaticBoxSizer(sb_3d, wx.VERTICAL)

        txt_info = wx.StaticText(
            parent,
            label="调用 PyVista 交互式三维视窗，在独立窗口中自由旋转、平移、缩放三维地质体，并显示实体钻孔管道与孔口球体。",
        )
        txt_info.Wrap(380)
        sb_3d_sizer.Add(txt_info, 0, wx.ALL, 6)

        btn_launch_3d = wx.Button(parent, label="🚀 启动三维交互窗口 (PyVista)", size=(-1, 42))
        btn_launch_3d.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btn_launch_3d.Bind(wx.EVT_BUTTON, lambda e: self.on_open_3d_viewer())
        sb_3d_sizer.Add(btn_launch_3d, 0, wx.EXPAND | wx.ALL, 6)

        self.chk_3d_topo = wx.CheckBox(parent, label="在 3D 视窗中呈现地表三维起伏网格")
        self.chk_3d_topo.SetValue(True)
        sb_3d_sizer.Add(self.chk_3d_topo, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        sizer.Add(sb_3d_sizer, 0, wx.EXPAND | wx.ALL, 4)

        sb_data_exp = wx.StaticBox(parent, label="地学成果与模型数据导出")
        sb_data_exp_sizer = wx.StaticBoxSizer(sb_data_exp, wx.VERTICAL)

        btn_load_gp = wx.Button(parent, label="📂 载入已有地质模型 (.gempy)...")
        btn_load_gp.Bind(wx.EVT_BUTTON, lambda e: self.on_load_gempy_model())
        sb_data_exp_sizer.Add(btn_load_gp, 0, wx.EXPAND | wx.ALL, 4)

        btn_save_gp = wx.Button(parent, label="💾 保存当前地质模型 (.gempy)...")
        btn_save_gp.Bind(wx.EVT_BUTTON, lambda e: self.on_save_gempy_model())
        sb_data_exp_sizer.Add(btn_save_gp, 0, wx.EXPAND | wx.ALL, 4)

        btn_save_cfg = wx.Button(parent, label="💾 导出建模参数方案 (JSON)...")
        btn_save_cfg.Bind(wx.EVT_BUTTON, lambda e: self.on_save_config_profile())
        sb_data_exp_sizer.Add(btn_save_cfg, 0, wx.EXPAND | wx.ALL, 4)

        btn_exp_all_vtk = wx.Button(parent, label="📦 导出所有地层曲面真实坐标 VTK 网格...")
        btn_exp_all_vtk.Bind(wx.EVT_BUTTON, lambda e: self.on_export_vtk())
        sb_data_exp_sizer.Add(btn_exp_all_vtk, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(sb_data_exp_sizer, 0, wx.EXPAND | wx.ALL, 4)

        parent.SetSizer(sizer)

    # -------------------------------------------------------------------------
    # 右侧视图与画布构建
    # -------------------------------------------------------------------------
    def _build_right_panel(self):
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        self.canvas_container = wx.Panel(self.right_panel)
        self.csizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas_container.SetSizer(self.csizer)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.canvas_container, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        self.csizer.Add(self.toolbar, 0, wx.EXPAND)
        self.csizer.Add(self.canvas, 1, wx.EXPAND)
        right_sizer.Add(self.canvas_container, 1, wx.EXPAND)

        # 运行日志
        sb_log = wx.StaticBox(self.right_panel, label="系统运行日志与地学分析")
        sb_log_sizer = wx.StaticBoxSizer(sb_log, wx.VERTICAL)

        self.txt_log = wx.TextCtrl(
            self.right_panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
            size=(-1, 160),
        )
        font = wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.txt_log.SetFont(font)
        sb_log_sizer.Add(self.txt_log, 1, wx.EXPAND)

        right_sizer.Add(sb_log_sizer, 0, wx.EXPAND | wx.ALL, 4)
        self.right_panel.SetSizer(right_sizer)

    # =========================================================================
    # 步骤向导与工作流调度
    # =========================================================================
    def goto_step(self, step_idx: int):
        """跳转至指定步骤页面并联动更新视图"""
        step_idx = max(0, min(step_idx, 7))
        self.current_step = step_idx
        self.step_bar.set_active_step(step_idx)

        if self.step_notebook.GetSelection() != step_idx:
            self.step_notebook.ChangeSelection(step_idx)

        # 导航按钮文字更新
        step_names = [
            "步骤 2: 黑白Mask提取",
            "步骤 3: DEM高程提取",
            "步骤 4: 边界断层分离",
            "步骤 5: 三维控制点",
            "步骤 6: 地层人工合并",
            "步骤 7: 产状解算核对",
            "步骤 8: 三维建模展示",
            "完成",
        ]
        self.btn_prev_step.Enable(step_idx > 0)
        if step_idx < 7:
            self.btn_next_step.SetLabel(f"下一步: {step_names[step_idx]} ▶")
            self.btn_next_step.Enable(True)
        else:
            self.btn_next_step.SetLabel("✓ 当前已在建模阶段")
            self.btn_next_step.Disable()

        self.status_bar.SetStatusText(f"当前步骤: {StepWizardBar.STEP_NAMES[step_idx]}", 0)

        # 步骤联动动作
        if step_idx == 4 and not self.features_meta:
            self.on_extract_surface_points_step()
        elif step_idx == 6:
            # 步骤 7 联动：确保所有选取的点（断层 + 全部地层）均已解算产状并展示
            needed_forms = set()
            if self.features_meta:
                needed_forms = {self.layer_mapping.get(k, k) for k in self.features_meta.keys()}
            elif self.surface_points_df is not None and len(self.surface_points_df) > 0:
                needed_forms = set(self.surface_points_df["formation"].dropna().unique())

            existing_forms = {a.get("formation") for a in self.orientations_list}
            if (not self.orientations_list) or (needed_forms and not needed_forms.issubset(existing_forms)):
                self.on_compute_attitudes_step(silent=True)
        elif step_idx == 7:
            if self.orientations_list:
                self.on_save_orientations_csv(show_msg=False)
            self.on_auto_detect_extent(show_msg=False)

        self._render_current_step_canvas()

    def on_prev_step(self):
        if self.current_step > 0:
            self.goto_step(self.current_step - 1)

    def on_next_step(self):
        if self.current_step < 7:
            self.goto_step(self.current_step + 1)

    def _on_notebook_page_changed(self, event):
        # 严格过滤：仅响应顶层步骤主 Notebook 的切换，防止步骤 8 内部子选项卡冒泡导致跳页
        if hasattr(self, "step_notebook") and event.GetEventObject() != self.step_notebook:
            event.Skip()
            return

        idx = event.GetSelection()
        if idx != self.current_step:
            self.goto_step(idx)
        event.Skip()

    # =========================================================================
    # 右侧多维成果画布渲染 (分步呈现各阶段产物)
    # =========================================================================
    def _render_current_step_canvas(self):
        """根据当前步骤动态重绘右侧 Matplotlib 画布"""
        setup_chinese_font()
        fig = plt.figure(figsize=(9, 6), dpi=120)

        if self.current_step == 0:
            self._render_step1_canvas(fig)
        elif self.current_step == 1:
            self._render_step2_canvas(fig)
        elif self.current_step == 2:
            self._render_step3_canvas(fig)
        elif self.current_step == 3:
            self._render_step4_canvas(fig)
        elif self.current_step == 4:
            self._render_step5_canvas(fig)
        elif self.current_step == 5:
            self._render_step6_canvas(fig)
        elif self.current_step == 6:
            self._render_step7_canvas(fig)
        elif self.current_step == 7:
            self._render_step8_canvas(fig)

        self._update_canvas(fig)

    def _render_step1_canvas(self, fig: plt.Figure):
        """步骤 1: 渲染原始平面地质图与空间坐标轴"""
        ax = fig.add_subplot(1, 1, 1)
        map_p = self.txt_map_path.GetValue().strip()
        if os.path.exists(map_p):
            img = cv2.imread(map_p)
            h, w = img.shape[:2]
            scale = self.scale_gsd
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, w * scale, 0, h * scale])
            ax.set_title(f"步骤 1: 原始地质图输入与空间尺度标定 (总宽 {w*scale:.0f}m, 总高 {h*scale:.0f}m)", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("X (东向, 米)", fontsize=9)
            ax.set_ylabel("Y (北向, 米)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.4, color="gray")
        else:
            ax.text(0.5, 0.5, "未找到地质图文件，请在左侧指定图片路径", ha="center", va="center", color="red")
            ax.set_xticks([])
            ax.set_yticks([])

    def _render_step2_canvas(self, fig: plt.Figure):
        """步骤 2: 查看黑白综合掩膜 (test_mask.png) 提取成果与原图对比"""
        ax = fig.add_subplot(1, 1, 1)
        sel = self.choice_step2_view.GetSelection() if hasattr(self, "choice_step2_view") else 1
        map_p = self.txt_map_path.GetValue().strip()
        mask_p = os.path.join(self.cur_dir, "test_mask.png")
        scale = self.scale_gsd

        if sel == 0:
            # 原始彩色地质图
            if os.path.exists(map_p):
                img = cv2.imread(map_p)
                h, w = img.shape[:2]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, w * scale, 0, h * scale])
                ax.set_title("步骤 2: 原始输入地质图 (RGB)", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel("X (东向, 米)", fontsize=9)
                ax.set_ylabel("Y (北向, 米)", fontsize=9)
            else:
                ax.text(0.5, 0.5, "未找到地质图文件", ha="center", va="center", color="red")
        elif sel == 1:
            # 暗色综合黑白掩膜 (test_mask.png)
            if os.path.exists(mask_p):
                mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                h, w = mask.shape[:2]
                ax.imshow(mask, cmap="gray", extent=[0, w * scale, 0, h * scale])
                ax.set_title(f"步骤 2: 暗色综合黑白掩膜 test_mask.png ({w}×{h} px)", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel("X (东向, 米)", fontsize=9)
                ax.set_ylabel("Y (北向, 米)", fontsize=9)
            else:
                ax.text(0.5, 0.5, "未找到掩膜文件 test_mask.png\n请点击左侧【⚡ 运行/重新提取黑白综合掩膜】按钮生成", ha="center", va="center", color="blue")
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # 掩膜与原图叠加对比图 (Overlay)
            if os.path.exists(map_p) and os.path.exists(mask_p):
                img = cv2.imread(map_p)
                mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb[mask > 128] = [255, 40, 40]
                ax.imshow(rgb, extent=[0, w * scale, 0, h * scale])
                ax.set_title("步骤 2: 黑白掩膜在原始地质图上的叠加比对图 (红色为提取掩膜)", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel("X (东向, 米)", fontsize=9)
                ax.set_ylabel("Y (北向, 米)", fontsize=9)
            elif os.path.exists(mask_p):
                mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                h, w = mask.shape[:2]
                ax.imshow(mask, cmap="gray", extent=[0, w * scale, 0, h * scale])
                ax.set_title("步骤 2: 黑白掩膜", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel("X (东向, 米)", fontsize=9)
                ax.set_ylabel("Y (北向, 米)", fontsize=9)
            else:
                ax.text(0.5, 0.5, "未找到原图或掩膜文件", ha="center", va="center", color="red")
                ax.set_xticks([])
                ax.set_yticks([])

    def _render_step3_canvas(self, fig: plt.Figure):
        """步骤 3: 渲染全自动提取的连续 DEM 高程模型成果"""
        ax = fig.add_subplot(1, 1, 1)
        sel = self.choice_step3_view.GetSelection() if hasattr(self, "choice_step3_view") else 0
        dem_file = self.txt_dem_path.GetValue().strip() if hasattr(self, "txt_dem_path") else ""
        if not os.path.exists(dem_file):
            for cand in [
                os.path.join(self.cur_dir, "dem_output", "dem_auto.npy"),
                os.path.join(self.cur_dir, "results", "dem_auto.npy"),
            ]:
                if os.path.exists(cand):
                    dem_file = cand
                    break

        if sel == 0:
            # 连续平滑 DEM 彩色高程热力图 (含等高线)
            if os.path.exists(dem_file):
                dem = np.load(dem_file)
                h, w = dem.shape
                scale = self.scale_gsd
                extent = [0, w * scale, 0, h * scale]
                dem_disp = np.flipud(dem)
                im = ax.imshow(dem_disp, cmap="terrain", extent=extent, origin="lower")
                cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
                cb.set_label("高程 Z (米)", fontsize=9)

                X = np.linspace(0, w * scale, w)
                Y = np.linspace(0, h * scale, h)
                ci = float(self.spin_contour_interval.GetValue()) if hasattr(self, "spin_contour_interval") else 10.0
                ci = max(1.0, ci)
                if np.nanmax(dem) > np.nanmin(dem):
                    min_level = np.ceil(np.nanmin(dem) / ci) * ci
                    max_level = np.floor(np.nanmax(dem) / ci) * ci
                    levels = np.arange(min_level, max_level + ci * 0.5, ci)
                    if len(levels) > 0:
                        cs = ax.contour(X, Y, dem_disp, levels=levels, colors="black", alpha=0.45, linewidths=0.7)
                        ax.clabel(cs, inline=True, fontsize=8, fmt="%.0fm")

                ax.set_title(
                    f"步骤 3: 全自动连续 DEM 彩色高程模型 (标高 {np.nanmin(dem):.1f}m ~ {np.nanmax(dem):.1f}m, 等高距={ci:.0f}m)",
                    fontsize=11,
                    fontweight="bold",
                    pad=10,
                )
                ax.set_xlabel("X (东向, 米)", fontsize=9)
                ax.set_ylabel("Y (北向, 米)", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "未找到 DEM 高程数据 dem_auto.npy\n请点击左侧【⚡ 运行全自动连续 DEM 重建】按钮生成",
                    ha="center",
                    va="center",
                    color="blue",
                    fontsize=11,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        elif sel == 1:
            # 全自动 3D 连续地貌透视图
            view_3d = os.path.join(self.cur_dir, "dem_output", "dem_3d_view.png")
            if not os.path.exists(view_3d):
                view_3d = os.path.join(self.cur_dir, "results", "dem_3d_view.png")
            if os.path.exists(view_3d):
                img = cv2.imread(view_3d)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title("步骤 3: 全自动 3D 连续地貌透视三维视图", fontsize=11, fontweight="bold", pad=10)
                ax.axis("off")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "未找到 3D 地貌透视图 dem_3d_view.png\n请先点击左侧【⚡ 运行全自动连续 DEM 重建】",
                    ha="center",
                    va="center",
                    color="blue",
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # DEM 成果 4合1 科研大看板
            dash_p = os.path.join(self.cur_dir, "dem_output", "demo_result_dashboard.png")
            if not os.path.exists(dash_p):
                dash_p = os.path.join(self.cur_dir, "results", "demo_result_dashboard.png")
            if os.path.exists(dash_p):
                img = cv2.imread(dash_p)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title("步骤 3: DEM 连续高程模型 4合1 诊断看板", fontsize=11, fontweight="bold", pad=10)
                ax.axis("off")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "未找到 4合1 看板 demo_result_dashboard.png\n请先点击左侧【⚡ 运行全自动连续 DEM 重建】",
                    ha="center",
                    va="center",
                    color="blue",
                )
                ax.set_xticks([])
                ax.set_yticks([])

    def _render_step4_canvas(self, fig: plt.Figure):
        """步骤 4: 切换查看粗细特征分离的地质界线与断层线骨架成果"""
        ax = fig.add_subplot(1, 1, 1)
        sel = self.choice_step4_view.GetSelection() if hasattr(self, "choice_step4_view") else 2

        file_map = {
            0: os.path.join(self.cur_dir, "boundary_output", "boundary_skeleton.png"),
            1: os.path.join(self.cur_dir, "fault_output", "fault_skeleton.png"),
            2: os.path.join(self.cur_dir, "boundary_output", "boundary_overlay.png"),
        }
        title_map = {
            0: "① 细地层边界纯净单像素骨架 (boundary_skeleton.png)",
            1: "② 粗断层构造中心走向骨架 (fault_skeleton.png)",
            2: "③ 地质界线与断层线粗细分离综合对比大图 (Overlay)",
        }

        target_file = file_map.get(sel, "")
        if not os.path.exists(target_file):
            if sel == 2:
                target_file = os.path.join(self.cur_dir, "fault_output", "fault_overlay.png")

        if os.path.exists(target_file):
            img = cv2.imread(target_file)
            h, w = img.shape[:2]
            scale = self.scale_gsd
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, w * scale, 0, h * scale])
            ax.set_title(f"步骤 4 产出成果: {title_map.get(sel, '')}", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("X (东向, 米)", fontsize=9)
            ax.set_ylabel("Y (北向, 米)", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                f"未找到产出文件: {os.path.basename(target_file)}\n请点击左侧【⚡ 运行/重新分离提取地质界线与断层骨架】按钮生成",
                ha="center",
                va="center",
                color="blue",
            )
            ax.set_xticks([])
            ax.set_yticks([])

    def _render_step5_canvas(self, fig: plt.Figure):
        """步骤 5: 渲染提取的全部构造线与三维采样控制点 (结合步骤 3 DEM 标高)"""
        ax = fig.add_subplot(1, 1, 1)
        map_p = self.txt_map_path.GetValue().strip()

        if os.path.exists(map_p):
            img = cv2.imread(map_p)
            h, w = img.shape[:2]
            scale = self.scale_gsd
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, w * scale, 0, h * scale], alpha=0.75)
        else:
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 1326)

        if self.features_meta:
            for name, feat in self.features_meta.items():
                col = feat.get("color", "#377eb8")
                pts = feat["sampled_points"]
                xs = [p["X"] for p in pts]
                ys = [p["Y"] for p in pts]
                lbl = f"{name} ({feat['type']}, {len(pts)}点)"
                ax.plot(xs, ys, color=col, linewidth=2.4, label=lbl)
                ax.scatter(xs, ys, color=col, s=35, edgecolors="white", zorder=5)

            ax.set_title("步骤 5: 提取的独立构造线与控制点空间采样 (连通段 Layer 视图)", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("X (东向, 米)", fontsize=9)
            ax.set_ylabel("Y (北向, 米)", fontsize=9)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "暂未提取控制点，请点击左侧【⚡ 重新提取控制点】", ha="center", va="center")

    def _render_step6_canvas(self, fig: plt.Figure):
        """步骤 6: 人工交互合并画布 (高亮突出显示当前选中的连通段，支持原图/Mask底图切换)"""
        ax = fig.add_subplot(1, 1, 1)
        map_p = self.txt_map_path.GetValue().strip()
        mask_p = os.path.join(self.cur_dir, "test_mask.png")
        if not os.path.exists(mask_p):
            mask_p = os.path.join(self.cur_dir, "boundary_output", "boundary_mask.png")

        bg_sel = getattr(self, "choice_step6_bg", getattr(self, "choice_step4_bg", None))
        show_mask = (bg_sel is not None and bg_sel.GetSelection() == 1)

        scale = self.scale_gsd
        bg_loaded = False
        img_h = 663

        if show_mask and os.path.exists(mask_p):
            img_m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if img_m is not None:
                img_h, img_w = img_m.shape[:2]
                ax.imshow(img_m, cmap="gray", extent=[0, img_w * scale, 0, img_h * scale], alpha=0.88)
                bg_loaded = True

        if not bg_loaded and os.path.exists(map_p):
            img_c = cv2.imread(map_p)
            if img_c is not None:
                img_h, img_w = img_c.shape[:2]
                ax.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB), extent=[0, img_w * scale, 0, img_h * scale], alpha=0.82)
                bg_loaded = True

        if not bg_loaded:
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 1326)

        sel_name = self.selected_layer_name or (list(self.features_meta.keys())[0] if self.features_meta else None)

        if self.features_meta:
            for name, feat in self.features_meta.items():
                pts = feat["sampled_points"]
                col = feat.get("color", "gray")

                # 提取完整空间轨迹
                if "path_px" in feat and feat["path_px"]:
                    path_xs = [p[1] * scale for p in feat["path_px"]]
                    path_ys = [(img_h - 1 - p[0]) * scale for p in feat["path_px"]]
                else:
                    path_xs = [p["X"] for p in pts]
                    path_ys = [p["Y"] for p in pts]

                xs = [p["X"] for p in pts]
                ys = [p["Y"] for p in pts]

                if name == sel_name:
                    # 强对比荧光粗线高亮当前选中要素及其连续骨架边界
                    ax.plot(path_xs, path_ys, color="yellow", linewidth=5.8, zorder=8)
                    ax.plot(path_xs, path_ys, color="magenta", linewidth=3.0, zorder=9, label=f"★ 当前高亮: [{name}]")
                    ax.scatter(xs, ys, color="yellow", s=75, edgecolors="black", linewidths=1.2, zorder=10)

                    # 标注起始与几何范围提示框
                    mid_idx = len(xs) // 2
                    target = self.layer_mapping.get(name, name)
                    ax.annotate(
                        f"  【{name}】\n-> 归属: {target}\n(长 {feat['length_px']}px, {len(pts)}点)",
                        xy=(xs[mid_idx], ys[mid_idx]),
                        xytext=(xs[mid_idx] + 30, ys[mid_idx] + 30),
                        fontsize=9,
                        fontweight="bold",
                        color="navy",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="magenta", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="magenta", lw=1.5),
                        zorder=15,
                    )
                else:
                    # 其余要素以半透明弱化显示
                    ax.plot(path_xs, path_ys, color=col, linewidth=1.6, alpha=0.45, label=f"{name} -> {self.layer_mapping.get(name, name)}")
                    ax.scatter(xs, ys, color=col, s=15, alpha=0.35, zorder=4)

            bg_title = "二值Mask底图" if show_mask else "原始彩色地质图"
            ax.set_title(f"步骤 6: 地层人工识别与合并 - 当前高亮聚焦 [{sel_name}] ({bg_title})", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("X (东向, 米)", fontsize=9)
            ax.set_ylabel("Y (北向, 米)", fontsize=9)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.88)
        else:
            ax.text(0.5, 0.5, "暂无地层连通段，请先完成步骤 5 控制点提取", ha="center", va="center")

    def _render_step7_canvas(self, fig: plt.Figure):
        """步骤 7: 渲染地质产状 Strike-Dip 符号，高亮标红倾向突变异常点"""
        ax = fig.add_subplot(1, 1, 1)
        map_p = self.txt_map_path.GetValue().strip()

        if os.path.exists(map_p):
            img = cv2.imread(map_p)
            h, w = img.shape[:2]
            scale = self.scale_gsd
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, w * scale, 0, h * scale], alpha=0.65)
        else:
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 1326)

        # 绘制构造线轮廓背景
        if self.features_meta:
            for name, feat in self.features_meta.items():
                pts = feat["sampled_points"]
                ax.plot([p["X"] for p in pts], [p["Y"] for p in pts], color="gray", lw=1.2, linestyle="--", alpha=0.5)

        # 绘制标准地质产状符号 (走向线 + 倾向短垂直短杠)
        # 绘制标准地质产状符号 (走向线 + 倾向短垂直短杠)
        if self.orientations_list:
            sym_len = 70.0  # 走向线基准长度 (米)
            has_fault = False
            has_strata = False
            has_mut = False

            for idx, att in enumerate(self.orientations_list):
                x = att["X"]
                y = att["Y"]
                az = att["azimuth"]
                dip = att["dip"]
                is_mut = att.get("is_mutation", False)
                form_name = str(att.get("formation", ""))
                is_fault = any(k in form_name.lower() for k in ["fault", "断层"])

                strike = (az - 90.0) % 360.0
                rad_s = np.radians(strike)
                rad_d = np.radians(az)

                # 区分颜色：突变异常纯红；断层深酒红；地层深蓝色
                if is_mut:
                    color = "red"
                    lw = 2.6
                    has_mut = True
                elif is_fault:
                    color = "#B22222"  # 深红
                    lw = 2.0
                    has_fault = True
                else:
                    color = "#00558F"  # 深蓝
                    lw = 1.8
                    has_strata = True

                # 走向短实线
                dx_s = (sym_len / 2.0) * np.sin(rad_s)
                dy_s = (sym_len / 2.0) * np.cos(rad_s)
                ax.plot([x - dx_s, x + dx_s], [y - dy_s, y + dy_s], color=color, linewidth=lw, zorder=10)

                # 倾向短垂直短杠 (指向下倾方向)
                tick_len = sym_len * 0.42
                dx_d = tick_len * np.sin(rad_d)
                dy_d = tick_len * np.cos(rad_d)
                ax.plot([x, x + dx_d], [y, y + dy_d], color=color, linewidth=lw, zorder=10)

                # 中心控制点
                pt_size = 50 if is_mut else 32
                ax.scatter([x], [y], color=color, s=pt_size, edgecolors="white", zorder=11)

                # 产状数值与突变标记
                txt = f"∠{dip:.0f}°∠{az:.0f}°"
                if is_mut:
                    txt += " [异常]"
                    ax.text(x + dx_d * 1.35, y + dy_d * 1.35, txt, color="red", fontsize=8, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", edgecolor="red", alpha=0.9), zorder=15)
                else:
                    ax.text(x + dx_d * 1.3, y + dy_d * 1.3, txt, color=color, fontsize=8, fontweight="bold", ha="center", va="center", zorder=12)

            # 图例
            import matplotlib.lines as mlines
            legends = []
            if has_fault:
                legends.append(mlines.Line2D([], [], color="#B22222", marker="o", markersize=6, label="断层产状控制点"))
            if has_strata:
                legends.append(mlines.Line2D([], [], color="#00558F", marker="o", markersize=6, label="地层产状控制点"))
            if has_mut:
                legends.append(mlines.Line2D([], [], color="red", marker="o", markersize=7, label="异常预警点 (可智能修复)"))
            if legends:
                ax.legend(handles=legends, loc="upper right", framealpha=0.92, fontsize=8)

            ax.set_title("步骤 7: 全要素多点空间产状与倾向突变质检 (红:断层, 蓝:地层, 黄标:异常)", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("X (东向, 米)", fontsize=9)
            ax.set_ylabel("Y (北向, 米)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "暂无产状数据，请点击左侧【⚡ 重新解算产状与突变检测】", ha="center", va="center")

    def _render_step8_canvas(self, fig: plt.Figure):
        """步骤 8: 渲染剖面切割图或三维建模就绪概览"""
        ax = fig.add_subplot(1, 1, 1)
        if self.engine.is_computed:
            # 若已计算，显示当前剖面
            sec_name = self.txt_sec_name.GetValue().strip() or "Section_A_A"
            if sec_name in self.customizer.sections_config:
                try:
                    self.customizer.render_cross_section(
                        section_name=sec_name,
                        show_boreholes=self.chk_sec_bh.IsChecked(),
                        show_topography=self.chk_sec_topo.IsChecked(),
                        vertical_exaggeration=float(self.txt_sec_ve.GetValue()),
                        ax=ax,
                    )
                    return
                except Exception:
                    pass

        ax.text(
            0.5,
            0.5,
            "【质检完成，就绪建模】\n\n地层界面人工合并与产状质检已全部确认完毕!\n\n"
            "点击左侧绿色按钮【开始构建并计算模型】\n"
            "系统将启动 GemPy 3.x 隐式势场求解器进行三维建模并自动切割 2D 地质剖面。",
            ha="center",
            va="center",
            fontsize=12,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f8f0", edgecolor="#2e8b57"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # =========================================================================
    # 建模与产出管理核心业务 (保留全部已有功能并兼容已有测试)
    # =========================================================================
    def log(self, message: str):
        def _append():
            self.txt_log.AppendText(message + "\n")
        if wx.IsMainThread():
            _append()
        else:
            wx.CallAfter(_append)

    def _initial_setup(self):
        """系统初始化启动检查与预设恢复"""
        self.log("系统启动中...")
        self.log(f"工作目录: {self.cur_dir}")
        self.log(f"默认地质图: {self.default_map}")
        self.log(f"默认 DEM: {self.default_dem}")

        # 检测并尝试自动载入已有工程配置
        cfg_file = os.path.join(self.cur_dir, "project_config.json")
        if os.path.exists(cfg_file):
            try:
                self.on_load_config_profile(cfg_file)
                self.log("✓ 已自动恢复建模工程配置: project_config.json")
            except Exception as e:
                self.log(f"配置恢复提示: {e}，将采用自适应参数")
                self.on_auto_detect_extent(show_msg=False)
        else:
            self.on_auto_detect_extent(show_msg=False)

        self.on_resolution_or_extent_changed()

        # 加载示例钻孔
        try:
            z_max = float(self.txt_zmax.GetValue())
            z_min = float(self.txt_zmin.GetValue())
        except Exception:
            z_max, z_min = 1000.0, 600.0

        self._add_borehole_to_list("ZK01", 600.0, 600.0, z_max - 30.0, z_min + 30.0, "firebrick")
        self._add_borehole_to_list("ZK02", 1400.0, 1000.0, z_max - 30.0, z_min + 30.0, "darkgreen")

        # 尝试提前载入已有的骨架与控制点
        try:
            b_skel = os.path.join(self.cur_dir, "boundary_output", "boundary_skeleton.png")
            f_skel = os.path.join(self.cur_dir, "fault_output", "fault_skeleton.png")
            dem_file = self.default_dem
            if os.path.exists(b_skel) and os.path.exists(f_skel) and os.path.exists(dem_file):
                df, meta = extract_surface_points_structured(
                    fault_skel_path=f_skel,
                    boundary_skel_path=b_skel,
                    dem_path=dem_file,
                    scale=self.scale_gsd,
                    points_per_feature=20,
                    output_csv=None,
                )
                self.features_meta = meta["features"]
                self.surface_points_df = df
                self._sync_features_to_ui()
                self.log(f"✓ 已载入当前项目 {len(self.features_meta)} 个构造与地层要素数据。")
        except Exception as e:
            self.log(f"初始骨架数据加载提示: {e}")

        # 尝试提前载入已有的产状数据库并初始化质检
        try:
            if os.path.exists(self.default_or):
                df_or = pd.read_csv(self.default_or, comment="#")
                if len(df_or) > 0 and {"X", "Y", "Z", "azimuth", "dip", "formation"}.issubset(df_or.columns):
                    init_atts = []
                    for _, r in df_or.iterrows():
                        init_atts.append({
                            "X": float(r["X"]),
                            "Y": float(r["Y"]),
                            "Z": float(r["Z"]),
                            "azimuth": float(r["azimuth"]),
                            "dip": float(r["dip"]),
                            "strike": float(r.get("strike", (float(r["azimuth"]) - 90.0) % 360.0)),
                            "formation": str(r["formation"]),
                            "polarity": float(r.get("polarity", 1.0)),
                        })
                    az_th = float(self.spin_att_az_threshold.GetValue()) if hasattr(self, "spin_att_az_threshold") else 45.0
                    dip_th = float(self.spin_att_dip_threshold.GetValue()) if hasattr(self, "spin_att_dip_threshold") else 20.0
                    self.orientations_list = detect_attitude_mutations(init_atts, az_th, dip_th)
                    if hasattr(self, "list_attitudes"):
                        self._refresh_step5_table()
                    self.log(f"✓ 成功载入已有产状数据库: {self.default_or} (共 {len(self.orientations_list)} 个控制点)")
        except Exception as e:
            self.log(f"初始产状加载提示: {e}")

        # 初始刷新第一步画布
        self.goto_step(0)

    def _choose_file(self, target_ctrl: wx.TextCtrl, title: str, wildcard: str):
        dlg = wx.FileDialog(self, title, self.cur_dir, "", wildcard, wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            target_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    def _add_borehole_to_list(self, name, x, y, z_top, z_bottom, color="royalblue"):
        depth = z_top - z_bottom
        self.customizer.add_virtual_borehole(name, x, y, z_top, z_bottom, color=color)

        idx = self.list_bh.GetItemCount()
        self.list_bh.InsertItem(idx, name)
        self.list_bh.SetItem(idx, 1, f"{x:.1f}")
        self.list_bh.SetItem(idx, 2, f"{y:.1f}")
        self.list_bh.SetItem(idx, 3, f"{z_top:.1f}")
        self.list_bh.SetItem(idx, 4, f"{z_bottom:.1f}")
        self.list_bh.SetItem(idx, 5, f"{depth:.1f}")

    def on_auto_detect_extent(self, show_msg: bool = False):
        sp_path = self.txt_sp.GetValue().strip()
        or_path = self.txt_or.GetValue().strip()
        if not os.path.exists(sp_path) or not os.path.exists(or_path):
            if show_msg:
                wx.MessageBox("找不到数据文件，请先选择合法的界面点与产状 CSV 文件！", "提示", wx.OK | wx.ICON_WARNING)
            return

        try:
            info = self.engine.inspect_csv_files(sp_path, or_path)
            sug = info["suggested_extent"]
            bounds = info["actual_data_bounds"]

            self.txt_xmin.SetValue(f"{sug[0]:.1f}")
            self.txt_xmax.SetValue(f"{sug[1]:.1f}")
            self.txt_ymin.SetValue(f"{sug[2]:.1f}")
            self.txt_ymax.SetValue(f"{sug[3]:.1f}")
            self.txt_zmin.SetValue(f"{sug[4]:.1f}")
            self.txt_zmax.SetValue(f"{sug[5]:.1f}")

            self.txt_sec_start.SetValue(f"{bounds['X'][0] + 100:.0f}, {bounds['Y'][0] + 100:.0f}")
            self.txt_sec_stop.SetValue(f"{bounds['X'][1] - 100:.0f}, {bounds['Y'][1] - 100:.0f}")
            self.on_resolution_or_extent_changed()

            self.log(f"✓ 已根据输入数据自适应更新 Extent 空间范围:")
            self.log(f"  建议范围: X[{sug[0]}, {sug[1]}], Y[{sug[2]}, {sug[3]}], Z[{sug[4]}, {sug[5]}]")

            if show_msg:
                wx.MessageBox(
                    f"已成功自动识别数据范围并更新参数!\n\n"
                    f"• 实际数据极值: X({bounds['X'][0]:.0f}~{bounds['X'][1]:.0f}), Y({bounds['Y'][0]:.0f}~{bounds['Y'][1]:.0f}), Z({bounds['Z'][0]:.0f}~{bounds['Z'][1]:.0f})\n"
                    f"• 建议 Extent: X({sug[0]}~{sug[1]}), Y({sug[2]}~{sug[3]}), Z({sug[4]}~{sug[5]})\n"
                    f"• 识别断层: {', '.join(info['faults']) if info['faults'] else '无'}\n"
                    f"• 识别地层: {', '.join(info['stratigraphy'])}",
                    "自适应范围成功",
                    wx.OK | wx.ICON_INFORMATION,
                )
        except Exception as ex:
            if show_msg:
                wx.MessageBox(f"无法分析数据文件: {ex}", "错误", wx.OK | wx.ICON_ERROR)

    def on_compute_model(self):
        """重新计算模型 (多线程后台执行)"""
        try:
            sp_path = self.txt_sp.GetValue().strip()
            or_path = self.txt_or.GetValue().strip()
            extent = [
                float(self.txt_xmin.GetValue()),
                float(self.txt_xmax.GetValue()),
                float(self.txt_ymin.GetValue()),
                float(self.txt_ymax.GetValue()),
                float(self.txt_zmin.GetValue()),
                float(self.txt_zmax.GetValue()),
            ]
            resolution = [
                int(self.txt_nx.GetValue()),
                int(self.txt_ny.GetValue()),
                int(self.txt_nz.GetValue()),
            ]
            refine = int(self.spin_refine.GetValue())
            surf_refine = int(self.spin_surf_refine.GetValue())
            # 自动核验 Extent 是否完整覆盖当前三维控制点数据范围
            if os.path.exists(sp_path):
                try:
                    df_sp_chk = pd.read_csv(sp_path, comment="#")
                    if len(df_sp_chk) > 0 and {"X", "Y", "Z"}.issubset(df_sp_chk.columns):
                        sp_xmin = float(df_sp_chk["X"].min())
                        sp_xmax = float(df_sp_chk["X"].max())
                        sp_ymin = float(df_sp_chk["Y"].min())
                        sp_ymax = float(df_sp_chk["Y"].max())
                        sp_zmin = float(df_sp_chk["Z"].min())
                        sp_zmax = float(df_sp_chk["Z"].max())

                        # 如果当前 Extent 无法覆盖实际数据点，自动自适应扩展并回填控件
                        if (extent[0] > sp_xmin or extent[1] < sp_xmax or
                            extent[2] > sp_ymin or extent[3] < sp_ymax or
                            extent[4] > sp_zmin or extent[5] < sp_zmax):
                            dx = max((sp_xmax - sp_xmin) * 0.05, 50.0)
                            dy = max((sp_ymax - sp_ymin) * 0.05, 50.0)
                            dz = max((sp_zmax - sp_zmin) * 0.15, 50.0)
                            extent[0] = float(np.floor((sp_xmin - dx) / 50.0) * 50.0)
                            extent[1] = float(np.ceil((sp_xmax + dx) / 50.0) * 50.0)
                            extent[2] = float(np.floor((sp_ymin - dy) / 50.0) * 50.0)
                            extent[3] = float(np.ceil((sp_ymax + dy) / 50.0) * 50.0)
                            extent[4] = float(np.floor((sp_zmin - dz) / 50.0) * 50.0)
                            extent[5] = float(np.ceil((sp_zmax + dz) / 50.0) * 50.0)

                            self.txt_xmin.ChangeValue(f"{extent[0]:.1f}")
                            self.txt_xmax.ChangeValue(f"{extent[1]:.1f}")
                            self.txt_ymin.ChangeValue(f"{extent[2]:.1f}")
                            self.txt_ymax.ChangeValue(f"{extent[3]:.1f}")
                            self.txt_zmin.ChangeValue(f"{extent[4]:.1f}")
                            self.txt_zmax.ChangeValue(f"{extent[5]:.1f}")
                            self.on_resolution_or_extent_changed()
                            self.log(f"⚡ 自动自适应 Extent: 扩展模型三维空间范围至 X[{extent[0]}~{extent[1]}], Y[{extent[2]}~{extent[3]}], Z[{extent[4]}~{extent[5]}]，确保所有控制点 100% 包含于三维实体内！")
                except Exception as e:
                    self.log(f"自适应 Extent 检查提示: {e}")
        except ValueError as ex:
            wx.MessageBox(f"参数格式有误，请输入合法数值: {ex}", "错误", wx.OK | wx.ICON_ERROR)
            return

        is_val, errs, warns = GeologicalModelEngine.validate_parameters(
            extent=extent,
            resolution=resolution,
            refinement=refine,
            surface_refinement=surf_refine,
        )
        if not is_val:
            wx.MessageBox("参数校验未通过:\n• " + "\n• ".join(errs), "参数错误", wx.OK | wx.ICON_ERROR)
            return

        total_voxels = resolution[0] * resolution[1] * resolution[2]
        if total_voxels >= 350000 or refine >= 6:
            msg = (
                f"【高精细度计算提示】\n"
                f"当前正交网格体素总数为 {total_voxels:,}，细化等级为 {refine} (曲面={surf_refine})。\n"
                f"预计计算耗时 15~45 秒。是否确认继续执行？"
            )
            ret = wx.MessageBox(msg, "高精细度确认", wx.YES_NO | wx.ICON_QUESTION)
            if ret != wx.YES:
                return

        self.status_bar.SetStatusText("正在计算模型...", 0)
        self.btn_calc.SetLabel("⏳ 正在构建地质模型，请稍候...")
        self.btn_calc.Disable()
        self.log(f"\n>>> 开始初始化并求解地质场 (网格: {resolution}, 细化: {refine}, 曲面: {surf_refine})...")

        topo_sel = self.choice_topo_mode.GetSelection()
        if topo_sel == 1:
            topo_mode = "file"
            topo_file = self.txt_topo_path.GetValue().strip()
            topo_params = None
        elif topo_sel == 2:
            topo_mode = "random"
            topo_file = None
            try:
                topo_params = {
                    "d_z": [float(self.txt_topo_zmin.GetValue()), float(self.txt_topo_zmax.GetValue())],
                    "fractal_dimension": 1.2,
                    "topography_resolution": [40, 40],
                }
            except Exception:
                topo_params = {"d_z": [extent[4] + (extent[5] - extent[4]) * 0.7, extent[5]]}
        else:
            topo_mode = "none"
            topo_file = None
            topo_params = None

        preset_idx = self.choice_preset.GetSelection()
        p_name = self.preset_options[preset_idx][0] if 0 <= preset_idx < len(self.preset_options) else "自定义"

        try:
            sec_name = self.txt_sec_name.GetValue().strip() or "Section_Custom"
            sec_start = [float(v.strip()) for v in self.txt_sec_start.GetValue().split(",")]
            sec_stop = [float(v.strip()) for v in self.txt_sec_stop.GetValue().split(",")]
            sec_res = [int(v.strip()) for v in self.txt_sec_res.GetValue().split(",")]
            has_sec = (len(sec_start) == 2 and len(sec_stop) == 2 and len(sec_res) == 2)
        except Exception:
            has_sec = False
            sec_name, sec_start, sec_stop, sec_res = None, None, None, None

        def _worker():
            try:
                # 构建基于步骤 4 人工编排与合并的年代层序映射
                series_mapping = None
                fault_series = None
                if self.strat_order:
                    strata = [s for s in self.strat_order if not self.is_fault_map.get(s, False) and not str(s).lower().startswith("fault")]
                    faults = [s for s in self.strat_order if self.is_fault_map.get(s, False) or str(s).lower().startswith("fault")]
                    series_mapping = {}
                    if faults:
                        series_mapping["Fault_Series"] = tuple(faults) if len(faults) > 1 else faults[0]
                    if strata:
                        series_mapping["Strat_Series"] = tuple(strata) if len(strata) > 1 else strata[0]
                    fault_series = ["Fault_Series"] if faults else []

                self.engine.initialize_model(
                    surface_points_path=sp_path,
                    orientations_path=or_path,
                    series_mapping=series_mapping,
                    fault_series=fault_series,
                    extent=extent,
                    resolution=resolution,
                    refinement=refine,
                    surface_refinement=surf_refine,
                    topography_mode=topo_mode,
                    topography_filepath=topo_file,
                    topography_params=topo_params,
                )

                if hasattr(self, "customizer") and has_sec:
                    self.customizer.sections_config[sec_name] = {
                        "start": sec_start,
                        "stop": sec_stop,
                        "resolution": sec_res,
                    }
                    section_dict = {
                        k: (v["start"], v["stop"], v["resolution"])
                        for k, v in self.customizer.sections_config.items()
                    }
                    gp.set_section_grid(grid=self.engine.geo_model.grid, section_dict=section_dict)

                res = self.engine.compute()
                self.log(f"✓ 模型计算成功! 耗时: {res['compute_time_seconds']} 秒")
                self.log(f"  网格单元: {res['total_voxels']:,} (尺寸: {res['voxel_size']}m)")
                self.log(f"  曲面细化: 体网格={res['refinement']}, 曲面={res['surface_refinement']}")
                self.log(f"  曲面三角网: {res['surfaces_mesh_count']} 个地质界面 (顶点: {res['total_mesh_vertices']:,})")

                try:
                    self.engine.export_config(
                        os.path.join(self.cur_dir, "project_config.json"),
                        extra_meta={"preset_name": p_name},
                    )
                except Exception:
                    pass

                def _finish():
                    self._on_compute_finished()
                    self.status_bar.SetStatusText(
                        f"✓ 计算完成 | 网格 {resolution[0]}×{resolution[1]}×{resolution[2]} | 细化 {refine} | 耗时 {res['compute_time_seconds']}s",
                        0,
                    )
                wx.CallAfter(_finish)
            except Exception as e:
                self.log(f"✖ 计算失败: {e}")
                def _reset_error():
                    self.btn_calc.SetLabel("▶ 开始构建并计算模型")
                    self.btn_calc.Enable()
                    self.status_bar.SetStatusText("计算失败", 0)
                    wx.MessageBox(f"计算发生错误: {e}", "计算错误", wx.OK | wx.ICON_ERROR)
                wx.CallAfter(_reset_error)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_compute_finished(self):
        self.btn_calc.SetLabel("▶ 开始构建并计算模型")
        self.btn_calc.Enable()
        self.status_bar.SetStatusText("🎉 模型计算完成，剖面已自动渲染", 0)
        self.goto_step(7)
        self.on_cut_section(None)

    def on_cut_section(self, event=None):
        """切割并展示 2D 剖面"""
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return

        try:
            name = self.txt_sec_name.GetValue().strip() or "Section_Custom"
            start_vals = [float(v.strip()) for v in self.txt_sec_start.GetValue().split(",")]
            stop_vals = [float(v.strip()) for v in self.txt_sec_stop.GetValue().split(",")]
            res_vals = [int(v.strip()) for v in self.txt_sec_res.GetValue().split(",")]
            ve = float(self.txt_sec_ve.GetValue())
            show_bh = self.chk_sec_bh.IsChecked()
            show_topo = self.chk_sec_topo.IsChecked()
        except Exception as ex:
            wx.MessageBox(f"剖面参数解析错误: {ex}", "错误", wx.OK | wx.ICON_ERROR)
            return

        try:
            self.customizer.add_cross_section(
                name=name,
                start_xy=(start_vals[0], start_vals[1]),
                stop_xy=(stop_vals[0], stop_vals[1]),
                resolution=(res_vals[0], res_vals[1]),
            )
            fig, ax = self.customizer.render_cross_section(
                section_name=name,
                show_boreholes=show_bh,
                show_topography=show_topo,
                vertical_exaggeration=ve,
            )
            self._update_canvas(fig)
            self.log(f"✓ 剖面 '{name}' 切割并渲染完成。")
            self.status_bar.SetStatusText(f"已展示剖面: {name}", 0)
        except Exception as e:
            self.log(f"✖ 剖面生成失败: {e}")
            wx.MessageBox(f"剖面切割失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

    def on_quick_orthogonal(self, direction: str):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        try:
            fig, ax = self.customizer.render_orthogonal_section(direction=direction, cell_number="mid")
            self._update_canvas(fig)
            self.log(f"✓ 已切换至正交切片视图 (沿 {direction.upper()} 轴)")
        except Exception as e:
            self.log(f"✖ 正交切片生成失败: {e}")

    def on_export_section_image(self, event):
        dlg = wx.FileDialog(
            self,
            "保存剖面图片",
            self.cur_dir,
            f"{self.txt_sec_name.GetValue().strip()}.png",
            "PNG 图片 (*.png)|*.png|JPEG 图片 (*.jpg)|*.jpg|PDF 矢量图 (*.pdf)|*.pdf",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                self.customizer.export_section_image(self.fig, path)
                self.log(f"✓ 剖面图已导出至: {path}")
                wx.MessageBox(f"剖面图导出成功!\n路径: {path}", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"导出图片失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def on_add_borehole(self, event):
        try:
            name = self.txt_bh_name.GetValue().strip()
            x = float(self.txt_bh_x.GetValue())
            y = float(self.txt_bh_y.GetValue())
            ztop = float(self.txt_bh_ztop.GetValue())
            zbot = float(self.txt_bh_zbot.GetValue())
            if not name:
                raise ValueError("孔号不能为空")
            if ztop <= zbot:
                raise ValueError("孔口标高必须大于孔底标高")
        except Exception as ex:
            wx.MessageBox(f"钻孔参数有误: {ex}", "错误", wx.OK | wx.ICON_ERROR)
            return

        found_idx = -1
        for i in range(self.list_bh.GetItemCount()):
            if self.list_bh.GetItemText(i) == name:
                found_idx = i
                break

        depth = ztop - zbot
        self.customizer.add_virtual_borehole(name, x, y, ztop, zbot)

        if found_idx >= 0:
            self.list_bh.SetItem(found_idx, 1, f"{x:.1f}")
            self.list_bh.SetItem(found_idx, 2, f"{y:.1f}")
            self.list_bh.SetItem(found_idx, 3, f"{ztop:.1f}")
            self.list_bh.SetItem(found_idx, 4, f"{zbot:.1f}")
            self.list_bh.SetItem(found_idx, 5, f"{depth:.1f}")
            self.log(f"✓ 已更新钻孔 {name} (X={x}, Y={y}, 孔深={depth}m)")
        else:
            self._add_borehole_to_list(name, x, y, ztop, zbot)
            self.log(f"✓ 已新增钻孔 {name} (X={x}, Y={y}, 孔深={depth}m)")

    def on_delete_borehole(self, event):
        sel = self.list_bh.GetFirstSelected()
        if sel < 0:
            wx.MessageBox("请先在列表中选中要删除的钻孔", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        name = self.list_bh.GetItemText(sel)
        self.list_bh.DeleteItem(sel)
        self.customizer.boreholes = [b for b in self.customizer.boreholes if b["name"] != name]
        self.log(f"✓ 已移除钻孔: {name}")

    def on_plot_borehole_stratigraphy(self, event):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        sel = self.list_bh.GetFirstSelected()
        if sel < 0:
            wx.MessageBox("请先在列表中选中一个钻孔", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        name = self.list_bh.GetItemText(sel)
        try:
            fig, ax = self.customizer.plot_borehole_stratigraphy(name)
            self._update_canvas(fig)
            self.log(f"✓ 钻孔 {name} 柱状图绘制完毕。")
            self.status_bar.SetStatusText(f"已展示钻孔柱状图: {name}", 0)
        except Exception as e:
            self.log(f"✖ 柱状图生成失败: {e}")
            wx.MessageBox(f"柱状图生成失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

    def on_export_borehole_csv(self, event):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        dlg = wx.DirDialog(self, "选择导出目录", self.cur_dir, wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            out_dir = dlg.GetPath()
            try:
                saved = self.customizer.export_borehole_csv(out_dir)
                msg = "\n".join(saved)
                self.log(f"✓ 钻孔分层数据已成功导出至:\n{msg}")
                wx.MessageBox(f"钻孔分层数据已成功导出至:\n{msg}", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"导出 CSV 失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def on_open_3d_viewer(self):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        self.log("正在构建交互式三维地质体与实体钻孔 (PyVista)...")
        try:
            show_topo = self.chk_3d_topo.IsChecked()
            plotter = self.customizer.get_pyvista_plotter(
                show_lith=True,
                show_surfaces=True,
                show_boreholes=True,
                show_topography=show_topo,
            )
            if sys.platform == "darwin":
                plotter.show()
            else:
                threading.Thread(target=plotter.show, daemon=True).start()
        except Exception as e:
            self.log(f"✖ 唤起三维视窗失败: {e}")
            wx.MessageBox(f"唤起三维视窗失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

    def on_export_vtk(self):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，请先点击【开始构建并计算模型】", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        dlg = wx.DirDialog(self, "选择 VTK 网格导出目录", self.cur_dir, wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            out_dir = dlg.GetPath()
            try:
                saved = self.customizer.export_surfaces_to_vtk(out_dir)
                msg = "\n".join(saved)
                self.log(f"✓ 真实坐标 VTK 曲面网格已导出至:\n{msg}")
                wx.MessageBox(f"已导出 {len(saved)} 个地层界面的 VTK 网格:\n{msg}", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"导出 VTK 失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def on_save_gempy_model(self):
        if not self.engine.is_computed:
            wx.MessageBox("模型尚未计算，无法保存", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        dlg = wx.FileDialog(
            self,
            "保存 GemPy 模型",
            self.cur_dir,
            "geological_model.gempy",
            "GemPy 模型 (*.gempy)|*.gempy",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                self.engine.save_model(path)
                self.log(f"✓ 模型已保存至: {path}")
                wx.MessageBox(f"模型已保存至: {path}", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"保存失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    # -------------------------------------------------------------------------
    # 精细度控制与工程配置持久化
    # -------------------------------------------------------------------------
    def on_resolution_input_changed(self, event=None):
        self._check_and_mark_custom_preset()
        self.on_resolution_or_extent_changed()

    def on_refine_changed(self, event=None):
        val = self.spin_refine.GetValue()
        if self.chk_sync_refine.IsChecked():
            self.spin_surf_refine.SetValue(val)
        self._check_and_mark_custom_preset()
        self.on_resolution_or_extent_changed()

    def on_surf_refine_changed(self, event=None):
        if self.spin_surf_refine.GetValue() != self.spin_refine.GetValue():
            self.chk_sync_refine.SetValue(False)
        self._check_and_mark_custom_preset()
        self.on_resolution_or_extent_changed()

    def on_toggle_sync_refine(self, event=None):
        if self.chk_sync_refine.IsChecked():
            self.spin_surf_refine.SetValue(self.spin_refine.GetValue())
            self.on_resolution_or_extent_changed()

    def on_preset_selected(self, event=None):
        idx = self.choice_preset.GetSelection()
        if 0 <= idx < len(self.preset_options):
            opt = self.preset_options[idx]
            res, ref, surf = opt[1], opt[2], opt[3]
            if res is not None:
                self.txt_nx.ChangeValue(str(res[0]))
                self.txt_ny.ChangeValue(str(res[1]))
                self.txt_nz.ChangeValue(str(res[2]))
                self.spin_refine.SetValue(ref)
                self.spin_surf_refine.SetValue(surf)
                self.chk_sync_refine.SetValue(ref == surf)
                self.on_resolution_or_extent_changed()

    def _check_and_mark_custom_preset(self):
        try:
            cur_res = [int(self.txt_nx.GetValue()), int(self.txt_ny.GetValue()), int(self.txt_nz.GetValue())]
            cur_ref = int(self.spin_refine.GetValue())
            cur_surf = int(self.spin_surf_refine.GetValue())
            matched = False
            for i, opt in enumerate(self.preset_options[:-1]):
                if opt[1] == cur_res and opt[2] == cur_ref and opt[3] == cur_surf:
                    self.choice_preset.SetSelection(i)
                    matched = True
                    break
            if not matched:
                self.choice_preset.SetSelection(len(self.preset_options) - 1)
        except Exception:
            pass

    def _match_preset_selection(self, res, refine, surf_refine):
        matched = False
        if res is not None and refine is not None:
            for i, opt in enumerate(self.preset_options[:-1]):
                if opt[1] == list(res) and opt[2] == int(refine) and (surf_refine is None or opt[3] == int(surf_refine)):
                    self.choice_preset.SetSelection(i)
                    matched = True
                    break
        if not matched:
            self.choice_preset.SetSelection(len(self.preset_options) - 1)

    def on_resolution_or_extent_changed(self, event=None):
        try:
            xmin = float(self.txt_xmin.GetValue())
            xmax = float(self.txt_xmax.GetValue())
            ymin = float(self.txt_ymin.GetValue())
            ymax = float(self.txt_ymax.GetValue())
            zmin = float(self.txt_zmin.GetValue())
            zmax = float(self.txt_zmax.GetValue())
            nx = int(self.txt_nx.GetValue())
            ny = int(self.txt_ny.GetValue())
            nz = int(self.txt_nz.GetValue())

            if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                self.lbl_voxel_info.SetForegroundColour(wx.Colour(190, 0, 0))
                self.lbl_voxel_info.SetLabel("⚠️ 范围参数非法: Xmax/Ymax/Zmax 必须分别大于 Xmin/Ymin/Zmin")
                self.pnl_voxel_metric.Layout()
                return

            if nx < 5 or ny < 5 or nz < 5:
                self.lbl_voxel_info.SetForegroundColour(wx.Colour(190, 0, 0))
                self.lbl_voxel_info.SetLabel("⚠️ 网格分辨率过低: Nx/Ny/Nz 各轴至少为 5")
                self.pnl_voxel_metric.Layout()
                return

            dx = (xmax - xmin) / float(nx)
            dy = (ymax - ymin) / float(ny)
            dz = (zmax - zmin) / float(nz)
            total = nx * ny * nz

            if total > 2000000:
                eval_tag = "⛔ 超出安全上限 (总单元 > 200万，严禁计算以防崩)"
                txt_color = wx.Colour(190, 0, 0)
            elif total > 350000:
                eval_tag = "🔥 密集超高 (预计 > 30 秒)"
                txt_color = wx.Colour(190, 0, 0)
            elif total > 120000:
                eval_tag = "⚠️ 较高负荷 (预计 10~25 秒)"
                txt_color = wx.Colour(180, 100, 0)
            elif total >= 35000:
                eval_tag = "✓ 标准平衡 (推荐，预计 4~10 秒)"
                txt_color = wx.Colour(20, 80, 160)
            else:
                eval_tag = "⚡ 极速轻量 (预计 < 4 秒)"
                txt_color = wx.Colour(0, 130, 0)

            ref_val = self.spin_refine.GetValue()
            surf_val = self.spin_surf_refine.GetValue()

            text = (
                f"📐 单体素尺寸: ΔX = {dx:.1f} m | ΔY = {dy:.1f} m | ΔZ = {dz:.1f} m\n"
                f"📊 网格总数: {total:,} 单元 | 细化等级: 体网格 = {ref_val}, 曲面 = {surf_val}\n"
                f"⏱ 综合评估: {eval_tag}"
            )
            self.lbl_voxel_info.SetForegroundColour(txt_color)
            self.lbl_voxel_info.SetLabel(text)
            self.pnl_voxel_metric.Layout()
        except Exception:
            self.lbl_voxel_info.SetForegroundColour(wx.Colour(190, 0, 0))
            self.lbl_voxel_info.SetLabel("⚠️ 参数输入未完成或格式不合法 (请输入数字)")
            self.pnl_voxel_metric.Layout()

    def on_reset_default_precision(self):
        self.choice_preset.SetSelection(1)
        self.on_preset_selected()
        self.log("已恢复至推荐标准平衡参数 (Nx=40, Ny=40, Nz=30 | Refinement=4)")

    def on_save_config_profile(self):
        dlg = wx.FileDialog(
            self,
            "保存建模参数配置方案",
            self.cur_dir,
            "geology_config.json",
            "JSON 配置文件 (*.json)|*.json",
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                extent = [
                    float(self.txt_xmin.GetValue()),
                    float(self.txt_xmax.GetValue()),
                    float(self.txt_ymin.GetValue()),
                    float(self.txt_ymax.GetValue()),
                    float(self.txt_zmin.GetValue()),
                    float(self.txt_zmax.GetValue()),
                ]
                resolution = [
                    int(self.txt_nx.GetValue()),
                    int(self.txt_ny.GetValue()),
                    int(self.txt_nz.GetValue()),
                ]
                refine = int(self.spin_refine.GetValue())
                surf_refine = int(self.spin_surf_refine.GetValue())

                self.engine.extent = extent
                self.engine.resolution = resolution
                self.engine.refinement = refine
                self.engine.surface_refinement = surf_refine
                self.engine.surface_points_path = self.txt_sp.GetValue().strip()
                self.engine.orientations_path = self.txt_or.GetValue().strip()

                preset_idx = self.choice_preset.GetSelection()
                preset_name = (
                    self.preset_options[preset_idx][0]
                    if 0 <= preset_idx < len(self.preset_options)
                    else "自定义"
                )

                saved = self.engine.export_config(path, extra_meta={"preset_name": preset_name})
                self.log(f"✓ 建模配置方案已成功保存至: {saved}")
                wx.MessageBox(f"建模配置方案已成功保存至:\n{saved}", "保存成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"保存配置失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def on_load_config_profile(self, filepath: Optional[str] = None):
        if filepath is None:
            dlg = wx.FileDialog(
                self,
                "载入建模参数配置方案",
                self.cur_dir,
                "",
                "JSON 配置文件 (*.json)|*.json|所有文件 (*.*)|*.*",
                wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            )
            if dlg.ShowModal() != wx.ID_OK:
                dlg.Destroy()
                return
            path = dlg.GetPath()
            dlg.Destroy()
        else:
            path = filepath

        try:
            cfg = self.engine.load_config(path)
            if "extent" in cfg and len(cfg["extent"]) == 6:
                ext = cfg["extent"]
                self.txt_xmin.ChangeValue(str(ext[0]))
                self.txt_xmax.ChangeValue(str(ext[1]))
                self.txt_ymin.ChangeValue(str(ext[2]))
                self.txt_ymax.ChangeValue(str(ext[3]))
                self.txt_zmin.ChangeValue(str(ext[4]))
                self.txt_zmax.ChangeValue(str(ext[5]))

            if "resolution" in cfg and len(cfg["resolution"]) == 3:
                res = cfg["resolution"]
                self.txt_nx.ChangeValue(str(res[0]))
                self.txt_ny.ChangeValue(str(res[1]))
                self.txt_nz.ChangeValue(str(res[2]))

            if "refinement" in cfg:
                self.spin_refine.SetValue(int(cfg["refinement"]))
            if "surface_refinement" in cfg:
                self.spin_surf_refine.SetValue(int(cfg["surface_refinement"]))
            elif "refinement" in cfg:
                self.spin_surf_refine.SetValue(int(cfg["refinement"]))

            if "refinement" in cfg and "surface_refinement" in cfg:
                self.chk_sync_refine.SetValue(int(cfg["refinement"]) == int(cfg["surface_refinement"]))

            if "surface_points_path" in cfg and cfg["surface_points_path"]:
                self.txt_sp.SetValue(cfg["surface_points_path"])
            if "orientations_path" in cfg and cfg["orientations_path"]:
                self.txt_or.SetValue(cfg["orientations_path"])

            if "topography" in cfg and isinstance(cfg["topography"], dict):
                topo = cfg["topography"]
                t_mode = topo.get("mode", "none")
                if t_mode == "file":
                    self.choice_topo_mode.SetSelection(1)
                    if topo.get("filepath"):
                        self.txt_topo_path.SetValue(str(topo["filepath"]))
                elif t_mode == "random":
                    self.choice_topo_mode.SetSelection(2)
                    t_params = topo.get("params") or {}
                    if "d_z" in t_params and len(t_params["d_z"]) == 2:
                        self.txt_topo_zmin.SetValue(str(t_params["d_z"][0]))
                        self.txt_topo_zmax.SetValue(str(t_params["d_z"][1]))
                else:
                    self.choice_topo_mode.SetSelection(0)
                self.on_topo_mode_changed()

            self._match_preset_selection(
                cfg.get("resolution"),
                cfg.get("refinement"),
                cfg.get("surface_refinement"),
            )
            self.on_resolution_or_extent_changed()

            self.log(f"✓ 成功载入参数配置方案: {os.path.basename(path)}")
            if filepath is None:
                wx.MessageBox(f"已成功加载配置方案:\n{path}", "载入成功", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            self.log(f"✖ 载入配置失败: {e}")
            if filepath is None:
                wx.MessageBox(f"载入配置方案失败: {e}", "错误", wx.OK | wx.ICON_ERROR)

    def on_load_gempy_model(self):
        dlg = wx.FileDialog(
            self,
            "选择 GemPy 模型文件",
            self.cur_dir,
            "",
            "GemPy 模型 (*.gempy)|*.gempy|所有文件 (*.*)|*.*",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                self.engine.load_model(path)
                if self.engine.extent and len(self.engine.extent) == 6:
                    ext = self.engine.extent
                    self.txt_xmin.ChangeValue(str(ext[0]))
                    self.txt_xmax.ChangeValue(str(ext[1]))
                    self.txt_ymin.ChangeValue(str(ext[2]))
                    self.txt_ymax.ChangeValue(str(ext[3]))
                    self.txt_zmin.ChangeValue(str(ext[4]))
                    self.txt_zmax.ChangeValue(str(ext[5]))

                if self.engine.resolution and len(self.engine.resolution) == 3:
                    res = self.engine.resolution
                    self.txt_nx.ChangeValue(str(res[0]))
                    self.txt_ny.ChangeValue(str(res[1]))
                    self.txt_nz.ChangeValue(str(res[2]))

                if self.engine.refinement:
                    self.spin_refine.SetValue(int(self.engine.refinement))
                if self.engine.surface_refinement:
                    self.spin_surf_refine.SetValue(int(self.engine.surface_refinement))
                if self.engine.refinement and self.engine.surface_refinement:
                    self.chk_sync_refine.SetValue(int(self.engine.refinement) == int(self.engine.surface_refinement))

                self._match_preset_selection(
                    self.engine.resolution,
                    self.engine.refinement,
                    self.engine.surface_refinement,
                )
                self.on_resolution_or_extent_changed()

                self.log(f"✓ 成功载入 GemPy 模型: {path}")
                self.status_bar.SetStatusText(f"已载入模型: {os.path.basename(path)}", 0)
                wx.MessageBox(f"已成功加载 GemPy 模型:\n{path}", "成功", wx.OK | wx.ICON_INFORMATION)
                self.goto_step(7)
                self.on_cut_section(None)
            except Exception as e:
                wx.MessageBox(f"加载模型失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def on_show_about(self, event):
        info = (
            "【基于平面地质图的智能化三维构建】\n"
            "智能化三维地质建模系统 (MVP 原型)\n\n"
            "【分步引导式交互工作流】\n"
            "1. 原始平面地质图输入与空间测绘尺度标定\n"
            "2. 双边滤波与距离变换暗色界线/粗断线分离\n"
            "3. DFS 测地拓扑排序提取三维界面控制点与连通段\n"
            "4. 人工交互合并断层错断地层与年代层序编排\n"
            "5. SVD 曲面产状解算、倾向跳变质检与 180° 极性平滑修复\n"
            "6. GemPy 隐式位势场求解、剖面切割、虚拟钻孔与 PyVista 3D 导出\n"
        )
        wx.MessageBox(info, "关于本系统", wx.OK | wx.ICON_INFORMATION)

    def _update_canvas(self, new_fig: plt.Figure):
        self.fig = new_fig
        self.csizer.Clear(delete_windows=True)
        self.canvas = FigureCanvas(self.canvas_container, -1, new_fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        self.csizer.Add(self.toolbar, 0, wx.EXPAND)
        self.csizer.Add(self.canvas, 1, wx.EXPAND)
        self.canvas_container.Layout()
        self.right_panel.Layout()
        self.canvas.draw()
        self.canvas.Refresh()


def main():
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
