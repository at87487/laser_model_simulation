import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numba import njit, prange

# --- 1. 預設參數 (更正為物理合理的初值，避免初次載入時能量密度過高) ---
DEFAULTS = {
    'v_stage': -1.0,       # mm/s
    'v_scan': 10.0,        # mm/s
    'f_base_khz': 1000,    # kHz
    'divider': 10,         # 實際發射頻率 100 kHz
    'num_cycles': 20,
    'passes': 2,           # 加工次數
    'a_um': 4.0,           # μm
    'b_um': 8.0,           # μm
    'phase_shift_deg': 180.0, # Pass 間相位錯位角度 (度)

    'wavelength_nm': 257.5,
    'M2': 1.2,             
    'input_D_mm': 2.0,     # 入射光徑改為 2.0 mm
    'focal_length_mm': 10.0, # 透鏡焦距改為 10.0 mm (初始 Beam 2w0 ~ 2.0 μm)
    'defocus_um': 0.0,
    'pulse_width_fs': 800,

    'P_avg_W': 0.05,       # 初始功率調為 0.05W，避免初次載入時過度燒蝕
    'F_th_1': 1.8,         # SiO2 典型 257.5nm 燒蝕閾值
    'S_inc': 0.80,         # 孵化係數
    'delta_um': 0.025,     # 穿透深度 (25 nm)
    'D_sat': 12.0,         # 幾何飽和深度 (μm)

    'grid_res': 180,
    'elev': 30,
    'azim': -60,
    'slice_x_um': 0.0,
    'slice_y_um': 0.0,
    'show_spots': True,

    'exp_target_depth_um': 18.0, # 實驗目標深度 (μm)
    'exp_target_spot_um': 1.4    # 實驗目標光斑 (μm)
}

SIM_CACHE = {
    'X': None, 'Y': None,
    'Total_Depth_um': None,
    'x_grid_um': None, 'y_grid_um': None,
    'all_pass_spots': [],
    'w0_um': 0, 'd0_um': 0, 'd_eff_um': 0, 'pitch_stage_um': 0, 'overlap_rate': 0,
    'pass_depth_history': [],
    'f_laser': 0, 'F0_z': 0, 'E_p': 0
}

# --- 2. 物理燒蝕核心 (Numba 加速) ---
@njit(parallel=True, fastmath=True)
def compute_single_pass_ablation_experiment_matched(
    x_grid_um, y_grid_um, x_spots_um, y_spots_um, 
    current_depth, F0, effective_F_th, delta_um, 
    w_spot_um, zR_um, D_sat=12.0, f_laser_khz=100.0
):
    nx = len(x_grid_um)
    ny = len(y_grid_um)
    n_spots = len(x_spots_um)

    updated_depth = current_depth.copy()
    w_sq = w_spot_um * w_spot_um
    
    # 熱擴散區域抹平
    effective_w_sq = w_sq * 2.2  
    cutoff_r_sq = 3.0 * 3.0 * effective_w_sq

    # 頻率熱累積效應
    heat_accum_factor = 1.0 + 0.12 * (f_laser_khz ** 0.65)

    for s in prange(n_spots):
        xs = x_spots_um[s]
        ys = y_spots_um[s]

        for i in range(ny):
            dy = y_grid_um[i] - ys
            dy_sq = dy * dy
            if dy_sq > cutoff_r_sq:
                continue

            for j in range(nx):
                dx = x_grid_um[j] - xs
                r_sq = dx * dx + dy_sq

                if r_sq < cutoff_r_sq:
                    current_d = updated_depth[i, j]

                    # 深溝槽波導自聚焦增益
                    trapping_gain = 1.0 + 0.25 * current_d 

                    w_deeper_sq = effective_w_sq * (1.0 + (current_d / zR_um)**2)
                    F0_deeper = F0 * (effective_w_sq / w_deeper_sq) * trapping_gain

                    att_factor = np.exp(-current_d / D_sat)
                    fluence = F0_deeper * np.exp(-2.0 * r_sq / w_deeper_sq) * att_factor

                    if fluence > effective_F_th:
                        d_k = (delta_um * heat_accum_factor) * np.log(fluence / effective_F_th)
                        updated_depth[i, j] += d_k

    return updated_depth


class LaserAblationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SiO2 飛秒雷射多 Pass 燒蝕與光學熱效應模擬器 (排版修正版)")
        self.geometry("1450x920")

        self.params = {k: tk.DoubleVar(value=v) if isinstance(v, float) else tk.IntVar(value=v) 
                       for k, v in DEFAULTS.items()}
        self.chk_show_spots = tk.BooleanVar(value=DEFAULTS['show_spots'])

        self.setup_ui()
        self.run_simulation()
        self.render_plots()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左側控制面板 (可捲動)
        left_frame = ttk.Frame(main_frame, width=420)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        canvas_scroll = tk.Canvas(left_frame, width=400)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)

        scrollable_frame.bind(
            "<" + "Configure" + ">",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 控制項群組 ---
        lf_motion = ttk.LabelFrame(scrollable_frame, text="🌀 運動控制與掃描軌跡", padding="5")
        lf_motion.pack(fill=tk.X, pady=5)
        self.add_slider(lf_motion, 'v_stage', 'v_stage (mm/s)', -10.0, 10.0, 0.5)
        self.add_slider(lf_motion, 'v_scan', 'v_scan (mm/s)', -100.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'f_base_khz', 'Base Rep (kHz)', 400, 1000, 100, is_int=True)
        self.add_slider(lf_motion, 'divider', 'Divider', 1, 1000, 1, is_int=True)
        self.add_slider(lf_motion, 'num_cycles', 'Cycles/Pass', 1, 40, 1, is_int=True)
        self.add_slider(lf_motion, 'passes', 'Passes (次數)', 1, 50, 1, is_int=True)
        self.add_slider(lf_motion, 'a_um', 'Major a (μm)', 1.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'b_um', 'Minor b (μm)', 1.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'phase_shift_deg', 'Pass 相位差 (°)', 0.0, 360.0, 15.0)

        lf_optics = ttk.LabelFrame(scrollable_frame, text="🔍 光學與焦區系統", padding="5")
        lf_optics.pack(fill=tk.X, pady=5)
        self.add_slider(lf_optics, 'wavelength_nm', '波長 λ (nm)', 257.5, 1070.0, 0.5)
        self.add_slider(lf_optics, 'M2', '光束質量 M²', 1.0, 3.0, 0.05)
        self.add_slider(lf_optics, 'input_D_mm', '入射光徑 D (mm)', 1.0, 20.0, 0.1)
        self.add_slider(lf_optics, 'focal_length_mm', '透鏡焦距 f (mm)', 1.0, 200.0, 1.0)
        self.add_slider(lf_optics, 'defocus_um', '離焦量 Δz (μm)', -100.0, 100.0, 1.0)
        self.add_slider(lf_optics, 'pulse_width_fs', '脈寬 τ (fs)', 50, 5000, 50, is_int=True)

        lf_physics = ttk.LabelFrame(scrollable_frame, text="⚡ 雷射功率與 SiO2 物理參數", padding="5")
        lf_physics.pack(fill=tk.X, pady=5)
        self.add_slider(lf_physics, 'P_avg_W', 'Power (W)', 0.01, 10.0, 0.01)
        self.add_slider(lf_physics, 'F_th_1', 'F_th_1 (J/cm²)', 0.5, 5.0, 0.1)
        self.add_slider(lf_physics, 'S_inc', '孵化係數 S', 0.70, 0.95, 0.01)
        self.add_slider(lf_physics, 'delta_um', 'delta (μm)', 0.005, 0.150, 0.001)
        self.add_slider(lf_physics, 'D_sat', 'D_sat (μm)', 1.0, 50.0, 0.5)

        lf_scf = ttk.LabelFrame(scrollable_frame, text="🎯 物理約束 SCF 實驗雙參數擬合", padding="5")
        lf_scf.pack(fill=tk.X, pady=5)
        self.add_slider(lf_scf, 'exp_target_depth_um', '實驗底部深度 (μm)', 1.0, 100.0, 0.5)
        self.add_slider(lf_scf, 'exp_target_spot_um', '實際加工光斑 (μm)', 0.2, 10.0, 0.1)
        
        btn_scf = ttk.Button(lf_scf, text="🔄 執行物理約束 SCF 擬合", command=self.run_scf_fitting)
        btn_scf.pack(fill=tk.X, pady=5)

        lf_view = ttk.LabelFrame(scrollable_frame, text="🔪 視角與剖面切面控制", padding="5")
        lf_view.pack(fill=tk.X, pady=5)
        self.add_slider(lf_view, 'grid_res', '網格解析度', 100, 300, 25, is_int=True, auto_render=True)
        self.add_slider(lf_view, 'elev', '3D 俯角', 0, 90, 5, is_int=True, auto_render=True)
        self.add_slider(lf_view, 'azim', '3D 方位', -180, 180, 5, is_int=True, auto_render=True)
        
        self.slice_x_slider = self.add_slider(lf_view, 'slice_x_um', 'X 切面位置 (μm)', -20.0, 20.0, 0.1, auto_render=True)
        self.slice_y_slider = self.add_slider(lf_view, 'slice_y_um', 'Y 切面位置 (μm)', -20.0, 20.0, 0.1, auto_render=True)
        
        chk_spots = ttk.Checkbutton(lf_view, text="顯示軌跡與脈衝點", variable=self.chk_show_spots, command=self.render_plots)
        chk_spots.pack(anchor=tk.W, pady=2)

        btn_run = ttk.Button(scrollable_frame, text="🚀 開始模擬", command=self.on_btn_run)
        btn_run.pack(fill=tk.X, pady=10)

        self.lbl_status = ttk.Label(scrollable_frame, text="狀態：準備就緒", wraplength=380, foreground="blue")
        self.lbl_status.pack(fill=tk.X, pady=5)

        # 右側 Matplotlib 繪圖區
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(13, 8.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()

    def add_slider(self, parent, param_key, label_text, from_, to, resolution, is_int=False, auto_render=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        lbl = ttk.Label(frame, text=label_text, width=18, anchor=tk.W)
        lbl.pack(side=tk.LEFT)

        var = self.params[param_key]
        
        def on_change(val):
            val_num = int(float(val)) if is_int else round(float(val), 4)
            var.set(val_num)
            val_lbl.config(text=str(val_num))
            if auto_render:
                self.render_plots()

        scale = ttk.Scale(frame, from_=from_, to=to, value=var.get(), command=on_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        val_lbl = ttk.Label(frame, text=str(var.get()), width=8, anchor=tk.E)
        val_lbl.pack(side=tk.RIGHT)

        return scale

    def get_val(self, key):
        return self.params[key].get()

    def set_val(self, key, val):
        self.params[key].set(val)

    def on_btn_run(self):
        self.lbl_status.config(text="狀態：⚡ 正在計算物理修正版 SiO2 多 Pass 燒蝕...")
        self.update_idletasks()
        self.run_simulation()
        self.render_plots()
        d0 = SIM_CACHE['d0_um']
        d_eff = SIM_CACHE['d_eff_um']
        self.lbl_status.config(text=f"狀態：✅ 模擬完成！| Beam 2w0: {d0:.2f}μm | Eff Spot: {d_eff:.2f}μm | M²: {self.get_val('M2'):.2f}")

    def run_simulation(self):
        wavelength_m = self.get_val('wavelength_nm') * 1e-9
        M2 = self.get_val('M2')
        D_m = self.get_val('input_D_mm') * 1e-3
        f_m = self.get_val('focal_length_mm') * 1e-3
        defocus_m = self.get_val('defocus_um') * 1e-6

        w0_m = (2.0 * wavelength_m * f_m * M2) / (np.pi * D_m)
        w0_um = w0_m * 1e6
        d0_um = 2.0 * w0_um

        zR_m = (np.pi * w0_m**2) / (wavelength_m * M2)
        zR_um = zR_m * 1e6
        w_z_m = w0_m * np.sqrt(1.0 + (defocus_m / zR_m)**2)
        w_z_um = w_z_m * 1e6

        f_laser = (self.get_val('f_base_khz') * 1000.0) / float(self.get_val('divider'))
        f_laser_khz = f_laser / 1000.0
        E_p = self.get_val('P_avg_W') / f_laser
        w_z_cm = w_z_m * 100.0
        F0_z = (2.0 * E_p) / (np.pi * (w_z_cm**2))

        F_th_1 = self.get_val('F_th_1')
        if F0_z > F_th_1:
            d_eff_um = 2.0 * w_z_um * np.sqrt(0.5 * np.log(F0_z / F_th_1))
        else:
            d_eff_um = 0.0

        v_stage_um_s = abs(self.get_val('v_stage')) * 1000.0
        pitch_stage_um = v_stage_um_s / f_laser
        overlap_rate = (1.0 - (pitch_stage_um / d0_um)) * 100.0

        a_um, b_um = self.get_val('a_um'), self.get_val('b_um')
        a_mm, b_mm = a_um / 1000.0, b_um / 1000.0
        h = ((a_mm - b_mm)**2) / ((a_mm + b_mm)**2 + 1e-12)
        ellipse_perimeter_mm = np.pi * (a_mm + b_mm) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        v_scan_val = self.get_val('v_scan')
        f_scan = 1e-5 if abs(v_scan_val) < 1e-5 else v_scan_val / ellipse_perimeter_mm

        period = 1.0 / abs(f_scan)
        total_time = self.get_val('num_cycles') * period
        dt = 1.0 / f_laser
        t = np.arange(0, total_time, dt)
        if len(t) > 30000: t = t[:30000]

        total_passes = self.get_val('passes')
        phase_shift_rad = np.radians(self.get_val('phase_shift_deg'))

        all_pass_spots = []
        all_x_spots = []
        all_y_spots = []

        for p in range(total_passes):
            current_phase = p * phase_shift_rad
            x_p = np.ascontiguousarray(a_um * np.cos(2 * np.pi * f_scan * t + current_phase) + (self.get_val('v_stage') * 1000.0) * t)
            y_p = np.ascontiguousarray(b_um * np.sin(2 * np.pi * f_scan * t + current_phase))

            all_pass_spots.append((x_p, y_p))
            all_x_spots.extend(x_p)
            all_y_spots.extend(y_p)

        margin_um = max(w_z_um * 3.5, 3.0)
        x_min, x_max = np.min(all_x_spots) - margin_um, np.max(all_x_spots) + margin_um
        y_min, y_max = np.min(all_y_spots) - margin_um, np.max(all_y_spots) + margin_um

        res = self.get_val('grid_res')
        x_grid_um = np.ascontiguousarray(np.linspace(x_min, x_max, res))
        y_grid_um = np.ascontiguousarray(np.linspace(y_min, y_max, res))

        current_depth = np.zeros((res, res), dtype=np.float64)
        pass_history = []

        S_inc = self.get_val('S_inc')
        delta_um = self.get_val('delta_um')
        D_sat = self.get_val('D_sat')

        for p in range(total_passes):
            effective_F_th = max(F_th_1 * ((p + 1) ** (S_inc - 1.0)), F_th_1 * 0.35)
            x_p, y_p = all_pass_spots[p]

            current_depth = compute_single_pass_ablation_experiment_matched(
                x_grid_um, y_grid_um, x_p, y_p,
                current_depth, F0_z, effective_F_th,
                delta_um, w_z_um, zR_um,
                D_sat=D_sat, f_laser_khz=f_laser_khz
            )
            pass_history.append(float(np.max(current_depth)))

        X, Y = np.meshgrid(x_grid_um, y_grid_um)

        SIM_CACHE['X'] = X
        SIM_CACHE['Y'] = Y
        SIM_CACHE['Total_Depth_um'] = current_depth
        SIM_CACHE['x_grid_um'] = x_grid_um
        SIM_CACHE['y_grid_um'] = y_grid_um
        SIM_CACHE['all_pass_spots'] = all_pass_spots
        SIM_CACHE['w0_um'] = w0_um
        SIM_CACHE['d0_um'] = d0_um
        SIM_CACHE['d_eff_um'] = d_eff_um
        SIM_CACHE['pitch_stage_um'] = pitch_stage_um
        SIM_CACHE['overlap_rate'] = overlap_rate
        SIM_CACHE['pass_depth_history'] = pass_history
        SIM_CACHE['f_laser'] = f_laser
        SIM_CACHE['F0_z'] = F0_z
        SIM_CACHE['E_p'] = E_p

    def run_scf_fitting(self):
        target_depth = self.get_val('exp_target_depth_um')
        target_spot = self.get_val('exp_target_spot_um')

        self.lbl_status.config(text=f"狀態：🔄 啟動物理約束 SCF 擬合 (目標: 深度={target_depth}μm, 光斑痕跡={target_spot}μm)...")
        self.update_idletasks()

        tol = 0.015

        for i in range(30):
            self.run_simulation()

            Total_Depth = SIM_CACHE['Total_Depth_um']
            ny, nx = Total_Depth.shape
            center_region = Total_Depth[ny//4:3*ny//4, nx//4:3*nx//4]
            sim_flat_depth = np.mean(center_region)
            sim_spot = SIM_CACHE['d_eff_um']

            if sim_spot <= 0 or sim_flat_depth <= 0:
                self.set_val('M2', max(self.get_val('M2') * 0.8, 1.0))
                self.set_val('delta_um', min(self.get_val('delta_um') * 1.2, 0.10))
                continue

            err_depth = (sim_flat_depth - target_depth) / target_depth
            err_spot = (sim_spot - target_spot) / target_spot

            if abs(err_depth) < tol and abs(err_spot) < tol:
                self.render_plots()
                msg = (f"狀態：✅ SCF 於第 {i+1} 代成功收斂！\n"
                       f"擬合結果: M² = {self.get_val('M2'):.2f}, delta = {self.get_val('delta_um'):.4f} μm")
                self.lbl_status.config(text=msg)
                return

            spot_ratio = target_spot / sim_spot
            new_m2 = np.clip(self.get_val('M2') * spot_ratio, 1.0, 3.0)
            self.set_val('M2', float(round(new_m2, 2)))

            depth_ratio = target_depth / sim_flat_depth
            new_delta = np.clip(self.get_val('delta_um') * depth_ratio, 0.005, 0.120)
            self.set_val('delta_um', float(round(new_delta, 4)))

            self.update_idletasks()

        self.render_plots()
        msg = f"狀態：⚠️ SCF 完成迭代，當前平坦深度: {sim_flat_depth:.2f} μm, 有效光斑: {SIM_CACHE['d_eff_um']:.2f} μm"
        self.lbl_status.config(text=msg)

    def render_plots(self):
        if SIM_CACHE['Total_Depth_um'] is None:
            return

        self.fig.clear()

        # 使用 GridSpec 進行緊湊且不重疊的版面控制
        gs = gridspec.GridSpec(2, 3, figure=self.fig, wspace=0.35, hspace=0.38)

        X, Y = SIM_CACHE['X'], SIM_CACHE['Y']
        Total_Depth_um = SIM_CACHE['Total_Depth_um']
        x_grid_um = SIM_CACHE['x_grid_um']
        y_grid_um = SIM_CACHE['y_grid_um']
        all_pass_spots = SIM_CACHE['all_pass_spots']

        ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[0, 2])
        ax4 = self.fig.add_subplot(gs[1, 0])
        ax5 = self.fig.add_subplot(gs[1, 1])
        ax6 = self.fig.add_subplot(gs[1, 2])

        # 1. 3D Surface
        ax1.plot_surface(X, Y, -Total_Depth_um, cmap='viridis', edgecolor='none', alpha=0.95)
        ax1.view_init(elev=self.get_val('elev'), azim=self.get_val('azim'))

        ny, nx = Total_Depth_um.shape
        center_region = Total_Depth_um[ny//4:3*ny//4, nx//4:3*nx//4]
        flat_bottom_depth = np.mean(center_region)

        ax1.set_title(f"3D Surface (Depth: {flat_bottom_depth:.2f} μm)", fontsize=9, pad=2)
        ax1.set_xlabel("X (μm)", fontsize=8)
        ax1.set_ylabel("Y (μm)", fontsize=8)
        ax1.set_zlabel("Depth", fontsize=8)

        # 2. 2D Top-View
        c = ax2.contourf(X, Y, Total_Depth_um, levels=50, cmap='inferno')
        if self.chk_show_spots.get():
            colors = ['cyan', 'magenta', 'lime', 'yellow', 'white']
            for p_idx, (xs, ys) in enumerate(all_pass_spots):
                xs_arr = np.atleast_1d(xs).flatten()
                ys_arr = np.atleast_1d(ys).flatten()
                if len(xs_arr) == len(ys_arr) and len(xs_arr) > 0:
                    col = colors[p_idx % len(colors)]
                    ax2.plot(xs_arr, ys_arr, color=col, linestyle='--', linewidth=0.8, alpha=0.7)

        idx_x = int(np.clip((np.abs(x_grid_um - self.get_val('slice_x_um'))).argmin(), 0, len(x_grid_um) - 1))
        idx_y = int(np.clip((np.abs(y_grid_um - self.get_val('slice_y_um'))).argmin(), 0, len(y_grid_um) - 1))

        ax2.axhline(y_grid_um[idx_y], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.axvline(x_grid_um[idx_x], color='cyan', linestyle='--', linewidth=1.2, alpha=0.7)

        f_laser_khz = SIM_CACHE['f_laser'] / 1000.0
        ax2.set_title(f"2D Top-View (Rep: {f_laser_khz:.0f}kHz)\nSpot: {SIM_CACHE['d_eff_um']:.2f}μm", fontsize=9)
        ax2.set_xlabel("X (μm)", fontsize=8)
        ax2.set_ylabel("Y (μm)", fontsize=8)
        cbar = self.fig.colorbar(c, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # 3. X Profile
        x_prof = Total_Depth_um[idx_y, :]
        ax3.plot(x_grid_um, x_prof, 'r-', linewidth=1.5)
        ax3.set_title(f"X-Profile (Y = {y_grid_um[idx_y]:.1f} μm)", fontsize=9)
        ax3.set_xlabel("X (μm)", fontsize=8)
        ax3.set_ylabel("Depth (μm)", fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.invert_yaxis()

        # 4. Y Profile
        y_prof = Total_Depth_um[:, idx_x]
        ax4.plot(y_grid_um, y_prof, 'c-', linewidth=1.5)
        ax4.set_title(f"Y-Profile (X = {x_grid_um[idx_x]:.1f} μm)", fontsize=9)
        ax4.set_xlabel("Y (μm)", fontsize=8)
        ax4.set_ylabel("Depth (μm)", fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.invert_yaxis()

        # 5. Y 軸平均深度
        mean_depth_y = np.mean(Total_Depth_um, axis=1)
        ax5.plot(y_grid_um, mean_depth_y, 'g-', linewidth=1.8, label='Mean Depth')
        ax5.axvline(y_grid_um[idx_y], color='red', linestyle='--', alpha=0.5, label='Current Y Slice')
        ax5.set_title("Average Depth vs. Y", fontsize=9)
        ax5.set_xlabel("Y (μm)", fontsize=8)
        ax5.set_ylabel("Avg Depth (μm)", fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.invert_yaxis()
        ax5.legend(fontsize=7)

        # 6. Pass 深度累積
        pass_depths = SIM_CACHE['pass_depth_history']
        p_range = np.arange(1, len(pass_depths) + 1, dtype=int)
        ax6.plot(p_range, pass_depths, 'bo-', linewidth=2, markersize=6, label='Simulated Depth')
        ax6.set_title("Depth vs. Passes", fontsize=9)
        ax6.set_xlabel("Passes", fontsize=8)
        ax6.set_ylabel("Max Depth (μm)", fontsize=8)
        ax6.grid(True, linestyle=':', alpha=0.6)
        ax6.legend(fontsize=7)

        self.canvas.draw()


if __name__ == "__main__":
    app = LaserAblationApp()
    app.mainloop()
