import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numba import njit, prange

# --- 1. 預設參數 ---
DEFAULTS = {
    'v_stage': -1.0,
    'v_scan': 10.0,
    'f_base_khz': 1000,
    'divider': 10,
    'num_cycles': 20,
    'passes': 2,
    'a_um': 4.0,
    'b_um': 8.0,
    'phase_shift_deg': 180.0,

    'wavelength_nm': 257.5,
    'M2': 1.2,             
    'input_D_mm': 3.8,
    'focal_length_mm': 5.0,
    'defocus_um': 0.0,
    'pulse_width_fs': 800,

    'P_avg_W': 0.25,
    'F_th_1': 1.8,         
    'S_inc': 0.80,         
    'delta_um': 0.025,     
    'D_sat': 12.0,          

    'grid_res': 200,
    'elev': 30,
    'azim': -60,
    'slice_x_um': 0.0,
    'slice_y_um': 0.0,
    'show_spots': True,

    'exp_target_depth_um': 18.0, 
    'exp_target_spot_um': 1.4    
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

# 物理修正版：包含深溝槽波導聚焦與焦區熱平滑效應
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
    
    effective_w_sq = w_sq * 2.8  
    cutoff_r_sq = 3.5 * 3.5 * effective_w_sq
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
                    trapping_gain = 1.0 + 0.35 * current_d 

                    w_deeper_sq = effective_w_sq * (1.0 + (current_d / zR_um)**2)
                    F0_deeper = F0 * (effective_w_sq / w_deeper_sq) * trapping_gain

                    att_factor = np.exp(-current_d / D_sat)
                    fluence = F0_deeper * np.exp(-2.0 * r_sq / w_deeper_sq) * att_factor

                    if fluence > effective_F_th:
                        d_k = (delta_um * heat_accum_factor) * np.log(fluence / effective_F_th)
                        updated_depth[i, j] += d_k

    return updated_depth


class LaserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SiO2 飛秒雷射加工物理模擬與 SCF 擬合系統")
        self.root.geometry("1450\\times950")
        
        self.is_destroyed = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 變數字典初始化
        self.vars = {}
        for key, val in DEFAULTS.items():
            if isinstance(val, bool):
                self.vars[key] = tk.BooleanVar(value=val)
            elif isinstance(val, int):
                self.vars[key] = tk.IntVar(value=val)
            else:
                self.vars[key] = tk.DoubleVar(value=val)

        self.create_widgets()
        
        # 初始執行一次模擬與渲染
        self.run_simulation()
        self.render_plots()

    def create_widgets(self):
        # 左右分割：左側控制面板、右側 Matplotlib 畫布
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(control_frame, weight=1)

        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=3)

        # 左側使用 Notebook (標籤分頁取代 Accordion)
        self.notebook = ttk.Notebook(control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 分頁 1: 基礎運動學
        tab_base = ttk.Frame(self.notebook)
        self.notebook.add(tab_base, text='運動學與加工')
        self.build_base_controls(tab_base)

        # 分頁 2: 光學與離焦
        tab_optics = ttk.Frame(self.notebook)
        self.notebook.add(tab_optics, text='光學與離焦')
        self.build_optics_controls(tab_optics)

        # 分頁 3: 物理參數與熱效應
        tab_physics = ttk.Frame(self.notebook)
        self.notebook.add(tab_physics, text='物理與熱累積')
        self.build_physics_controls(tab_physics)

        # 分頁 4: SCF 擬合
        tab_scf = ttk.Frame(self.notebook)
        self.notebook.add(tab_scf, text='SCF 參數擬合')
        self.build_scf_controls(tab_scf)

        # 分頁 5: 顯示與切面
        tab_slice = ttk.Frame(self.notebook)
        self.notebook.add(tab_slice, text='視角與切面')
        self.build_slice_controls(tab_slice)

        # 底部按鈕與狀態列
        bottom_frame = ttk.Frame(control_frame)
        bottom_frame.pack(fill=tk.X, padx=5, pady=10)

        self.btn_run = ttk.Button(bottom_frame, text="🚀 開始模擬", command=self.on_run_clicked)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.status_var = tk.StringVar(value="狀態：就緒")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, wraplength=400, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

        # 右側 Matplotlib 畫布設置
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_slider(self, parent, label_text, var_key, min_v, max_v, step_v):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        lbl = ttk.Label(frame, text=label_text, width=22, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        
        val_lbl = ttk.Label(frame, textvariable=self.vars[var_key], width=8, anchor=tk.E)
        val_lbl.pack(side=tk.RIGHT)
        
        scale = ttk.Scale(frame, from_=min_v, to=max_v, variable=self.vars[var_key], 
                          command=lambda v: self.on_slider_changed())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def build_base_controls(self, parent):
        self.add_slider(parent, "v_stage (mm/s)", 'v_stage', -10.0, 10.0, 0.5)
        self.add_slider(parent, "v_scan (mm/s)", 'v_scan', -100.0, 100.0, 1.0)
        self.add_slider(parent, "Base Rep (kHz)", 'f_base_khz', 400, 1000, 100)
        self.add_slider(parent, "Divider", 'divider', 1, 100, 1)
        self.add_slider(parent, "Cycles/Pass", 'num_cycles', 1, 40, 1)
        self.add_slider(parent, "Passes", 'passes', 1, 50, 1)
        self.add_slider(parent, "Major a (μm)", 'a_um', 1.0, 100.0, 1.0)
        self.add_slider(parent, "Minor b (μm)", 'b_um', 1.0, 100.0, 1.0)
        self.add_slider(parent, "Pass 相位差 (°)", 'phase_shift_deg', 0.0, 360.0, 15.0)

    def build_optics_controls(self, parent):
        self.add_slider(parent, "波長 λ (nm)", 'wavelength_nm', 257.5, 1070.0, 0.5)
        self.add_slider(parent, "光束質量 M²", 'M2', 1.0, 3.0, 0.05)
        self.add_slider(parent, "入射光徑 D(mm)", 'input_D_mm', 1.0, 20.0, 0.1)
        self.add_slider(parent, "透鏡焦距 f(mm)", 'focal_length_mm', 1.0, 200.0, 1.0)
        self.add_slider(parent, "離焦量 Δz(μm)", 'defocus_um', -100.0, 100.0, 1.0)
        self.add_slider(parent, "脈寬 τ (fs)", 'pulse_width_fs', 50, 5000, 50)

    def build_physics_controls(self, parent):
        self.add_slider(parent, "Power (W)", 'P_avg_W', 0.01, 10.0, 0.05)
        self.add_slider(parent, "F_th_1 (J/cm²)", 'F_th_1', 0.5, 5.0, 0.1)
        self.add_slider(parent, "孵化係數 S", 'S_inc', 0.70, 0.95, 0.01)
        self.add_slider(parent, "delta (μm)", 'delta_um', 0.005, 0.150, 0.001)
        self.add_slider(parent, "D_sat (μm)", 'D_sat', 1.0, 50.0, 0.5)

    def build_scf_controls(self, parent):
        self.add_slider(parent, "實驗目標底部深度(μm)", 'exp_target_depth_um', 1.0, 100.0, 0.5)
        self.add_slider(parent, "實際加工光斑(μm)", 'exp_target_spot_um', 0.2, 10.0, 0.1)
        
        btn_scf = ttk.Button(parent, text="🔄 執行物理約束 SCF 擬合", command=self.on_scf_clicked)
        btn_scf.pack(fill=tk.X, padx=10, pady=20)

    def build_slice_controls(self, parent):
        self.add_slider(parent, "網格解析度", 'grid_res', 100, 300, 25)
        self.add_slider(parent, "3D 俯角", 'elev', 0, 90, 5)
        self.add_slider(parent, "3D 方位", 'azim', -180, 180, 5)
        self.add_slider(parent, "X切面位置(μm)", 'slice_x_um', -20.0, 20.0, 0.1)
        self.add_slider(parent, "Y切面位置(μm)", 'slice_y_um', -20.0, 20.0, 0.1)
        
        chk = ttk.Checkbutton(parent, text="顯示軌跡與脈衝點", variable=self.vars['show_spots'], command=self.on_slider_changed)
        chk.pack(anchor=tk.W, padx=10, pady=10)

    def on_slider_changed(self):
        # 為了順暢，數值滑動時可選擇是否即時更新（此處採用即時或可改為輕量渲染）
        pass

    def run_simulation(self):
        wavelength_m = self.vars['wavelength_nm'].get() * 1e-9
        M2 = self.vars['M2'].get()
        D_m = self.vars['input_D_mm'].get() * 1e-3
        f_m = self.vars['focal_length_mm'].get() * 1e-3
        defocus_m = self.vars['defocus_um'].get() * 1e-6

        w0_m = (2.0 * wavelength_m * f_m * M2) / (np.pi * D_m)
        w0_um = w0_m * 1e6
        d0_um = 2.0 * w0_um

        zR_m = (np.pi * w0_m**2) / (wavelength_m * M2)
        zR_um = zR_m * 1e6
        w_z_m = w0_m * np.sqrt(1.0 + (defocus_m / zR_m)**2)
        w_z_um = w_z_m * 1e6

        f_laser = (self.vars['f_base_khz'].get() * 1000.0) / float(self.vars['divider'].get())
        f_laser_khz = f_laser / 1000.0
        E_p = self.vars['P_avg_W'].get() / f_laser
        w_z_cm = w_z_m * 100.0
        F0_z = (2.0 * E_p) / (np.pi * (w_z_cm**2))

        F_th_1 = self.vars['F_th_1'].get()
        if F0_z > F_th_1:
            d_eff_um = 2.0 * w_z_um * np.sqrt(0.5 * np.log(F0_z / F_th_1))
        else:
            d_eff_um = 0.0

        v_stage_um_s = abs(self.vars['v_stage'].get()) * 1000.0
        pitch_stage_um = v_stage_um_s / f_laser
        overlap_rate = (1.0 - (pitch_stage_um / d0_um)) * 100.0

        a_um, b_um = self.vars['a_um'].get(), self.vars['b_um'].get()
        a_mm, b_mm = a_um / 1000.0, b_um / 1000.0
        h = ((a_mm - b_mm)**2) / ((a_mm + b_mm)**2 + 1e-12)
        ellipse_perimeter_mm = np.pi * (a_mm + b_mm) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        v_scan_val = self.vars['v_scan'].get()
        f_scan = 1e-5 if abs(v_scan_val) < 1e-5 else v_scan_val / ellipse_perimeter_mm

        period = 1.0 / abs(f_scan)
        total_time = self.vars['num_cycles'].get() * period
        dt = 1.0 / f_laser
        t = np.arange(0, total_time, dt)
        if len(t) > 30000: t = t[:30000]

        total_passes = self.vars['passes'].get()
        phase_shift_rad = np.radians(self.vars['phase_shift_deg'].get())

        all_pass_spots = []
        all_x_spots = []
        all_y_spots = []

        for p in range(total_passes):
            current_phase = p * phase_shift_rad
            x_p = np.ascontiguousarray(a_um * np.cos(2 * np.pi * f_scan * t + current_phase) + (self.vars['v_stage'].get() * 1000.0) * t)
            y_p = np.ascontiguousarray(b_um * np.sin(2 * np.pi * f_scan * t + current_phase))

            all_pass_spots.append((x_p, y_p))
            all_x_spots.extend(x_p)
            all_y_spots.extend(y_p)

        margin_um = w_z_um * 3.5
        x_min, x_max = np.min(all_x_spots) - margin_um, np.max(all_x_spots) + margin_um
        y_min, y_max = np.min(all_y_spots) - margin_um, np.max(all_y_spots) + margin_um

        res = self.vars['grid_res'].get()
        x_grid_um = np.ascontiguousarray(np.linspace(x_min, x_max, res))
        y_grid_um = np.ascontiguousarray(np.linspace(y_min, y_max, res))

        current_depth = np.zeros((res, res), dtype=np.float64)
        pass_history = []

        for p in range(total_passes):
            effective_F_th = max(self.vars['F_th_1'].get() * ((p + 1) ** (self.vars['S_inc'].get() - 1.0)), self.vars['F_th_1'].get() * 0.35)
            x_p, y_p = all_pass_spots[p]

            current_depth = compute_single_pass_ablation_experiment_matched(
                x_grid_um, y_grid_um, x_p, y_p,
                current_depth, F0_z, effective_F_th,
                self.vars['delta_um'].get(), w_z_um, zR_um,
                D_sat=self.vars['D_sat'].get(), f_laser_khz=f_laser_khz
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

    def render_plots(self):
        if self.is_destroyed or SIM_CACHE['Total_Depth_um'] is None:
            return

        self.fig.clear()

        X, Y = SIM_CACHE['X'], SIM_CACHE['Y']
        Total_Depth_um = SIM_CACHE['Total_Depth_um']
        x_grid_um = np.atleast_1d(SIM_CACHE['x_grid_um']).flatten()
        y_grid_um = np.atleast_1d(SIM_CACHE['y_grid_um']).flatten()
        all_pass_spots = SIM_CACHE['all_pass_spots']

        ax1 = self.fig.add_subplot(2, 3, 1, projection='3d')
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax3 = self.fig.add_subplot(2, 3, 3)
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax6 = self.fig.add_subplot(2, 3, 6)

        # 1. 3D Surface
        ax1.plot_surface(X, Y, -Total_Depth_um, cmap='viridis', edgecolor='none', alpha=0.95)
        ax1.view_init(elev=self.vars['elev'].get(), azim=self.vars['azim'].get())

        ny, nx = Total_Depth_um.shape
        center_region = Total_Depth_um[ny//4:3*ny//4, nx//4:3*nx//4]
        flat_bottom_depth = np.mean(center_region)

        ax1.set_title(f"3D Surface (Depth: {flat_bottom_depth:.2f} μm)", fontsize=9)
        ax1.set_xlabel("X (μm)"); ax1.set_ylabel("Y (μm)"); ax1.set_zlabel("Depth")

        # 2. 2D Top-View
        c = ax2.contourf(X, Y, Total_Depth_um, levels=50, cmap='inferno')
        if self.vars['show_spots'].get():
            colors = ['cyan', 'magenta', 'lime', 'yellow', 'white']
            for p_idx, (xs, ys) in enumerate(all_pass_spots):
                xs_arr = np.atleast_1d(xs).flatten()
                ys_arr = np.atleast_1d(ys).flatten()
                if len(xs_arr) > 0:
                    col = colors[p_idx % len(colors)]
                    ax2.plot(xs_arr, ys_arr, color=col, linestyle='--', linewidth=0.8, alpha=0.7)
                    ax2.scatter(xs_arr, ys_arr, color='white', edgecolors='none', s=6, alpha=0.9)

        idx_x = int(np.clip((np.abs(x_grid_um - self.vars['slice_x_um'].get())).argmin(), 0, len(x_grid_um) - 1))
        idx_y = int(np.clip((np.abs(y_grid_um - self.vars['slice_y_um'].get())).argmin(), 0, len(y_grid_um) - 1))

        ax2.axhline(y_grid_um[idx_y], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.axvline(x_grid_um[idx_x], color='cyan', linestyle='--', linewidth=1.2, alpha=0.7)

        f_laser_khz = SIM_CACHE['f_laser'] / 1000.0
        ax2.set_title(f"2D Top-View (Rep: {f_laser_khz:.1f} kHz | M²: {self.vars['M2'].get():.2f})", fontsize=9)
        ax2.set_xlabel("X (μm)"); ax2.set_ylabel("Y (μm)")
        self.fig.colorbar(c, ax=ax2, label='Depth')

        # 3. X Profile
        x_prof = Total_Depth_um[idx_y, :]
        ax3.plot(x_grid_um, x_prof, 'r-', linewidth=1.5)
        ax3.set_title(f"X-Profile (Y = {y_grid_um[idx_y]:.2f} μm)", fontsize=9)
        ax3.set_xlabel("X (μm)"); ax3.set_ylabel("Depth (μm)")
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.invert_yaxis()

        # 4. Y Profile
        y_prof = Total_Depth_um[:, idx_x]
        ax4.plot(y_grid_um, y_prof, 'c-', linewidth=1.5)
        ax4.set_title(f"Y-Profile (X = {x_grid_um[idx_x]:.2f} μm)", fontsize=9)
        ax4.set_xlabel("Y (μm)"); ax4.set_ylabel("Depth (μm)")
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.invert_yaxis()

        # 5. Y 平均深度
        mean_depth_y = np.mean(Total_Depth_um, axis=1)
        ax5.plot(y_grid_um, mean_depth_y, 'g-', linewidth=1.8)
        ax5.set_title("Average Depth vs. Y", fontsize=9)
        ax5.set_xlabel("Y (μm)"); ax5.set_ylabel("Depth (μm)")
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.invert_yaxis()

        # 6. Pass 累積
        pass_depths = SIM_CACHE['pass_depth_history']
        ax6.plot(np.arange(1, len(pass_depths) + 1), pass_depths, 'bo-', linewidth=2, markersize=5)
        ax6.set_title("Depth vs. Passes", fontsize=9)
        ax6.set_xlabel("Passes"); ax6.set_ylabel("Max Depth (μm)")
        ax6.grid(True, linestyle=':', alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def on_run_clicked(self):
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("狀態：⚡ 正在計算物理修正版 SiO2 模擬...")
        self.root.update_idletasks()

        def background_task():
            try:
                self.run_simulation()
                if not self.is_destroyed:
                    self.root.after(0, self.update_ui_after_run)
            except Exception as e:
                if not self.is_destroyed:
                    self.root.after(0, lambda: messagebox.showerror("錯誤", str(e)))

        threading.Thread(target=background_task, daemon=True).start()

    def update_ui_after_run(self):
        if self.is_destroyed: return
        self.render_plots()
        self.btn_run.config(state=tk.NORMAL)
        d0 = SIM_CACHE['d0_um']
        d_eff = SIM_CACHE['d_eff_um']
        self.status_var.set(f"狀態：✅ 模擬完成！ | Beam 2w0: {d0:.2f}μm | Eff. Spot: {d_eff:.2f}μm")

    def on_scf_clicked(self):
        target_depth = self.vars['exp_target_depth_um'].get()
        target_spot = self.vars['exp_target_spot_um'].get()
        self.status_var.set(f"狀態：🔄 啟動物理約束 SCF 擬合...")
        
        def scf_task():
            tol = 0.015
            for i in range(30):
                if self.is_destroyed: break
                self.run_simulation()
                Total_Depth = SIM_CACHE['Total_Depth_um']
                ny, nx = Total_Depth.shape
                center_region = Total_Depth[ny//4:3*ny//4, nx//4:3*ny//4]
                sim_flat_depth = np.mean(center_region)
                sim_spot = SIM_CACHE['d_eff_um']

                if sim_spot <= 0 or sim_flat_depth <= 0:
                    self.vars['M2'].set(max(self.vars['M2'].get() * 0.8, 1.0))
                    self.vars['delta_um'].set(min(self.vars['delta_um'].get() * 1.2, 0.10))
                    continue

                err_depth = (sim_flat_depth - target_depth) / target_depth
                err_spot = (sim_spot - target_spot) / target_spot

                if abs(err_depth) < tol and abs(err_spot) < tol:
                    msg = f"✅ SCF 於第 {i+1} 代收斂！M²={self.vars['M2'].get():.2f}, delta={self.vars['delta_um'].get():.4f}μm"
                    if not self.is_destroyed:
                        self.root.after(0, lambda: self.finish_scf(msg))
                    return

                spot_ratio = target_spot / sim_spot
                new_m2 = np.clip(self.vars['M2'].get() * spot_ratio, 1.0, 3.0)
                self.vars['M2'].set(float(round(new_m2, 2)))

                depth_ratio = target_depth / sim_flat_depth
                new_delta = np.clip(self.vars['delta_um'].get() * depth_ratio, 0.005, 0.120)
                self.vars['delta_um'].set(float(round(new_delta, 4)))

            if not self.is_destroyed:
                self.root.after(0, lambda: self.finish_scf("⚠️ SCF 迭代完成"))

        threading.Thread(target=scf_task, daemon=True).start()

    def finish_scf(self, msg):
        if self.is_destroyed: return
        self.render_plots()
        self.status_var.set(f"狀態：{msg}")

    def on_close(self):
        self.is_destroyed =- True
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LaserApp(root)
    root.mainloop()
