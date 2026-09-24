import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt

# 設定 Matplotlib 支援 macOS 中文字型與正確顯示負號
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numba import njit, prange

# --- 1. 預設參數 ---
DEFAULTS = {
    'v_stage': -1.0,        # mm/s
    'v_scan': 10.0,          # mm/s
    'f_base_khz': 1000,      # kHz
    'divider': 10,           # 實際發射頻率 100 kHz
    'num_cycles': 20,
    'passes': 2,             # 加工次數
    'a_um': 4.0,             # μm
    'b_um': 8.0,             # μm
    'phase_shift_deg': 180.0, # Pass 間相位錯位角度 (度)

    'wavelength_nm': 257.5,
    'M2': 1.2,              
    'input_D_mm': 2.0,       # 入射光徑 (mm)
    'focal_length_mm': 10.0, # 透鏡焦距 (mm)
    'defocus_um': 0.0,
    'pulse_width_fs': 800,

    'P_avg_W': 0.05,         # 平均功率 (W)
    'F_th_1': 1.8,           # SiO2 燒蝕閾值 (J/cm²)
    'S_inc': 0.80,           # 孵化係數
    'delta_um': 0.025,       # 穿透深度 (μm)
    'D_sat': 12.0,           # 飽和深度 (μm)

    'grid_res': 100,         # 預設稍微調降以兼顧流暢度
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
@njit(fastmath=True)
def compute_single_pass_ablation_experiment_matched(
    X, Y, spot_centers, E_pulse, w0, alpha, F_th_0, S_inc,
    k_inc, k_thermal, enable_inc, enable_thermal,
    enable_aspect, aspect_factor, enable_attenuation, alpha_medium, current_depth
):
    nx, ny = X.shape[0], X.shape[1]
    N_shots_map = np.zeros((nx, ny), dtype=np.float64)
    
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    x_min, y_min = X[0, 0], Y[0, 0]
    
    n_spots = spot_centers.shape[0]
    
    for s in range(n_spots):
        xc = spot_centers[s, 0]
        yc = spot_centers[s, 1]
        
        r_cut = 3.0 * w0
        i_min = max(0, int(np.floor((xc - r_cut - x_min) / dx)))
        i_max = min(nx, int(np.ceil((xc + r_cut - x_min) / dx)))
        j_min = max(0, int(np.floor((yc - r_cut - y_min) / dy)))
        j_max = min(ny, int(np.ceil((yc + r_cut - y_min) / dy)))
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                x = X[i, j]
                y = Y[i, j]
                r2 = (x - xc)**2 + (y - yc)**2
                
                if r2 <= r_cut**2:
                    F_0 = (2.0 * E_pulse) / (np.pi * (w0**2))
                    F = F_0 * np.exp(-2.0 * r2 / (w0**2))
                    
                    if enable_attenuation and alpha_medium > 0.0:
                        d_curr = current_depth[i, j]
                        F = F * np.exp(-alpha_medium * d_curr)
                    
                    N_curr = N_shots_map[i, j]
                    if enable_inc:
                        F_th = F_th_0 * (1.0 + (S_inc - 1.0) * (1.0 - np.exp(-k_inc * N_curr)))
                    else:
                        F_th = F_th_0
                        
                    if enable_thermal and N_curr > 0:
                        F_th_eff = F_th / (1.0 + k_thermal * np.log1p(N_curr))
                    else:
                        F_th_eff = F_th
                        
                    if F > F_th_eff:
                        d_k = (1.0 / alpha) * np.log(F / F_th_eff)
                        
                        if enable_aspect:
                            d_curr = current_depth[i, j]
                            aspect_ratio = d_curr / (2.0 * w0)
                            d_k = d_k * np.exp(-aspect_factor * aspect_ratio)
                            
                        current_depth[i, j] += d_k
                        N_shots_map[i, j] += 1.0
                        
    return current_depth


class LaserAblationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SiO2 飛秒雷射多 Pass 燒蝕模擬器")
        self.geometry("1450x900")

        self.is_destroyed = False
        self.params = {k: tk.DoubleVar(value=float(v)) if isinstance(v, (float, int)) else tk.IntVar(value=v) 
                       for k, v in DEFAULTS.items()}
        self.chk_show_spots = tk.BooleanVar(value=DEFAULTS['show_spots'])

        self.setup_ui()
        self.show_blank_canvas()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.is_destroyed = True
        self.destroy()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame, width=440)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        self.canvas_scroll = tk.Canvas(left_frame, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.canvas_scroll.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas_scroll)

        CONFIG_EVENT = "<" + "Configure" + ">"

        self.scrollable_frame.bind(
            CONFIG_EVENT, 
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )

        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_scroll.bind(CONFIG_EVENT, self._on_canvas_configure)

        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bind_mousewheel_recursive(self.scrollable_frame)

        # 1. 運動控制
        lf_motion = ttk.LabelFrame(self.scrollable_frame, text="運動控制與掃描軌跡", padding="5")
        lf_motion.pack(fill=tk.X, pady=5, padx=5)
        self.add_slider(lf_motion, 'v_stage', 'v_stage (mm/s)', -10.0, 10.0, 0.5)
        self.add_slider(lf_motion, 'v_scan', 'v_scan (mm/s)', -100.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'f_base_khz', 'Base Rep (kHz)', 400, 1000, 100, is_int=True)
        self.add_slider(lf_motion, 'divider', 'Divider', 1, 1000, 1, is_int=True)
        self.add_slider(lf_motion, 'num_cycles', 'Cycles/Pass', 1, 40, 1, is_int=True)
        self.add_slider(lf_motion, 'passes', 'Passes (次數)', 1, 50, 1, is_int=True)
        self.add_slider(lf_motion, 'a_um', 'Major a (μm)', 1.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'b_um', 'Minor b (μm)', 1.0, 100.0, 1.0)
        self.add_slider(lf_motion, 'phase_shift_deg', 'Pass 相位差 (°)', 0.0, 360.0, 15.0)

        # 2. 光學系統
        lf_optics = ttk.LabelFrame(self.scrollable_frame, text="光學與焦區系統", padding="5")
        lf_optics.pack(fill=tk.X, pady=5, padx=5)
        self.add_slider(lf_optics, 'wavelength_nm', '波長 λ (nm)', 257.5, 1070.0, 0.5)
        self.add_slider(lf_optics, 'M2', '光束質量 M²', 1.0, 3.0, 0.05)
        self.add_slider(lf_optics, 'input_D_mm', '入射光徑 D (mm)', 1.0, 20.0, 0.1)
        self.add_slider(lf_optics, 'focal_length_mm', '透鏡焦距 f (mm)', 1.0, 200.0, 1.0)
        self.add_slider(lf_optics, 'defocus_um', '離焦量 Δz (μm)', -100.0, 100.0, 1.0)
        self.add_slider(lf_optics, 'pulse_width_fs', '脈寬 τ (fs)', 50, 5000, 50, is_int=True)

        # 3. 雷射功率與材料參數
        lf_physics = ttk.LabelFrame(self.scrollable_frame, text="雷射功率與 SiO2 物理參數", padding="5")
        lf_physics.pack(fill=tk.X, pady=5, padx=5)
        self.add_slider(lf_physics, 'P_avg_W', 'Power (W)', 0.01, 10.0, 0.01)
        self.add_slider(lf_physics, 'F_th_1', 'F_th_1 (J/cm²)', 0.5, 5.0, 0.1)
        self.add_slider(lf_physics, 'S_inc', '孵化係數 S', 0.70, 0.95, 0.01)
        self.add_slider(lf_physics, 'delta_um', 'delta (μm)', 0.005, 0.150, 0.001)
        self.add_slider(lf_physics, 'D_sat', 'D_sat (μm)', 1.0, 50.0, 0.5)

        # 4. SCF 擬合控制
        lf_scf = ttk.LabelFrame(self.scrollable_frame, text="物理約束 SCF 實驗雙參數擬合", padding="5")
        lf_scf.pack(fill=tk.X, pady=5, padx=5)
        self.add_slider(lf_scf, 'exp_target_depth_um', '實驗底部深度 (μm)', 1.0, 100.0, 0.5)
        self.add_slider(lf_scf, 'exp_target_spot_um', '實際加工光斑 (μm)', 0.2, 10.0, 0.1)
        
        self.btn_scf = ttk.Button(lf_scf, text="執行物理約束 SCF 擬合", command=self.on_btn_scf)
        self.btn_scf.pack(fill=tk.X, pady=5)

        # 5. 視角與切面控制
        lf_view = ttk.LabelFrame(self.scrollable_frame, text="視角與剖面切面控制", padding="5")
        lf_view.pack(fill=tk.X, pady=5, padx=5)
        self.add_slider(lf_view, 'grid_res', '網格解析度', 100, 300, 20, is_int=True, auto_render=True)
        self.add_slider(lf_view, 'elev', '3D 俯角', 0, 90, 5, is_int=True, auto_render=True)
        self.add_slider(lf_view, 'azim', '3D 方位', -180, 180, 5, is_int=True, auto_render=True)
        
        self.slice_x_slider = self.add_slider(lf_view, 'slice_x_um', 'X 切面位置 (μm)', -20.0, 20.0, 0.1, auto_render=True)
        self.slice_y_slider = self.add_slider(lf_view, 'slice_y_um', 'Y 切面位置 (μm)', -20.0, 20.0, 0.1, auto_render=True)
        
        chk_spots = ttk.Checkbutton(lf_view, text="顯示軌跡與脈衝點", variable=self.chk_show_spots, command=self.render_plots)
        chk_spots.pack(anchor=tk.W, pady=2)

        self.btn_run = ttk.Button(self.scrollable_frame, text="開始模擬", command=self.on_btn_run)
        self.btn_run.pack(fill=tk.X, pady=15, padx=5)

        self.lbl_status = ttk.Label(self.scrollable_frame, text="狀態：等待啟動...", wraplength=380, foreground="blue")
        self.lbl_status.pack(fill=tk.X, pady=5, padx=5)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(11, 7.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()

    def _on_canvas_configure(self, event):
        self.canvas_scroll.itemconfig(self.canvas_window, width=event.width)

    def bind_mousewheel_recursive(self, widget):
        ENTER_EVENT = "<" + "Enter" + ">"
        LEAVE_EVENT = "<" + "Leave" + ">"
        widget.bind(ENTER_EVENT, lambda e: self._bind_mousewheel())
        widget.bind(LEAVE_EVENT, lambda e: self._unbind_mousewheel())
        for child in widget.winfo_children():
            self.bind_mousewheel_recursive(child)

    def _bind_mousewheel(self):
        WHEEL_EVENT = "<" + "MouseWheel" + ">"
        BTN4_EVENT = "<" + "Button-4" + ">"
        BTN5_EVENT = "<" + "Button-5" + ">"
        if sys.platform.startswith('darwin'):
            self.canvas_scroll.bind_all(WHEEL_EVENT, lambda e: self.canvas_scroll.yview_scroll(-1 * e.delta, "units"))
        elif sys.platform.startswith('win'):
            self.canvas_scroll.bind_all(WHEEL_EVENT, lambda e: self.canvas_scroll.yview_scroll(-1 * int(e.delta / 120), "units"))
        else:
            self.canvas_scroll.bind_all(BTN4_EVENT, lambda e: self.canvas_scroll.yview_scroll(-1, "units"))
            self.canvas_scroll.bind_all(BTN5_EVENT, lambda e: self.canvas_scroll.yview_scroll(1, "units"))

    def _unbind_mousewheel(self):
        WHEEL_EVENT = "<" + "MouseWheel" + ">"
        BTN4_EVENT = "<" + "Button-4" + ">"
        BTN5_EVENT = "<" + "Button-5" + ">"
        if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
            self.canvas_scroll.unbind_all(WHEEL_EVENT)
        else:
            self.canvas_scroll.unbind_all(BTN4_EVENT)
            self.canvas_scroll.unbind_all(BTN5_EVENT)

    def add_slider(self, parent, param_key, label_text, from_, to, resolution, is_int=False, auto_render=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        lbl = ttk.Label(frame, text=label_text, width=17, anchor=tk.W)
        lbl.pack(side=tk.LEFT)

        var = self.params[param_key]
        
        def on_change(val):
            val_num = int(float(val)) if is_int else round(float(val), 4)
            var.set(val_num)
            val_lbl.config(text=str(val_num))
            if auto_render and SIM_CACHE['Total_Depth_um'] is not None:
                self.render_plots()

        scale = ttk.Scale(frame, from_=from_, to=to, value=var.get(), command=on_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        val_lbl = ttk.Label(frame, text=str(var.get()), width=7, anchor=tk.E)
        val_lbl.pack(side=tk.RIGHT)

        return scale

    def get_val(self, key):
        return self.params[key].get()

    def set_val(self, key, val):
        self.params[key].set(val)

    def show_blank_canvas(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "歡迎使用 SiO2 飛秒雷射模擬器！\n\n請點擊左側「開始模擬」或\n「執行物理約束 SCF 擬合」按鈕開始計算", 
                ha='center', va='center', fontsize=14, color='gray', multialignment='center')
        ax.axis('off')
        self.canvas.draw()

    def set_ui_state(self, state="normal"):
        if not self.is_destroyed:
            self.btn_run.config(state=state)
            self.btn_scf.config(state=state)

    def on_btn_run(self):
        self.set_ui_state("disabled")
        self.lbl_status.config(text="狀態：背景運算中：正在計算物理修正版 SiO2 多 Pass 燒蝕...")
        threading.Thread(target=self._run_simulation_task, daemon=True).start()

    def _run_simulation_task(self):
        try:
            print("開始執行單次模擬運算...")
            self.run_simulation()
            print("單次模擬運算完成，準備更新 UI...")
            if not self.is_destroyed:
                self.after(0, self._finalize_simulation)
        except Exception as err:
            err_str = str(err)
            import traceback
            traceback.print_exc()
            if not self.is_destroyed:
                self.after(0, lambda msg=err_str: self.lbl_status.config(text=f"狀態：錯誤: {msg}"))
        finally:
            if not self.is_destroyed:
                self.after(0, lambda: self.set_ui_state("normal"))

    def _finalize_simulation(self):
        x_min, x_max = SIM_CACHE['x_grid_um'][0], SIM_CACHE['x_grid_um'][-1]
        y_min, y_max = SIM_CACHE['y_grid_um'][0], SIM_CACHE['y_grid_um'][-1]
        self.slice_x_slider.config(from_=x_min, to=x_max)
        self.slice_y_slider.config(from_=y_min, to=y_max)

        self.render_plots()
        d0 = SIM_CACHE['d0_um']
        d_eff = SIM_CACHE['d_eff_um']
        self.lbl_status.config(text=f"狀態：模擬完成！\nBeam 2w0: {d0:.2f}μm | Eff Spot: {d_eff:.2f}μm")

    def run_simulation(self):
        # 強制全部轉為 float 並加強防禦性保護
        wavelength_m = float(self.get_val('wavelength_nm')) * 1e-9
        M2 = max(float(self.get_val('M2')), 1e-6)
        D_m = max(float(self.get_val('input_D_mm')) * 1e-3, 1e-6)
        f_m = max(float(self.get_val('focal_length_mm')) * 1e-3, 1e-6)
        defocus_m = float(self.get_val('defocus_um')) * 1e-6

        w0_m = (2.0 * wavelength_m * f_m * M2) / max(np.pi * D_m, 1e-12)
        w0_um = w0_m * 1e6
        d0_um = 2.0 * w0_um

        zR_m = max((np.pi * w0_m**2) / max(wavelength_m * M2, 1e-12), 1e-12)
        w_z_m = w0_m * np.sqrt(max(1.0 + (defocus_m / zR_m)**2, 1e-12))
        w_z_um = w_z_m * 1e6

        divider_val = max(float(self.get_val('divider')), 1e-6)
        f_laser = max((float(self.get_val('f_base_khz')) * 1000.0) / divider_val, 1e-6)
        E_p = float(self.get_val('P_avg_W')) / f_laser
        w_z_cm = w_z_m * 100.0
        F0_z = (2.0 * E_p) / max(np.pi * (w_z_cm**2), 1e-12)

        F_th_1 = float(self.get_val('F_th_1'))
        if F0_z > F_th_1:
            d_eff_um = 2.0 * w_z_um * np.sqrt(max(0.5 * np.log(F0_z / F_th_1), 0.0))
        else:
            d_eff_um = 0.0

        v_stage_um_s = abs(float(self.get_val('v_stage'))) * 1000.0
        pitch_stage_um = v_stage_um_s / f_laser
        overlap_rate = (1.0 - (pitch_stage_um / max(d0_um, 1e-6))) * 100.0

        a_um = float(self.get_val('a_um'))
        b_um = float(self.get_val('b_um'))
        a_mm = max(a_um / 1000.0, 1e-6)
        b_mm = max(b_um / 1000.0, 1e-6)
        
        # 嚴格的 Ramanujan 橢圓周長計算與強制絕對安全保底
        h = ((a_mm - b_mm) / max(a_mm + b_mm, 1e-12))**2
        ellipse_perimeter_mm = np.pi * (a_mm + b_mm) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(max(4.0 - 3.0 * h, 1e-6))))
        ellipse_perimeter_mm = max(ellipse_perimeter_mm, 1e-3)
        
        v_scan_val = float(self.get_val('v_scan'))
        f_scan = 1e-5 if abs(v_scan_val) < 1e-5 else v_scan_val / ellipse_perimeter_mm
        period = 1.0 / max(abs(f_scan), 1e-6)
        total_time = float(self.get_val('num_cycles')) * period
        dt = 1.0 / f_laser
        t = np.arange(0, total_time, dt)
        if len(t) > 20000: t = t[:20000]

        total_passes = int(float(self.get_val('passes')))
        phase_shift_rad = np.radians(float(self.get_val('phase_shift_deg')))

        all_pass_spots = []
        all_x_spots = []
        all_y_spots = []

        for p in range(total_passes):
            current_phase = p * phase_shift_rad
            x_p = np.ascontiguousarray(a_um * np.cos(2 * np.pi * f_scan * t + current_phase) + (float(self.get_val('v_stage')) * 1000.0) * t)
            y_p = np.ascontiguousarray(b_um * np.sin(2 * np.pi * f_scan * t + current_phase))

            all_pass_spots.append((x_p, y_p))
            all_x_spots.extend(x_p)
            all_y_spots.extend(y_p)

        margin_um = max(w_z_um * 3.5, 3.0)
        x_min, x_max = np.min(all_x_spots) - margin_um, np.max(all_x_spots) + margin_um
        y_min, y_max = np.min(all_y_spots) - margin_um, np.max(all_y_spots) + margin_um

        res = int(float(self.get_val('grid_res')))
        x_grid_um = np.ascontiguousarray(np.linspace(x_min, x_max, res))
        y_grid_um = np.ascontiguousarray(np.linspace(y_min, y_max, res))
        X, Y = np.meshgrid(x_grid_um, y_grid_um)

        current_depth = np.zeros((res, res), dtype=np.float64)
        pass_history = []

        S_inc = float(self.get_val('S_inc'))
        delta_um = max(float(self.get_val('delta_um')), 1e-6)

        for p in range(total_passes):
            x_p, y_p = all_pass_spots[p]
            spot_centers = np.column_stack((x_p, y_p))

            current_depth = compute_single_pass_ablation_experiment_matched(
                X, Y, spot_centers, E_p, w_z_um, 1.0/delta_um, F_th_1, S_inc,
                0.1, 0.05, True, False, False, 0.0, False, 0.0, current_depth
            )
            ny, nx = current_depth.shape
            center_region = current_depth[ny//4:3*ny//4, nx//4:3*nx//4]
            pass_history.append(float(np.mean(center_region)))

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

    def on_btn_scf(self):
        self.set_ui_state("disabled")
        target_depth = float(self.get_val('exp_target_depth_um'))
        target_spot = float(self.get_val('exp_target_spot_um'))
        self.lbl_status.config(text=f"狀態：背景執行中：物理約束 SCF 擬合 (目標深度={target_depth}μm)...")
        threading.Thread(target=self._run_scf_task, daemon=True).start()

    def _run_scf_task(self):
        tol = 0.015
        sim_flat_depth, sim_spot = 0.0, 0.0
        try:
            for i in range(15):
                if self.is_destroyed:
                    return

                print(f"SCF 擬合第 {i+1} 代進行中...")
                self.run_simulation()

                Total_Depth = SIM_CACHE['Total_Depth_um']
                if Total_Depth is None:
                    print("警告：Total_Depth 為空！")
                    break

                ny, nx = Total_Depth.shape
                center_region = Total_Depth[ny//4:3*ny//4, nx//4:3*nx//4]
                sim_flat_depth = float(np.mean(center_region))
                sim_spot = float(SIM_CACHE['d_eff_um'])

                if sim_spot <= 0 or sim_flat_depth <= 0 or np.isnan(sim_flat_depth):
                    print("數值異常，調整參數重試...")
                    self.set_val('M2', max(float(self.get_val('M2')) * 0.8, 1.0))
                    self.set_val('delta_um', min(float(self.get_val('delta_um')) * 1.2, 0.10))
                    continue

                err_depth = (sim_flat_depth - target_depth) / max(target_depth, 1e-6)
                err_spot = (sim_spot - target_spot) / max(target_spot, 1e-6)

                if abs(err_depth) < tol and abs(err_spot) < tol:
                    print(f"SCF 於第 {i+1} 代收斂！")
                    if not self.is_destroyed:
                        self.after(0, self._finalize_scf_success, i+1)
                    return

                spot_ratio = target_spot / max(sim_spot, 1e-6)
                new_m2 = np.clip(float(self.get_val('M2')) * spot_ratio, 1.0, 3.0)
                self.set_val('M2', float(round(new_m2, 2)))

                depth_ratio = target_depth / max(sim_flat_depth, 1e-6)
                new_delta = np.clip(float(self.get_val('delta_um')) * depth_ratio, 0.005, 0.120)
                self.set_val('delta_um', float(round(new_delta, 4)))

            if not self.is_destroyed:
                self.after(0, self._finalize_scf_finish, sim_flat_depth)
        except Exception as err:
            err_str = str(err)
            import traceback
            traceback.print_exc()
            if not self.is_destroyed:
                self.after(0, lambda msg=err_str: self.lbl_status.config(text=f"狀態：SCF 擬合錯誤: {msg}"))
        finally:
            if not self.is_destroyed:
                self.after(0, lambda: self.set_ui_state("normal"))

    def _finalize_scf_success(self, generations):
        self.render_plots()
        msg = (f"狀態：SCF 於第 {generations} 代成功收斂！\n"
               f"擬合結果: M² = {self.get_val('M2'):.2f}, delta = {self.get_val('delta_um'):.4f} μm")
        self.lbl_status.config(text=msg)

    def _finalize_scf_finish(self, sim_flat_depth):
        self.render_plots()
        msg = f"狀態：SCF 完成疊代，當前平坦深度: {sim_flat_depth:.2f} μm, 有效光斑: {SIM_CACHE['d_eff_um']:.2f} μm"
        self.lbl_status.config(text=msg)

    def render_plots(self):
        if SIM_CACHE['Total_Depth_um'] is None:
            return

        self.fig.clear()

        gs = gridspec.GridSpec(2, 3, figure=self.fig, wspace=0.4, hspace=0.45)

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

        ax1.plot_surface(X, Y, -Total_Depth_um, cmap='viridis', edgecolor='none', alpha=0.95)
        ax1.view_init(elev=int(float(self.get_val('elev'))), azim=int(float(self.get_val('azim'))))

        ny, nx = Total_Depth_um.shape
        center_region = Total_Depth_um[ny//4:3*ny//4, nx//4:3*nx//4]
        flat_bottom_depth = np.mean(center_region)

        ax1.set_title(f"3D Surface\n(Flat Depth: {flat_bottom_depth:.2f} μm)", fontsize=9, pad=3)
        ax1.set_xlabel("X (μm)", fontsize=8)
        ax1.set_ylabel("Y (μm)", fontsize=8)
        ax1.set_zlabel("Depth", fontsize=8)

        c = ax2.contourf(X, Y, Total_Depth_um, levels=50, cmap='inferno')
        if self.chk_show_spots.get():
            colors = ['cyan', 'magenta', 'lime', 'yellow', 'white']
            for p_idx, (xs, ys) in enumerate(all_pass_spots):
                xs_arr = np.atleast_1d(xs).flatten()
                ys_arr = np.atleast_1d(ys).flatten()
                if len(xs_arr) == len(ys_arr) and len(xs_arr) > 0:
                    col = colors[p_idx % len(colors)]
                    ax2.plot(xs_arr, ys_arr, color=col, linestyle='--', linewidth=0.8, alpha=0.7)

        idx_x = int(np.clip((np.abs(x_grid_um - float(self.get_val('slice_x_um')))).argmin(), 0, len(x_grid_um) - 1))
        idx_y = int(np.clip((np.abs(y_grid_um - float(self.get_val('slice_y_um')))).argmin(), 0, len(y_grid_um) - 1))

        ax2.axhline(y_grid_um[idx_y], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.axvline(x_grid_um[idx_x], color='cyan', linestyle='--', linewidth=1.2, alpha=0.7)

        f_laser_khz = SIM_CACHE['f_laser'] / 1000.0
        ax2.set_title(f"2D Top-View (Rep: {f_laser_khz:.0f}kHz)\nEff Spot: {SIM_CACHE['d_eff_um']:.2f}μm", fontsize=9)
        ax2.set_xlabel("X (μm)", fontsize=8)
        ax2.set_ylabel("Y (μm)", fontsize=8)
        cbar = self.fig.colorbar(c, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        x_prof = Total_Depth_um[idx_y, :]
        ax3.plot(x_grid_um, x_prof, 'r-', linewidth=1.5)
        ax3.set_title(f"X-Profile\n(Y = {y_grid_um[idx_y]:.1f} μm)", fontsize=9)
        ax3.set_xlabel("X (μm)", fontsize=8)
        ax3.set_ylabel("Depth (μm)", fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.invert_yaxis()

        y_prof = Total_Depth_um[:, idx_x]
        ax4.plot(y_grid_um, y_prof, 'c-', linewidth=1.5)
        ax4.set_title(f"Y-Profile\n(X = {x_grid_um[idx_x]:.1f} μm)", fontsize=9)
        ax4.set_xlabel("Y (μm)", fontsize=8)
        ax4.set_ylabel("Depth (μm)", fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.invert_yaxis()

        mean_depth_y = np.mean(Total_Depth_um, axis=1)
        ax5.plot(y_grid_um, mean_depth_y, 'g-', linewidth=1.8, label='Mean Depth')
        ax5.axvline(y_grid_um[idx_y], color='red', linestyle='--', alpha=0.5, label='Current Y Slice')
        ax5.set_title("Average Depth vs. Y", fontsize=9)
        ax5.set_xlabel("Y (μm)", fontsize=8)
        ax5.set_ylabel("Avg Depth (μm)", fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.invert_yaxis()
        ax5.legend(fontsize=7)

        pass_depths = SIM_CACHE['pass_depth_history']
        p_range = np.arange(1, len(pass_depths) + 1, dtype=int)
        ax6.plot(p_range, pass_depths, 'bo-', linewidth=2, markersize=6, label='Simulated Depth')
        ax6.set_title("Depth vs. Passes", fontsize=9)
        ax6.set_xlabel("Passes", fontsize=8)
        ax6.set_ylabel("Max Depth (μm)", fontsize=8)
        ax6.grid(True, linestyle=':', alpha=0.6)
        ax6.legend(fontsize=7)

        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
        self.canvas.draw()


if __name__ == "__main__":
    app = LaserAblationApp()
    app.mainloop()
