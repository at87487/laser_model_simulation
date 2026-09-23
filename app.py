import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numba import njit, prange

# --- 1. 預設參數 (針對 SiO2 材料進行校正) ---
DEFAULTS = {
    'v_stage': -1.0,       # mm/s
    'v_scan': 10.0,        # mm/s
    'f_base_khz': 1000,    # kHz
    'divider': 10,         # 實際發射頻率 1 kHz
    'num_cycles': 20,      
    'passes': 2,           # 加工兩次
    'a_um': 4.0,           # μm
    'b_um': 8.0,           # μm
    'phase_shift_deg': 180.0, # Pass 間相位錯位角度 (度)

    'wavelength_nm': 257.5,
    'M2': 1.2,             
    'input_D_mm': 3.8,     
    'focal_length_mm': 5.0,
    'defocus_um': 0.0,     
    'pulse_width_fs': 800, 

    'P_avg_W': 0.25,       
    'F_th_1': 1.0,         # SiO2 典型單脈衝燒蝕閾值
    'S_inc': 0.7,          # SiO2 典型孵化係數
    'delta_um': 0.02,      # 微觀單脈衝穿透深度初始值
    'D_sat': 12.0,         # 飽和吸收深度 (μm)

    'grid_res': 200,
    'elev': 30,
    'azim': -60,
    'slice_x_um': 0.0,
    'slice_y_um': 0.0,
    'show_spots': True,

    'exp_target_depth_um': 18.0, # 實驗目標底部平整深度 (μm)
    'exp_target_spot_um': 1.4    # 實驗量測有效加工光斑直徑 (μm)
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

# --- 2. Numba 燒蝕核心 (帶 D_sat 飽和衰減) ---
@njit(parallel=True, fastmath=True)
def compute_single_pass_ablation(x_grid_um, y_grid_um, x_spots_um, y_spots_um, current_depth, F0, effective_F_th, delta_um, w_spot_um, zR_um, D_sat=12.0):
    nx = len(x_grid_um)
    ny = len(y_grid_um)
    n_spots = len(x_spots_um)

    updated_depth = current_depth.copy()
    w_sq = w_spot_um * w_spot_um
    cutoff_r_sq = 3.5 * 3.5 * w_sq

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
                    
                    w_deeper_sq = w_sq * (1.0 + (current_d / zR_um)**2)
                    F0_deeper = F0 * (w_sq / w_deeper_sq)

                    att_factor = np.exp(-current_d / D_sat)
                    fluence = F0_deeper * np.exp(-2.0 * r_sq / w_deeper_sq) * att_factor

                    if fluence > effective_F_th:
                        d_k = delta_um * np.log(fluence / effective_F_th)
                        updated_depth[i, j] += d_k

    return updated_depth


# --- 3. GUI 主介面類別 (Tkinter) ---
class LaserAblationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SiO2 Laser Ablation Simulator (SCF M² & Delta Fitting Edition)")
        self.geometry("1450x920")

        self.widgets_dict = {}
        self.setup_ui()

    def setup_ui(self):
        # 左側控制面板與滾動機制
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(control_frame, width=340)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. 基礎運動參數面板
        lf_base = ttk.LabelFrame(scrollable_frame, text="⚙️ 基礎運動與加工參數", padding="5")
        lf_base.pack(fill=tk.X, pady=5)
        self.add_slider(lf_base, 'v_stage', 'v_stage(mm/s)', -10.0, 10.0, 0.5, DEFAULTS['v_stage'])
        self.add_slider(lf_base, 'v_scan', 'v_scan(mm/s)', -100.0, 100.0, 1.0, DEFAULTS['v_scan'])
        self.add_slider(lf_base, 'f_base_khz', 'Base Rep(kHz)', 400, 1000, 100, DEFAULTS['f_base_khz'], is_int=True)
        self.add_slider(lf_base, 'divider', 'Divider', 1, 1000, 1, DEFAULTS['divider'], is_int=True)
        self.add_slider(lf_base, 'num_cycles', 'Cycles/Pass', 1, 40, 1, DEFAULTS['num_cycles'], is_int=True)
        self.add_slider(lf_base, 'passes', '加工次數 Pass', 1, 50, 1, DEFAULTS['passes'], is_int=True)
        self.add_slider(lf_base, 'a_um', 'Major a (μm)', 1.0, 100.0, 1.0, DEFAULTS['a_um'])
        self.add_slider(lf_base, 'b_um', 'Minor b (μm)', 1.0, 100.0, 1.0, DEFAULTS['b_um'])
        self.add_slider(lf_base, 'phase_shift_deg', 'Pass 相位差 (°)', 0.0, 360.0, 15.0, DEFAULTS['phase_shift_deg'])

        # 2. 光學參數面板
        lf_optics = ttk.LabelFrame(scrollable_frame, text="🔍 光學與離焦", padding="5")
        lf_optics.pack(fill=tk.X, pady=5)
        self.add_slider(lf_optics, 'wavelength_nm', '波長 λ (nm)', 257.5, 1070.0, 0.5, DEFAULTS['wavelength_nm'])
        self.add_slider(lf_optics, 'M2', '光束質量 M²', 1.0, 6.0, 0.05, DEFAULTS['M2'])
        self.add_slider(lf_optics, 'input_D_mm', '入射光徑 D(mm)', 1.0, 20.0, 0.1, DEFAULTS['input_D_mm'])
        self.add_slider(lf_optics, 'focal_length_mm', '透鏡焦距 f(mm)', 1.0, 200.0, 1.0, DEFAULTS['focal_length_mm'])
        self.add_slider(lf_optics, 'defocus_um', '離焦量 Δz(μm)', -100.0, 100.0, 1.0, DEFAULTS['defocus_um'])

        # 3. SiO2 物理屬性面板
        lf_physics = ttk.LabelFrame(scrollable_frame, text="⚡ 雷射功率與 SiO2 物理", padding="5")
        lf_physics.pack(fill=tk.X, pady=5)
        self.add_slider(lf_physics, 'P_avg_W', 'Power (W)', 0.01, 10.0, 0.05, DEFAULTS['P_avg_W'])
        self.add_slider(lf_physics, 'F_th_1', 'F_th_1(J/cm²)', 0.1, 10.0, 0.1, DEFAULTS['F_th_1'])
        self.add_slider(lf_physics, 'S_inc', '孵化係數 S', 0.3, 1.0, 0.02, DEFAULTS['S_inc'])
        self.add_slider(lf_physics, 'delta_um', 'delta (μm)', 0.0001, 0.5, 0.001, DEFAULTS['delta_um'])
        self.add_slider(lf_physics, 'D_sat', 'D_sat 飽和深度(μm)', 1.0, 50.0, 1.0, DEFAULTS['D_sat'])

        # 4. ★ SCF 實驗雙參數擬合控制面板
        lf_scf = ttk.LabelFrame(scrollable_frame, text="🎯 SiO2 SCF 實驗雙參數擬合", padding="5")
        lf_scf.pack(fill=tk.X, pady=5)
        self.add_slider(lf_scf, 'exp_target_depth_um', '實驗目標深度(μm)', 1.0, 100.0, 0.5, DEFAULTS['exp_target_depth_um'])
        self.add_slider(lf_scf, 'exp_target_spot_um', '實際加工光斑(μm)', 0.2, 10.0, 0.1, DEFAULTS['exp_target_spot_um'])
        
        btn_scf = ttk.Button(lf_scf, text="🔄 執行 SiO2 M² & Delta SCF 擬合", command=self.run_scf_fitting)
        btn_scf.pack(fill=tk.X, pady=5)

        # 5. 切面與 Zoom 面板
        lf_view = ttk.LabelFrame(scrollable_frame, text="🔪 Zoom In 與視角控制", padding="5")
        lf_view.pack(fill=tk.X, pady=5)
        self.add_slider(lf_view, 'grid_res', '網格解析度', 100, 300, 25, DEFAULTS['grid_res'], is_int=True)
        self.add_slider(lf_view, 'slice_x_um', 'X切面位置(μm)', -20.0, 20.0, 0.1, DEFAULTS['slice_x_um'], command=self.on_render_only)
        self.add_slider(lf_view, 'slice_y_um', 'Y切面位置(μm)', -20.0, 20.0, 0.1, DEFAULTS['slice_y_um'], command=self.on_render_only)
        
        self.add_slider(lf_view, 'zoom_x_min', 'Zoom X Min(μm)', -100.0, 100.0, 0.5, -20.0, command=self.on_render_only)
        self.add_slider(lf_view, 'zoom_x_max', 'Zoom X Max(μm)', -100.0, 100.0, 0.5, 20.0, command=self.on_render_only)
        self.add_slider(lf_view, 'zoom_y_min', 'Zoom Y Min(μm)', -50.0, 50.0, 0.5, -10.0, command=self.on_render_only)
        self.add_slider(lf_view, 'zoom_y_max', 'Zoom Y Max(μm)', -50.0, 50.0, 0.5, 10.0, command=self.on_render_only)

        self.add_slider(lf_view, 'elev', '3D 俯角', 0, 90, 5, DEFAULTS['elev'], is_int=True, command=self.on_render_only)
        self.add_slider(lf_view, 'azim', '3D 方位', -180, 180, 5, DEFAULTS['azim'], is_int=True, command=self.on_render_only)

        self.var_show_spots = tk.BooleanVar(value=DEFAULTS['show_spots'])
        chk_spots = ttk.Checkbutton(lf_view, text="顯示軌跡與脈衝點", variable=self.var_show_spots, command=self.on_render_only)
        chk_spots.pack(anchor=tk.W, pady=2)

        btn_run = ttk.Button(scrollable_frame, text="🚀 開始模擬", command=self.run_simulation_ui)
        btn_run.pack(fill=tk.X, pady=10)

        self.lbl_status = ttk.Label(scrollable_frame, text="狀態：請點擊「🚀 開始模擬」", wraplength=300)
        self.lbl_status.pack(fill=tk.X, pady=5)

        # 右側 Matplotlib 畫面
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        self.canvas_matplotlib = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_matplotlib.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_matplotlib, plot_frame)
        toolbar.update()

    def add_slider(self, parent, key, label_text, min_val, max_val, step, default_val, is_int=False, command=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(frame, text=f"{label_text}: {default_val}")
        lbl.pack(anchor=tk.W)

        var = tk.IntVar(value=int(default_val)) if is_int else tk.DoubleVar(value=float(default_val))

        def update_lbl(*args):
            val = var.get()
            v = int(val) if is_int else round(float(val), 4)
            lbl.config(text=f"{label_text}: {v}")
            if command:
                command()

        scale = ttk.Scale(
            frame, from_=min_val, to=max_val, variable=var,
            command=update_lbl
        )
        scale.pack(fill=tk.X)
        self.widgets_dict[key] = {'var': var, 'scale': scale, 'lbl': lbl, 'is_int': is_int, 'label_text': label_text}

    def get_val(self, key):
        val = self.widgets_dict[key]['var'].get()
        return int(val) if self.widgets_dict[key]['is_int'] else float(val)

    def set_val(self, key, val):
        self.widgets_dict[key]['var'].set(val)
        v = int(val) if self.widgets_dict[key]['is_int'] else round(float(val), 4)
        self.widgets_dict[key]['lbl'].config(text=f"{self.widgets_dict[key]['label_text']}: {v}")

    def set_slider_range(self, key, min_val, max_val, current_val):
        self.widgets_dict[key]['scale'].config(from_=min_val, to=max_val)
        self.set_val(key, current_val)

    def run_simulation(self):
        """ 核心模擬推導與計算 (邏輯與 Jupyter 版 100% 一致) """
        wavelength_m = self.get_val('wavelength_nm') * 1e-9
        M2 = self.get_val('M2')
        D_m = self.get_val('input_D_mm') * 1e-3
        f_m = self.get_val('focal_length_mm') * 1e-3
        defocus_m = self.get_val('defocus_um') * 1e-6

        # 計算高斯光束焦點直徑 2w0
        w0_m = (2.0 * wavelength_m * f_m * M2) / (np.pi * D_m)
        w0_um = w0_m * 1e6
        d0_um = 2.0 * w0_um

        zR_m = (np.pi * w0_m**2) / (wavelength_m * M2)
        zR_um = zR_m * 1e6
        w_z_m = w0_m * np.sqrt(1.0 + (defocus_m / zR_m)**2)
        w_z_um = w_z_m * 1e6

        f_laser = (self.get_val('f_base_khz') * 1000.0) / float(self.get_val('divider'))
        E_p = self.get_val('P_avg_W') / f_laser
        w_z_cm = w_z_m * 100.0
        F0_z = (2.0 * E_p) / (np.pi * (w_z_cm**2))

        # ★ 推導 Effective Spot Size (依據 SiO2 燒蝕閾值計算)
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
        v_scan = self.get_val('v_scan')
        f_scan = 1e-5 if abs(v_scan) < 1e-5 else v_scan / ellipse_perimeter_mm

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

        margin_um = w_z_um * 3.5
        x_min, x_max = np.min(all_x_spots) - margin_um, np.max(all_x_spots) + margin_um
        y_min, y_max = np.min(all_y_spots) - margin_um, np.max(all_y_spots) + margin_um

        res = self.get_val('grid_res')
        x_grid_um = np.ascontiguousarray(np.linspace(x_min, x_max, res))
        y_grid_um = np.ascontiguousarray(np.linspace(y_min, y_max, res))

        self.set_slider_range('slice_x_um', float(x_min), float(x_max), float((x_min + x_max) / 2.0))
        self.set_slider_range('slice_y_um', float(y_min), float(y_max), 0.0 if (y_min <= 0 <= y_max) else float((y_min + y_max) / 2.0))

        self.set_slider_range('zoom_x_min', float(x_min), float(x_max), float(x_min))
        self.set_slider_range('zoom_x_max', float(x_min), float(x_max), float(x_max))
        self.set_slider_range('zoom_y_min', float(y_min), float(y_max), float(y_min))
        self.set_slider_range('zoom_y_max', float(y_min), float(y_max), float(y_max))

        current_depth = np.zeros((res, res), dtype=np.float64)
        pass_history = []

        for p in range(total_passes):
            effective_F_th = max(self.get_val('F_th_1') * ((p + 1) ** (self.get_val('S_inc') - 1.0)), self.get_val('F_th_1') * 0.3)
            x_p, y_p = all_pass_spots[p]

            current_depth = compute_single_pass_ablation(
                x_grid_um, y_grid_um, x_p, y_p, 
                current_depth, F0_z, effective_F_th, 
                self.get_val('delta_um'), w_z_um, zR_um, self.get_val('D_sat')
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

    def run_simulation_ui(self):
        self.lbl_status.config(text="狀態：⚡ 正在計算 SiO2 多 Pass 燒蝕與深度累積...")
        self.update_idletasks()

        try:
            self.run_simulation()
            self.render_plots()

            d0 = SIM_CACHE['d0_um']
            d_eff = SIM_CACHE['d_eff_um']
            status_msg = (f"狀態：✅ SiO2 模擬完成！\n"
                          f"Beam 2w0: {d0:.2f} μm | Effective Spot: {d_eff:.2f} μm\n"
                          f"M²: {self.get_val('M2'):.2f} | Pitch: {SIM_CACHE['pitch_stage_um']:.3f} μm")
            self.lbl_status.config(text=status_msg)

        except Exception as e:
            messagebox.showerror("模擬出錯", str(e))
            self.lbl_status.config(text="狀態：❌ 計算過程發生錯誤。")

    # ★ 6. M² & Delta 雙變數 SCF 自洽擬合邏輯
    def run_scf_fitting(self):
        target_depth = self.get_val('exp_target_depth_um')
        target_spot = self.get_val('exp_target_spot_um')

        self.lbl_status.config(text=f"狀態：🔄 啟動 SiO2 SCF 擬合 (目標: 深度={target_depth}μm, 有效痕跡={target_spot}μm)...")
        self.update_idletasks()

        tol = 0.01  # 1.0% 容許誤差

        for i in range(25):
            self.run_simulation()

            # 取底部中央 50% 區域的平均深度（代表平整底部深度）
            Total_Depth = SIM_CACHE['Total_Depth_um']
            ny, nx = Total_Depth.shape
            center_region = Total_Depth[ny//4:3*ny//4, nx//4:3*nx//4]
            sim_flat_depth = np.mean(center_region)
            sim_spot = SIM_CACHE['d_eff_um']

            if sim_spot <= 0 or sim_flat_depth <= 0:
                self.set_val('M2', max(self.get_val('M2') * 0.8, 1.0))
                self.set_val('delta_um', min(self.get_val('delta_um') * 1.5, 0.5))
                continue

            err_depth = (sim_flat_depth - target_depth) / target_depth
            err_spot = (sim_spot - target_spot) / target_spot

            # 兩者誤差皆小於門檻時結束收斂
            if abs(err_depth) < tol and abs(err_spot) < tol:
                self.render_plots()
                msg = (f"狀態：✅ SCF 於第 {i+1} 代成功收斂！\n"
                       f"SiO2 擬合結果: M² = {self.get_val('M2'):.2f}, delta = {self.get_val('delta_um'):.4f} μm")
                self.lbl_status.config(text=msg)
                return

            # 修正 M² (主導 Effective Spot Size)
            spot_ratio = target_spot / sim_spot
            new_m2 = np.clip(self.get_val('M2') * spot_ratio, 1.0, 6.0)
            self.set_val('M2', float(round(new_m2, 2)))

            # 修正 delta_um (主導總平坦深度)
            depth_ratio = target_depth / sim_flat_depth
            new_delta = np.clip(self.get_val('delta_um') * depth_ratio, 0.0001, 0.5)
            self.set_val('delta_um', float(round(new_delta, 4)))

            self.update_idletasks()

        self.render_plots()
        msg = f"狀態：⚠️ SCF 完成迭代，當前平坦深度: {sim_flat_depth:.2f} μm, 有效光斑: {SIM_CACHE['d_eff_um']:.2f} μm"
        self.lbl_status.config(text=msg)

    def on_render_only(self, *args):
        if SIM_CACHE['Total_Depth_um'] is not None:
            self.render_plots()

    def render_plots(self):
        if SIM_CACHE['Total_Depth_um'] is None:
            return

        self.fig.clf()

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

        # 1. 3D Surface & Flat Bottom Depth
        ax1.plot_surface(X, Y, -Total_Depth_um, cmap='viridis', edgecolor='none', alpha=0.95)
        ax1.view_init(elev=self.get_val('elev'), azim=self.get_val('azim'))
        
        ny, nx = Total_Depth_um.shape
        center_region = Total_Depth_um[ny//4:3*ny//4, nx//4:3*nx//4]
        flat_bottom_depth = np.mean(center_region)

        ax1.set_title(f"3D Surface (Flat Bottom Depth: {flat_bottom_depth:.2f} μm)", fontsize=8)
        ax1.set_xlabel("X (μm)", fontsize=7); ax1.set_ylabel("Y (μm)", fontsize=7); ax1.set_zlabel("Depth (μm)", fontsize=7)

        # 2. 2D Top-View
        c = ax2.contourf(X, Y, Total_Depth_um, levels=50, cmap='inferno')
        if self.var_show_spots.get():
            colors = ['cyan', 'magenta', 'lime', 'yellow', 'white']
            for p_idx, (xs, ys) in enumerate(all_pass_spots):
                xs_arr = np.atleast_1d(xs).flatten()
                ys_arr = np.atleast_1d(ys).flatten()
                if len(xs_arr) == len(ys_arr) and len(xs_arr) > 0:
                    col = colors[p_idx % len(colors)]
                    ax2.plot(xs_arr, ys_arr, color=col, linestyle='--', linewidth=0.8, alpha=0.7)
                    ax2.scatter(xs_arr, ys_arr, color='white', edgecolors='none', s=8, alpha=0.9, zorder=3)

        idx_x = int(np.clip((np.abs(x_grid_um - self.get_val('slice_x_um'))).argmin(), 0, len(x_grid_um) - 1))
        idx_y = int(np.clip((np.abs(y_grid_um - self.get_val('slice_y_um'))).argmin(), 0, len(y_grid_um) - 1))

        ax2.axhline(y_grid_um[idx_y], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.axvline(x_grid_um[idx_x], color='cyan', linestyle='--', linewidth=1.2, alpha=0.7)

        zx_min, zx_max = min(self.get_val('zoom_x_min'), self.get_val('zoom_x_max')), max(self.get_val('zoom_x_min'), self.get_val('zoom_x_max'))
        zy_min, zy_max = min(self.get_val('zoom_y_min'), self.get_val('zoom_y_max')), max(self.get_val('zoom_y_min'), self.get_val('zoom_y_max'))
        
        ax2.set_xlim(zx_min, zx_max)
        ax2.set_ylim(zy_min, zy_max)

        f_laser_khz = SIM_CACHE['f_laser'] / 1000.0
        title_str = (f"2D Top-View (Rep: {f_laser_khz:.2f} kHz | M²: {self.get_val('M2'):.2f})\n"
                     f"Beam 2w0: {SIM_CACHE['d0_um']:.2f}μm | Eff. Spot: {SIM_CACHE['d_eff_um']:.2f}μm")
        ax2.set_title(title_str, fontsize=8)
        ax2.set_xlabel("X (μm)", fontsize=7); ax2.set_ylabel("Y (μm)", fontsize=7)
        self.fig.colorbar(c, ax=ax2, label='Depth (μm)')

        # 3. X Profile
        x_prof = Total_Depth_um[idx_y, :]
        ax3.plot(x_grid_um, x_prof, 'r-', linewidth=1.5)
        ax3.set_title(f"X-Profile (at Y = {y_grid_um[idx_y]:.2f} μm)", fontsize=8)
        ax3.set_xlabel("X (μm)", fontsize=7); ax3.set_ylabel("Depth (μm)", fontsize=7)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.invert_yaxis()

        # 4. Y Profile
        y_prof = Total_Depth_um[:, idx_x]
        ax4.plot(y_grid_um, y_prof, 'c-', linewidth=1.5)
        ax4.set_title(f"Y-Profile (at X = {x_grid_um[idx_x]:.2f} μm)", fontsize=8)
        ax4.set_xlabel("Y (μm)", fontsize=7); ax4.set_ylabel("Depth (μm)", fontsize=7)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.invert_yaxis()

        # 5. Y 軸平均深度
        mean_depth_y = np.mean(Total_Depth_um, axis=1)
        ax5.plot(y_grid_um, mean_depth_y, 'g-', linewidth=1.8, label='Mean Depth')
        ax5.axvline(y_grid_um[idx_y], color='red', linestyle='--', alpha=0.5, label='Current Y Slice')
        ax5.set_title("Average Depth vs. Y-Position", fontsize=8)
        ax5.set_xlabel("Y (μm)", fontsize=7); ax5.set_ylabel("Average Depth (μm)", fontsize=7)
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.invert_yaxis()
        ax5.legend(fontsize=7)

        # 6. Pass 深度累積
        pass_depths = SIM_CACHE['pass_depth_history']
        p_range = np.arange(1, len(pass_depths) + 1, dtype=int)
        p_depths_arr = np.array(pass_depths, dtype=float)

        ax6.plot(p_range, p_depths_arr, 'bo-', linewidth=2, markersize=6, label='Simulated Depth')
        ax6.set_title("Depth vs. Passes", fontsize=8)
        ax6.set_xlabel("Passes", fontsize=7); ax6.set_ylabel("Max Depth (μm)", fontsize=7)
        ax6.grid(True, linestyle=':', alpha=0.6)
        ax6.legend(fontsize=7)

        self.fig.tight_layout()
        self.canvas_matplotlib.draw()

if __name__ == '__main__':
    app = LaserAblationApp()
    app.mainloop()
