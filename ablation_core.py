import numpy as np
from numba import njit, prange

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
