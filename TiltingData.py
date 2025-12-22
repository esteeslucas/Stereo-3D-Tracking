import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, CheckButtons
import re

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CSV_FILENAME = 'tracking_data_robust.csv'
GCODE_FILENAME = 'file.gcode'
OUTPUT_FILENAME = 'processed_trajectory.csv'
RESAMPLE_COUNT = 2000     
DTW_CALC_RES = 600       

# ==============================================================================
# 2. DATA LOADING & PARSING
# ==============================================================================
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} rows from CSV.")
        return df
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return None

def parse_gcode(filename):
    points = []
    current_pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        print(f"Parsing G-code '{filename}'...")
        
        for line in lines:
            line = line.split(';')[0].strip().upper()
            if not line: continue
            
            if line.startswith('G0') or line.startswith('G1'):
                has_move = False
                for axis in ['X', 'Y', 'Z']:
                    match = re.search(f'{axis}([-+]?[0-9]*\.?[0-9]+)', line)
                    if match:
                        current_pos[axis] = float(match.group(1))
                        has_move = True
                if has_move:
                    points.append([current_pos['X'], current_pos['Y'], current_pos['Z']])
                    
        if len(points) == 0:
            print("Warning: No G0/G1 coordinates found in G-code.")
            return None

        pts_array = np.array(points)
        pts_array = pts_array - pts_array[0] 
        print(f"Loaded {len(pts_array)} G-code points.")
        return pts_array
        
    except FileNotFoundError:
        print(f"Warning: '{filename}' not found. G-code overlay disabled.")
        return None

# ==============================================================================
# 3. MATH & DTW FUNCTIONS
# ==============================================================================
def smooth_data(data, window_size):
    if window_size <= 1: return data
    s = pd.Series(data)
    return s.rolling(window=int(window_size), min_periods=1, center=True).mean().to_numpy()

def rotate_points(points, angle_x, angle_y, angle_z):
    rad_x, rad_y, rad_z = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    Rx = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx @ points.T).T

def resample_path_by_length(points, num_points):
    if len(points) < 2:
        return np.resize(points, (num_points, 3))

    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative_dist = np.zeros(len(points))
    cumulative_dist[1:] = np.cumsum(dists)
    
    total_len = cumulative_dist[-1]
    if total_len == 0:
        return np.resize(points, (num_points, 3))
        
    normalized_dist = cumulative_dist / total_len
    target_dist = np.linspace(0, 1, num_points)
    
    new_points = np.zeros((num_points, 3))
    for dim in range(3): # X, Y, Z
        new_points[:, dim] = np.interp(target_dist, normalized_dist, points[:, dim])
        
    return new_points

def compute_dtw_numpy(seq_a, seq_b):
    N, M = len(seq_a), len(seq_b)
    dist_mat = np.linalg.norm(seq_a[:, None, :] - seq_b[None, :, :], axis=2)
    cost_mat = np.zeros((N, M))
    cost_mat[0, 0] = dist_mat[0, 0]
    
    for i in range(1, N): cost_mat[i, 0] = cost_mat[i-1, 0] + dist_mat[i, 0]
    for j in range(1, M): cost_mat[0, j] = cost_mat[0, j-1] + dist_mat[0, j]
        
    for i in range(1, N):
        for j in range(1, M):
            cost_mat[i, j] = dist_mat[i, j] + min(cost_mat[i-1, j], cost_mat[i, j-1], cost_mat[i-1, j-1])
            
    path = []
    i, j = N-1, M-1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            min_val = min(cost_mat[i-1, j], cost_mat[i, j-1], cost_mat[i-1, j-1])
            if min_val == cost_mat[i-1, j-1]: i, j = i-1, j-1
            elif min_val == cost_mat[i-1, j]: i -= 1
            else: j -= 1
        path.append((i, j))
        
    return np.array(path[::-1])

# ==============================================================================
# 4. MAIN VISUALIZER
# ==============================================================================
def main():
    # 1. Load Data
    df = load_data(CSV_FILENAME)
    if df is None: return
    gcode_points = parse_gcode(GCODE_FILENAME)
    
    # Raw Recorded Data
    raw_x = df['X (mm)'].to_numpy()
    raw_y = df['Y (mm)'].to_numpy()
    raw_z = df['Z (mm)'].to_numpy()
    raw_time = df['Time (sec)'].to_numpy()
    total_points = len(raw_x)

    # State
    state = {
        'current_points': None,
        'current_time': None,
        'active_gcode': gcode_points, 
        'heatmap_line': None, 
        'colorbar': None,
        'resampled_track': None, 
        'resampled_ref': None,
        'resampled_errors': None,
        'resampled_diffs': None,
        'is_calculated': False
    }

    # Setup Figure
    fig = plt.figure(figsize=(16, 10))
    plt.subplots_adjust(bottom=0.28, left=0.05, right=0.95, top=0.95)
    ax = fig.add_subplot(111, projection='3d')

    # --- PLOT OBJECTS ---
    if gcode_points is not None:
        gcode_line_obj, = ax.plot(gcode_points[:,0], gcode_points[:,1], gcode_points[:,2], 
                color='orange', linestyle='--', linewidth=1.5, label='G-code')
    else:
        gcode_line_obj = None

    line, = ax.plot([], [], [], lw=2.0, color='blue', label='Recorded Path')
    
    start_scatter = ax.scatter([], [], [], color='green', s=60, label='Start')
    end_scatter = ax.scatter([], [], [], color='red', s=60, label='End')

    # SELECTION MARKERS
    click_marker_track = ax.scatter([], [], [], s=100, color='black', marker='x', zorder=100)
    click_marker_ref = ax.scatter([], [], [], s=100, color='black', marker='o', zorder=100)
    click_connector, = ax.plot([], [], [], color='black', linestyle=':', linewidth=1)
    
    # Top-Left: Error Info
    error_text = ax.text2D(0.02, 0.97, "Click 'Calc Error' to Resample & Fit", transform=ax.transAxes, 
                           fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.9),
                           horizontalalignment='left', verticalalignment='top')
    
    # --- TABLE RIGHT (Inspection) ---
    ax_table = plt.axes([0.76, 0.82, 0.22, 0.15]) 
    ax_table.axis('off')
    col_labels = ['Ref (mm)', 'Track (mm)', 'Error (mm)']
    row_labels = ['X', 'Y', 'Z', 'Total']
    table_data = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    
    the_table = ax_table.table(cellText=table_data,
                               rowLabels=row_labels,
                               colLabels=col_labels,
                               loc='center',
                               cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)

    # --- TABLE LEFT (Statistics) ---
    ax_table_left = plt.axes([0.02, 0.82, 0.18, 0.15]) 
    ax_table_left.axis('off')
    col_labels_left = ['Mean Abs Error']
    row_labels_left = ['X', 'Y', 'Z', 'Total']
    table_data_left = [['-'], ['-'], ['-'], ['-']]
    
    the_table_left = ax_table_left.table(cellText=table_data_left,
                                         rowLabels=row_labels_left,
                                         colLabels=col_labels_left,
                                         loc='center',
                                         cellLoc='center')
    the_table_left.auto_set_font_size(False)
    the_table_left.set_fontsize(9)
    the_table_left.scale(1, 1.5)

    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)
    ax.set_zlabel('Z (mm)', fontsize=9)
    ax.set_title('Trajectory Alignment Tool (DTW)', fontsize=11)
    ax.view_init(elev=30, azim=-60)

    # --- UI LAYOUT ---
    y_start = 0.22
    y_step = 0.025
    slider_height = 0.015
    
    ax_trim_g = plt.axes([0.15, y_start, 0.55, slider_height])
    ax_trim_s = plt.axes([0.15, y_start - 1*y_step, 0.55, slider_height])
    ax_trim_e = plt.axes([0.15, y_start - 2*y_step, 0.55, slider_height])
    ax_smooth = plt.axes([0.15, y_start - 3*y_step, 0.55, slider_height])
    ax_rot_x  = plt.axes([0.15, y_start - 4*y_step, 0.55, slider_height])
    ax_rot_y  = plt.axes([0.15, y_start - 5*y_step, 0.55, slider_height])
    ax_rot_z  = plt.axes([0.15, y_start - 6*y_step, 0.55, slider_height])
    ax_index  = plt.axes([0.15, y_start - 7*y_step, 0.55, slider_height])

    # Sliders
    gc_max = len(gcode_points) if gcode_points is not None else 100
    s_trim_g = Slider(ax_trim_g, 'Trim G-Start', 0, gc_max-2, valinit=0, valstep=1, color='gold')
    s_trim_s = Slider(ax_trim_s, 'Rec Start', 0, total_points-10, valinit=0, valstep=1, color='orange')
    s_trim_e = Slider(ax_trim_e, 'Rec End', 0, total_points-10, valinit=0, valstep=1, color='red')
    s_smooth = Slider(ax_smooth, 'Filter', 1, 50, valinit=1, valstep=1, color='green')
    s_rot_x  = Slider(ax_rot_x, 'Tilt X', -90, 90, valinit=0, valstep=0.5, color='blue')
    s_rot_y  = Slider(ax_rot_y, 'Tilt Y', -90, 90, valinit=0, valstep=0.5, color='blue')
    s_rot_z  = Slider(ax_rot_z, 'Tilt Z', -180, 180, valinit=0, valstep=0.5, color='purple')
    s_index  = Slider(ax_index, 'Inspect Pt', 0, RESAMPLE_COUNT-1, valinit=0, valstep=1, color='black')

    for s in [s_trim_g, s_trim_s, s_trim_e, s_smooth, s_rot_x, s_rot_y, s_rot_z, s_index]:
        s.label.set_fontsize(8)
        s.valtext.set_fontsize(8)

    ax_check = plt.axes([0.02, 0.05, 0.08, 0.12])
    check = CheckButtons(ax_check, ['Inv X', 'Inv Y', 'Inv Z'], [False, False, False])
    for l in check.labels: l.set_fontsize(8)

    # --- UPDATED VISIBILITY CHECKBOXES ---
    ax_vis = plt.axes([0.85, 0.12, 0.10, 0.08])
    # Added 'Heatmap' to the list
    check_vis = CheckButtons(ax_vis, ['G-code', 'Rec', 'Heatmap'], [True, True, True])
    for l in check_vis.labels: l.set_fontsize(8)

    ax_calc = plt.axes([0.85, 0.065, 0.10, 0.04])
    btn_calc = Button(ax_calc, 'Calc Error', color='gold', hovercolor='yellow')
    btn_calc.label.set_fontsize(9)

    ax_save = plt.axes([0.85, 0.015, 0.10, 0.04])
    btn_save = Button(ax_save, 'Save CSV', color='lightblue', hovercolor='0.9')
    btn_save.label.set_fontsize(9)

    ax_reset = plt.axes([0.02, 0.015, 0.06, 0.03])
    btn_reset = Button(ax_reset, 'Reset', hovercolor='0.975')
    btn_reset.label.set_fontsize(8)

    # --- LOGIC ---
    
    def toggle_visibility(event):
        """New function to handle visibility without re-calculating data."""
        show_gcode, show_rec, show_heatmap = check_vis.get_status()
        
        # 1. G-Code Visibility
        if gcode_line_obj:
            gcode_line_obj.set_visible(show_gcode)
            
        # 2. Recorded Path (Blue Line) Visibility
        # If Heatmap is active, usually we hide Rec path, but we respect the checkbox
        line.set_visible(show_rec)
        
        # 3. Heatmap Visibility (only if calculation exists)
        if state['heatmap_line']:
            state['heatmap_line'].set_visible(show_heatmap)
            
        fig.canvas.draw_idle()

    def update_markers(idx):
        if not state['is_calculated']: return
        
        track_pts = state['resampled_track']
        ref_pts = state['resampled_ref']
        errors = state['resampled_errors']
        diffs = state['resampled_diffs']
        
        if track_pts is None: return

        idx = min(idx, len(track_pts)-1)
        idx = max(idx, 0)

        p_track = track_pts[idx]
        p_ref = ref_pts[idx]
        err_total = errors[idx]
        d = diffs[idx] # [dx, dy, dz]
        
        # Update 3D Markers
        click_marker_track._offsets3d = ([p_track[0]], [p_track[1]], [p_track[2]])
        click_marker_ref._offsets3d = ([p_ref[0]], [p_ref[1]], [p_ref[2]])
        click_connector.set_data([p_track[0], p_ref[0]], [p_track[1], p_ref[1]])
        click_connector.set_3d_properties([p_track[2], p_ref[2]])
        
        # --- UPDATE RIGHT TABLE ---
        the_table[(1, 0)].get_text().set_text(f"{p_ref[0]:.2f}")
        the_table[(1, 1)].get_text().set_text(f"{p_track[0]:.2f}")
        the_table[(1, 2)].get_text().set_text(f"{d[0]:.2f}")
        
        the_table[(2, 0)].get_text().set_text(f"{p_ref[1]:.2f}")
        the_table[(2, 1)].get_text().set_text(f"{p_track[1]:.2f}")
        the_table[(2, 2)].get_text().set_text(f"{d[1]:.2f}")
        
        the_table[(3, 0)].get_text().set_text(f"{p_ref[2]:.2f}")
        the_table[(3, 1)].get_text().set_text(f"{p_track[2]:.2f}")
        the_table[(3, 2)].get_text().set_text(f"{d[2]:.2f}")
        
        the_table[(4, 0)].get_text().set_text("-")
        the_table[(4, 1)].get_text().set_text("-")
        the_table[(4, 2)].get_text().set_text(f"{err_total:.2f}")

        fig.canvas.draw_idle()

    def update_slider_change(val):
        idx = int(s_index.val)
        update_markers(idx)

    def update(val):
        # NOTE: This function resets the calculation because geometry changed.
        if state['heatmap_line']:
            state['heatmap_line'].remove()
            state['heatmap_line'] = None
        if state['colorbar']:
            state['colorbar'].remove()
            state['colorbar'] = None
        
        state['is_calculated'] = False

        click_marker_track._offsets3d = ([], [], [])
        click_marker_ref._offsets3d = ([], [], [])
        click_connector.set_data([], [])
        click_connector.set_3d_properties([])
        
        # Reset tables
        for r in range(1, 5):
            for c in range(3):
                the_table[(r, c)].get_text().set_text("-")
            the_table_left[(r, 0)].get_text().set_text("-")

        # --- 1. Process G-Code (Reference) ---
        gc_center_x, gc_center_y = 0.0, 0.0
        
        if gcode_points is not None:
            g_start = int(s_trim_g.val)
            if g_start >= len(gcode_points): g_start = len(gcode_points) - 1
            trimmed_gcode = gcode_points[g_start:]
            
            if len(trimmed_gcode) > 0:
                # Force G-code to start at 0,0,0
                trimmed_gcode = trimmed_gcode - trimmed_gcode[0]
                
                # Calculate G-code Center (XY) for alignment
                gc_min = np.min(trimmed_gcode, axis=0)
                gc_max = np.max(trimmed_gcode, axis=0)
                gc_center_x = (gc_min[0] + gc_max[0]) / 2
                gc_center_y = (gc_min[1] + gc_max[1]) / 2
            
            state['active_gcode'] = trimmed_gcode
            if gcode_line_obj:
                gcode_line_obj.set_data_3d(trimmed_gcode[:,0], trimmed_gcode[:,1], trimmed_gcode[:,2])

        # --- 2. Process Recorded Path ---
        trim_s = int(s_trim_s.val)
        trim_e = int(s_trim_e.val)
        w_size = s_smooth.val
        deg_x, deg_y, deg_z = s_rot_x.val, s_rot_y.val, s_rot_z.val
        inv_x, inv_y, inv_z = check.get_status()
        
        # We manually call toggle_visibility logic here to ensure lines are correct
        # but we don't call the function directly as it needs an event
        show_gcode, show_rec, _ = check_vis.get_status()
        if gcode_line_obj: gcode_line_obj.set_visible(show_gcode)
        line.set_visible(show_rec)

        if trim_s + trim_e >= total_points: return 
        
        end_idx = total_points - trim_e
        curr_x = raw_x[trim_s : end_idx]
        curr_y = raw_y[trim_s : end_idx]
        curr_z = raw_z[trim_s : end_idx]
        curr_t = raw_time[trim_s : end_idx]

        if inv_x: curr_x = curr_x * -1
        if inv_y: curr_y = curr_y * -1
        if inv_z: curr_z = curr_z * -1

        # A. Local Normalization (for intuitive rotation)
        curr_x = curr_x - curr_x[0]
        curr_y = curr_y - curr_y[0]
        curr_z = curr_z - curr_z[0]

        sx = smooth_data(curr_x, w_size)
        sy = smooth_data(curr_y, w_size)
        sz = smooth_data(curr_z, w_size)
        
        # B. Apply Rotation
        smoothed_points = np.column_stack((sx, sy, sz))
        final_points = rotate_points(smoothed_points, deg_x, deg_y, deg_z)
        
        # C. Re-Zero Z (User Requirement: Z starts at 0)
        final_points[:, 2] = final_points[:, 2] - final_points[0, 2]
        
        # D. Center XY Alignment (User Requirement: Center on G-code square)
        if gcode_points is not None and len(state['active_gcode']) > 0:
            rec_min = np.min(final_points, axis=0)
            rec_max = np.max(final_points, axis=0)
            rec_center_x = (rec_min[0] + rec_max[0]) / 2
            rec_center_y = (rec_min[1] + rec_max[1]) / 2
            
            # Shift Rec Center to match G-code Center
            offset_x = gc_center_x - rec_center_x
            offset_y = gc_center_y - rec_center_y
            
            final_points[:, 0] += offset_x
            final_points[:, 1] += offset_y
        
        state['current_points'] = final_points
        state['current_time'] = curr_t

        error_text.set_text("Modified.\nClick 'Calc Error' to run DTW.")
        error_text.set_color('gray')

        xs, ys, zs = final_points[:,0], final_points[:,1], final_points[:,2]
        line.set_data_3d(xs, ys, zs)
        
        start_scatter._offsets3d = ([xs[0]], [ys[0]], [zs[0]])
        end_scatter._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])

        # Auto-Scale logic
        all_x, all_y, all_z = xs, ys, zs
        if gcode_points is not None and show_gcode:
            active_gc = state['active_gcode']
            if len(active_gc) > 0:
                all_x = np.concatenate((xs, active_gc[:,0]))
                all_y = np.concatenate((ys, active_gc[:,1]))
                all_z = np.concatenate((zs, active_gc[:,2]))
            
        margin = 5.0
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_zlim(all_z.min() - margin, all_z.max() + margin)

        fig.canvas.draw_idle()

    def calculate_error_click(event):
        final_points = state['current_points']
        active_gc = state['active_gcode']
        
        if final_points is None: return
        if active_gc is None or len(active_gc) < 2:
            error_text.set_text("G-code Empty or Not Loaded")
            return

        error_text.set_text("Calculating DTW... (Please wait)")
        plt.pause(0.01)
            
        track_high = resample_path_by_length(final_points, RESAMPLE_COUNT)
        ref_high = resample_path_by_length(active_gc, RESAMPLE_COUNT) 
        
        track_low = resample_path_by_length(final_points, DTW_CALC_RES)
        ref_low = resample_path_by_length(active_gc, DTW_CALC_RES)
        
        path_low = compute_dtw_numpy(track_low, ref_low)
        
        x_indices = path_low[:, 0]
        y_indices = path_low[:, 1]
        unique_x, unique_indices = np.unique(x_indices, return_index=True)
        mapped_y = y_indices[unique_indices]
        high_res_query = np.linspace(0, DTW_CALC_RES-1, RESAMPLE_COUNT)
        low_res_ref_float = np.interp(high_res_query, unique_x, mapped_y)
        scale_factor = RESAMPLE_COUNT / DTW_CALC_RES
        high_res_ref_indices = np.clip(np.round(low_res_ref_float * scale_factor).astype(int), 0, RESAMPLE_COUNT-1)
        
        matched_ref_points = ref_high[high_res_ref_indices]
        
        diffs = track_high - matched_ref_points 
        errors = np.linalg.norm(diffs, axis=1)
        
        # --- CALCULATE STATISTICS ---
        mae_x = np.mean(np.abs(diffs[:, 0]))
        mae_y = np.mean(np.abs(diffs[:, 1]))
        mae_z = np.mean(np.abs(diffs[:, 2]))
        mean_total = np.mean(errors)
        
        # --- UPDATE LEFT TABLE ---
        the_table_left[(1, 0)].get_text().set_text(f"{mae_x:.2f}")
        the_table_left[(2, 0)].get_text().set_text(f"{mae_y:.2f}")
        the_table_left[(3, 0)].get_text().set_text(f"{mae_z:.2f}")
        the_table_left[(4, 0)].get_text().set_text(f"{mean_total:.2f}")

        # SAVE DATA
        state['resampled_track'] = track_high
        state['resampled_ref'] = matched_ref_points
        state['resampled_errors'] = errors
        state['resampled_diffs'] = diffs 
        state['is_calculated'] = True
        
        error_text.set_text(f"Mean: {mean_total:.2f}mm\n(DTW Aligned)")
        error_text.set_color('black')
        
        # Check user visibility preferences
        _, show_rec, show_heatmap = check_vis.get_status()
        
        line.set_visible(show_rec)
        
        points_reshaped = track_high.reshape(-1, 1, 3)
        segments = np.concatenate([points_reshaped[:-1], points_reshaped[1:]], axis=1)
        norm = plt.Normalize(0, np.max(errors))
        lc = Line3DCollection(segments, cmap=plt.cm.RdYlGn_r, norm=norm, picker=5)
        lc.set_array(errors[:-1]) 
        lc.set_linewidth(2)
        lc.set_visible(show_heatmap) # Apply visibility based on checkbox
        ax.add_collection(lc)
        
        state['heatmap_line'] = lc
        
        if state['colorbar']: state['colorbar'].remove()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Error (mm)')
        state['colorbar'] = cbar
        
        update_markers(int(s_index.val))

        fig.canvas.draw_idle()

    def on_pick(event):
        if event.artist != state['heatmap_line']:
            return
        idx = event.ind[0]
        s_index.set_val(idx) 

    def save_config(event):
        if state['is_calculated']:
            ref = state['resampled_ref']
            track = state['resampled_track']
            errs = state['resampled_errors']
            diffs = state['resampled_diffs']
            
            df_out = pd.DataFrame({
                'Sample_Index': np.arange(len(ref)),
                'Ref_X': ref[:,0], 'Ref_Y': ref[:,1], 'Ref_Z': ref[:,2],
                'Track_X': track[:,0], 'Track_Y': track[:,1], 'Track_Z': track[:,2],
                'Err_X': diffs[:,0], 'Err_Y': diffs[:,1], 'Err_Z': diffs[:,2],
                'Total_Error_mm': errs
            })
            print(f"✅ Saved DTW aligned data to '{OUTPUT_FILENAME}'")
        else:
            if state['current_points'] is None: return
            pts = state['current_points']
            df_out = pd.DataFrame({
                'Time (sec)': state['current_time'],
                'X (mm)': pts[:,0], 'Y (mm)': pts[:,1], 'Z (mm)': pts[:,2]
            })
            print(f"✅ Saved MODIFIED RAW data to '{OUTPUT_FILENAME}'")
            
        df_out.to_csv(OUTPUT_FILENAME, index=False)

    def reset(event):
        s_trim_g.reset()
        s_trim_s.reset()
        s_trim_e.reset()
        s_smooth.reset()
        s_rot_x.reset()
        s_rot_y.reset()
        s_rot_z.reset()
        s_index.reset()
        update(None)

    s_trim_g.on_changed(update)
    s_trim_s.on_changed(update)
    s_trim_e.on_changed(update)
    s_smooth.on_changed(update)
    s_rot_x.on_changed(update)
    s_rot_y.on_changed(update)
    s_rot_z.on_changed(update)
    s_index.on_changed(update_slider_change)
    check.on_clicked(update)
    
    # Wired to toggle_visibility instead of update so we don't lose calculated data
    check_vis.on_clicked(toggle_visibility) 
    
    btn_save.on_clicked(save_config)
    btn_calc.on_clicked(calculate_error_click)
    btn_reset.on_clicked(reset)
    
    fig.canvas.mpl_connect('pick_event', on_pick)

    update(None)
    plt.show()

if __name__ == "__main__":
    main()
