import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import csv

# ==============================================================================
# 1. USER PARAMETERS
# ==============================================================================

# --- MODE SELECTION ---
HEADLESS_MODE = False 

# --- INPUT SOURCES ---
CAM_SOURCE_BOTTOM = 'UTB0EdditedMirrored.mp4'
CAM_SOURCE_FRONT  = 'UTT0Eddited.mp4'

# --- MARKER CONFIGURATION ---
MARKER_ID_BOTTOM = 2  
MARKER_ID_FRONT  = 3  
MARKER_SIZE = 38.0 

# --- OFFSETS ---
OFFSET_BOT_Z = 9.0  
OFFSET_FRONT_Z = 20.0 - 9.0       
OFFSET_FRONT_Y = 93.3 - 8.86/2 

# --- FILTERS ---
# Lower value = smoother but more lag. Higher = faster but jittery.
SMOOTHING_FACTOR = 0.1 
# Maximum allowed movement per frame (mm). Jumps larger than this are ignored (glitch protection).
MAX_JUMP_PER_FRAME = 20.0 

# --- AXIS INVERSION FLAGS ---
INVERT_X = True   
INVERT_Y = False
INVERT_Z = True   

# --- OUTPUT ---
CSV_FILENAME = 'tracking_data_robust.csv'


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_camera_matrices(frame_width, frame_height):
    focal_length = frame_width 
    center_x = frame_width / 2
    center_y = frame_height / 2
    K = np.array([[focal_length, 0, center_x],
                  [0, focal_length, center_y],
                  [0, 0, 1]], dtype=float)
    dist = np.zeros((5, 1)) 
    return K, dist

def get_object_points():
    ms = MARKER_SIZE / 2.0
    # Centered marker points
    obj_points_bottom = np.array([
        [-ms,  ms, OFFSET_BOT_Z], [ ms,  ms, OFFSET_BOT_Z], 
        [ ms, -ms, OFFSET_BOT_Z], [-ms, -ms, OFFSET_BOT_Z]  
    ], dtype=np.float32)
    
    obj_points_front = np.array([
        [-ms, OFFSET_FRONT_Y, OFFSET_FRONT_Z + ms], 
        [ ms, OFFSET_FRONT_Y, OFFSET_FRONT_Z + ms], 
        [ ms, OFFSET_FRONT_Y, OFFSET_FRONT_Z - ms], 
        [ ms, OFFSET_FRONT_Y, OFFSET_FRONT_Z - ms]  
    ], dtype=np.float32)
    return obj_points_bottom, obj_points_front

def apply_smoothing(current_val, new_target, alpha, max_jump):
    """
    Applies Low-Pass Filter with Outlier Rejection.
    If the new target is too far from current (glitch), we ignore it.
    """
    if current_val is None:
        return new_target
        
    # Outlier check
    if abs(new_target - current_val) > max_jump:
        # Return old value (ignore glitch)
        return current_val
        
    # Exponential Moving Average
    return (alpha * new_target) + ((1 - alpha) * current_val)

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================
def main():
    cap_bot = cv2.VideoCapture(CAM_SOURCE_BOTTOM)
    cap_front = cv2.VideoCapture(CAM_SOURCE_FRONT)

    if not cap_bot.isOpened() or not cap_front.isOpened():
        print("Error: Could not open video files.")
        return

    total_frames = int(cap_bot.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- DETECTOR CONFIGURATION ---
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
    # Increase error correction to detect markers even if slightly occluded/noisy
    parameters.errorCorrectionRate = 0.8 
    
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    pts_bot_ref, pts_front_ref = get_object_points()
    
    # State Variables
    # Offsets (Zero point)
    offset_x, offset_y, offset_z = None, None, None
    
    # Filtered positions (Current Estimates)
    # Initialize to 0.0 or None. We will start writing once initialized.
    est_x, est_y, est_z = 0.0, 0.0, 0.0
    
    trajectory = []
    frame_count = 0
    is_initialized = False

    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (sec)", "X (mm)", "Y (mm)", "Z (mm)", "Valid_Bot", "Valid_Front"])

        print(f"Processing {total_frames} frames...")

        while True:
            ret1, frame_bot = cap_bot.read()
            ret2, frame_front = cap_front.read()

            if not ret1 or not ret2:
                break
            
            frame_count += 1
            timestamp = cap_bot.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if HEADLESS_MODE and frame_count % 100 == 0:
                print(f"Progress: {(frame_count / total_frames) * 100:.1f}%")

            h, w = frame_bot.shape[:2]
            K, D = get_camera_matrices(w, h)

            corners_bot, ids_bot, _ = detector.detectMarkers(frame_bot)
            corners_front, ids_front, _ = detector.detectMarkers(frame_front)

            # --- 1. EXTRACT RAW DATA FROM CAMERAS ---
            
            # Bottom Camera Data (Provides X and Y)
            raw_bot_x, raw_bot_y = None, None
            has_bot = False
            if ids_bot is not None:
                for i in range(len(ids_bot)):
                    if ids_bot[i][0] == MARKER_ID_BOTTOM:
                        _, rvec, tvec = cv2.solvePnP(pts_bot_ref, corners_bot[i], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                        raw_bot_x = tvec[0][0]
                        raw_bot_y = tvec[1][0]
                        has_bot = True
                        if not HEADLESS_MODE:
                            cv2.drawFrameAxes(frame_bot, K, D, rvec, tvec, 20.0)
                            aruco.drawDetectedMarkers(frame_bot, corners_bot, ids_bot)

            # Front Camera Data (Provides X and Z)
            raw_front_x, raw_front_z = None, None
            has_front = False
            if ids_front is not None:
                for i in range(len(ids_front)):
                    if ids_front[i][0] == MARKER_ID_FRONT:
                        _, rvec, tvec = cv2.solvePnP(pts_front_ref, corners_front[i], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                        raw_front_x = tvec[0][0]
                        raw_front_z = tvec[1][0]
                        has_front = True
                        if not HEADLESS_MODE:
                            cv2.drawFrameAxes(frame_front, K, D, rvec, tvec, 20.0)
                            aruco.drawDetectedMarkers(frame_front, corners_front, ids_front)

            # --- 2. INITIALIZATION (Zeroing) ---
            # We need ONE good frame with BOTH cameras to set the zero point correctly.
            if not is_initialized:
                if has_bot and has_front:
                    # Initialize offsets
                    offset_x = (raw_bot_x + raw_front_x) / 2.0 # Average starting X
                    offset_y = raw_bot_y
                    offset_z = raw_front_z
                    is_initialized = True
                    print(f"System Initialized at Frame {frame_count}")
                else:
                    # Wait for good lock
                    if not HEADLESS_MODE:
                        cv2.putText(frame_bot, "WAITING FOR LOCK...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.imshow('Bottom View', frame_bot)
                        cv2.imshow('Front View', frame_front)
                        cv2.waitKey(1)
                    continue

            # --- 3. SENSOR FUSION & FILTERING ---
            
            # A. Process X (Available in both)
            x_candidates = []
            if has_bot: x_candidates.append(raw_bot_x)
            if has_front: x_candidates.append(raw_front_x)
            
            if len(x_candidates) > 0:
                # Average available sources
                avg_raw_x = sum(x_candidates) / len(x_candidates)
                rel_x = avg_raw_x - offset_x
                if INVERT_X: rel_x *= -1
                est_x = apply_smoothing(est_x, rel_x, SMOOTHING_FACTOR, MAX_JUMP_PER_FRAME)
            else:
                # NO DATA: Hold last value (Zero Order Hold)
                pass 

            # B. Process Y (Available only in Bottom)
            if has_bot:
                rel_y = raw_bot_y - offset_y
                if INVERT_Y: rel_y *= -1
                est_y = apply_smoothing(est_y, rel_y, SMOOTHING_FACTOR, MAX_JUMP_PER_FRAME)
            else:
                # NO DATA: Hold last value
                pass

            # C. Process Z (Available only in Front)
            if has_front:
                rel_z = raw_front_z - offset_z
                if INVERT_Z: rel_z *= -1
                est_z = apply_smoothing(est_z, rel_z, SMOOTHING_FACTOR, MAX_JUMP_PER_FRAME)
            else:
                # NO DATA: Hold last value
                pass

            # --- 4. DATA LOGGING & DISPLAY ---
            trajectory.append([est_x, est_y, est_z])
            writer.writerow([f"{timestamp:.3f}", f"{est_x:.3f}", f"{est_y:.3f}", f"{est_z:.3f}", has_bot, has_front])

            if not HEADLESS_MODE:
                dashboard = np.zeros((220, 400, 3), dtype=np.uint8)
                
                # Status Colors
                col_x = (0, 255, 0) if len(x_candidates) > 0 else (0, 0, 255)
                col_y = (0, 255, 0) if has_bot else (0, 0, 255) # Red if holding value
                col_z = (0, 255, 0) if has_front else (0, 0, 255)

                cv2.putText(dashboard, f"X: {est_x:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_x, 2)
                cv2.putText(dashboard, f"Y: {est_y:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_y, 2)
                cv2.putText(dashboard, f"Z: {est_z:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_z, 2)
                
                status_text = "TRACKING"
                if not has_bot and not has_front: status_text = "LOST - HOLDING"
                elif not has_bot or not has_front: status_text = "PARTIAL"
                
                cv2.putText(dashboard, f"Status: {status_text}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow('Bottom View', frame_bot)
                cv2.imshow('Front View', frame_front)
                cv2.imshow('Results', dashboard)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap_bot.release()
    cap_front.release()
    cv2.destroyAllWindows()
    print("Processing Complete.")

    # --- 5. PLOTTING ---
    if len(trajectory) > 0:
        
        print(f"Plotting {len(trajectory)} points...")
        data = np.array(trajectory)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:,0], data[:,1], data[:,2], label='Fused Path', color='blue', linewidth=2)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    main()
