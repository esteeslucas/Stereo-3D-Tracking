import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import math
import csv

# ==============================================================================
# 1. USER PARAMETERS
# ==============================================================================

# --- MODE SELECTION ---
HEADLESS_MODE = True 

# --- INPUT SOURCES ---
CAM_SOURCE_BOTTOM = 'UT90BottomVideo1.mp4'
CAM_SOURCE_FRONT  = 'UT90TopVideo1.mp4'

# --- MARKER CONFIGURATION ---
MARKER_ID_BOTTOM = 2  
MARKER_ID_FRONT  = 3  
MARKER_SIZE = 38.0 

# --- OFFSETS ---
OFFSET_BOT_Z = 9.0  
OFFSET_FRONT_Z = 20.0 - 9.0       
OFFSET_FRONT_Y = 93.3 - 8.86/2 

# --- FILTERS ---
SMOOTHING_FACTOR = 0.1 
MAX_JUMP_POS = 20.0  # Max mm jump per frame
MAX_JUMP_ROT = 10.0  # Max degree jump per frame

# --- AXIS INVERSION FLAGS ---
INVERT_Z_HEIGHT = True   
INVERT_ROT_X = False 
INVERT_ROT_Y = False 
INVERT_ROT_Z = False 

# --- OUTPUT ---
CSV_FILENAME = '3_axis_rotation_vs_height.csv'


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

def get_euler_angles(rvec):
    """
    Converts Rotation Vector (rvec) to Euler Angles (Pitch, Yaw, Roll).
    Returns a tuple (rot_x, rot_y, rot_z) in degrees.
    """
    R, _ = cv2.Rodrigues(rvec)
    
    # Safe unpacking for different OpenCV versions
    decomp = cv2.RQDecomp3x3(R)
    euler_angles = decomp[0] # The first element is always the angles
    
    return euler_angles[0], euler_angles[1], euler_angles[2]

def apply_smoothing(current_val, new_target, alpha, max_jump):
    if current_val is None:
        return new_target
        
    # Outlier check
    if abs(new_target - current_val) > max_jump:
        return current_val
        
    # Exponential Moving Average
    return (alpha * new_target) + ((1 - alpha) * current_val)

def normalize_angle(angle):
    """Keeps angle within reasonable bounds if needed (e.g. -180 to 180)."""
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

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
    parameters.errorCorrectionRate = 0.8 
    
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    pts_bot_ref, pts_front_ref = get_object_points()
    
    # State Variables
    offset_z_height = None
    offset_rot = [0.0, 0.0, 0.0] # [x, y, z]
    
    est_z_height = 0.0
    est_rot = [0.0, 0.0, 0.0]    # [x, y, z]
    
    # Log structure: [rot_x, rot_y, rot_z, height_z]
    data_log = [] 
    frame_count = 0
    is_initialized = False

    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (sec)", "Rot_X (deg)", "Rot_Y (deg)", "Rot_Z (deg)", "Height_Z (mm)"])

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

            # Detect Markers
            corners_bot, ids_bot, _ = detector.detectMarkers(frame_bot)
            corners_front, ids_front, _ = detector.detectMarkers(frame_front)

            # --- PROCESS FRONT CAMERA (Primary for Height & Inclination) ---
            raw_z_height = None
            raw_rot = [0,0,0]
            has_front = False

            if ids_front is not None:
                for i in range(len(ids_front)):
                    if ids_front[i][0] == MARKER_ID_FRONT:
                        # Solve PnP
                        _, rvec, tvec = cv2.solvePnP(pts_front_ref, corners_front[i], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                        
                        # Extract Height (Y-component of tvec)
                        raw_z_height = tvec[1][0] 
                        
                        # Extract Rotation (All 3 axes)
                        rx, ry, rz = get_euler_angles(rvec)
                        raw_rot = [rx, ry, rz]
                        
                        has_front = True
                        
                        if not HEADLESS_MODE:
                            cv2.drawFrameAxes(frame_front, K, D, rvec, tvec, 20.0)
                            aruco.drawDetectedMarkers(frame_front, corners_front, ids_front)

            # --- INITIALIZATION (Zeroing) ---
            if not is_initialized:
                if has_front:
                    offset_z_height = raw_z_height
                    offset_rot = raw_rot # Set current orientation as "zero"
                    is_initialized = True
                    print(f"System Initialized at Frame {frame_count}")
                else:
                    if not HEADLESS_MODE:
                        cv2.putText(frame_front, "WAITING FOR LOCK...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.imshow('Front View', frame_front)
                        cv2.waitKey(1)
                    continue

            # --- DATA PROCESSING ---
            if has_front:
                # 1. Calculate Relative Z Height
                rel_z = raw_z_height - offset_z_height
                if INVERT_Z_HEIGHT: rel_z *= -1
                est_z_height = apply_smoothing(est_z_height, rel_z, SMOOTHING_FACTOR, MAX_JUMP_POS)

                # 2. Calculate Relative Rotations for X, Y, Z
                current_relative_rot = [0.0, 0.0, 0.0]
                inversions = [INVERT_ROT_X, INVERT_ROT_Y, INVERT_ROT_Z]

                for i in range(3):
                    # Relative to start
                    val = raw_rot[i] - offset_rot[i]
                    val = normalize_angle(val)
                    if inversions[i]: val *= -1
                    
                    # Smooth
                    est_rot[i] = apply_smoothing(est_rot[i], val, SMOOTHING_FACTOR, MAX_JUMP_ROT)

                # Log data
                data_log.append([est_rot[0], est_rot[1], est_rot[2], est_z_height])
                writer.writerow([f"{timestamp:.3f}", f"{est_rot[0]:.3f}", f"{est_rot[1]:.3f}", f"{est_rot[2]:.3f}", f"{est_z_height:.3f}"])

            # --- DISPLAY ---
            if not HEADLESS_MODE:
                dashboard = np.zeros((180, 400, 3), dtype=np.uint8)
                col = (0, 255, 0) if has_front else (0, 0, 255)

                cv2.putText(dashboard, f"Height (Z): {est_z_height:.2f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.putText(dashboard, f"Rot X: {est_rot[0]:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2) # Red
                cv2.putText(dashboard, f"Rot Y: {est_rot[1]:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2) # Green
                cv2.putText(dashboard, f"Rot Z: {est_rot[2]:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2) # Blue
                
                cv2.imshow('Front View', frame_front)
                cv2.imshow('Data', dashboard)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap_bot.release()
    cap_front.release()
    cv2.destroyAllWindows()
    print("Processing Complete.")

    # --- 5. PLOTTING ---
    if len(data_log) > 0:
        print(f"Plotting {len(data_log)} points...")
        data = np.array(data_log)
        
        # data[:, 0] = Rot X, data[:, 1] = Rot Y, data[:, 2] = Rot Z
        # data[:, 3] = Height Z
        
        plt.figure(figsize=(10, 8))
        
        # Plot X Rotation vs Height
        plt.plot(data[:, 0], data[:, 3], color='red', label='X-Axis Rotation', linewidth=1.5, alpha=0.8)
        
        # Plot Y Rotation vs Height
        plt.plot(data[:, 1], data[:, 3], color='green', label='Y-Axis Rotation', linewidth=1.5, alpha=0.8)
        
        # Plot Z Rotation vs Height
        plt.plot(data[:, 2], data[:, 3], color='blue', label='Z-Axis Rotation', linewidth=1.5, alpha=0.8)
        
        plt.title('3-Axis Object Inclination vs Height')
        plt.xlabel('Inclination (Degrees)')
        plt.ylabel('Z-Axis Height (mm)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
