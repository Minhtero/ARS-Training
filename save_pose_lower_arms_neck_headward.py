# save_pose_lower_arms_neck_headward.py
import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np

# --------- CẤU HÌNH ----------
CAMERA_INDEX = 0
OUTPUT_VIDEO = r"D:\2 Code\0 Computer Vision\Coding\test2\mediasquat_lower_arms_neck_headward.mp4"
SAVE_LANDMARKS = True
LANDMARK_CSV = r"D:\2 Code\0 Computer Vision\Coding\test2\landmarks_lower_arms_neck_headward.csv"
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 0
SHOW_WINDOW = True
PROCESS_EVERY_N_FRAMES = 1
# -----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

SELECTED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

LANDMARK_GROUP = {
    'arm': {'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST'},
    'leg': {'LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX'},
    'torso': {'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_HIP','RIGHT_HIP'},
    'head': {'NOSE', 'HEAD_TOP'}
}

CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Không thể mở camera index={CAMERA_INDEX}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

csv_file = None
csv_writer = None
if SAVE_LANDMARKS:
    os.makedirs(os.path.dirname(LANDMARK_CSV) or ".", exist_ok=True)
    csv_file = open(LANDMARK_CSV, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "lm_name", "lm_index", "group", "x_norm", "y_norm", "z_norm", "x_px", "y_px"])

def safe_get(results, enum):
    try:
        return results.pose_landmarks.landmark[enum]
    except Exception:
        return None

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=MODEL_COMPLEXITY,
                  min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                  min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được frame, thoát.")
                break

            do_process = (frame_idx % PROCESS_EVERY_N_FRAMES) == 0
            annotated = frame.copy()

            neck_x = neck_y = neck_z = None
            head_top_x = head_top_y = head_top_z = None

            if do_process:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = pose.process(img_rgb)
                img_rgb.flags.writeable = True

                if results.pose_landmarks:
                    l_sh = safe_get(results, mp_pose.PoseLandmark.LEFT_SHOULDER)
                    r_sh = safe_get(results, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                    nose = safe_get(results, mp_pose.PoseLandmark.NOSE)
                    l_hip = safe_get(results, mp_pose.PoseLandmark.LEFT_HIP)
                    r_hip = safe_get(results, mp_pose.PoseLandmark.RIGHT_HIP)

                    # compute neck biased toward head (nose) when available
                    if l_sh and r_sh and nose:
                        mid_sh_x = (l_sh.x + r_sh.x) / 2.0
                        mid_sh_y = (l_sh.y + r_sh.y) / 2.0
                        mid_sh_z = (getattr(l_sh,"z",0.0)+getattr(r_sh,"z",0.0))/2.0
                        # neck = 60% shoulders midpoint + 40% nose  => moves neck upward toward the head
                        neck_x = 0.6*mid_sh_x + 0.4*float(nose.x)
                        neck_y = 0.6*mid_sh_y + 0.4*float(nose.y)
                        neck_z = 0.6*mid_sh_z + 0.4*float(getattr(nose,"z",0.0))
                    elif l_sh and r_sh:
                        # no nose: fallback to midpoint between shoulders
                        neck_x = (l_sh.x + r_sh.x) / 2.0
                        neck_y = (l_sh.y + r_sh.y) / 2.0
                        neck_z = (getattr(l_sh,"z",0.0)+getattr(r_sh,"z",0.0))/2.0
                    elif (l_sh or r_sh) and nose:
                        # single shoulder + nose: average shoulder and nose
                        sh = l_sh if l_sh else r_sh
                        neck_x = 0.6*float(sh.x) + 0.4*float(nose.x)
                        neck_y = 0.6*float(sh.y) + 0.4*float(nose.y)
                        neck_z = 0.6*getattr(sh,"z",0.0) + 0.4*getattr(nose,"z",0.0)
                    elif nose and (l_hip and r_hip):
                        # no shoulders but hips + nose -> midpoint hips toward nose
                        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
                        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
                        neck_x = 0.3*mid_hip_x + 0.7*float(nose.x)  # lean more to nose
                        neck_y = 0.3*mid_hip_y + 0.7*float(nose.y)
                        neck_z = 0.3*(getattr(l_hip,"z",0.0)+getattr(r_hip,"z",0.0))/2.0 + 0.7*getattr(nose,"z",0.0)
                    else:
                        neck_x = neck_y = neck_z = None

                    # compute head_top point above nose
                    if nose and neck_x is not None:
                        # head_top = nose + vector from neck to nose, extended upward
                        neck_to_nose_x = float(nose.x) - neck_x
                        neck_to_nose_y = float(nose.y) - neck_y
                        neck_to_nose_z = float(getattr(nose,"z",0.0)) - neck_z
                        
                        # Extend the vector upward by 1.5x the neck-to-nose distance
                        head_top_x = float(nose.x) + 1.5 * neck_to_nose_x
                        head_top_y = float(nose.y) + 1.5 * neck_to_nose_y
                        head_top_z = float(getattr(nose,"z",0.0)) + 1.5 * neck_to_nose_z
                    elif nose:
                        # If no neck, estimate head_top above nose
                        head_top_x = float(nose.x)
                        head_top_y = float(nose.y) - 0.1  # Move up by 10% of frame height
                        head_top_z = float(getattr(nose,"z",0.0))
                    else:
                        head_top_x = head_top_y = head_top_z = None

                    # draw selected landmarks (no head points)
                    for lm_enum in SELECTED_LANDMARKS:
                        lm = safe_get(results, lm_enum)
                        if lm is None:
                            continue
                        x_norm, y_norm = float(lm.x), float(lm.y)
                        x_px = int(x_norm * width + 0.5)
                        y_px = int(y_norm * height + 0.5)
                        name = lm_enum.name
                        if name in LANDMARK_GROUP['leg']:
                            color = (0,160,255)
                        elif name in LANDMARK_GROUP['arm']:
                            color = (0,220,0)
                        elif name in LANDMARK_GROUP['torso']:
                            color = (0,180,255)
                        else:
                            color = (200,200,200)
                        cv2.circle(annotated, (x_px, y_px), 4, color, -1)
                        cv2.putText(annotated, name.split('_')[-1].lower(), (x_px+4, y_px-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # draw neck if available
                    if neck_x is not None:
                        neck_px = (int(neck_x * width + 0.5), int(neck_y * height + 0.5))
                        cv2.circle(annotated, neck_px, 5, (255,0,0), -1)
                        cv2.putText(annotated, "neck", (neck_px[0]+4, neck_px[1]-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)

                    # draw head_top if available
                    if head_top_x is not None:
                        head_top_px = (int(head_top_x * width + 0.5), int(head_top_y * height + 0.5))
                        cv2.circle(annotated, head_top_px, 6, (255,255,0), -1)  # Yellow color for head_top
                        cv2.putText(annotated, "head_top", (head_top_px[0]+4, head_top_px[1]-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

                    # draw connections
                    for a_enum, b_enum in CONNECTIONS:
                        a = safe_get(results, a_enum)
                        b = safe_get(results, b_enum)
                        if a is None or b is None:
                            continue
                        xa = int(a.x * width + 0.5); ya = int(a.y * height + 0.5)
                        xb = int(b.x * width + 0.5); yb = int(b.y * height + 0.5)
                        if a_enum in (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                                      mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                                      mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX):
                            line_color = (200,100,0)
                        else:
                            line_color = (180,180,180)
                        cv2.line(annotated, (xa,ya), (xb,yb), line_color, 2)

                    # draw connection between neck and head_top
                    if neck_x is not None and head_top_x is not None:
                        neck_px = (int(neck_x * width + 0.5), int(neck_y * height + 0.5))
                        head_top_px = (int(head_top_x * width + 0.5), int(head_top_y * height + 0.5))
                        cv2.line(annotated, neck_px, head_top_px, (255,255,0), 3)  # Yellow line connecting neck to head_top

                    # save CSV for selected landmarks
                    if SAVE_LANDMARKS and csv_writer is not None:
                        for lm_enum in SELECTED_LANDMARKS:
                            lm = safe_get(results, lm_enum)
                            if lm is None:
                                continue
                            x_norm, y_norm, z_norm = float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))
                            x_px = int(x_norm * width + 0.5); y_px = int(y_norm * height + 0.5)
                            name = lm_enum.name
                            if name in LANDMARK_GROUP['leg']:
                                group = 'leg'
                            elif name in LANDMARK_GROUP['arm']:
                                group = 'arm'
                            elif name in LANDMARK_GROUP['torso']:
                                group = 'torso'
                            else:
                                group = 'other'
                            csv_writer.writerow([frame_idx, name, lm_enum.value, group, x_norm, y_norm, z_norm, x_px, y_px])
                        
                        # write NECK row
                        if neck_x is None:
                            csv_writer.writerow([frame_idx, "NECK", -1, "neck", float('nan'), float('nan'), float('nan'), '', ''])
                        else:
                            neck_px = (int(neck_x * width + 0.5), int(neck_y * height + 0.5))
                            csv_writer.writerow([frame_idx, "NECK", -1, "neck", neck_x, neck_y, neck_z, neck_px[0], neck_px[1]])
                        
                        # write HEAD_TOP row
                        if head_top_x is None:
                            csv_writer.writerow([frame_idx, "HEAD_TOP", -2, "head", float('nan'), float('nan'), float('nan'), '', ''])
                        else:
                            head_top_px = (int(head_top_x * width + 0.5), int(head_top_y * height + 0.5))
                            csv_writer.writerow([frame_idx, "HEAD_TOP", -2, "head", head_top_x, head_top_y, head_top_z, head_top_px[0], head_top_px[1]])

            # FPS overlay
            if frame_idx % 10 == 0:
                elapsed = time.time() - t0 + 1e-6
                fps_est = frame_idx / elapsed
                cv2.putText(annotated, f"Frame: {frame_idx}  FPS: {fps_est:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(annotated)
            if SHOW_WINDOW:
                display = cv2.resize(annotated, (int(width*0.8), int(height*0.8)))
                cv2.imshow("Pose - Lower body + arms + neck(headward) + head_top", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("Ngắt bởi người dùng")

    finally:
        cap.release()
        out.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

print("Hoàn tất. Video đã lưu:", OUTPUT_VIDEO)
if SAVE_LANDMARKS:
    print("CSV đã lưu:", LANDMARK_CSV)