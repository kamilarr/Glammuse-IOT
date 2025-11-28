import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import mediapipe as mp
import traceback

# ---------------------------------
# CONFIG
# ---------------------------------
BASE_FOLDER = "Dataset"   
OUTPUT_CSV = "skin_dataset_no_clahe.csv"
DEBUG_SAVE = True
DEBUG_FOLDER = "debug_output_no_clahe"
MIN_BBOX_SIZE = 30
GRABCUT_ITER = 5
PAD_RATIO = 0.20
KMEANS_K = 2
KMEANS_N_INIT = 10

# ---------------------------------
# Mediapipe face detector
# ---------------------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ---------------------------------
# Create debug folder
# ---------------------------------
if DEBUG_SAVE and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)


# ---------------------------------
# 1. Face Detection (Mediapipe + Padding)
# ---------------------------------
def detect_face_mediapipe(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        return None

    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box

    ih, iw = img.shape[:2]
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    # Add padding
    pad = int(PAD_RATIO * h)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + pad * 2)
    h = min(ih - y, h + pad * 2)

    if w < 1 or h < 1:
        return None

    return (x, y, w, h)


# ---------------------------------
# 2. GrabCut with fallback
# ---------------------------------
def apply_grabcut_with_fallback(img, face_box):
    x, y, w, h = face_box

    # If bbox too small
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        fallback_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        fallback_mask[y:y+h, x:x+w] = 255
        return img * (fallback_mask[:, :, None] // 255), (fallback_mask // 255).astype(np.uint8)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (x, y, w, h)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 1, 0).astype("uint8")

        img_fg = img * mask2[:, :, None]

        if mask2.sum() < 50:
            raise ValueError("GrabCut empty")

        return img_fg, mask2

    except:
        # fallback: simple rectangle mask
        fallback_mask = np.zeros(img.shape[:2], np.uint8)
        fallback_mask[y:y+h, x:x+w] = 255
        img_fg = img * (fallback_mask[:, :, None] // 255)
        return img_fg, (fallback_mask // 255).astype(np.uint8)


# ---------------------------------
# 3. Skin mask (HSV + YCrCb)
# ---------------------------------
def initial_skin_mask(face_roi):
    if face_roi is None or face_roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([25, 180, 255]))

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)

    return mask


# ---------------------------------
# 4. Dominant Color via KMeans
# ---------------------------------
def dominant_color(img, mask):
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return None

    pixels = pixels.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=KMEANS_K, n_init=KMEANS_N_INIT, random_state=42)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dom = kmeans.cluster_centers_[np.argmax(counts)]

    return dom.astype(int)


# ---------------------------------
# 5. Process Folder to CSV
# ---------------------------------
def process_folder_to_csv(base_folder, output_csv):
    rows = []
    total = 0
    failed = 0

    for label in sorted(os.listdir(base_folder)):
        folder = os.path.join(base_folder, label)
        if not os.path.isdir(folder):
            continue

        print(f"\n=== Folder: {label} ===")

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total += 1
            path = os.path.join(folder, fname)
            print(f"Processing {label}/{fname} ...", end=" ")

            try:
                img = cv2.imread(path)
                if img is None:
                    print("Read error")
                    rows.append([fname, label, None, None, None, "read_error"])
                    failed += 1
                    continue

                # 1. detect face
                face_box = detect_face_mediapipe(img)
                if face_box is None:
                    print("No face")
                    rows.append([fname, label, None, None, None, "no_face"])
                    failed += 1
                    continue

                # 2. GrabCut
                grab_img, grab_mask = apply_grabcut_with_fallback(img, face_box)

                # 3. Color-mask inside face ROI
                x, y, w, h = face_box
                color_mask_patch = initial_skin_mask(grab_img[y:y+h, x:x+w])

                color_mask = np.zeros(img.shape[:2], np.uint8)
                if color_mask_patch.size != 0:
                    color_mask[y:y+h, x:x+w] = color_mask_patch

                # combine mask
                grab_mask255 = (grab_mask * 255).astype(np.uint8)
                combined_mask = cv2.bitwise_and(grab_mask255, color_mask)

                if combined_mask.sum() < 50:
                    combined_mask = grab_mask255

                # 4. dominant color
                dom = dominant_color(img, combined_mask)

                if dom is None:
                    print("No skin")
                    rows.append([fname, label, None, None, None, "no_skin"])
                    failed += 1
                    continue

                r, g, b = dom
                print(f"RGB({r},{g},{b})")
                rows.append([fname, label, r, g, b, "ok"])

                # Debug save
                if DEBUG_SAVE:
                    save_debug_visuals(img, face_box, grab_mask, combined_mask, dom, label, fname)

            except Exception as e:
                print("Error")
                traceback.print_exc()
                rows.append([fname, label, None, None, None, "exception"])
                failed += 1

    df = pd.DataFrame(rows, columns=["filename", "label", "R", "G", "B", "status"])
    df.to_csv(output_csv, index=False)
    print(f"\nDONE. Total={total}, Failed={failed}. Saved: {output_csv}")


# ---------------------------------
# Debug Visualization
# ---------------------------------
def save_debug_visuals(img, face_box, grab_mask, combined_mask, dom_rgb, label, fname):
    try:
        base = os.path.join(DEBUG_FOLDER, label)
        if not os.path.exists(base):
            os.makedirs(base)

        x, y, w, h = face_box
        img_vis = img.copy()
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        grab_vis = (grab_mask * 255).astype(np.uint8)
        grab_color = cv2.cvtColor(grab_vis, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_vis, 0.7, grab_color, 0.3, 0)

        comb_vis = cv2.bitwise_and(img, img, mask=combined_mask)
        patch = np.full((100, 100, 3), dom_rgb, np.uint8)

        top = np.hstack([img_vis, overlay])
        bottom = np.hstack([comb_vis, patch])
        out = np.vstack([top, bottom])

        cv2.imwrite(os.path.join(base, f"debug_{fname}"), out)

    except Exception as e:
        print("Debug save failed:", e)


# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":
    process_folder_to_csv(BASE_FOLDER, OUTPUT_CSV)
