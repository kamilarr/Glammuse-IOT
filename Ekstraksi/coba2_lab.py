import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import mediapipe as mp
import traceback

# -----------------------
# CONFIG
# -----------------------
BASE_FOLDER = "Dataset"            # folder berisi subfolder Black,Brown,White
OUTPUT_CSV = "skin_dataset_results_grabcut.csv"
DEBUG_SAVE = True                  # kalau True simpan gambar debug ke folder debug_output/
DEBUG_FOLDER = "debug_output"
MIN_BBOX_SIZE = 30                 # minimal width/height bbox untuk GrabCut
GRABCUT_ITER = 5                   # iterasi GrabCut
PAD_RATIO = 0.20                   # padding relatif terhadap height bbox (20%)
KMEANS_K = 2                       # jumlah cluster KMeans
KMEANS_N_INIT = 10                 # n_init untuk KMeans

# -----------------------
# MediaPipe detector (buat sekali)
# -----------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)


# -----------------------
# Util: buat folder debug
# -----------------------
if DEBUG_SAVE and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)


# -----------------------
# 1) Deteksi wajah pake MediaPipe + padding
# -----------------------
def detect_face_mediapipe(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        return None

    best = max(results.detections, key=lambda d: d.score[0] if d.score else 0.0)
    bbox = best.location_data.relative_bounding_box

    ih, iw = img.shape[:2]
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    pad = int(PAD_RATIO * h)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + pad * 2)
    h = min(ih - y, h + pad * 2)

    if w < 1 or h < 1:
        return None

    return (x, y, w, h)


# -----------------------
# 2) GrabCut dengan fallback
# -----------------------
def apply_grabcut_with_fallback(img, face_box):
    x, y, w, h = face_box

    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        fallback_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        fallback_mask[y:y+h, x:x+w] = 255
        return img * (fallback_mask[:, :, np.newaxis] // 255), (fallback_mask // 255).astype(np.uint8)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 1, 0).astype('uint8')
        img_fg = img * mask2[:, :, np.newaxis]

        if mask2.sum() < 50:
            raise ValueError("GrabCut produced very small foreground; fallback.")

        return img_fg, mask2

    except Exception:
        try:
            fallback_mask = initial_skin_mask(img[y:y+h, x:x+w])
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = fallback_mask
            img_fg = img * (full_mask[:, :, np.newaxis] // 255)
            return img_fg, (full_mask // 255).astype(np.uint8)
        except Exception:
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = 1
            img_fg = img * full_mask[:, :, np.newaxis]
            return img_fg, full_mask.astype(np.uint8)


# -----------------------
# 3) CLAHE in YCrCb
# -----------------------
def apply_clahe_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    merged = cv2.merge([y_clahe, cr, cb])
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return img_clahe, merged


# -----------------------
# 4) Initial skin mask (HSV + YCrCb)
# -----------------------
def initial_skin_mask(face_roi):
    if face_roi is None or face_roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([25, 180, 255]))

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    return mask


# -----------------------
# 5) K-Means dominant color
# -----------------------
def dominant_color(img, mask, k=KMEANS_K):
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return None

    pixels = pixels.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=42)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant.astype(int)


# -----------------------
# 6) Process folder to CSV (LAB added)
# -----------------------
def process_folder_to_csv(base_folder=BASE_FOLDER, output_csv=OUTPUT_CSV):
    rows = []
    total = 0
    failed = 0

    for label in sorted(os.listdir(base_folder)):
        folder = os.path.join(base_folder, label)
        if not os.path.isdir(folder):
            continue
        print(f"\n=== Memproses folder: {label} ===")

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            total += 1
            path = os.path.join(folder, fname)
            print(f"Processing: {label}/{fname} ...", end=" ")

            try:
                img = cv2.imread(path)
                if img is None:
                    print("Gagal baca gambar.")
                    rows.append([fname, label, None, None, None, None, None, None, "read_error"])
                    failed += 1
                    continue

                face_box = detect_face_mediapipe(img)
                if face_box is None:
                    print("No face.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_face"])
                    failed += 1
                    continue

                grab_img, grab_mask = apply_grabcut_with_fallback(img, face_box)

                clahe_img, ycrcb = apply_clahe_ycrcb(grab_img)

                x, y, w, h = face_box
                color_mask_patch = initial_skin_mask(clahe_img[y:y+h, x:x+w])
                color_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                if color_mask_patch.size != 0:
                    color_mask[y:y+h, x:x+w] = color_mask_patch

                grab_mask255 = (grab_mask * 255).astype(np.uint8)
                combined_mask = cv2.bitwise_and(grab_mask255, color_mask)

                if combined_mask.sum() < 50:
                    combined_mask = grab_mask255

                dom = dominant_color(clahe_img, combined_mask)
                if dom is None:
                    print("No skin pixels.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_skin_pixels"])
                    failed += 1
                else:
                    r, g, b = int(dom[0]), int(dom[1]), int(dom[2])

                    # ---------------------------------
                    # NEW: Convert RGB -> LAB
                    # ---------------------------------
                    bgr_pixel = np.uint8([[[b, g, r]]])  
                    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
                    L, A, B_lab = lab_pixel[0][0]

                    print(f"OK -> RGB({r},{g},{b}) LAB({L},{A},{B_lab})")

                    rows.append([fname, label, r, g, b, L, A, B_lab, "ok"])

                    if DEBUG_SAVE:
                        save_debug_visuals(path, img, face_box, grab_mask, combined_mask, (r, g, b), label, fname)

            except Exception as e:
                print("Error:", str(e))
                traceback.print_exc()
                rows.append([fname, label, None, None, None, None, None, None, "exception"])
                failed += 1

    df = pd.DataFrame(rows, columns=[
        "filename", "label",
        "R", "G", "B",
        "L", "A", "B_lab",
        "status"
    ])
    df.to_csv(output_csv, index=False)
    print(f"\nDONE. Total: {total}, Failed: {failed}. CSV: {output_csv}")


# -----------------------
# debug save function
# -----------------------
def save_debug_visuals(original_path, orig_img, face_box, grab_mask, combined_mask, dom_rgb, label, fname):
    try:
        base = os.path.join(DEBUG_FOLDER, label)
        if not os.path.exists(base):
            os.makedirs(base)

        img_vis = orig_img.copy()
        x, y, w, h = face_box
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        grab_vis = (grab_mask * 255).astype(np.uint8)
        grab_color = cv2.cvtColor(grab_vis, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_vis, 0.7, grab_color, 0.3, 0)

        comb_vis = cv2.bitwise_and(orig_img, orig_img, mask=combined_mask)

        dom_patch = np.full((100, 100, 3), 
                            (int(dom_rgb[0]), int(dom_rgb[1]), int(dom_rgb[2])), dtype=np.uint8)

        top = np.hstack([img_vis, overlay])
        bottom = np.hstack([comb_vis, dom_patch])
        out = np.vstack([top, bottom])

        save_path = os.path.join(base, f"debug_{fname}")
        cv2.imwrite(save_path, out)

    except Exception as e:
        print("Failed to save debug visuals:", e)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    process_folder_to_csv(BASE_FOLDER, OUTPUT_CSV)
