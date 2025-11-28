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
MIN_SKIN_PIXELS = 500              # minimal pixel valid skin untuk dipakai KMeans (tuning)

# -----------------------
# MediaPipe detector (buat sekali)
# -----------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# -----------------------
# Haarcascade Fallback Detector
# -----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
# Fallback: Haarcascade + padding
# -----------------------
def detect_face_fallback(img):
    """
    Coba MediaPipe dulu.
    Jika gagal → fallback Haarcascade.
    """
    # 1) coba mediapipe
    mp_box = detect_face_mediapipe(img)
    if mp_box is not None:
        return mp_box

    # 2) Haarcascade fallback
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # ambil wajah terbesar (paling masuk akal)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # tambahkan padding biar mirip mediapipe
    ih, iw = img.shape[:2]
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
            # fallback: gunakan skin mask di dalam bounding box
            fallback_mask = initial_skin_mask(img[y:y+h, x:x+w])
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            if fallback_mask is not None and fallback_mask.size != 0:
                # fallback_mask currently 0/255 -> normalize to 0/1
                if fallback_mask.max() > 1:
                    fallback_mask_norm = (fallback_mask // 255).astype(np.uint8)
                else:
                    fallback_mask_norm = fallback_mask.astype(np.uint8)
                full_mask[y:y+h, x:x+w] = fallback_mask_norm
            img_fg = img * (full_mask[:, :, np.newaxis])
            return img_fg, full_mask.astype(np.uint8)
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
# 4) Initial skin mask (HSV + YCrCb) - improved ranges + cleaning
# -----------------------
def initial_skin_mask(face_roi):
    if face_roi is None or face_roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    # Convert
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)

    # HSV: generous but avoids blues; tuned for wide skin tones
    mask_hsv = cv2.inRange(hsv, np.array([0, 15, 30], np.uint8), np.array([35, 255, 255], np.uint8))

    # YCrCb: common skin detection band (papers often use Cr between ~135-180)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 135, 85], np.uint8), np.array([255, 180, 135], np.uint8))

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # clean small holes and noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # ensure binary 0/255
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask_bin


# -----------------------
# 5) K-Means dominant color (works on BGR pixels)
# -----------------------
def dominant_color(img, mask, k=KMEANS_K):
    # mask expected 0 or 255; select pixels where mask>0
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return None

    pixels = pixels.reshape(-1, 3).astype(np.float32)
    # optionally filter extremely dark/bright pixels (remove specular)
    # keep as-is for now
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=42)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    # dominant is in BGR order (since img is BGR)
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

                face_box = detect_face_fallback(img)
                if face_box is None:
                    print("No face.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_face"])
                    failed += 1
                    continue

                grab_img, grab_mask = apply_grabcut_with_fallback(img, face_box)
                # grab_mask currently 0/1 or 0/255 depending on branch. Normalize to 0/255
                if grab_mask.max() <= 1:
                    grab_mask255 = (grab_mask * 255).astype(np.uint8)
                else:
                    grab_mask255 = grab_mask.astype(np.uint8)

                clahe_img, ycrcb = apply_clahe_ycrcb(grab_img)

                x, y, w, h = face_box
                # create color_mask_patch in CLAHE-ed image coordinates (use ROI)
                roi_for_mask = clahe_img[y:y+h, x:x+w]
                color_mask_patch = initial_skin_mask(roi_for_mask)
                color_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                if color_mask_patch.size != 0:
                    # color_mask_patch is 0/255, place into full mask
                    color_mask[y:y+h, x:x+w] = color_mask_patch

                combined_mask = cv2.bitwise_and(grab_mask255, color_mask)

                # If combined mask too small, prefer color_mask (skin detector) rather than raw grabcut
                if int(np.count_nonzero(combined_mask)) < MIN_SKIN_PIXELS:
                    # fallback to color_mask if that has enough pixels
                    if int(np.count_nonzero(color_mask)) >= MIN_SKIN_PIXELS:
                        combined_mask = color_mask.copy()
                    else:
                        # else fallback to grab_mask255 if grabcut is big enough
                        if int(np.count_nonzero(grab_mask255)) >= MIN_SKIN_PIXELS:
                            combined_mask = grab_mask255.copy()
                        else:
                            # final fallback: use a slightly eroded center rectangle (avoid background)
                            cx = x + w // 2
                            cy = y + h // 2
                            small_w = max(10, w // 4)
                            small_h = max(10, h // 4)
                            final_mask = np.zeros_like(combined_mask)
                            sx = max(0, cx - small_w)
                            sy = max(0, cy - small_h)
                            ex = min(img.shape[1], cx + small_w)
                            ey = min(img.shape[0], cy + small_h)
                            final_mask[sy:ey, sx:ex] = 255
                            combined_mask = final_mask

                # compute number of pixels used
                used_pixels = int(np.count_nonzero(combined_mask))

                dom = dominant_color(clahe_img, combined_mask)
                if dom is None:
                    print("No skin pixels.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_skin_pixels"])
                    failed += 1
                else:
                    # dom is BGR
                    b, g, r = int(dom[0]), int(dom[1]), int(dom[2])

                    # Convert to RGB for CSV (standard)
                    R_csv, G_csv, B_csv = r, g, b

                    # LAB from BGR pixel (OpenCV expects BGR)
                    bgr_pixel = np.uint8([[[b, g, r]]])
                    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
                    L, A, B_lab = lab_pixel[0][0]

                    print(f"OK -> RGB({R_csv},{G_csv},{B_csv}) LAB({L},{A},{B_lab}) used_pixels={used_pixels}")

                    rows.append([fname, label, R_csv, G_csv, B_csv, L, A, B_lab, "ok"])

                    if DEBUG_SAVE:
                        save_debug_visuals(path, img, face_box, grab_mask255, combined_mask, (R_csv, G_csv, B_csv), label, fname)

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
# debug save function (fixed BGR/RGB order and robust stacking)
# -----------------------
def save_debug_visuals(original_path, orig_img, face_box, grab_mask255, combined_mask, dom_rgb, label, fname):
    try:
        base = os.path.join(DEBUG_FOLDER, label)
        if not os.path.exists(base):
            os.makedirs(base)

        # Visuals: draw bbox on original
        img_vis = orig_img.copy()
        x, y, w, h = face_box
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Grab mask visualization (convert to 3-channel)
        grab_vis = grab_mask255.copy()
        if grab_vis.max() <= 1:
            grab_vis = (grab_vis * 255).astype(np.uint8)
        grab_color = cv2.cvtColor(grab_vis, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_vis, 0.7, grab_color, 0.3, 0)

        # Combined mask applied to original image
        comb_vis = cv2.bitwise_and(orig_img, orig_img, mask=combined_mask)

        # Create dom patch: dom_rgb is (R,G,B) for CSV; convert to BGR for OpenCV display
        R, G, B = dom_rgb
        dom_patch = np.full((100, 100, 3), (int(B), int(G), int(R)), dtype=np.uint8)

        # Resize panels to common height for stacking
        height = img_vis.shape[0]
        # make sure other images have same size as original for neat stacking
        def ensure_size(img, target_shape):
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

        grab_vis_resized = ensure_size(overlay, img_vis.shape)
        comb_vis_resized = ensure_size(comb_vis, img_vis.shape)
        dom_patch_resized = ensure_size(dom_patch, (img_vis.shape[0], img_vis.shape[1]//4))

        # Build layout: left-to-right: original with bbox | overlay | combined ; bottom: dom patch
        top = np.hstack([img_vis, grab_vis_resized, comb_vis_resized])
        # pad dom_patch_resized width to match top width
        dom_full = np.zeros_like(top)
        w_dom = dom_patch_resized.shape[1]
        dom_full[:dom_patch_resized.shape[0], :w_dom] = dom_patch_resized
        out = np.vstack([top, dom_full])

        save_path = os.path.join(base, f"debug_{fname}")
        cv2.imwrite(save_path, out)

    except Exception as e:
        print("Failed to save debug visuals:", e)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    process_folder_to_csv(BASE_FOLDER, OUTPUT_CSV)