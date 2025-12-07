import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import mediapipe as mp
import traceback

# CONFIG
BASE_FOLDER = "Dataset"
OUTPUT_CSV = "Ekstraksi/CLAHE_skin_dataset_results_new_ori.csv"
DEBUG_SAVE = True
DEBUG_FOLDER = "debug_output_ekstraksiori"
MIN_BBOX_SIZE = 30
GRABCUT_ITER = 5
PAD_RATIO = 0.20
KMEANS_K = 2
KMEANS_N_INIT = 10
MIN_SKIN_PIXELS = 500

# MediaPipe detector (face detection only)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Haarcascade Fallback Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure debug folder exists
if DEBUG_SAVE and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

# -------------------------
# Helpers: face detection
# -------------------------
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

def detect_face_fallback(img):
    mp_box = detect_face_mediapipe(img)
    if mp_box is not None:
        return mp_box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40,40))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    ih, iw = img.shape[:2]
    pad = int(PAD_RATIO * h)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + pad * 2)
    h = min(ih - y, h + pad * 2)
    if w < 1 or h < 1:
        return None
    return (x, y, w, h)

# -------------------------
# GrabCut with fallback
# -------------------------
def apply_grabcut_with_fallback(img, face_box):
    x, y, w, h = face_box
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        fallback_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        fallback_mask[y:y+h, x:x+w] = 255
        return img * (fallback_mask[:, :, np.newaxis] // 255), (fallback_mask // 255).astype(np.uint8)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (x, y, w, h)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==cv2.GC_PR_FGD)|(mask==cv2.GC_FGD), 1, 0).astype('uint8')
        img_fg = img * mask2[:, :, np.newaxis]
        if mask2.sum() < 50:
            raise ValueError("GrabCut produced very small foreground; fallback.")
        return img_fg, mask2
    except Exception:
        try:
            fb = initial_skin_mask(img[y:y+h, x:x+w])
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            if fb is not None and fb.size != 0:
                if fb.max() > 1:
                    fb_norm = (fb // 255).astype(np.uint8)
                else:
                    fb_norm = fb.astype(np.uint8)
                full_mask[y:y+h, x:x+w] = fb_norm
            img_fg = img * (full_mask[:, :, np.newaxis])
            return img_fg, full_mask.astype(np.uint8)
        except Exception:
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = 1
            img_fg = img * full_mask[:, :, np.newaxis]
            return img_fg, full_mask.astype(np.uint8)

# -------------------------
# CLAHE on Y channel
# -------------------------
def apply_clahe_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_clahe = clahe.apply(y)
    merged = cv2.merge([y_clahe, cr, cb])
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return img_clahe, merged

# -------------------------
# HSV + YCrCb threshold masks (small morph)
# -------------------------
def hsv_ycrcb_masks(img):
    # img expected BGR (CLAHE output)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # tuned ranges (empirical) - kamu bisa adjust
    lower_hsv = np.array([0, 15, 30], np.uint8)
    upper_hsv = np.array([35, 200, 255], np.uint8)

    lower_y = np.array([0, 135, 85], np.uint8)   # [Y,Cr,Cb] lower
    upper_y = np.array([255, 180, 135], np.uint8) # [Y,Cr,Cb] upper

    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_y, upper_y)

    # small morphology to remove tiny holes / speckles, avoid over-dilate
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, k, iterations=1)
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_CLOSE, k, iterations=1)

    # final binary
    _, mask_hsvb = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
    _, mask_ycb = cv2.threshold(mask_ycrcb, 10, 255, cv2.THRESH_BINARY)

    return mask_hsvb, mask_ycb

# -------------------------
# Initial skin mask fallback (keperluan GrabCut fallback)
# -------------------------
def initial_skin_mask(face_roi):
    if face_roi is None or face_roi.size == 0:
        return np.zeros((0,0), dtype=np.uint8)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    mask_hsv = cv2.inRange(hsv, np.array([0,15,30], np.uint8), np.array([35,200,255], np.uint8))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0,135,85], np.uint8), np.array([255,180,135], np.uint8))
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask_bin

# -------------------------
# Dominant color via KMeans
# -------------------------
def dominant_color(img, mask, k=KMEANS_K):
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return None
    pixels = pixels.reshape(-1,3).astype(np.float32)
    if pixels.shape[0] > 20000:
        idx = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=42)
    kmeans.fit(pixels_sample)
    labels_full = kmeans.predict(pixels)
    counts = np.bincount(labels_full)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant.astype(int)

# -------------------------
# Main processing
# -------------------------
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
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
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

                # 1) detect face to init grabcut rect (we still use face_box only for grabcut)
                face_box = detect_face_fallback(img)
                if face_box is None:
                    print("No face.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_face"])
                    failed += 1
                    continue

                # 2) GrabCut to remove background coarse
                grab_img, grab_mask = apply_grabcut_with_fallback(img, face_box)
                # normalize grab_mask to 0/255 uint8
                if grab_mask.max() <= 1:
                    grab_mask255 = (grab_mask * 255).astype(np.uint8)
                else:
                    grab_mask255 = grab_mask.astype(np.uint8)

                # 3) CLAHE on grabbed image
                clahe_img, ycrcb_merged = apply_clahe_ycrcb(grab_img)

                # 4) Otsu on grayscale of CLAHE image
                gray_clahe = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
                _, otsu_mask = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # apply small morphological clean to Otsu
                ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, ksmall, iterations=1)

                # 5) HSV + YCrCb masks (on CLAHE image)
                mask_hsv, mask_ycrcb = hsv_ycrcb_masks(clahe_img)

                # 6) Final mask = GrabCut ∩ Otsu ∩ HSV ∩ YCrCb
                # ensure all masks are 0/255
                def to255(m):
                    if m is None:
                        return np.zeros(img.shape[:2], dtype=np.uint8)
                    if m.max() <= 1:
                        return (m * 255).astype(np.uint8)
                    return m.astype(np.uint8)

                m_grab = to255(grab_mask255)
                m_otsu = to255(otsu_mask)
                m_hsv = to255(mask_hsv)
                m_ycb = to255(mask_ycrcb)

                final_mask = cv2.bitwise_and(m_grab, m_otsu)
                final_mask = cv2.bitwise_and(final_mask, m_hsv)
                final_mask = cv2.bitwise_and(final_mask, m_ycb)

                # 7) If final mask too small, fallback strategies (keamanan)
                if int(np.count_nonzero(final_mask)) < MIN_SKIN_PIXELS:
                    # try: Otsu ∩ (HSV ∪ YCrCb) ∩ GrabCut
                    union_color = cv2.bitwise_or(m_hsv, m_ycb)
                    candidate = cv2.bitwise_and(m_grab, cv2.bitwise_and(m_otsu, union_color))
                    if int(np.count_nonzero(candidate)) >= MIN_SKIN_PIXELS:
                        final_mask = candidate
                    else:
                        # try color-only ∩ Otsu
                        candidate2 = cv2.bitwise_and(m_otsu, union_color)
                        if int(np.count_nonzero(candidate2)) >= MIN_SKIN_PIXELS:
                            final_mask = candidate2
                        else:
                            # try color-only (hsv & ycrcb union)
                            if int(np.count_nonzero(union_color)) >= MIN_SKIN_PIXELS:
                                final_mask = union_color
                            else:
                                # fallback to grabcut if it has enough pixels
                                if int(np.count_nonzero(m_grab)) >= MIN_SKIN_PIXELS:
                                    final_mask = m_grab.copy()
                                else:
                                    # last-resort: small central patch inside face_box
                                    x, y, w, h = face_box
                                    cx = x + w//2
                                    cy = y + h//2
                                    small_w = max(10, w//6)
                                    small_h = max(10, h//6)
                                    patch = np.zeros_like(final_mask)
                                    sx = max(0, cx - small_w)
                                    sy = max(0, cy - small_h)
                                    ex = min(img.shape[1], cx + small_w)
                                    ey = min(img.shape[0], cy + small_h)
                                    patch[sy:ey, sx:ex] = 255
                                    final_mask = patch

                used_pixels = int(np.count_nonzero(final_mask))

                # 8) Dominant color computed from CLAHE image masked by final_mask
                dom = dominant_color(img, final_mask)
                if dom is None:
                    print("No skin pixels.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_skin_pixels"])
                    failed += 1
                else:
                    b, g, r = int(dom[0]), int(dom[1]), int(dom[2])
                    R_csv, G_csv, B_csv = r, g, b
                    bgr_pixel = np.uint8([[[b,g,r]]])
                    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
                    L, A, B_lab = lab_pixel[0][0]

                    print(f"OK -> RGB({R_csv},{G_csv},{B_csv}) LAB({L},{A},{B_lab}) used_pixels={used_pixels}")

                    rows.append([fname, label, R_csv, G_csv, B_csv, L, A, B_lab, "ok"])

                    if DEBUG_SAVE:
                        # ensure we pass masks in original shapes
                        save_debug_visuals(path, img.copy(), face_box, m_grab.copy(), final_mask.copy(), (R_csv, G_csv, B_csv), label, fname, clahe_img.copy(), mask_hsv.copy(), mask_ycrcb.copy(), otsu_mask.copy())

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
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDONE. Total: {total}, Failed: {failed}. CSV: {output_csv}")

# ------------------------------------------------------
def save_debug_visuals(
    original_path, orig_img, face_box, grab_mask255, final_mask,
    dom_rgb, label, fname,
    clahe_img=None, mask_hsv=None, mask_ycrcb=None, otsu_mask=None):

    try:
        base = os.path.join(DEBUG_FOLDER, label)
        os.makedirs(base, exist_ok=True)

        H, W = orig_img.shape[:2]

        # --- 1. Original with face bbox ---
        img_vis = orig_img.copy()
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(img_vis, (x,y), (x+w, y+h), (0,255,0), 2)

        # --- 2. Grabcut applied to original image ---
        grab_vis = cv2.bitwise_and(orig_img, orig_img, mask=grab_mask255.astype(np.uint8))

        # --- 3. CLAHE result ---
        clahe_vis = clahe_img.copy()

        # --- 4. Final mask applied to CLAHE image ---
        final_clahe_vis = cv2.bitwise_and(clahe_img, clahe_img, mask=final_mask.astype(np.uint8))

        # --- 5. Final mask applied to ORIGINAL image ---
        final_orig_vis = cv2.bitwise_and(orig_img, orig_img, mask=final_mask.astype(np.uint8))

        # --- 6. Color patch ---
        R, G, B = dom_rgb
        dom_patch = np.full((H//3, W//5, 3), (int(B), int(G), int(R)), dtype=np.uint8)

        # list urutan debug images
        debug_imgs = [
            img_vis,
            grab_vis,
            clahe_vis,
            final_clahe_vis,
            final_orig_vis,
            dom_patch
        ]

        # Resize semua ke tinggi seragam
        TARGET_H = int(np.clip(H, 250, 600))
        resized = []
        for im in debug_imgs:
            h, w = im.shape[:2]
            scale = TARGET_H / h
            new_w = int(w * scale)
            resized.append(cv2.resize(im, (new_w, TARGET_H)))

        # Buat grid 2 baris
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:])

        # Sesuaikan lebar baris
        w1 = row1.shape[1]
        w2 = row2.shape[1]
        if w1 > w2:
            row2 = np.hstack([row2, np.zeros((TARGET_H, w1-w2, 3), dtype=np.uint8)])
        else:
            row1 = np.hstack([row1, np.zeros((TARGET_H, w2-w1, 3), dtype=np.uint8)])

        grid = np.vstack([row1, row2])

        # Save final debug image
        save_path = os.path.join(base, f"debug_{os.path.splitext(fname)[0]}.png")
        cv2.imwrite(save_path, grid)

    except Exception as e:
        print("Failed to save debug visuals:", e)
        traceback.print_exc()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    process_folder_to_csv(BASE_FOLDER, OUTPUT_CSV)
