import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import mediapipe as mp
import traceback

# CONFIG
BASE_FOLDER = "Dataset"
OUTPUT_CSV = "Ekstraksi/HE_skin_dataset_results.csv"
DEBUG_SAVE = True
DEBUG_FOLDER = "debug_output"
MIN_BBOX_SIZE = 30
GRABCUT_ITER = 5
PAD_RATIO = 0.20
KMEANS_K = 2
KMEANS_N_INIT = 10
MIN_SKIN_PIXELS = 500

# MediaPipe detector (buat sekali)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# MediaPipe Face Mesh (landmarks)
mp_mesh = mp.solutions.face_mesh
face_mesh = mp_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5)

# Haarcascade Fallback Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Util: buat folder debug
if DEBUG_SAVE and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

# Landmark index groups (MediaPipe 468 / 478 refined)
FACE_OVAL = [10,338,297,332,284,251,389,356,454,
             323,361,288,397,365,379,378,400,
             377,152,148,176,149,150,136,172,
             58,132,93,234,127,162,21,54,103,67]

LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70,63,105,66,107]
RIGHT_BROW = [336,296,334,293,300]
OUTER_LIPS = [61,146,91,181,84,17,314,405,321,375,291,308]

# Helpers: convert landmarks to pixel points
def landmarks_to_points(landmarks, idxs, img_w, img_h):
    pts = []
    for i in idxs:
        lm = landmarks[i]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        pts.append([x, y])
    return np.array(pts, np.int32)

# Deteksi wajah pake MediaPipe + padding
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

# Fallback: Haarcascade + padding
def detect_face_fallback(img):
    mp_box = detect_face_mediapipe(img)
    if mp_box is not None:
        return mp_box

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
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

# GrabCut dengan fallback
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

# Histogram Equalization (HE) in YCrCb
def apply_he_ycrcb(img):
    # Histogram Equalization (HE) pada channel Y (brightness)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_he = cv2.equalizeHist(y)

    merged = cv2.merge([y_he, cr, cb])
    img_he = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return img_he, merged

# Initial skin mask (HSV + YCrCb) - improved ranges + cleaning
def initial_skin_mask(face_roi):
    if face_roi is None or face_roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)

    mask_hsv = cv2.inRange(hsv, np.array([0, 15, 30], np.uint8), np.array([35, 255, 255], np.uint8))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 135, 85], np.uint8), np.array([255, 180, 135], np.uint8))

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask_bin

# Dominant color
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

# Build masks from FaceMesh (full-image masks)
def build_masks_from_mesh(img, face_box):
    ih, iw = img.shape[:2]
    # run face_mesh on RGB image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    face_mask_full = np.zeros((ih, iw), dtype=np.uint8)
    feature_mask_full = np.zeros((ih, iw), dtype=np.uint8)
    mesh_draw = img.copy()
    mask_eyes = np.zeros_like(feature_mask_full)
    mask_brows = np.zeros_like(feature_mask_full)
    mask_mouth = np.zeros_like(feature_mask_full)

    if not results.multi_face_landmarks:
        return face_mask_full, feature_mask_full, mesh_draw

    lm = results.multi_face_landmarks[0].landmark

    # face oval
    try:
        oval_pts = landmarks_to_points(lm, FACE_OVAL, iw, ih)
        if oval_pts.shape[0] >= 3:
            # optionally use convex hull to ensure closed area
            hull = cv2.convexHull(oval_pts)
            cv2.fillPoly(face_mask_full, [hull], 255)
            cv2.polylines(mesh_draw, [hull], True, (0,255,0), 1)
    except Exception:
        pass

    # eyes, brows, lips -> feature mask (to be removed)
    try:
        le_pts = landmarks_to_points(lm, LEFT_EYE, iw, ih)
        re_pts = landmarks_to_points(lm, RIGHT_EYE, iw, ih)
        lb_pts = landmarks_to_points(lm, LEFT_BROW, iw, ih)
        rb_pts = landmarks_to_points(lm, RIGHT_BROW, iw, ih)
        lip_pts = landmarks_to_points(lm, OUTER_LIPS, iw, ih)

        if le_pts.shape[0] >= 3:
            cv2.fillPoly(mask_eyes, [le_pts], 255)
            cv2.polylines(mesh_draw, [le_pts], True, (0,0,255), 1)
        if re_pts.shape[0] >= 3:
            cv2.fillPoly(mask_eyes, [re_pts], 255)
            cv2.polylines(mesh_draw, [re_pts], True, (0,0,255), 1)
        if lb_pts.shape[0] >= 3:
            cv2.fillPoly(mask_brows, [lb_pts], 255)
            cv2.polylines(mesh_draw, [lb_pts], True, (255,0,0), 1)
        if rb_pts.shape[0] >= 3:
            cv2.fillPoly(mask_brows, [rb_pts], 255)
            cv2.polylines(mesh_draw, [rb_pts], True, (255,0,0), 1)
        if lip_pts.shape[0] >= 3:
            cv2.fillPoly(mask_mouth, [lip_pts], 255)
            cv2.polylines(mesh_draw, [lip_pts], True, (0,255,255), 1)
    except Exception:
        pass

    kernel_eye = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_eyes = cv2.dilate(mask_eyes, kernel_eye, iterations=1)

    feature_mask_full = cv2.bitwise_or(mask_eyes, mask_brows)
    feature_mask_full = cv2.bitwise_or(feature_mask_full, mask_mouth)

    return face_mask_full, feature_mask_full, mesh_draw

# Process folder to CSV (LAB added)
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
                # normalize grab_mask to 0/255
                if grab_mask.max() <= 1:
                    grab_mask255 = (grab_mask * 255).astype(np.uint8)
                else:
                    grab_mask255 = grab_mask.astype(np.uint8)

                he_img, ycrcb = apply_he_ycrcb(grab_img)

                x, y, w, h = face_box
                roi_for_mask = he_img[y:y+h, x:x+w]
                color_mask_patch = initial_skin_mask(roi_for_mask)
                color_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                if color_mask_patch.size != 0:
                    color_mask[y:y+h, x:x+w] = color_mask_patch

                combined_mask = cv2.bitwise_and(grab_mask255, color_mask)

                # If combined mask too small, prefer color_mask or grabcut as before
                if int(np.count_nonzero(combined_mask)) < MIN_SKIN_PIXELS:
                    if int(np.count_nonzero(color_mask)) >= MIN_SKIN_PIXELS:
                        combined_mask = color_mask.copy()
                    else:
                        if int(np.count_nonzero(grab_mask255)) >= MIN_SKIN_PIXELS:
                            combined_mask = grab_mask255.copy()
                        else:
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

                # --- NEW: refine masks using Face Mesh ---
                face_mask_full, feature_mask_full, mesh_draw = build_masks_from_mesh(img, face_box)
                # final face-only mask = face_mask - feature_mask
                refined_face = cv2.subtract(face_mask_full, feature_mask_full)
                # intersect refined_face with combined_mask to avoid background leakage
                # ensure both are 0/255
                if refined_face.max() > 0:
                    combined_mask = cv2.bitwise_and(combined_mask, refined_face)

                # if combined mask ends up tiny, fallback to previous logic (avoid losing everything)
                if int(np.count_nonzero(combined_mask)) < MIN_SKIN_PIXELS:
                    # try using refined_face alone
                    if int(np.count_nonzero(refined_face)) >= MIN_SKIN_PIXELS:
                        combined_mask = refined_face.copy()
                    else:
                        # keep previous combined_mask as-is (already computed)
                        pass

                used_pixels = int(np.count_nonzero(combined_mask))

                dom = dominant_color(he_img, combined_mask)
                if dom is None:
                    print("No skin pixels.")
                    rows.append([fname, label, None, None, None, None, None, None, "no_skin_pixels"])
                    failed += 1
                else:
                    b, g, r = int(dom[0]), int(dom[1]), int(dom[2])
                    R_csv, G_csv, B_csv = r, g, b
                    bgr_pixel = np.uint8([[[b, g, r]]])
                    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
                    L, A, B_lab = lab_pixel[0][0]

                    print(f"OK -> RGB({R_csv},{G_csv},{B_csv}) LAB({L},{A},{B_lab}) used_pixels={used_pixels}")

                    rows.append([fname, label, R_csv, G_csv, B_csv, L, A, B_lab, "ok"])

                    if DEBUG_SAVE:
                        save_debug_visuals(path, img, face_box, grab_mask255, combined_mask, (R_csv, G_csv, B_csv), label, fname, mesh_draw, refined_face)

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

# debug save function (updated)
def save_debug_visuals(original_path, orig_img, face_box, grab_mask255, combined_mask, dom_rgb, label, fname, mesh_draw=None, refined_face=None):
    try:
        base = os.path.join(DEBUG_FOLDER, label)
        if not os.path.exists(base):
            os.makedirs(base)

        img_vis = orig_img.copy()
        x, y, w, h = face_box
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        grab_vis = grab_mask255.copy()
        if grab_vis.max() <= 1:
            grab_vis = (grab_vis * 255).astype(np.uint8)
        grab_color = cv2.cvtColor(grab_vis, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_vis, 0.7, grab_color, 0.3, 0)

        comb_vis = cv2.bitwise_and(orig_img, orig_img, mask=combined_mask)

        R, G, B = dom_rgb
        dom_patch = np.full((100, 100, 3), (int(B), int(G), int(R)), dtype=np.uint8)

        # mesh overlay if exists
        if mesh_draw is None:
            mesh_draw = img_vis.copy()

        # refined face mask visualization (convert to BGR)
        if refined_face is None:
            refined_face_vis = np.zeros_like(img_vis)
        else:
            rf = refined_face.copy()
            if rf.max() <= 1:
                rf = (rf * 255).astype(np.uint8)
            refined_face_vis = cv2.bitwise_and(orig_img, orig_img, mask=rf)
            # draw semi-transparent overlay
            overlay2 = img_vis.copy()
            mask_color = np.zeros_like(img_vis)
            mask_color[:, :] = (0, 128, 255)
            alpha = 0.5
            mask3ch = cv2.cvtColor(rf, cv2.COLOR_GRAY2BGR)
            overlay2 = np.where(mask3ch==255, (overlay2 * (1 - alpha) + mask_color * alpha).astype(np.uint8), overlay2)

        # resize for stacking
        h0, w0 = img_vis.shape[:2]
        def ensure_size(img, target_shape):
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

        mesh_vis_resized = ensure_size(mesh_draw, img_vis.shape)
        comb_vis_resized = ensure_size(comb_vis, img_vis.shape)
        refined_vis_resized = ensure_size(refined_face_vis, img_vis.shape)
        dom_patch_resized = ensure_size(dom_patch, (img_vis.shape[0]//4, img_vis.shape[1]//4))

        top = np.hstack([img_vis, mesh_vis_resized, comb_vis_resized])
        dom_full = np.zeros((dom_patch_resized.shape[0], top.shape[1], 3), dtype=np.uint8)
        dom_full[:dom_patch_resized.shape[0], :dom_patch_resized.shape[1]] = dom_patch_resized
        out = np.vstack([top, dom_full])

        save_path = os.path.join(base, f"debug_{fname}")
        cv2.imwrite(save_path, out)

    except Exception as e:
        print("Failed to save debug visuals:", e)

# MAIN
if __name__ == "__main__":
    process_folder_to_csv(BASE_FOLDER, OUTPUT_CSV)