import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import mediapipe as mp
import traceback

# =====================================================
# CONFIG
# =====================================================
BASE_FOLDER = "Dataset"
OUTPUT_CSV = "Z/CLAHE_skin_dataset_results.csv"

DEBUG_SAVE = True
DEBUG_FOLDER = "Z/zdebug_output"

PAD_RATIO = 0.20
GRABCUT_ITER = 5

KMEANS_K = 2
KMEANS_N_INIT = 10
MIN_SKIN_PIXELS = 500


# =====================================================
# FACE DETECTOR → MEDIA PIPE + HAAR
# =====================================================
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# =====================================================
# FUNCTIONS
# =====================================================
def detect_face(img):
    """Deteksi wajah utama menggunakan MediaPipe, fallback Haarcascade."""

    ih, iw = img.shape[:2]

    # --- MediaPipe detection ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_face.process(img_rgb)

    if result.detections:
        det = result.detections[0]
        box = det.location_data.relative_bounding_box

        x = int(box.xmin * iw)
        y = int(box.ymin * ih)
        w = int(box.width * iw)
        h = int(box.height * ih)

        pad = int(h * PAD_RATIO)

        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(iw - x, w + 2 * pad)
        h = min(ih - y, h + 2 * pad)

        if w > 0 and h > 0:
            return (x, y, w, h)

    # --- Haarcascade fallback ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    pad = int(h * PAD_RATIO)

    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + 2 * pad)
    h = min(ih - y, h + 2 * pad)

    return (x, y, w, h)


def apply_grabcut(img, face_box):
    """GrabCut untuk memisahkan foreground wajah."""
    x, y, w, h = face_box
    mask = np.zeros(img.shape[:2], np.uint8)

    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            img, mask, (x, y, w, h),
            bg, fg, GRABCUT_ITER,
            cv2.GC_INIT_WITH_RECT
        )

        mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
        fg_img = cv2.bitwise_and(img, img, mask=mask2)

        return fg_img, mask2

    except:
        # fallback bounding box
        fallback = np.zeros(img.shape[:2], np.uint8)
        fallback[y:y + h, x:x + w] = 255
        return img * (fallback[:, :, None] // 255), fallback


def apply_clahe(img):
    """CLAHE pada channel Y dari YCrCb."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y)

    merged = cv2.merge([y2, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def skin_mask(img):
    """Mask kulit menggunakan HSV + YCrCb + Otsu + morfologi."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 1. Rough HSV mask
    mask_hsv = cv2.inRange(hsv, (0, 30, 40), (35, 200, 255))

    # 2. Rough YCrCb mask
    mask_ycc = cv2.inRange(ycrcb, (60, 135, 85), (255, 170, 135))

    # combine
    combo = cv2.bitwise_and(mask_hsv, mask_ycc)

    if combo.sum() < 1000:
        return combo

    # 3. Otsu threshold pada Y channel
    y_channel = ycrcb[:, :, 0]
    y_masked = cv2.bitwise_and(y_channel, y_channel, mask=combo)
    _, otsu = cv2.threshold(
        y_masked, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_and(combo, otsu)

    # 4. Morphology cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Ambil region terbesar
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # 6. Hilangkan bagian gelap (mata, rambut)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 45)
    mask[dark > 0] = 0

    return mask


def dominant_color(img, mask):
    """Mengambil warna dominan dari area kulit."""
    pixels = img[mask > 0]

    if len(pixels) < 10:
        return None

    kmeans = KMeans(n_clusters=KMEANS_K, n_init=KMEANS_N_INIT)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dom = kmeans.cluster_centers_[np.argmax(counts)]

    return dom.astype(int)


# =====================================================
# DEBUG SAVE
# =====================================================
def save_debug(label, fname, orig, grabcut, clahe, mask, skin, patch):
    folder = os.path.join(DEBUG_FOLDER, label)
    os.makedirs(folder, exist_ok=True)

    H = orig.shape[0]

    def R(img): return cv2.resize(img, (H, H))

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    row1 = np.hstack([R(orig), R(grabcut), R(clahe)])
    row2 = np.hstack([R(mask_bgr), R(skin), R(patch)])
    out = np.vstack([row1, row2])

    cv2.imwrite(os.path.join(folder, f"debug_{fname}"), out)


# =====================================================
# MAIN PIPELINE
# =====================================================
def process_folder(base=BASE_FOLDER, output_csv=OUTPUT_CSV):
    rows = []
    total = 0
    failed = 0

    for label in sorted(os.listdir(base)):
        folder = os.path.join(base, label)
        if not os.path.isdir(folder):
            continue

        print(f"\n=== Folder: {label} ===")

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            total += 1
            path = os.path.join(folder, fname)

            img = cv2.imread(path)
            if img is None:
                print("Gagal membaca gambar.")
                continue

            face = detect_face(img)
            if face is None:
                print(f"{fname} → No face")
                failed += 1
                continue

            grab_img, grab_mask = apply_grabcut(img, face)
            clahe_img = apply_clahe(grab_img)

            mask = skin_mask(clahe_img)
            skin_only = cv2.bitwise_and(clahe_img, clahe_img, mask=mask)

            dom = dominant_color(clahe_img, mask)
            if dom is None:
                print(f"{fname} → No skin pixels")
                failed += 1
                continue

            b, g, r = dom
            rgb = (r, g, b)

            # Convert to LAB
            lab = cv2.cvtColor(
                np.uint8([[[b, g, r]]]),
                cv2.COLOR_BGR2LAB
            )[0][0]

            rows.append([fname, label, r, g, b, lab[0], lab[1], lab[2]])

            print(f"{fname} → RGB{rgb}")

            # Debug image
            if DEBUG_SAVE:
                patch = np.full((100, 100, 3), (b, g, r), np.uint8)
                save_debug(label, fname, img, grab_img, clahe_img,
                           mask, skin_only, patch)

    # Save CSV
    df = pd.DataFrame(rows, columns=[
        "filename", "label", "R", "G", "B",
        "L", "A", "B_lab"
    ])
    df.to_csv(output_csv, index=False)

    print(f"\nDONE. Total={total}, Failed={failed}")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    process_folder()
