import cv2
import numpy as np
import os
import csv
import mediapipe as mp
from sklearn.cluster import KMeans

# ============================================================
# 1. DETEKSI WAJAH DENGAN MEDIAPIPE
# ============================================================

def detect_face_mediapipe(img):
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        return None

    d = results.detections[0]
    bbox = d.location_data.relative_bounding_box

    ih, iw, _ = img.shape
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    x = max(0, x)
    y = max(0, y)

    return (x, y, w, h)


# ============================================================
# 2. GRABCUT
# ============================================================

def apply_grabcut(img, face_box):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    x, y, w, h = face_box
    rect = (x, y, w, h)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype("uint8")
    img_fg = img * mask2[:, :, np.newaxis]

    return img_fg, mask2


# ============================================================
# 3. CLAHE pada channel Y (YCrCb)
# ============================================================

def apply_clahe_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    return img_clahe, ycrcb_clahe


# ============================================================
# 4. MASK KULIT HSV + YCrCb
# ============================================================

def skin_mask_hsv_ycrcb(img_rgb, ycrcb_img):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    # HSV range kulit
    lower_hsv = np.array([0, 30, 20])
    upper_hsv = np.array([25, 180, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb range kulit
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    mask_ycrcb = cv2.inRange(ycrcb_img, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    return mask


# ============================================================
# 5. K-MEANS DOMINANT SKIN COLOR
# ============================================================

def dominant_color(img, mask, k=2):
    pixels = img[mask > 0]

    if len(pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dom = kmeans.cluster_centers_[np.argmax(counts)]

    return dom.astype(int)  # RGB int


# ============================================================
# 6. PROSES FOLDER & SIMPAN CSV
# ============================================================

def process_folder_to_csv(folder_path, output_csv="hasil_kmeans_mediapipe.csv"):
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            print(f"\n=== Memproses {filename} ===")
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(">> Gambar tidak bisa dibaca.")
                results.append([filename, None, None, None])
                continue

            # 1. Deteksi wajah
            face = detect_face_mediapipe(img)
            if face is None:
                print(">> Tidak ada wajah terdeteksi.")
                results.append([filename, None, None, None])
                continue

            # 2. GrabCut
            grabcut_img, grab_mask = apply_grabcut(img, face)

            # 3. CLAHE
            clahe_img, ycrcb_img = apply_clahe_ycrcb(grabcut_img)

            # 4. Mask kulit
            skin_mask = skin_mask_hsv_ycrcb(clahe_img, ycrcb_img)

            # 5. Dominant color
            dom = dominant_color(img, skin_mask, k=2)

            if dom is None:
                print(">> Tidak ditemukan piksel kulit.")
                results.append([filename, None, None, None])
            else:
                print(">> Dominant RGB:", dom)
                results.append([filename, dom[0], dom[1], dom[2]])

    # Simpan CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "R", "G", "B"])
        writer.writerows(results)

    print("\nSelesai! Hasil CSV tersimpan di:", output_csv)

process_folder_to_csv("Dataset/White")
