import pandas as pd
import numpy as np
from skimage.color import deltaE_ciede2000

# =====================================================
# 1. LOAD GROUNDTRUTH (3 FILE)
# =====================================================

gt_black = pd.read_csv(r"Convert/hex_to_lab_output_black.csv")
gt_brown = pd.read_csv(r"Convert/hex_to_lab_output_brown.csv")
gt_white = pd.read_csv(r"Convert/hex_to_lab_output_white.csv")

gt = pd.concat([gt_black, gt_brown, gt_white], ignore_index=True)

# =====================================================
# 2. FUNGSI HITUNG DELTA E
# =====================================================

def compute_deltaE(row):
    L1, a1, b1 = row["L_grabcut"], row["A_grabcut"], row["B_lab_grabcut"]
    L2, a2, b2 = row["L_gt"], row["A_gt"], row["B_lab_gt"]

    color1 = np.array([[L1, a1, b1]])
    color2 = np.array([[L2, a2, b2]])

    delta = deltaE_ciede2000(color1, color2)
    return float(delta)

# =====================================================
# 3. FUNGSI EVALUASI UNTUK SATU FILE
# =====================================================

def evaluate_file(extract_csv_path, output_path):
    print(f"\n=== Evaluasi: {extract_csv_path} ===")

    # load hasil ekstraksi
    df_extract = pd.read_csv(extract_csv_path)

    # merge dengan groundtruth
    merged = pd.merge(
        df_extract,
        gt,
        left_on="filename",
        right_on="Filename",
        how="inner",
        suffixes=("_grabcut", "_gt")
    )

    print("Jumlah data berhasil di-merge:", len(merged))

    # hitung deltaE
    merged["DeltaE"] = merged.apply(compute_deltaE, axis=1).round(2)

    # simpan hasil
    merged.to_csv(output_path, index=False)
    print("Selesai! Disimpan ke:", output_path)

# =====================================================
# 4. JALANKAN UNTUK HE & CLAHE
# =====================================================

evaluate_file(
    extract_csv_path="Ekstraksi/HE_skin_dataset_results.csv",
    output_path="Evaluasi/evaluation_deltaE_HE.csv"
)

evaluate_file(
    extract_csv_path="Ekstraksi/CLAHE_skin_dataset_results.csv",
    output_path="Evaluasi/evaluation_deltaE_CLAHE.csv"
)
