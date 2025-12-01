import pandas as pd
import numpy as np
from skimage.color import deltaE_ciede2000

# Load Groundtruth Skin Tone
gt_black = pd.read_csv(r"Convert/hex_to_lab_output_black.csv")
gt_brown = pd.read_csv(r"Convert/hex_to_lab_output_brown.csv")
gt_white = pd.read_csv(r"Convert/hex_to_lab_output_white.csv")

gt = pd.concat([gt_black, gt_brown, gt_white], ignore_index=True)

# Hitung DeltaE
def compute_deltaE(row):
    L1, a1, b1 = row["L_grabcut"], row["A_grabcut"], row["B_lab_grabcut"]
    L2, a2, b2 = row["L_gt"], row["A_gt"], row["B_lab_gt"]

    color1 = np.array([[L1, a1, b1]])
    color2 = np.array([[L2, a2, b2]])

    delta = deltaE_ciede2000(color1, color2)
    return float(delta)

# Evaluasi
def evaluate_file(extract_csv_path, output_path):
    print(f"\n=== Evaluasi: {extract_csv_path} ===")

    # Load Hasil Ekstraksi
    df_extract = pd.read_csv(extract_csv_path)

    # Merge dengan Groundtruth
    merged = pd.merge(
        df_extract,
        gt,
        left_on="filename",
        right_on="Filename",
        how="inner",
        suffixes=("_grabcut", "_gt")
    )

    print("Jumlah data berhasil di-merge:", len(merged))

    # Hitung DeltaE
    merged["DeltaE"] = merged.apply(compute_deltaE, axis=1).round(2)

    # Save Hasil
    merged.to_csv(output_path, index=False)
    print("Selesai! Disimpan ke:", output_path)

# Evaluasi 2 File Sekaligus
evaluate_file(
    extract_csv_path="Ekstraksi/HE_skin_dataset_results.csv",
    output_path="Evaluasi/evaluation_deltaE_HE.csv"
)

evaluate_file(
    extract_csv_path="Ekstraksi/CLAHE_skin_dataset_results.csv",
    output_path="Evaluasi/evaluation_deltaE_CLAHE.csv"
)
