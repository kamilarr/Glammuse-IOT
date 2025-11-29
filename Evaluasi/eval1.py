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
# 2. LOAD HASIL GRABCUT
# =====================================================

grabcut = pd.read_csv(r"Ekstraksi/skin_dataset_results_grabcut2_with_mesh.csv")

# =====================================================
# 3. MERGE (tanpa rename apa pun)
# =====================================================

merged = pd.merge(
    grabcut,
    gt,
    left_on="filename",
    right_on="Filename",
    how="inner",
    suffixes=("_grabcut", "_gt")
)

print("Jumlah data berhasil di-merge:", len(merged))

# =====================================================
# 4. HITUNG DeltaE CIEDE2000 (pakai CIELAB dari CSV)
# =====================================================

def compute_deltaE(row):
    # dari hasil ekstraksi
    L1, a1, b1 = row["L_grabcut"], row["A_grabcut"], row["B_lab_grabcut"]

    # dari groundtruth HEX→LAB
    L2, a2, b2 = row["L_gt"], row["A_gt"], row["B_lab_gt"]

    color1 = np.array([[L1, a1, b1]])
    color2 = np.array([[L2, a2, b2]])

    delta = deltaE_ciede2000(color1, color2)
    return float(delta)


merged["DeltaE"] = merged.apply(compute_deltaE, axis=1).round(2)

# =====================================================
# 5. SIMPAN HASIL
# =====================================================

merged.to_csv("evaluation_deltaE_results2.csv", index=False)
print("Evaluasi selesai! Hasil disimpan di evaluation_deltaE_results2.csv")
