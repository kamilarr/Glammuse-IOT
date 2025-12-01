import pandas as pd
import matplotlib.pyplot as plt

# ================================
# LOAD FILE
# ================================
clahe = pd.read_csv(r"Evaluasi/evaluation_deltaE_Clahe.csv")
he    = pd.read_csv(r"Evaluasi/evaluation_deltaE_HE.csv")

# ================================
# FUNGSI HITUNG PERSENTASE
# ================================

def compute_percentage(df):
    total = len(df)
    below_20 = (df["DeltaE"] <= 20).sum()
    above_20 = (df["DeltaE"] > 20).sum()

    print("Max DeltaE:", df["DeltaE"].max())
    print(df[df["DeltaE"] >= 20])

    return {
        "total": total,
        "below_20_count": below_20,
        "above_20_count": above_20,
        "below_20_percent": (below_20 / total) * 100,
        "above_20_percent": (above_20 / total) * 100
    }

clahe_pct = compute_percentage(clahe)
he_pct = compute_percentage(he)

# ================================
# TAMPILKAN HASIL
# ================================
print("\n===== PERBANDINGAN PERSENTASE DELTAE (<20 vs ≥20) =====\n")

print(">> CLAHE")
print(f"DeltaE <= 20 : {clahe_pct['below_20_count']} data ({clahe_pct['below_20_percent']:.2f}%)")
print(f"DeltaE > 20: {clahe_pct['above_20_count']} data ({clahe_pct['above_20_percent']:.2f}%)")

print("\n>> HE")
print(f"DeltaE <= 20 : {he_pct['below_20_count']} data ({he_pct['below_20_percent']:.2f}%)")
print(f"DeltaE > 20: {he_pct['above_20_count']} data ({he_pct['above_20_percent']:.2f}%)")

# ================================
# KESIMPULAN
# ================================
print("\n===== KESIMPULAN =====")

if clahe_pct["below_20_percent"] > he_pct["below_20_percent"]:
    print("CLAHE lebih baik: lebih banyak data dengan DeltaE < 20.")
else:
    print("HE lebih baik: lebih banyak data dengan DeltaE < 20.")

# ================================
# VISUALISASI GRAFIK
# ================================

methods = ["CLAHE", "HE"]

# ===== Bar chart: DeltaE < 20 =====
plt.figure(figsize=(6,4))
plt.bar(methods, [clahe_pct["below_20_percent"], he_pct["below_20_percent"]])
plt.ylabel("Percentage (%)")
plt.title("Persentase DeltaE < 20 • CLAHE vs HE")
plt.tight_layout()
plt.show()

# ===== Bar chart: DeltaE ≥ 20 =====
plt.figure(figsize=(6,4))
plt.bar(methods, [clahe_pct["above_20_percent"], he_pct["above_20_percent"]])
plt.ylabel("Percentage (%)")
plt.title("Persentase DeltaE ≥ 20 • CLAHE vs HE")
plt.tight_layout()
plt.show()

