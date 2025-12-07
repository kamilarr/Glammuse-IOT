import pandas as pd
import matplotlib.pyplot as plt

# Load File CSV
clahe = pd.read_csv(r"Evaluasi/evaluation_deltaE_CLAHE_new_ori.csv")
he    = pd.read_csv(r"Evaluasi/evaluation_deltaE_HE.csv")

# Hitung Presentase + Statistik
def compute_stats(df):
    total = len(df)
    below_20 = (df["DeltaE"] <= 20).sum()
    above_20 = (df["DeltaE"] > 20).sum()

    return {
        "total": total,
        "below_20_count": below_20,
        "above_20_count": above_20,
        "below_20_percent": (below_20 / total) * 100,
        "above_20_percent": (above_20 / total) * 100,

        # Statistik
        "mean": df["DeltaE"].mean(),
        "median": df["DeltaE"].median(),
        "std": df["DeltaE"].std(),

        # Data Terbesar dan Terkecil
        "min": df["DeltaE"].min(),
        "max": df["DeltaE"].max()
    }

clahe_stats = compute_stats(clahe)
he_stats = compute_stats(he)

# Hasil Persentase
print("\nPerbandingan Persentase DeltaE\n")

print(">> CLAHE")
print(f"DeltaE <= 20 : {clahe_stats['below_20_count']} data ({clahe_stats['below_20_percent']:.2f}%)")
print(f"DeltaE > 20 : {clahe_stats['above_20_count']} data ({clahe_stats['above_20_percent']:.2f}%)")

print("\n>> HE")
print(f"DeltaE <= 20 : {he_stats['below_20_count']} data ({he_stats['below_20_percent']:.2f}%)")
print(f"DeltaE > 20 : {he_stats['above_20_count']} data ({he_stats['above_20_percent']:.2f}%)")

# Statistik
print("\nStatistik Nilai DeltaE")
print("\n>> CLAHE")
print(f"Mean DeltaE   : {clahe_stats['mean']:.2f}")
print(f"Median DeltaE : {clahe_stats['median']:.2f}")
print(f"Std Dev       : {clahe_stats['std']:.2f}")
print(f"Min DeltaE    : {clahe_stats['min']:.2f}")
print(f"Max DeltaE    : {clahe_stats['max']:.2f}")

print("\n>> HE")
print(f"Mean DeltaE   : {he_stats['mean']:.2f}")
print(f"Median DeltaE : {he_stats['median']:.2f}")
print(f"Std Dev       : {he_stats['std']:.2f}")
print(f"Min DeltaE    : {he_stats['min']:.2f}")
print(f"Max DeltaE    : {he_stats['max']:.2f}")

# Kesimpulan
print("\nKesimpulan")

if clahe_stats["mean"] < he_stats["mean"]:
    print("CLAHE lebih akurat karena rata-rata DeltaE lebih kecil.")
else:
    print("HE lebih akurat karena rata-rata DeltaE lebih kecil.")

if clahe_stats["below_20_percent"] > he_stats["below_20_percent"]:
    print("CLAHE juga lebih stabil karena persentase DeltaE ≤ 20 lebih tinggi.")
else:
    print("HE lebih stabil.")

# Visualisasi Grafik
methods = ["CLAHE", "HE"]

plt.figure(figsize=(6,4))
plt.bar(methods, [clahe_stats["below_20_percent"], he_stats["below_20_percent"]])
plt.ylabel("Percentage (%)")
plt.title("Persentase DeltaE < 20 CLAHE vs HE")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(methods, [clahe_stats["above_20_percent"], he_stats["above_20_percent"]])
plt.ylabel("Percentage (%)")
plt.title("Persentase DeltaE ≥ 20 CLAHE vs HE")
plt.tight_layout()
plt.show()
