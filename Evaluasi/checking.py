import pandas as pd

# Load CSV
df = pd.read_csv("Evaluasi/evaluation_deltaE_results2.csv")

# Filter DeltaE > 20
df_outliers = df[df["DeltaE"] > 20]

# Print jumlah
print("Jumlah data DeltaE > 20:", len(df_outliers))

# Print filenamenya
print("\nDaftar filename dengan DeltaE > 20:")
print(df_outliers["filename"].tolist())

# Tampilkan dataframe
print("\nData lengkap:")
print(df_outliers)

# Simpan ke CSV baru
df_outliers.to_csv("Evaluasi/deltaE_above_20.csv", index=False)
print("\nHasil disimpan ke deltaE_above_20.csv")

import pandas as pd

# Load CSV
df = pd.read_csv("Evaluasi/evaluation_deltaE_results2.csv")

# Filter DeltaE > 20
df_outliers = df[df["DeltaE"] > 20]

# Hitung jumlah per label
counts = df_outliers["label"].value_counts()

print("Jumlah data DeltaE > 20 per kategori:\n")
print(counts)

print("\nRincian lengkap:")
print(df_outliers[["filename", "label", "DeltaE"]])