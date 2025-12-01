import pandas as pd

# 1. LOAD DATA
df = pd.read_csv("Matching/Skinshade_Matching_Result2.csv")

# Pastikan kolom DeltaE tipe float
delta_cols = ["recommend_1_deltaE", "recommend_2_deltaE", "recommend_3_deltaE"]
df[delta_cols] = df[delta_cols].astype(float)

# 2. Buat kolom info rekomendasi mana yang DeltaE > 10
df["deltaE_over10_in"] = (
    (df["recommend_1_deltaE"] > 10).map(lambda x: "1" if x else "") +
    (df["recommend_2_deltaE"] > 10).map(lambda x: " 2" if x else "") +
    (df["recommend_3_deltaE"] > 10).map(lambda x: " 3" if x else "")
).str.strip()

# Filter baris yang ada minimal satu DeltaE > 10
df_over10 = df[df["deltaE_over10_in"] != ""]

# 3. Cetak tabel rapi ke terminal
print("\n=== Daftar file dengan DeltaE > 10 ===")
print(
    df_over10[
        ["filename", "recommend_1_deltaE", "recommend_2_deltaE", "recommend_3_deltaE", "deltaE_over10_in"]
    ].to_string(index=False)
)

# 4. Hitung statistik
count_1 = (df["recommend_1_deltaE"] > 10).sum()
count_2 = (df["recommend_2_deltaE"] > 10).sum()
count_3 = (df["recommend_3_deltaE"] > 10).sum()

print("\n=== Statistik DeltaE > 10 ===")
print(f"Recommend 1 : {count_1}")
print(f"Recommend 2 : {count_2}")
print(f"Recommend 3 : {count_3}")
print(f"TOTAL unik file bermasalah : {len(df_over10)}")
