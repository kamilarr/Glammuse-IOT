import pandas as pd
import numpy as np

# Menghitung Delta E (CIELAB) 
def delta_e_cie76(lab1, lab2):
    return np.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )

# Load dataset
df_foundation = pd.read_csv(r"Dataset/LAB-Dataset-Shades.csv")    
df_skin = pd.read_csv(r"Evaluasi/evaluation_deltaE_HE.csv")  

# Convert OpenCV LAB ke True CIELAB
df_skin["L_cielab"] = (df_skin["L_grabcut"] / 255.0) * 100
df_skin["a_cielab"] = df_skin["A_grabcut"] - 128
df_skin["b_cielab"] = df_skin["B_lab_grabcut"] - 128

# Pembulatan opsional
df_skin["L_cielab"] = df_skin["L_cielab"].round(2)
df_skin["a_cielab"] = df_skin["a_cielab"].round(2)
df_skin["b_cielab"] = df_skin["b_cielab"].round(2)

# Prepare output rows
output_rows = []

# Matching setiap skin ke semua foundation
for idx, row in df_skin.iterrows():

    # LAB kulit (CIELAB ASLI)
    skin_lab = np.array([
        row["L_cielab"],
        row["a_cielab"],
        row["b_cielab"]
    ], dtype=float)

    # Hitung DeltaE ke seluruh foundation
    df_foundation["DeltaE"] = df_foundation.apply(
        lambda x: delta_e_cie76(
            skin_lab,
            np.array([x["L_lab"], x["a_lab"], x["b_lab"]], dtype=float)
        ),
        axis=1
    )

    # Ambil 3 shade terbaik dengan DeltaE terkecil
    top3 = df_foundation.nsmallest(3, "DeltaE")

    output_rows.append({
        "filename": row["filename"],

        "recommend_1_brand": top3.iloc[0]["brand"],
        "recommend_1_product": top3.iloc[0]["product"],
        "recommend_1_hex": top3.iloc[0]["hex"],
        "recommend_1_deltaE": round(top3.iloc[0]["DeltaE"], 2),

        "recommend_2_brand": top3.iloc[1]["brand"],
        "recommend_2_product": top3.iloc[1]["product"],
        "recommend_2_hex": top3.iloc[1]["hex"],
        "recommend_2_deltaE": round(top3.iloc[0]["DeltaE"], 2),

        "recommend_3_brand": top3.iloc[2]["brand"],
        "recommend_3_product": top3.iloc[2]["product"],
        "recommend_3_hex": top3.iloc[2]["hex"],
        "recommend_3_deltaE": round(top3.iloc[0]["DeltaE"], 2),
    })

# Save output
df_output = pd.DataFrame(output_rows)
df_output.to_csv("Matching/Skinshade_Matching_Result_HE.csv", index=False)

print("Matching selesai! Hasil tersimpan di Skinshade_Matching_Result.csv")
