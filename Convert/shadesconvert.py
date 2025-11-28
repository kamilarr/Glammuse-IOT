import cv2
import pandas as pd
import numpy as np

# ---- Convert HEX to RGB ----
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

# ---- Convert RGB to TRUE CIELAB ----
def rgb_to_cielab(rgb):
    rgb_np = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)

    L, a, b = lab[0][0]

    # Normalize OpenCV LAB -> true CIELAB
    L = (L / 255) * 100
    a = a - 128
    b = b - 128
    return L, a, b

# ---- Load your existing CSV ----
df = pd.read_csv("Dataset-Shades.csv")

# Add new columns
R_list, G_list, B_list = [], [], []
L_list, a_list, b_list = [], [], []

for hex_color in df["hex"]:
    rgb = hex_to_rgb(hex_color)
    R_list.append(rgb[0])
    G_list.append(rgb[1])
    B_list.append(rgb[2])

    L, a, b = rgb_to_cielab(rgb)
    L_list.append(L)
    a_list.append(a)
    b_list.append(b)

# Append to dataframe
df["R"] = R_list
df["G"] = G_list
df["B"] = B_list

# ---- Pembulatan nilai LAB ----
df["L_lab"] = pd.Series(L_list).round().astype(int)
df["a_lab"] = pd.Series(a_list).round().astype(int)
df["b_lab"] = pd.Series(b_list).round().astype(int)

# Save new CSV
df.to_csv("LAB-Dataset-Shades.csv", index=False)

print("Selesai! File tersimpan sebagai LAB-Dataset-Shades.csv")