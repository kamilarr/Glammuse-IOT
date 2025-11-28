import cv2
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
import re

# --- Extract nomor dari nama file (foto (1).jpg -> 1) ---
def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else float('inf')

# --- Convert RGB to HEX ---
def rgb_to_hex(rgb):
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# --- K-Means dominant color ---
def get_dominant_color(image, k=1):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[0]
    return tuple([int(x) for x in dominant])

# --- Convert RGB to TRUE CIELAB ---
def rgb_to_lab(rgb):
    rgb_np = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)

    L, a, b = lab[0][0]

    # Normalize to true CIELAB
    L = (L / 255) * 100
    a = a - 128
    b = b - 128

    return L, a, b

# --- Process folder ---
folder_path = r"Dataset\Black"
data = []

files = sorted(
    [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
    key=extract_number
)

for file in files:
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path)

    rgb = get_dominant_color(img)
    hex_color = rgb_to_hex(rgb)
    L, a, b = rgb_to_lab(rgb)

    data.append({
        "filename": file,
        "R": rgb[0],
        "G": rgb[1],
        "B": rgb[2],
        "HEX": hex_color,
        "L": round(L),
        "a": round(a),
        "b": round(b)
    })

df = pd.DataFrame(data)

df.to_csv("Black.csv", index=False)
print("Selesai! white.csv sudah disorting urut foto (1) → foto (150)")
