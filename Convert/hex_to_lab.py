import pandas as pd
import numpy as np
import cv2
import csv
import io
import os
import sys

INPUT = "Convert/hex_brown.csv"
OUTPUT = "Convert/hex_to_lab_output_brown.csv"

# --- helper: robust hex to rgb ---
def hex_to_rgb(hex_color):
    if pd.isna(hex_color):
        return (0,0,0)
    s = str(hex_color).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if '#' in s:
        idx = s.find('#')
        s = s[idx:idx+7]
    if not s.startswith('#'):
        s = '#' + s
    s = s.lstrip('#')
    if len(s) < 6:
        s = s.ljust(6, '0')
    try:
        return tuple(int(s[i:i+2], 16) for i in (0,2,4))
    except ValueError:
        return (0,0,0)

# --- FIXED: rgb -> LAB (OpenCV-compatible LAB with extractor pipeline) ---
def rgb_to_lab(rgb):
    # Convert input RGB → BGR
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])  # B,G,R
    
    # Convert using OpenCV BGR2LAB (same as your extractor)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Return L, a, b exactly like extractor (0–255 each)
    L, a, b = lab
    return int(L), int(a), int(b)


# --- detect delimiter ---
with open(INPUT, 'rb') as f:
    raw = f.read(4096)
try:
    sample = raw.decode('utf-8-sig')
except:
    sample = raw.decode('utf-8', errors='replace')

detected_delim = None
try:
    dialect = csv.Sniffer().sniff(sample)
    detected_delim = dialect.delimiter
except:
    detected_delim = ';' if ';' in sample and sample.count(';') > sample.count(',') else ','

print("Detected delimiter:", repr(detected_delim))

# read CSV
df = pd.read_csv(INPUT, sep=detected_delim, encoding='utf-8-sig', engine='python')
print("Columns after read:", df.columns.tolist())

# split single column case
if len(df.columns) == 1:
    col0 = df.columns[0]
    split_df = df[col0].astype(str).str.split(';', expand=True)
    if split_df.shape[1] >= 2:
        split_df.columns = ['Filename','Hex'] + [f'col{i}' for i in range(3, split_df.shape[1]+1)]
        df = split_df[['Filename','Hex']]
    else:
        sys.exit("ERROR: couldn't split CSV.")

# normalize cols
df.columns = [c.strip() for c in df.columns]

hex_col = None
for c in df.columns:
    try:
        if df[c].astype(str).str.contains('#').any():
            hex_col = c
            break
    except:
        pass

if hex_col is None:
    for c in df.columns:
        if 'hex' in c.lower():
            hex_col = c
            break

if hex_col is None:
    sys.exit("ERROR: Tidak menemukan kolom HEX.")

print("Using HEX column:", hex_col)

df[hex_col] = df[hex_col].astype(str).str.strip()

rgb_vals = df[hex_col].apply(hex_to_rgb)
df['R'] = rgb_vals.apply(lambda t: t[0])
df['G'] = rgb_vals.apply(lambda t: t[1])
df['B'] = rgb_vals.apply(lambda t: t[2])

lab_vals = df[['R','G','B']].apply(lambda row: rgb_to_lab((row['R'], row['G'], row['B'])), axis=1)
df[['L','A','B_lab']] = pd.DataFrame(lab_vals.tolist(), index=df.index)

ordered = []
if 'Filename' in df.columns: ordered.append('Filename')
ordered.append(hex_col)
for x in ('R','G','B','L','A','B_lab'):
    if x in df.columns: ordered.append(x)

df = df[ordered]
df.to_csv(OUTPUT, index=False)
print("Selesai. Output disimpan ke:", OUTPUT)