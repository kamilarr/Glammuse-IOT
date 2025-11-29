import pandas as pd
import numpy as np
import cv2
import csv
import io
import os
import sys

INPUT = "Convert/hex_black.csv"
OUTPUT = "Convert/hex_to_lab_output_black.csv"

# --- helper: robust hex to rgb ---
def hex_to_rgb(hex_color):
    if pd.isna(hex_color):
        return (0,0,0)
    s = str(hex_color).strip()
    # remove possible quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # if there is a leading filename (in malformed split), try to extract something like #RRGGBB
    if '#' in s:
        # take first occurrence of # and next 6 hex chars
        idx = s.find('#')
        s = s[idx:idx+7]
    # ensure starts with #
    if not s.startswith('#'):
        s = '#' + s
    s = s.lstrip('#')
    if len(s) < 6:
        s = s.ljust(6, '0')
    try:
        return tuple(int(s[i:i+2], 16) for i in (0,2,4))
    except ValueError:
        return (0,0,0)

# --- helper: rgb -> true CIELAB ---
def rgb_to_lab(rgb):
    rgb_np = np.uint8([[rgb]])  # 1x1x3
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)[0][0]
    L = (lab[0] / 255.0) * 100.0
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return round(L,2), round(a,2), round(b,2)

# --- detect delimiter using csv.Sniffer on a sample ---
with open(INPUT, 'rb') as f:
    raw = f.read(4096)
try:
    sample = raw.decode('utf-8-sig')  # try decode and remove BOM for sniffing
except:
    sample = raw.decode('utf-8', errors='replace')

detected_delim = None
try:
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)
    detected_delim = dialect.delimiter
except Exception:
    # fallback: if we see semicolons, use ';', else use ','
    detected_delim = ';' if ';' in sample and sample.count(';') > sample.count(',') else ','

print("Detected delimiter:", repr(detected_delim))

# read with detected delimiter (and handle BOM)
df = pd.read_csv(INPUT, sep=detected_delim, encoding='utf-8-sig', engine='python')

print("Columns after read:", df.columns.tolist())

# If there is a single column like 'Filename;Hex' (still), split it
if len(df.columns) == 1:
    col0 = df.columns[0]
    # try splitting the single column by semicolon
    split_df = df[col0].astype(str).str.split(';', expand=True)
    if split_df.shape[1] >= 2:
        # assume first two are filename and hex
        split_df.columns = ['Filename','Hex'] + [f'col{i}' for i in range(3, split_df.shape[1]+1)]
        df = split_df[['Filename','Hex']]
        print("Split single column into Filename and Hex.")
    else:
        print("ERROR: couldn't split single-column CSV. Please check file format.")
        sys.exit(1)

# Normalize column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

# find hex column (look for a column that contains '#' in at least one cell)
hex_col = None
for c in df.columns:
    try:
        if df[c].astype(str).str.contains('#').any():
            hex_col = c
            break
    except Exception:
        continue

# fallback: find column name containing 'hex' (case-insensitive)
if hex_col is None:
    for c in df.columns:
        if 'hex' in c.lower():
            hex_col = c
            break

if hex_col is None:
    print("ERROR: Tidak menemukan kolom HEX. Kolom yang ada:", df.columns.tolist())
    sys.exit(1)

print("Using HEX column:", hex_col)

# Clean hex column and compute RGB and LAB
df[hex_col] = df[hex_col].astype(str).str.strip()
rgb_vals = df[hex_col].apply(hex_to_rgb)
df['R'] = rgb_vals.apply(lambda t: t[0])
df['G'] = rgb_vals.apply(lambda t: t[1])
df['B'] = rgb_vals.apply(lambda t: t[2])

lab_vals = df[['R','G','B']].apply(lambda row: rgb_to_lab((int(row['R']), int(row['G']), int(row['B']))), axis=1)
df[['L','a','b']] = pd.DataFrame(lab_vals.tolist(), index=df.index)

# reorder columns: Filename, Hex, R,G,B, L,a,b if present
cols = df.columns.tolist()
ordered = []
if 'Filename' in cols: ordered.append('Filename')
ordered.append(hex_col)
for x in ('R','G','B','L','a','b'):
    if x in cols and x not in ordered:
        ordered.append(x)

df = df[ordered]

df.to_csv(OUTPUT, index=False)
print("Selesai. Output disimpan ke:", OUTPUT)
