# ── Dataset statistics table ──────────────────────────────────────────────────
import pandas as pd
import os

dataset_files = {
    'BCWD 10%':       'bcwd_10.csv',
    'BCWD 25%':       'bcwd_25.csv',
    'BCWD 50%':       'bcwd_50.csv',
    'BCWP 10%':       'bcwp_10.csv',
    'BCWP 25%':       'bcwp_25.csv',
    'BCWP 50%':       'bcwp_50.csv',
    'BCC 10%':        'bcc_10.csv',
    'BCC 25%':        'bcc_25.csv',
    'BCC 50%':        'bcc_50.csv',
    'PIMA 10%':       'pima_10.csv',
    'PIMA 25%':       'pima_25.csv',
    'PIMA 50%':       'pima_50.csv',
    'HCV 10%':        'hcv_10.csv',
    'HCV 25%':        'hcv_25.csv',
    'HCV 50%':        'hcv_50.csv',
    'ILPD 5%':        'ilpd_5.csv',
    'ILPD 10%':       'ilpd_10.csv',
    'ILPD 15%':       'ilpd_15.csv',
    'Parkinsons 5%':  'parkinsons_5.csv',
    'Parkinsons 10%': 'parkinsons_10.csv',
    'Parkinsons 15%': 'parkinsons_15.csv',
}

def compute_dataset_stats(name, filepath):
    df      = pd.read_csv(filepath)
    target  = 'target'
    features = [c for c in df.columns if c != target]

    num_rows      = len(df)
    num_features  = len(features)
    num_classes   = df[target].nunique()
    counts        = df[target].value_counts()
    majority      = counts.iloc[0]
    minority      = counts.iloc[1]
    ir            = round(majority / minority, 2)

    # Manual override — these are encoded at runtime in the pipeline

    return {
        'Dataset':      name,
        'Rows':         num_rows,
        'Features':     num_features,
        'Classes':      num_classes,
        'Minority':     minority,
        'IR':           ir,
    }

rows = []
for name, filepath in dataset_files.items():
    if not os.path.exists(filepath):
        print(f"Missing: {filepath}")
        continue
    rows.append(compute_dataset_stats(name, filepath))

# ── Print table ───────────────────────────────────────────────────────────────
header = f"{'Dataset':<20} {'Rows':>6} {'Features':>10}  {'Classes':>8} {'Minority':>10} {'IR':>8}"
divider = "-" * len(header)

print("\n" + "=" * len(header))
print("DATASET STATISTICS")
print("=" * len(header))
print(header)
print(divider)
for r in rows:
    print(f"  {r['Dataset']:<20} {r['Rows']:>6} {r['Features']:>10} {r['Classes']:>8} {r['Minority']:>10} {r['IR']:>8}")
print(divider)

# ── Save to txt ───────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
with open('results/dataset_statistics.txt', 'w') as f:
    f.write("=" * len(header) + "\n")
    f.write("DATASET STATISTICS\n")
    f.write("=" * len(header) + "\n")
    f.write(header + "\n")
    f.write(divider + "\n")
    for r in rows:
        f.write(f"  {r['Dataset']:<20} {r['Rows']:>6} {r['Features']:>10} {r['Classes']:>8} {r['Minority']:>10} {r['IR']:>8}\n")
    f.write(divider + "\n")

print("/Users/susanazreis/Library/CloudStorage/OneDrive-NOVAIMS/thesis_test/results/dataset_statistics.txt")