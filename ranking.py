import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

dataset_groups = {
    'BCWD 10%': 'final_logs/bcwd_10_log.csv',
    'BCWD 25%': 'final_logs/bcwd_25_log.csv',
    'BCWD 50%': 'final_logs/bcwd_50_log.csv',
    'BCWP 10%': 'final_logs/bcwp_10_log.csv',
    'BCWP 25%': 'final_logs/bcwp_25_log.csv',
    'BCWP 50%': 'final_logs/bcwp_50_log.csv',
    'BCC 10%':  'final_logs/bcc_10_log.csv',
    'BCC 25%':  'final_logs/bcc_25_log.csv',
    'BCC 50%':  'final_logs/bcc_50_log.csv',
    'PIMA 10%': 'final_logs/pima_10_log.csv',
    'PIMA 25%': 'final_logs/pima_25_log.csv',
    'PIMA 50%': 'final_logs/pima_50_log.csv',
    'HCV 10%':  'final_logs/hcv_10_log.csv',
    'HCV 25%':  'final_logs/hcv_25_log.csv',
    'HCV 50%':  'final_logs/hcv_50_log.csv',
    'ILPD 5%':  'final_logs/ilpd_5_log.csv',
    'ILPD 10%': 'final_logs/ilpd_10_log.csv',
    'ILPD 15%': 'final_logs/ilpd_15_log.csv',
    'Parkinsons 5%':  'final_logs/parkinsons_5_log.csv',
    'Parkinsons 10%': 'final_logs/parkinsons_10_log.csv',
    'Parkinsons 15%': 'final_logs/parkinsons_15_log.csv',
}

tvae_datasets = [
    'BCWD 10%', 'BCWD 25%', 'BCWD 50%',
    'BCC 10%',  'BCC 25%',  'BCC 50%',
    'PIMA 25%', 'PIMA 50%',
    'ILPD 5%',  'ILPD 10%', 'ILPD 15%',
    'Parkinsons 5%', 'Parkinsons 10%', 'Parkinsons 15%',
]

cols = ['Mean_Knn', 'Mean_Ratio_Knn', 'Mean_Ratio_10Knn', 'mean_test_score']


def compute_scores(data, exclude_tvae=False):
    if exclude_tvae:
        data = data[data['Generator'] != 'tvae']

    data = data.copy()
    data['params'] = data['params'].astype(str)
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    t = data.groupby(["Generator", "Seed", "Model", "params", "fold"])[cols].mean().reset_index()
    u = t.groupby(["Generator", "Model", "Seed", "fold"])[cols].max().reset_index()
    y = t.merge(u, on=["Generator", "Model", "Seed", "fold"] + cols)[
        ["Generator", "Model", "Seed", "fold"] + cols
    ].drop_duplicates()
    y = y.groupby(["Generator", "Seed", "fold"])[cols].mean().reset_index()
    y.loc[y["Generator"] == "train", ["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn"]] = 0

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(y[["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn", "mean_test_score"]]),
        index=y.index,
        columns=["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn", "mean_test_score"]
    )
    scaled = scaled.merge(y[["Generator", "Seed", "fold"]], left_index=True, right_index=True)
    scaled["Privacy"] = (
        scaled["Mean_Knn"] + scaled["Mean_Ratio_Knn"] + scaled["Mean_Ratio_10Knn"]
    ) / 3
    scaled = scaled.rename({"mean_test_score": "ML-Utility"}, axis=1).drop(
        ["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn"], axis=1
    )

    scaler2 = MinMaxScaler()
    y_t = pd.DataFrame(
        scaler2.fit_transform(scaled[["Privacy", "ML-Utility"]]),
        index=scaled.index, columns=["Privacy", "ML-Utility"]
    )
    scaled = y_t.merge(scaled[["Generator", "Seed", "fold"]], left_index=True, right_index=True)
    scaled["Trade-off"] = (
        2 * scaled["ML-Utility"] * scaled["Privacy"]
    ) / (scaled["ML-Utility"] + scaled["Privacy"])

    return scaled


def write_overall_table(f, df, title):
    f.write("\n" + "=" * 70 + "\n")
    f.write(title + "\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Generator':<20} {'Trade-off':>10} {'Privacy':>10} {'ML-Utility':>10} {'1st Place':>10}\n")
    f.write("-" * 70 + "\n")
    for gen, row in df.iterrows():
        times = int(row.get("Times Ranked 1st", 0))
        f.write(f"{gen:<20} {row['Trade-off']:>10.3f} {row['Privacy']:>10.3f} {row['ML-Utility']:>10.3f} {times:>10}\n")
    f.write("-" * 70 + "\n")


def write_per_dataset_table(f, ranking_table):
    f.write("=" * 70 + "\n")
    f.write("GENERATOR RANKINGS PER DATASET\n")
    f.write("=" * 70 + "\n")
    for dataset_name in dataset_groups.keys():
        subset = ranking_table[ranking_table["Dataset"] == dataset_name]
        if subset.empty:
            continue
        f.write(f"\n  {dataset_name}\n")
        f.write(f"  {'Rank':<6} {'Generator':<20} {'Trade-off':>10} {'Privacy':>10} {'ML-Utility':>10}\n")
        f.write("  " + "-" * 60 + "\n")
        for _, row in subset.iterrows():
            f.write(f"  {int(row['Rank']):<6} {row['Generator']:<20} {row['Trade-off']:>10.3f} {row['Privacy']:>10.3f} {row['ML-Utility']:>10.3f}\n")


os.makedirs('results', exist_ok=True)

# ── Load all data once ────────────────────────────────────────────────────────
all_data = {}
for dataset_name, file in dataset_groups.items():
    if not os.path.exists(file):
        print(f"Missing: {file}")
        continue
    all_data[dataset_name] = pd.read_csv(file)


# ── Per dataset rankings ──────────────────────────────────────────────────────
results = []
for dataset_name, data in all_data.items():
    scaled  = compute_scores(data, exclude_tvae=False)
    summary = scaled.groupby("Generator")[["Trade-off", "Privacy", "ML-Utility"]].mean()
    summary["Dataset"] = dataset_name
    summary["Rank"]    = summary["Trade-off"].rank(ascending=False).astype(int)
    summary            = summary.sort_values("Rank")
    results.append(summary.reset_index())

ranking_table = pd.concat(results, ignore_index=True)
ranking_table = ranking_table[["Dataset", "Rank", "Generator", "Trade-off", "Privacy", "ML-Utility"]]
ranking_table = ranking_table.round(3)


# ── Overall ranking 1 — all 21 datasets, no TVAE ─────────────────────────────
results_no_tvae = []
for dataset_name, data in all_data.items():
    scaled  = compute_scores(data, exclude_tvae=True)
    summary = scaled.groupby("Generator")[["Trade-off", "Privacy", "ML-Utility"]].mean()
    summary["Dataset"] = dataset_name
    summary["Rank"]    = summary["Trade-off"].rank(ascending=False).astype(int)
    results_no_tvae.append(summary.reset_index())

ranking_no_tvae = pd.concat(results_no_tvae, ignore_index=True)
overall_no_tvae = (
    ranking_no_tvae
    .groupby("Generator")[["Trade-off", "Privacy", "ML-Utility"]]
    .mean()
    .round(3)
)
overall_no_tvae["Times Ranked 1st"] = (
    ranking_no_tvae[ranking_no_tvae["Rank"] == 1]
    .groupby("Generator")
    .size()
    .reindex(overall_no_tvae.index, fill_value=0)
)
overall_no_tvae = overall_no_tvae.sort_values("Trade-off", ascending=False)


# ── Overall ranking 2 — only datasets where TVAE ran ─────────────────────────
results_with_tvae = []
for dataset_name, data in all_data.items():
    if dataset_name not in tvae_datasets:
        continue
    scaled  = compute_scores(data, exclude_tvae=False)
    summary = scaled.groupby("Generator")[["Trade-off", "Privacy", "ML-Utility"]].mean()
    summary["Dataset"] = dataset_name
    summary["Rank"]    = summary["Trade-off"].rank(ascending=False).astype(int)
    results_with_tvae.append(summary.reset_index())

ranking_with_tvae = pd.concat(results_with_tvae, ignore_index=True)
overall_with_tvae = (
    ranking_with_tvae
    .groupby("Generator")[["Trade-off", "Privacy", "ML-Utility"]]
    .mean()
    .round(3)
)
overall_with_tvae["Times Ranked 1st"] = (
    ranking_with_tvae[ranking_with_tvae["Rank"] == 1]
    .groupby("Generator")
    .size()
    .reindex(overall_with_tvae.index, fill_value=0)
)
overall_with_tvae = overall_with_tvae.sort_values("Trade-off", ascending=False)


# ── Save everything to one file ───────────────────────────────────────────────
output_file = "results/rankings.txt"
with open(output_file, 'w') as f:

    write_per_dataset_table(f, ranking_table)

    write_overall_table(
        f, overall_no_tvae,
        "OVERALL RANKING — ALL 21 DATASETS (excluding TVAE)"
    )

    write_overall_table(
        f, overall_with_tvae,
        f"OVERALL RANKING — {len(tvae_datasets)} DATASETS WHERE TVAE RAN (all generators)"
    )

print(f"All rankings saved to {output_file}")