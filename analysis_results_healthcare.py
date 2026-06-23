import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import sys
import os


# ── Merge local + colab logs into final_logs/ ─────────────────────────────────
LOCAL_PATH  = 'logs/'
COLAB_PATH  = '/Users/susanazreis/Library/CloudStorage/GoogleDrive-susanazreis@gmail.com/My Drive/thesis_logs_tvae_gc/'
OUTPUT_PATH = 'final_logs/'

os.makedirs(OUTPUT_PATH, exist_ok=True)

datasets = [
    'bcwd_10', 'bcwd_25', 'bcwd_50',
    'bcwp_10', 'bcwp_25', 'bcwp_50',
    'bcc_10',  'bcc_25',  'bcc_50',
    'pima_10', 'pima_25', 'pima_50',
    'hcv_10',  'hcv_25',  'hcv_50',
    'ilpd_5',  'ilpd_10', 'ilpd_15',
    'parkinsons_5', 'parkinsons_10', 'parkinsons_15',
]

for dataset_name in datasets:
    local_file = LOCAL_PATH + f'{dataset_name}_log.csv'
    colab_file = COLAB_PATH + f'{dataset_name}_log_sdv.csv'
    parts = []

    if os.path.exists(local_file):
        local = pd.read_csv(local_file)
        parts.append(local)
    if os.path.exists(colab_file):
        colab = pd.read_csv(colab_file)
        colab = colab[colab['Generator'] != 'train']
        parts.append(colab)

    if len(parts) == 0:
        print(f"{dataset_name}: NO DATA — skipping")
        continue

    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(OUTPUT_PATH + f'{dataset_name}_log.csv', index=False)
    print(f"{dataset_name} MERGED: {merged['Generator'].unique()} — {merged.shape[0]} rows")

print("Done merging.\n")


# ── Helper: filter to datasets where TVAE ran ─────────────────────────────────
def files_with_tvae(result_files, min_runs=10):
    """
    Return only files where TVAE has sufficient runs.
    """
    valid = []
    for f in result_files:
        df = pd.read_csv(f)
        tvae_rows = df[df['Generator'] == 'tvae']
        if len(tvae_rows) == 0:
            continue
        tvae_combos = tvae_rows.groupby(['Seed', 'fold']).ngroups
        if tvae_combos >= min_runs:
            valid.append(f)
        else:
            print(f"Excluding {f} from TVAE analysis — only {tvae_combos} seed/fold combinations")
    return valid


# ── Helper: strip TVAE rows entirely (for the ALL-21 "excludes TVAE" analysis) ─
def strip_tvae(result_files, output_path='final_logs_no_tvae/'):
    """
    Write TVAE-free copies of each log file and return their paths.

    TVAE only has complete, reliable data for 14 of the 21 datasets (the other
    7 — BCWP 10/25/50%, PIMA 10%, HCV 10/25/50% — ran but did not converge
    fully). By design, the ALL-21 analysis excludes TVAE entirely (it gets its
    proper treatment in the dedicated TVAE subset below), so every consumer of
    the ALL-21 file list — descriptive (print_results, save_global_results,
    plot_results) and inferential (wilcoxon_*, friedman_ranks) — should see
    data with zero TVAE rows, not a mix of complete and partial ones.
    """
    os.makedirs(output_path, exist_ok=True)
    cleaned_files = []
    for f in result_files:
        df = pd.read_csv(f)
        df = df[df['Generator'] != 'tvae']
        out_path = output_path + os.path.basename(f)
        df.to_csv(out_path, index=False)
        cleaned_files.append(out_path)
    return cleaned_files


# ── Core aggregation function ─────────────────────────────────────────────────
def aggregate_healthcare(result_files, rank=False):

    cols = [
        'Mean_Knn', 'Mean_Ratio_Knn', 'Mean_Ratio_10Knn', 'mean_test_score'
    ]

    y_list = []

    for file in result_files:
        data = pd.read_csv(file)
        data['params'] = data['params'].astype(str)
        for col in cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        t = data.groupby(
            ["Generator", "Seed", "Model", "params", "fold"]
        )[cols].mean().reset_index()

        u = t.groupby(
            ["Generator", "Model", "Seed", "fold"]
        )[cols].max().reset_index()

        y = t.merge(
            u, on=["Generator", "Model", "Seed", "fold"] + cols
        )[["Generator", "Model", "Seed", "fold"] + cols].drop_duplicates()

        y = y.groupby(["Generator", "Seed", "fold"])[cols].mean().reset_index()

        y.loc[y["Generator"] == "train",
              ["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn"]] = 0

        y_list.append(y)

    mean_list = []

    for y in y_list:
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(
                y[["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn", "mean_test_score"]]
            ),
            index=y.index,
            columns=["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn", "mean_test_score"]
        )
        scaled = scaled.merge(
            y[["Generator", "Seed", "fold"]], left_index=True, right_index=True
        )

        scaled["Privacy"] = (
            scaled["Mean_Knn"] +
            scaled["Mean_Ratio_Knn"] +
            scaled["Mean_Ratio_10Knn"]
        ) / 3

        scaled = scaled.rename(
            {"mean_test_score": "ML-Utility"}, axis=1
        ).drop(["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn"], axis=1)

        scaler2 = MinMaxScaler()
        y_t = pd.DataFrame(
            scaler2.fit_transform(scaled[["Privacy", "ML-Utility"]]),
            index=scaled.index,
            columns=["Privacy", "ML-Utility"]
        )
        scaled = y_t.merge(
            scaled[["Generator", "Seed", "fold"]],
            left_index=True, right_index=True
        )

        scaled["Trade-off"] = (
            2 * scaled["ML-Utility"] * scaled["Privacy"]
        ) / (scaled["ML-Utility"] + scaled["Privacy"])

        mean_list.append(scaled)

    if not rank:
        mean_final = []
        std_final  = []
        for mean in mean_list:
            mean_f = mean.groupby("Generator").mean()[["Trade-off", "Privacy", "ML-Utility"]]
            std_f  = mean.groupby("Generator").std()[["Trade-off", "Privacy", "ML-Utility"]]
            mean_final.append(mean_f)
            std_final.append(std_f)
        return cols, std_final, mean_final

    else:
        mean_final = []
        std_final  = []
        for mean in mean_list:
            mean_i = mean[~mean["Generator"].isin(["train"])].copy()
            for col in ["Trade-off", "Privacy", "ML-Utility"]:
                mean_i[col + "_Rank"] = mean_i.groupby(
                    ["Seed", "fold"]
                )[col].rank(ascending=False)
            mean_f = mean_i.groupby("Generator").mean()[
                ["Trade-off_Rank", "Privacy_Rank", "ML-Utility_Rank"]
            ]
            std_f = mean_i.groupby("Generator").std()[
                ["Trade-off_Rank", "Privacy_Rank", "ML-Utility_Rank"]
            ]
            mean_final.append(mean_f)
            std_final.append(std_f)
        return cols, ["Trade-off_Rank", "Privacy_Rank", "ML-Utility_Rank"], std_final, mean_final


# ── Wilcoxon tests on trade-off score ─────────────────────────────────────────
def wilcoxon_tradeoff(result_files):
    cols, stds, means = aggregate_healthcare(result_files)

    print("=" * 60)
    print("WILCOXON SIGNED-RANK TEST — TRADE-OFF SCORE")
    print("Comparing each generator against UMAP-SMOTENC")
    print("Holm-corrected p-values")
    print("=" * 60)

    umap_smote_nc = []
    smote_nc      = []
    ctgan         = []
    copula        = []

    for mean in means:
        mean = mean.reset_index()
        [umap_smote_nc.append(x) for x in mean[mean.Generator == "umap_smote_nc"]["Trade-off"]]
        [smote_nc.append(x)      for x in mean[mean.Generator == "smote_nc"]["Trade-off"]]
        [ctgan.append(x)         for x in mean[mean.Generator == "ctgan"]["Trade-off"]]
        [copula.append(x)        for x in mean[mean.Generator == "copula"]["Trade-off"]]

    p_values = []
    names    = []

    for competitor, name in zip(
        [smote_nc, ctgan, copula],
        ["smote_nc", "ctgan", "copula"]
    ):
        if len(competitor) == len(umap_smote_nc) and len(competitor) > 0:
            diffs = [u - c for u, c in zip(umap_smote_nc, competitor)]
            if any(d != 0 for d in diffs):
                p_values.append(wilcoxon(diffs)[1])
                names.append(name)

    umap_for_tvae = []
    tvae_vals     = []
    for mean in means:
        mean = mean.reset_index()
        tvae_data = mean[mean.Generator == "tvae"]["Trade-off"].tolist()
        umap_data = mean[mean.Generator == "umap_smote_nc"]["Trade-off"].tolist()
        if len(tvae_data) == len(umap_data) and len(tvae_data) > 0:
            umap_for_tvae.extend(umap_data)
            tvae_vals.extend(tvae_data)

    if len(tvae_vals) > 1:
        diffs = [u - t for u, t in zip(umap_for_tvae, tvae_vals)]
        if any(d != 0 for d in diffs):
            p_values.append(wilcoxon(diffs)[1])
            names.append("tvae")

    if p_values:
        corrected = multipletests(p_values, method="holm")[1]
        for name, p in zip(names, corrected):
            sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
            print(f"  umap_smote_nc vs {name}: p = {p:.1e} — {sig}")
    print()


# ── Wilcoxon tests on raw privacy metrics ─────────────────────────────────────
def wilcoxon_privacy(result_files):
    privacy_cols = ["Mean_Knn", "Mean_Ratio_Knn", "Mean_Ratio_10Knn"]

    all_data = pd.concat([pd.read_csv(f) for f in result_files])

    # Note: no per-dataset TVAE exclusion needed here — when called on the
    # ALL-21 file list, TVAE rows have already been stripped entirely by
    # strip_tvae(); when called on the TVAE subset, only fully-converged
    # datasets are included in the first place via files_with_tvae().

    for col in privacy_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

    print("=" * 60)
    print("WILCOXON SIGNED-RANK TEST — RAW PRIVACY METRICS")
    print("Comparing each generator against train baseline")
    print("Holm-corrected p-values")
    print("=" * 60)

    for col in privacy_cols:
        print(f"\n  {col}:")
        grouped    = all_data.groupby(["Generator", "dataset"])[col].mean().reset_index()
        train_vals = grouped[grouped.Generator == "train"].set_index("dataset")[col]

        p_values = []
        names    = []

        for gen in ["umap_smote_nc", "smote_nc", "ctgan", "copula"]:
            gen_data = grouped[grouped.Generator == gen].set_index("dataset")[col]
            common   = train_vals.index.intersection(gen_data.index)
            if len(common) < 2:
                continue
            diffs = [gen_data[d] - train_vals[d] for d in common]
            if any(d != 0 for d in diffs):
                p_values.append(wilcoxon(diffs)[1])
                names.append(gen)

        tvae_data   = grouped[grouped.Generator == "tvae"].set_index("dataset")[col]
        common_tvae = train_vals.index.intersection(tvae_data.index)
        if len(common_tvae) >= 2:
            diffs = [tvae_data[d] - train_vals[d] for d in common_tvae]
            if any(d != 0 for d in diffs):
                p_values.append(wilcoxon(diffs)[1])
                names.append("tvae")

        if p_values:
            corrected = multipletests(p_values, method="holm")[1]
            for name, p in zip(names, corrected):
                sig = "✅" if p < 0.05 else "❌"
                print(f"    {name}: p = {p:.1e} {sig}")
    print()


# ── Wilcoxon tests on ML utility ──────────────────────────────────────────────
def wilcoxon_utility(result_files):
    """
    Tests whether each generator's utility (F1) is significantly different
    from UMAP-SMOTENC per classifier — equivalent to original paper's Table 5.
    """
    all_data = pd.concat([pd.read_csv(f) for f in result_files])

    # Note: no per-dataset TVAE exclusion needed here — see wilcoxon_privacy.

    all_data['mean_test_score'] = pd.to_numeric(all_data['mean_test_score'], errors='coerce')
    

    print("=" * 60)
    print("WILCOXON SIGNED-RANK TEST — ML UTILITY (F1) PER CLASSIFIER")
    print("Comparing each generator against UMAP-SMOTENC")
    print("Holm-corrected p-values")
    print("=" * 60)

    for classifier in ["lr", "gs", "dt", "rf", "xgb", "mlp"]:
        print(f"\n  Classifier: {classifier.upper()}")

        clf_data  = all_data[all_data["Model"] == classifier]
        grouped   = clf_data.groupby(["Generator", "dataset"])["mean_test_score"].mean().reset_index()
        umap_vals = grouped[grouped.Generator == "umap_smote_nc"].set_index("dataset")["mean_test_score"]

        p_values = []
        names    = []

        for gen in ["smote_nc", "ctgan", "copula"]:
            gen_data = grouped[grouped.Generator == gen].set_index("dataset")["mean_test_score"]
            common   = umap_vals.index.intersection(gen_data.index)
            if len(common) < 2:
                continue
            diffs = [umap_vals[d] - gen_data[d] for d in common]
            if any(d != 0 for d in diffs):
                p_values.append(wilcoxon(diffs)[1])
                names.append(gen)

        tvae_data   = grouped[grouped.Generator == "tvae"].set_index("dataset")["mean_test_score"]
        common_tvae = umap_vals.index.intersection(tvae_data.index)
        if len(common_tvae) >= 2:
            diffs = [umap_vals[d] - tvae_data[d] for d in common_tvae]
            if any(d != 0 for d in diffs):
                p_values.append(wilcoxon(diffs)[1])
                names.append("tvae")

        if p_values:
            corrected = multipletests(p_values, method="holm")[1]
            for name, p in zip(names, corrected):
                sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
                print(f"    umap_smote_nc vs {name}: p = {p:.1e} — {sig}")
    print()


# ── Friedman test on ranks ─────────────────────────────────────────────────────
def friedman_ranks(result_files):
    cols, cols_new, stds, means = aggregate_healthcare(result_files, rank=True)

    print("=" * 60)
    print("FRIEDMAN TEST — RANK SCORES")
    print("=" * 60)

    for col in ["Trade-off_Rank", "Privacy_Rank", "ML-Utility_Rank"]:
        umap_smote_nc = []
        smote_nc      = []
        ctgan         = []
        copula        = []

        for mean in means:
            mean    = mean.reset_index()
            grouped = mean.groupby("Generator")[[col]].mean().reset_index()
            [umap_smote_nc.append(x) for x in grouped[grouped.Generator == "umap_smote_nc"][col]]
            [smote_nc.append(x)      for x in grouped[grouped.Generator == "smote_nc"][col]]
            [ctgan.append(x)         for x in grouped[grouped.Generator == "ctgan"][col]]
            [copula.append(x)        for x in grouped[grouped.Generator == "copula"][col]]

        umap_for_tvae = []
        tvae_vals     = []
        for mean in means:
            mean      = mean.reset_index()
            grouped   = mean.groupby("Generator")[[col]].mean().reset_index()
            tvae_data = grouped[grouped.Generator == "tvae"][col].tolist()
            umap_data = grouped[grouped.Generator == "umap_smote_nc"][col].tolist()
            if len(tvae_data) == len(umap_data) and len(tvae_data) > 0:
                umap_for_tvae.extend(umap_data)
                tvae_vals.extend(tvae_data)

        groups      = [umap_smote_nc, smote_nc, ctgan, copula]
        group_names = ["umap_smote_nc", "smote_nc", "ctgan", "copula"]

        if len(tvae_vals) == len(umap_smote_nc):
            groups.append(tvae_vals)
            group_names.append("tvae")

        if all(len(g) > 1 for g in groups):
            _, p_value = stats.friedmanchisquare(*groups)
            sig = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
            print(f"  {col} ({', '.join(group_names)}): p = {p_value:.1e} — {sig}")
    print()


# ── Print and save functions ───────────────────────────────────────────────────
def print_results(result_files, dataset_names):
    cols, stds, means = aggregate_healthcare(result_files)

    print("=" * 60)
    print("ABSOLUTE SCORES")
    print("=" * 60)
    for name, mean, std in zip(dataset_names, means, stds):
        print(f"\n--- {name} ---")
        print("Mean:")
        print(mean.round(3))
        print("Std:")
        print(std.round(3))

    print("\n" + "=" * 60)
    print("AVERAGED ACROSS ALL DATASETS")
    print("=" * 60)
    combined_mean = sum(means) / len(means)
    print(combined_mean.round(3))


def plot_results(result_files, dataset_names, output_file="results_plot.png"):
    cols, stds, means = aggregate_healthcare(result_files)

    combined_mean = sum(means) / len(means)
    combined_std  = sum(stds)  / len(stds)

    color_map = {
        'umap_smote_nc': '#2196F3',
        'smote_nc':      '#4CAF50',
        'ctgan':         '#FF9800',
        'tvae':          '#9C27B0',
        'copula':        '#F44336',
        'train':         '#9E9E9E',
    }
    generators = combined_mean.index.tolist()
    colors     = [color_map.get(g, '#607D8B') for g in generators]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics   = ["Trade-off", "Privacy", "ML-Utility"]

    for ax, metric in zip(axes, metrics):
        values = combined_mean[metric]
        errors = combined_std[metric]
        ax.bar(generators, values, yerr=errors, capsize=5, color=colors)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=20)

    plt.suptitle(f"Privacy-Utility Trade-off — {', '.join(dataset_names)}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()


def print_utility_by_model(result_files, dataset_names):
    print("=" * 60)
    print("UTILITY (F1) BY MODEL AND GENERATOR")
    print("=" * 60)
    for name, file in zip(dataset_names, result_files):
        data = pd.read_csv(file)
        print(f"\n--- {name} ---")
        print(
            data.groupby(['Generator', 'Model'])['mean_test_score']
            .max().unstack().round(3)
        )


def print_privacy_metrics(result_files, dataset_names):
    print("=" * 60)
    print("RAW PRIVACY METRICS (Mean KNN + Duplicates)")
    print("=" * 60)
    for name, file in zip(dataset_names, result_files):
        data = pd.read_csv(file)
        print(f"\n--- {name} ---")
        print(
            data.groupby('Generator')[['Mean_Knn', 'Duplicates']]
            .mean().round(3).to_string()
        )
    print("\n--- Averaged across all datasets ---")
    all_data = pd.concat([pd.read_csv(f) for f in result_files])
    print(
        all_data.groupby('Generator')[['Mean_Knn', 'Duplicates']]
        .mean().round(3).to_string()
    )

def compute_utility_per_classifier(result_files):
    """
    For each generator/classifier: select best hyperparameter per seed/fold,
    average across seed/fold per dataset, then average across datasets.
    Returns raw (unscaled) F1 — methodologically consistent with aggregate_healthcare.
    """
    per_dataset_results = []

    for file in result_files:
        data = pd.read_csv(file)
        data['mean_test_score'] = pd.to_numeric(data['mean_test_score'], errors='coerce')

        best = data.groupby(
            ["Generator", "Model", "Seed", "fold"]
        )['mean_test_score'].max().reset_index()

        per_dataset = best.groupby(["Generator", "Model"])['mean_test_score'].mean().reset_index()
        per_dataset_results.append(per_dataset)

    combined = pd.concat(per_dataset_results)
    final = combined.groupby(["Generator", "Model"])['mean_test_score'].mean().reset_index()
    return final.pivot(index="Generator", columns="Model", values="mean_test_score").round(3)


def save_results(result_files, dataset_names, output_file='analysis_results.txt'):
    cols, stds, means = aggregate_healthcare(result_files)
    combined_mean = sum(means) / len(means)
    combined_std  = sum(stds)  / len(stds)

    with open(output_file, 'w') as f:

        f.write("=" * 60 + "\n")
        f.write("HEALTHCARE DATASETS — PRIVACY UTILITY ANALYSIS\n")
        f.write(f"Datasets: {', '.join(dataset_names)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("TRADE-OFF SCORES PER DATASET\n")
        f.write("=" * 60 + "\n")
        for name, mean, std in zip(dataset_names, means, stds):
            f.write(f"\n--- {name} ---\n")
            f.write("Mean:\n")
            f.write(mean.round(3).to_string())
            f.write("\nStd:\n")
            f.write(std.round(3).to_string())
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("TRADE-OFF SCORES (averaged across all datasets)\n")
        f.write("=" * 60 + "\n")
        f.write("\nMean:\n")
        f.write(combined_mean.round(3).to_string())
        f.write("\nStd:\n")
        f.write(combined_std.round(3).to_string())
        f.write("\n\n")

        f.write("=" * 60 + "\n")
        f.write("RAW PRIVACY METRICS PER DATASET\n")
        f.write("=" * 60 + "\n")
        for name, file in zip(dataset_names, result_files):
            data = pd.read_csv(file)
            f.write(f"\n--- {name} ---\n")
            f.write(
                data.groupby('Generator')[
                    ['Mean_Knn', 'Duplicates']
                ].mean().round(3).to_string()
            )
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("RAW PRIVACY METRICS (averaged across all datasets)\n")
        f.write("=" * 60 + "\n")
        all_data = pd.concat([pd.read_csv(f) for f in result_files])
        f.write(
            all_data.groupby('Generator')[
                ['Mean_Knn', 'Duplicates']
            ].mean().round(3).to_string()
        )
        f.write("\n")

    print(f"Results saved to {output_file}")


def save_global_results(result_files, label, output_file):
    """
    Save aggregated trade-off, privacy and utility averaged across a file set.
    Covers: scaled Trade-off/Privacy/ML-Utility, raw privacy metrics,
    and F1 per classifier.
    """
    cols, stds, means = aggregate_healthcare(result_files)
    combined_mean = sum(means) / len(means)
    combined_std  = sum(stds)  / len(stds)

    all_data = pd.concat([pd.read_csv(f) for f in result_files])
    for col in ['Mean_Knn', 'Mean_Ratio_Knn', 'Mean_Ratio_10Knn', 'mean_test_score']:
        all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

    with open(output_file, 'w') as f:

        f.write("=" * 60 + "\n")
        f.write(f"GLOBAL ANALYSIS — {label}\n")
        f.write(f"Datasets ({len(result_files)}): " + ", ".join(result_files) + "\n")
        f.write("=" * 60 + "\n\n")

        # ── Scaled composite scores ────────────────────────────────────────────
        f.write("TRADE-OFF | PRIVACY | ML-UTILITY  (averaged across datasets)\n")
        f.write("=" * 60 + "\n")
        f.write("\nMean:\n")
        f.write(combined_mean[["Trade-off", "Privacy", "ML-Utility"]].round(3).to_string())
        f.write("\n\nStd:\n")
        f.write(combined_std[["Trade-off", "Privacy", "ML-Utility"]].round(3).to_string())
        f.write("\n\n")

        # ── Raw privacy metrics ────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("RAW PRIVACY METRICS (averaged across datasets)\n")
        f.write("=" * 60 + "\n")
        f.write(
            all_data.groupby('Generator')[
                ['Mean_Knn', 'Mean_Ratio_Knn', 'Mean_Ratio_10Knn', 'Duplicates']  # ← add Duplicates
            ].mean().round(3).to_string()
        )
        f.write("\n\n")

        # ── ML utility (F1) per classifier ────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("ML UTILITY (F1) PER CLASSIFIER (averaged across datasets)\n")
        f.write("=" * 60 + "\n")
        utility_table = compute_utility_per_classifier(result_files)
        f.write(utility_table.to_string())
        f.write("\n\n")

    print(f"Global results saved to {output_file}")


# ── Run analysis ───────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

dataset_groups = {
    'bcwd': {
        'files': [
            'final_logs/bcwd_10_log.csv',
            'final_logs/bcwd_25_log.csv',
            'final_logs/bcwd_50_log.csv',
        ],
        'names':  ['BCWD 10%', 'BCWD 25%', 'BCWD 50%'],
        'output': 'results/bcwd_analysis_results.txt',
        'plot':   'results/bcwd_plot.png',
    },
    'bcwp': {
        'files': [
            'final_logs/bcwp_10_log.csv',
            'final_logs/bcwp_25_log.csv',
            'final_logs/bcwp_50_log.csv',
        ],
        'names':  ['BCWP 10%', 'BCWP 25%', 'BCWP 50%'],
        'output': 'results/bcwp_analysis_results.txt',
        'plot':   'results/bcwp_plot.png',
    },
    'bcc': {
        'files': [
            'final_logs/bcc_10_log.csv',
            'final_logs/bcc_25_log.csv',
            'final_logs/bcc_50_log.csv',
        ],
        'names':  ['BCC 10%', 'BCC 25%', 'BCC 50%'],
        'output': 'results/bcc_analysis_results.txt',
        'plot':   'results/bcc_plot.png',
    },
    'pima': {
        'files': [
            'final_logs/pima_10_log.csv',
            'final_logs/pima_25_log.csv',
            'final_logs/pima_50_log.csv',
        ],
        'names':  ['PIMA 10%', 'PIMA 25%', 'PIMA 50%'],
        'output': 'results/pima_analysis_results.txt',
        'plot':   'results/pima_plot.png',
    },
    'hcv': {
        'files': [
            'final_logs/hcv_10_log.csv',
            'final_logs/hcv_25_log.csv',
            'final_logs/hcv_50_log.csv',
        ],
        'names':  ['HCV 10%', 'HCV 25%', 'HCV 50%'],
        'output': 'results/hcv_analysis_results.txt',
        'plot':   'results/hcv_plot.png',
    },
    'parkinsons': {
        'files': [
            'final_logs/parkinsons_5_log.csv',
            'final_logs/parkinsons_10_log.csv',
            'final_logs/parkinsons_15_log.csv',
        ],
        'names':  ['Parkinsons 5%', 'Parkinsons 10%', 'Parkinsons 15%'],
        'output': 'results/parkinsons_analysis_results.txt',
        'plot':   'results/parkinsons_plot.png',
    },
    'ilpd': {
        'files': [
            'final_logs/ilpd_5_log.csv',
            'final_logs/ilpd_10_log.csv',
            'final_logs/ilpd_15_log.csv',
        ],
        'names':  ['ILPD 5%', 'ILPD 10%', 'ILPD 15%'],
        'output': 'results/ilpd_analysis_results.txt',
        'plot':   'results/ilpd_plot.png',
    },
}

# ── Per-group analysis ─────────────────────────────────────────────────────────
for group_name, config in dataset_groups.items():
    print("\n" + "=" * 60)
    print(f"GROUP: {group_name.upper()}")
    print("=" * 60)

    print_results(config['files'], config['names'])
    print_utility_by_model(config['files'], config['names'])
    print_privacy_metrics(config['files'], config['names'])  # ← correct place
    plot_results(config['files'], config['names'], output_file=config['plot'])
    save_results(config['files'], config['names'], output_file=config['output'])


# ── Build file lists ───────────────────────────────────────────────────────────
all_files       = [f for config in dataset_groups.values() for f in config['files']]
all_names       = [n for config in dataset_groups.values() for n in config['names']]
tvae_files      = files_with_tvae(all_files)
tvae_names      = [
    config['names'][config['files'].index(f)]
    for config in dataset_groups.values()
    for f in config['files']
    if f in tvae_files
]

print(f"\nAll datasets      : {len(all_files)}")
print(f"TVAE subset       : {len(tvae_files)}")
print(f"TVAE subset files : {tvae_files}")

# TVAE-free copies of all 21 logs — used for every ALL-21 "excludes TVAE" call below
all_files_no_tvae = strip_tvae(all_files)

os.makedirs('results/global', exist_ok=True)


# ── Global results: 21 datasets (TVAE excluded by design) ────────────────────
print("\n" + "=" * 60)
print(f"GLOBAL RESULTS — ALL {len(all_files)} DATASETS")
print("=" * 60)
print_results(all_files_no_tvae, all_names)
print_utility_by_model(all_files_no_tvae, all_names)
print_privacy_metrics(all_files_no_tvae, all_names)
plot_results(
    all_files_no_tvae, all_names,
    output_file=f"results/global/global_{len(all_files)}_plot.png"
)
save_global_results(
    all_files_no_tvae,
    label=f"ALL {len(all_files)} DATASETS",
    output_file=f"results/global/global_{len(all_files)}_results.txt"
)


# ── Global results: TVAE subset ────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"GLOBAL RESULTS — TVAE SUBSET ({len(tvae_files)} DATASETS)")
print("=" * 60)
print_results(tvae_files, tvae_names)
print_utility_by_model(tvae_files, tvae_names)
plot_results(
    tvae_files, tvae_names,
    output_file=f"results/global/global_{len(tvae_files)}_plot.png"
)
save_global_results(
    tvae_files,
    label=f"TVAE SUBSET ({len(tvae_files)} DATASETS)",
    output_file=f"results/global/global_{len(tvae_files)}_results.txt"
)


# ── Statistical tests: 21 datasets (TVAE excluded by design) ─────────────────
stats_file_21 = f"results/global/statistical_tests_{len(all_files)}.txt"
with open(stats_file_21, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    wilcoxon_tradeoff(all_files_no_tvae)
    wilcoxon_utility(all_files_no_tvae)
    wilcoxon_privacy(all_files_no_tvae)
    friedman_ranks(all_files_no_tvae)
    sys.stdout = original_stdout
print(f"\nStatistical tests ({len(all_files)} datasets) saved to {stats_file_21}")


# ── Statistical tests: TVAE subset ────────────────────────────────────────────
stats_file_tvae = f"results/global/statistical_tests_{len(tvae_files)}.txt"
with open(stats_file_tvae, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    wilcoxon_tradeoff(tvae_files)
    wilcoxon_utility(tvae_files)
    wilcoxon_privacy(tvae_files)
    friedman_ranks(tvae_files)
    sys.stdout = original_stdout
print(f"Statistical tests (TVAE subset, {len(tvae_files)} datasets) saved to {stats_file_tvae}")