import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from generative_functions_clean import (smotenc_model, smotenc_umap_model,)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier
from tqdm import tqdm
import time


os.makedirs('logs', exist_ok=True)

# Hyperparameter grids
param_grids_lr  = [{"solver": ["liblinear", "lbfgs"], "max_iter": [200, 500, 1500]}]
param_grids_rf  = [{"n_estimators": [150, 250], "max_depth": [15, 25, None]}]
param_grids_dt  = [{"max_depth": [15, 25, None]}]
param_grids_mlp = [{"learning_rate": ["constant", "adaptive"],
                    "hidden_layer_sizes": [(100,), (100, 100), (200,), (150, 100)],
                    "max_iter": [200, 500, 1500]}]
param_grids_xgb = [{"max_depth": [6, 10, 15], "n_estimators": [100, 150, 250]}]
param_grids_gs  = [{"var_smoothing": [1e-9]}]

target   = 'target'
cat_cols = []

def preprocess_default(filepath):
    return pd.read_csv(filepath).dropna().reset_index(drop=True)

def preprocess_hcv(filepath):
    df = pd.read_csv(filepath)
    # Remove uncertain outcomes
    df = df[df['Category'] != '0s=suspect Blood Donor'].copy()
    # Map to binary — combine all positive hepatitis stages
    df['target'] = df['Category'].map({
        '0=Blood Donor': 0,
        '1=Hepatitis':   1,
        '2=Fibrosis':    1,
        '3=Cirrhosis':   1,
    })
    # Drop original category and encode Sex
    df = df.drop(columns=['Category'])
    df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
    return df.dropna().reset_index(drop=True)

def preprocess_ilpd(filepath):
    df = pd.read_csv(filepath)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df.dropna().reset_index(drop=True)

datasets_config = {
    #  'bcwd_10': preprocess_default('bcwd_10.csv'),
    #  'bcwd_25': preprocess_default('bcwd_25.csv'),
    #  'bcwd_50': preprocess_default('bcwd_50.csv'),
    #  'bcwp_10': preprocess_default('bcwp_10.csv'),
    #  'bcwp_25': preprocess_default('bcwp_25.csv'),
    #  'bcwp_50': preprocess_default('bcwp_50.csv'),
    #  'bcc_10':  preprocess_default('bcc_10.csv'),
    #  'bcc_25':  preprocess_default('bcc_25.csv'),
    #  'bcc_50':  preprocess_default('bcc_50.csv'),
    #  'arcene_10': preprocess_default('arcene_10.csv'),
    #  'arcene_15': preprocess_default('arcene_15.csv'),
    #  'arcene_50': preprocess_default('arcene_50.csv'),
    #  'pima_10': preprocess_default('pima_10.csv'),
    #  'pima_25': preprocess_default('pima_25.csv'),
    #  'pima_50': preprocess_default('pima_50.csv'),
    #  'hcv_10':  preprocess_hcv('hcv_10.csv'),
    #  'hcv_25':  preprocess_hcv('hcv_25.csv'),
    #  'hcv_50':  preprocess_hcv('hcv_50.csv'),
    #  'ilpd_5':  preprocess_ilpd('ilpd_5.csv'),
    #  'ilpd_10': preprocess_ilpd('ilpd_10.csv'),
    #  'ilpd_15': preprocess_ilpd('ilpd_15.csv'),
      'parkinsons_5':  preprocess_default('parkinsons_5.csv'),
      'parkinsons_10': preprocess_default('parkinsons_10.csv'),
      'parkinsons_15': preprocess_default('parkinsons_15.csv'),
}

seeds = [220, 320, 420]

for dataset_name, data_raw in tqdm(datasets_config.items(), desc="Datasets"):

    if os.path.exists(f"logs/{dataset_name}_log.csv"):
        print(f"Skipping {dataset_name} — already done")
        continue

    data = data_raw.copy().reset_index(drop=True)
    data.columns = data.columns.astype(str)
    num_cols = [col for col in data.columns if col != target]
    storage = pd.DataFrame()

    for seed in tqdm(seeds, desc="Seeds", leave=False):
        counter = 0
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

        for train_index, test_index in tqdm(skf.split(
            data[[col for col in data.columns if col != target]],
            data[target]
        ), leave=False, desc="Folds"):

            time1 = time.time()

            train = data.loc[train_index, :].copy().reset_index(drop=True)
            test  = data.loc[test_index,  :].copy().reset_index(drop=True)

            minority_count = train[train[target] == 1].shape[0]
            n_neighbors = min(5, minority_count - 1)

            if minority_count < 2:
                print(f"Skipping {dataset_name} fold {counter} — too few minority instances ({minority_count})")
                counter += 1
                continue

            # ── SMOTE-NC ──────────────────────────────────────────────────────
            data_smote_nc = smotenc_model(
                train.copy(), num_cols.copy(), target, cat_cols.copy(),
                number_neighs=n_neighbors, seed=seed
            )
            data_smote_nc = data_smote_nc.dropna().reset_index(drop=True)
            for tipo, coluna in zip(train.dtypes, train.dtypes.index):
                col_str = str(coluna)
                if col_str in data_smote_nc.columns:
                    data_smote_nc[col_str] = data_smote_nc[col_str].astype(tipo)

            # ── UMAP-SMOTE-NC ─────────────────────────────────────────────────
            data_umap_smote_nc = smotenc_umap_model(
                train.copy(), num_cols.copy(), target, cat_cols.copy(),
                use_umap=True, number_neighs=n_neighbors, seed=seed
            )
            data_umap_smote_nc = data_umap_smote_nc.dropna().reset_index(drop=True)
            for tipo, coluna in zip(train.dtypes, train.dtypes.index):
                col_str = str(coluna)
                if col_str in data_umap_smote_nc.columns:
                    data_umap_smote_nc[col_str] = data_umap_smote_nc[col_str].astype(tipo)

            #-─ CTGAN ─────────────────────────────────
            # data_ctgan = generate_sdv_model(
            #     train.copy(), target, cat_cols.copy(), model="ctgan", seed=seed
            # )
            # data_ctgan = data_ctgan.dropna().reset_index(drop=True)

            #── TVAE ─────────────────────────────────
            # data_tvae = generate_sdv_model(
            #     train.copy(), target, cat_cols.copy(), model="tvae", seed=seed
            # )
            # data_tvae = data_tvae.dropna().reset_index(drop=True)

            # data_copula = generate_sdv_model(
            #     train.copy(), target, cat_cols.copy(), model="copula", seed=seed
            # )
            # data_copula = data_copula.dropna().reset_index(drop=True)
            # ──────────────────────────────────────────────────────────────────

            for dataset, title in tqdm(zip(
                [data_umap_smote_nc, data_smote_nc, train],
                ["umap_smote_nc",    "smote_nc",  "train"]
            ), leave=False, desc="Generators"):

                bin_dataset = dataset.copy()
                temp = pd.DataFrame()

                available_num_cols = [col for col in num_cols if col in bin_dataset.columns]

                bin_test_adapted = test.copy()
                bin_test_adapted[[col for col in bin_dataset.columns
                                   if col not in bin_test_adapted.columns]] = 0
                bin_test_adapted = bin_test_adapted[bin_dataset.columns]

                scaler = MinMaxScaler()
                scaler.fit(bin_dataset[available_num_cols])

                X_train_scaled = pd.DataFrame(
                    scaler.transform(bin_dataset[available_num_cols]),
                    index=bin_dataset.index, columns=available_num_cols
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(bin_test_adapted[available_num_cols]),
                    index=bin_test_adapted.index, columns=available_num_cols
                )

                X_train_scaled[target] = bin_dataset[target].values
                X_test_scaled[target]  = bin_test_adapted[target].values

                X = pd.concat([X_train_scaled, X_test_scaled]).reset_index(drop=True)
                y = X[target]
                X = X[[col for col in X.columns if col != target]]

                if X.isnull().any().any():
                    print(f"NaN found in X for {dataset_name} {title} fold {counter} — skipping")
                    continue

                separation = np.where(
                    X.index <= X_train_scaled.reset_index(drop=True).index.max(), -1, 0
                )
                ps = PredefinedSplit(separation)

                model1 = LogisticRegression(random_state=seed, n_jobs=-1)
                model2 = RandomForestClassifier(n_jobs=-1, random_state=seed)
                model3 = DecisionTreeClassifier(random_state=seed)
                model4 = MLPClassifier(random_state=seed)
                model5 = XGBClassifier(seed=seed)
                model6 = GaussianNB()

                for model, par, name_model in zip(
                    [model1, model2, model3, model4, model5, model6],
                    [param_grids_lr, param_grids_rf, param_grids_dt,
                     param_grids_mlp, param_grids_xgb, param_grids_gs],
                    ["lr", "rf", "dt", "mlp", "xgb", "gs"]
                ):
                    model_search_cv = GridSearchCV(
                        model, par, cv=ps, scoring="f1", n_jobs=1, refit=False
                    )
                    model_search_cv.fit(X, y)
                    temp_r = pd.DataFrame(model_search_cv.cv_results_)
                    temp_r["Model"] = name_model
                    temp = pd.concat([temp, temp_r], ignore_index=True)

                # Privacy metrics
                scaler_priv = MinMaxScaler()
                scaler_priv.fit(train[available_num_cols])
                original = scaler_priv.transform(train[available_num_cols])

                if title == "train":
                    searcher = scaler_priv.transform(test[available_num_cols])
                else:
                    searcher = scaler_priv.transform(dataset[available_num_cols])

                nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(original)
                distances, _ = nbrs.kneighbors(searcher)

                temp["Mean_Knn"]         = np.nanmean(distances[:, 0])
                temp["10_Knn"]           = np.nanquantile(distances[:, 0], 0.1)
                temp["Mean_Ratio_Knn"]   = np.nanmean(distances[:, 0] / distances[:, 1])
                temp["Mean_Ratio_10Knn"] = np.nanmean(distances[:, 0] / distances[:, 9])
                temp["10_Ratio_Knn"]     = np.nanquantile(distances[:, 0] / distances[:, 1], 0.1)
                temp["10_Ratio_10Knn"]   = np.nanquantile(distances[:, 0] / distances[:, 9], 0.1)
                temp["Duplicates"]       = (distances[:, 0] == 0).sum()
                temp["Generator"]        = title
                temp["Seed"]             = seed
                temp["fold"]             = counter
                temp["dataset"]          = dataset_name

                storage = pd.concat([storage, temp], ignore_index=True)

            time2 = time.time()
            print(f"{dataset_name} | Fold {counter} done in {time2-time1:.1f}s")
            counter += 1

    storage.to_csv(f"logs/{dataset_name}_log.csv", index=False)
    print(f"Done! Results saved to logs/{dataset_name}_log.csv")