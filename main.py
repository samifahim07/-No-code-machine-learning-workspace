"""
Step by step ML helper (Streamlit).

Run it with:
    pip install -r requirements.txt
    streamlit run main.py

Flow: upload csv -> pick target -> overview -> missing values -> duplicates
      -> visualization -> split/encode/scale -> train models -> compare -> save
Every step gives the user a choice before anything is changed.
"""
import io
import math
import pickle
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no popup windows, streamlit draws the figures itself
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, OrdinalEncoder, LabelEncoder,
                                   label_binarize)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score, r2_score,
                             mean_squared_error, mean_absolute_error)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

RANDOM_STATE = 42
SEPARATORS = {"Comma ( , )": ",", "Semicolon ( ; )": ";", "Tab": "\t", "Pipe ( | )": "|"}

STEPS = [
    "1. Upload and target",
    "2. Dataset overview",
    "3. Missing values",
    "4. Duplicate values",
    "5. Visualization",
    "6. Split, encode, scale",
    "7. Train models",
    "8. Compare models",
    "9. Save",
]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def init_state():
    # everything the app remembers between button clicks lives here
    defaults = {"df_raw": None, "df_work": None, "file_key": None, "target": None,
                "task": None, "prep": None, "results": {}, "log": [], "started": False}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(text):
    # the log is saved later in the "save entire work" option
    st.session_state.log.append(text)


def need(*keys):
    # stop the page politely if an earlier step is not finished yet
    for key in keys:
        if st.session_state.get(key) is None:
            st.warning("Please finish the earlier steps first.")
            st.stop()


def guess_task(series):
    # quick guess only, the user can always override it
    if series.dtype == "object" or series.dtype == "bool" or str(series.dtype) == "category":
        return "Classification"
    if series.nunique() <= 10:
        return "Classification"
    return "Regression"


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)  # avoids the "too many open figures" warning


# ----------------------------------------------------------------------------
# step 1: upload and target
# ----------------------------------------------------------------------------
def step_upload():
    st.header("Step 1: Upload your CSV and choose the target column")

    sep_label = st.selectbox("Separator used in your file", list(SEPARATORS))
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        key = (file.name, file.size, sep_label)
        if st.session_state.file_key != key:  # only re-read when something really changed
            try:
                df = pd.read_csv(file, sep=SEPARATORS[sep_label])
            except Exception as err:
                st.error(f"Could not read the file: {err}")
                return
            st.session_state.df_raw = df
            st.session_state.df_work = df.copy()
            st.session_state.file_key = key
            st.session_state.target = None
            st.session_state.task = None
            st.session_state.prep = None
            st.session_state.results = {}
            st.session_state.log = [f"Loaded {file.name} with shape {df.shape}"]
        # improvement idea: add an option to load excel or a sample dataset too

    if st.session_state.df_raw is None:
        st.info("Upload a file to begin.")
        return

    raw = st.session_state.df_raw
    st.write(f"Loaded data: {raw.shape[0]} rows and {raw.shape[1]} columns")
    st.dataframe(raw.head())

    cols = list(raw.columns)
    target = st.selectbox("Which column do you want to predict (target)?", cols, index=len(cols) - 1)
    guess = guess_task(raw[target])
    st.caption(f"This looks like a {guess.lower()} problem. You can change it below if it is wrong.")
    task = st.radio("Problem type", ["Classification", "Regression"],
                    index=0 if guess == "Classification" else 1,
                    key=f"task_{target}", horizontal=True)

    if task == "Classification" and raw[target].nunique() > 20:
        st.warning("This target has more than 20 different values. Regression may fit better.")

    if st.button("Confirm target and problem type"):
        st.session_state.target = target
        st.session_state.task = task
        st.session_state.df_work = raw.copy()  # start clean again from the original data
        st.session_state.prep = None
        st.session_state.results = {}
        add_log(f"Target = {target}, problem type = {task}")
        st.success(f"Target set to '{target}' ({task}). Go to the next step from the sidebar.")


# ----------------------------------------------------------------------------
# step 2: overview
# ----------------------------------------------------------------------------
def step_overview():
    st.header("Step 2: Dataset overview")
    need("df_work", "target")
    df = st.session_state.df_work

    n = st.slider("How many rows for head and tail?", 3, 20, 5)
    full_describe = st.checkbox("Include text columns in describe", value=False)

    tabs = st.tabs(["Head", "Tail", "Columns", "Shape", "Dtypes", "Info", "Describe"])
    with tabs[0]:
        st.dataframe(df.head(n))
    with tabs[1]:
        st.dataframe(df.tail(n))
    with tabs[2]:
        st.write(list(df.columns))
    with tabs[3]:
        st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
    with tabs[4]:
        st.dataframe(df.dtypes.astype(str).rename("dtype"))
    with tabs[5]:
        buf = io.StringIO()
        df.info(buf=buf)  # info() prints instead of returning, so we catch it
        st.text(buf.getvalue())
    with tabs[6]:
        st.dataframe(df.describe(include="all" if full_describe else None))
    # improvement idea: add a unique-values-per-column table, it helps a lot to spot id columns

    # ID-like columns hurt models, so let the user drop them right here
    drop = st.multiselect("Optional: drop useless columns (like id or name)",
                          [c for c in df.columns if c != st.session_state.target])
    if st.button("Drop selected columns") and drop:
        st.session_state.df_work = df.drop(columns=drop)
        add_log(f"Dropped columns: {drop}")
        st.success("Dropped. Reopen this step to see the update.")


# ----------------------------------------------------------------------------
# step 3: missing values
# ----------------------------------------------------------------------------
def step_missing():
    st.header("Step 3: Missing values")
    need("df_work", "target")
    df = st.session_state.df_work
    target = st.session_state.target

    miss = df.isna().sum()
    table = pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(2)})
    st.dataframe(table[table["missing"] > 0] if miss.sum() > 0 else table.head(0))
    if miss.sum() == 0:
        st.success("No missing values found.")

    choice = st.radio("What do you want to do?",
                      ["Only check (no changes)", "Handle missing values"])
    if choice == "Only check (no changes)":
        return

    threshold = st.slider("Drop a column if more than this % of it is missing", 10, 100, 60)
    num_strategy = st.selectbox("Numeric columns", ["Median", "Mean", "Mode", "Drop rows"])
    cat_strategy = st.selectbox("Text columns", ["Mode", "Fill with 'Unknown'", "Drop rows"])

    if st.button("Apply missing value handling"):
        out = df.copy()
        before = out.shape
        out = out.dropna(subset=[target])  # a row without a label is useless, never fill the target

        drop_cols = [c for c in out.columns if c != target and out[c].isna().mean() * 100 > threshold]
        out = out.drop(columns=drop_cols)

        num_cols = [c for c in out.select_dtypes(include="number").columns if c != target]
        cat_cols = [c for c in out.columns if c not in num_cols and c != target]

        if num_cols:
            if num_strategy == "Drop rows":
                out = out.dropna(subset=num_cols)
            elif num_strategy == "Mode":
                out[num_cols] = out[num_cols].fillna(out[num_cols].mode().iloc[0])
            else:
                fill = out[num_cols].median() if num_strategy == "Median" else out[num_cols].mean()
                out[num_cols] = out[num_cols].fillna(fill)

        for c in cat_cols:
            if cat_strategy == "Drop rows":
                out = out.dropna(subset=[c])
            elif cat_strategy == "Mode":
                mode = out[c].mode()
                if not mode.empty:
                    out[c] = out[c].fillna(mode.iloc[0])
            else:
                out[c] = out[c].fillna("Unknown")

        st.session_state.df_work = out
        st.session_state.prep = None      # old split is not valid anymore
        st.session_state.results = {}
        add_log(f"Missing values: numeric={num_strategy}, text={cat_strategy}, "
                f"dropped columns={drop_cols}, shape {before} -> {out.shape}")
        st.success(f"Done. Shape changed from {before} to {out.shape}. Dropped columns: {drop_cols}")
    # known weak point: we fill before splitting, which leaks a tiny bit of test info.
    # For a stricter version move the imputer into the split step and fit it on train only.


# ----------------------------------------------------------------------------
# step 4: duplicates
# ----------------------------------------------------------------------------
def step_duplicates():
    st.header("Step 4: Duplicate values")
    need("df_work", "target")
    df = st.session_state.df_work

    dup_count = int(df.duplicated().sum())
    st.write(f"Duplicate rows found: {dup_count}")
    if dup_count > 0:
        st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(50))

    choice = st.radio("What do you want to do?",
                      ["Only check (no changes)", "Drop duplicates, keep first", "Drop duplicates, keep last"])
    if choice != "Only check (no changes)" and st.button("Apply"):
        keep = "first" if "first" in choice else "last"
        out = df.drop_duplicates(keep=keep)
        st.session_state.df_work = out
        st.session_state.prep = None
        st.session_state.results = {}
        add_log(f"Duplicates: removed {len(df) - len(out)} rows (keep {keep})")
        st.success(f"Removed {len(df) - len(out)} rows. New shape: {out.shape}")


# ----------------------------------------------------------------------------
# step 5: visualization
# ----------------------------------------------------------------------------
def grid_plot(df, cols, kind):
    # draws many small plots in a 3 column grid
    per_row = 3
    rows = math.ceil(len(cols) / per_row)
    fig, axes = plt.subplots(rows, per_row, figsize=(5 * per_row, 3.6 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        if kind == "hist":
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
        elif kind == "box":
            sns.boxplot(x=df[col].dropna(), ax=ax)
        else:  # count plot, only top 10 values so long lists stay readable
            df[col].astype(str).value_counts().head(10).plot.bar(ax=ax)
        ax.set_title(str(col))
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    show_fig(fig)


def step_visualization():
    st.header("Step 5: Visualization")
    need("df_work", "target")
    df = st.session_state.df_work
    target = st.session_state.target
    task = st.session_state.task

    plots = st.multiselect("Which plots do you want?",
                           ["Target distribution", "Histograms (numeric)", "Boxplots (numeric)",
                            "Correlation heatmap", "Countplots (text columns)"],
                           default=["Target distribution", "Correlation heatmap"])
    limit = st.slider("Maximum columns per plot type", 3, 15, 6)

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    cat_cols = [c for c in df.columns if c not in num_cols and c != target]

    if "Target distribution" in plots:
        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if task == "Classification":
            df[target].astype(str).value_counts().plot.bar(ax=ax)
        else:
            sns.histplot(pd.to_numeric(df[target], errors="coerce").dropna(), kde=True, ax=ax)
        show_fig(fig)

    if "Histograms (numeric)" in plots and num_cols:
        st.subheader("Histograms")
        grid_plot(df, num_cols[:limit], "hist")

    if "Boxplots (numeric)" in plots and num_cols:
        st.subheader("Boxplots")
        grid_plot(df, num_cols[:limit], "box")

    if "Correlation heatmap" in plots:
        st.subheader("Correlation heatmap")
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=numeric_df.shape[1] <= 12, fmt=".2f",
                        cmap="coolwarm", ax=ax)
            show_fig(fig)
        else:
            st.info("Need at least two numeric columns.")

    if "Countplots (text columns)" in plots and cat_cols:
        st.subheader("Countplots")
        grid_plot(df, cat_cols[:limit], "count")
    # improvement idea: add feature-vs-target plots (scatter for regression, boxplot per class)


# ----------------------------------------------------------------------------
# step 6: split, encode, scale
# ----------------------------------------------------------------------------
def step_preprocess():
    st.header("Step 6: Train test split, encoding and scaling")
    need("df_work", "target")
    df = st.session_state.df_work
    target = st.session_state.target
    task = st.session_state.task

    c1, c2 = st.columns(2)
    test_size = c1.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    seed = c2.number_input("Random state", 0, 9999, RANDOM_STATE)
    stratify = st.checkbox("Keep class balance in both parts (stratify)",
                           value=(task == "Classification"), disabled=(task != "Classification"))
    enc_choice = st.radio("Encoding for text columns", ["One-Hot", "Ordinal (numbers per category)"], horizontal=True)
    scale_choice = st.radio("Scaling", ["Standard", "MinMax", "Robust", "None"], horizontal=True)

    if not st.button("Run split, encoding and scaling"):
        prep = st.session_state.prep
        if prep is None:
            return
    else:
        X = df.drop(columns=[target])
        y = df[target]

        if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
            st.error("There are still missing values. Please handle them in Step 3 first.")
            return

        # target: classification needs integer labels, regression needs numbers
        label_enc = None
        class_names = None
        if task == "Classification":
            label_enc = LabelEncoder()
            y = pd.Series(label_enc.fit_transform(y), index=y.index, name=target)
            class_names = [str(c) for c in label_enc.classes_]
        else:
            y = pd.to_numeric(y, errors="coerce")
            if y.isna().any():
                st.error("The target has non-numeric values, so it cannot be used for regression.")
                return

        cat_cols = [c for c in X.columns if c not in X.select_dtypes(include="number").columns]
        X[cat_cols] = X[cat_cols].astype(str)  # mixed types break the encoders

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(seed),
                stratify=y if (stratify and task == "Classification") else None)
        except ValueError as err:
            st.error(f"Split failed: {err}")
            return

        # encoders and scalers are fitted on TRAIN only, then applied to test (no leakage)
        encoder = None
        if cat_cols:
            if enc_choice == "One-Hot":
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                tr = encoder.fit_transform(X_train[cat_cols])
                te = encoder.transform(X_test[cat_cols])
                names = encoder.get_feature_names_out(cat_cols)
                X_train = pd.concat([X_train.drop(columns=cat_cols),
                                     pd.DataFrame(tr, columns=names, index=X_train.index)], axis=1)
                X_test = pd.concat([X_test.drop(columns=cat_cols),
                                    pd.DataFrame(te, columns=names, index=X_test.index)], axis=1)
            else:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
                X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        # improvement idea: warn when a text column has hundreds of unique values (one-hot explodes)

        scaler = None
        if scale_choice != "None":
            scaler = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}[scale_choice]()
            cols = X_train.columns
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)

        prep = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                "raw_columns": list(df.drop(columns=[target]).columns), "cat_cols": cat_cols,
                "encoder": encoder, "encoder_type": enc_choice if cat_cols else None,
                "scaler": scaler, "label_encoder": label_enc, "class_names": class_names}
        st.session_state.prep = prep
        st.session_state.results = {}
        add_log(f"Split test_size={test_size}, seed={int(seed)}, stratify={stratify}, "
                f"encoding={enc_choice if cat_cols else 'not needed'}, scaling={scale_choice}")

    # show the output of this step
    st.success("Preprocessing is ready.")
    st.write(f"X_train: {prep['X_train'].shape}, X_test: {prep['X_test'].shape}, "
             f"y_train: {prep['y_train'].shape}, y_test: {prep['y_test'].shape}")
    t1, t2, t3 = st.tabs(["X_train", "X_test", "y_train"])
    t1.dataframe(prep["X_train"].head(10))
    t2.dataframe(prep["X_test"].head(10))
    t3.dataframe(prep["y_train"].head(10))
    if prep["class_names"]:
        st.write("Class labels:", dict(enumerate(prep["class_names"])))


# ----------------------------------------------------------------------------
# step 7: models
# ----------------------------------------------------------------------------
def get_models(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, random_state=RANDOM_STATE),  # probability=True is needed for roc auc
            "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, allow_writing_files=False),
            "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
            "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, allow_writing_files=False),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
    }


# search spaces for the tuning option (random search, so it stays reasonably fast)
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Linear Regression": {"fit_intercept": [True, False]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
                      "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    "Decision Tree": {"max_depth": [None, 3, 5, 10, 20], "min_samples_split": [2, 5, 10],
                      "min_samples_leaf": [1, 2, 4]},
    "SVM": {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    "SVR": {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1],
                 "iterations": [200, 400], "l2_leaf_reg": [1, 3, 5, 9]},
    "AdaBoost": {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "LightGBM": {"n_estimators": [100, 200, 400], "learning_rate": [0.01, 0.05, 0.1],
                 "num_leaves": [15, 31, 63], "max_depth": [-1, 5, 10], "min_child_samples": [10, 20, 40]},
}


def specificity_sensitivity(cm):
    # binary: uses class 1 as the positive class
    # multiclass: one-vs-rest for each class, then the average
    sens, spec = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    if cm.shape[0] == 2:
        return sens[1], spec[1]
    return float(np.mean(sens)), float(np.mean(spec))


def evaluate_classification(model, prep):
    Xtr, Xte, ytr, yte = prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"]
    names = prep["class_names"]
    labels = list(range(len(names)))

    train_pred = np.ravel(model.predict(Xtr))  # catboost can return 2d arrays, ravel fixes it
    test_pred = np.ravel(model.predict(Xte))

    cm = confusion_matrix(yte, test_pred, labels=labels)
    report = pd.DataFrame(classification_report(yte, test_pred, labels=labels, target_names=names,
                                                output_dict=True, zero_division=0)).T
    sens, spec = specificity_sensitivity(cm)

    proba, auc = None, np.nan
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)
        try:
            if len(names) == 2:
                auc = roc_auc_score(yte, proba[:, 1])
            else:
                auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            auc = np.nan  # happens when a class is missing from the test part

    return {"train_acc": accuracy_score(ytr, train_pred), "test_acc": accuracy_score(yte, test_pred),
            "f1": f1_score(yte, test_pred, average="weighted", zero_division=0),
            "sensitivity": sens, "specificity": spec, "auc": auc,
            "cm": cm, "report": report, "proba": proba, "y_pred": test_pred}


def regression_scores(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.ravel(y_pred).astype(float)
    mean = y_true.mean()
    return {"r2": r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": mean_absolute_error(y_true, y_pred),
            # relative squared error and relative absolute error, compared to just predicting the mean
            "rse": float(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mean) ** 2)),
            "rae": float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - mean)))}


def evaluate_regression(model, prep):
    train = regression_scores(prep["y_train"], model.predict(prep["X_train"]))
    y_pred = np.ravel(model.predict(prep["X_test"]))
    test = regression_scores(prep["y_test"], y_pred)
    return {"train": train, "test": test, "y_pred": y_pred}


def train_one(name, mode, task, prep, n_iter, cv):
    model = get_models(task)[name]
    best_params, cv_score = None, None
    if mode == "Best hyperparameter tuning":
        grid = PARAM_GRIDS[name]
        size = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(model, grid, n_iter=min(n_iter, size), cv=cv,
                                    scoring="accuracy" if task == "Classification" else "r2",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(prep["X_train"], prep["y_train"])
        model, best_params, cv_score = search.best_estimator_, search.best_params_, search.best_score_
    else:
        model.fit(prep["X_train"], prep["y_train"])

    result = {"model": model, "mode": mode, "name": name, "best_params": best_params, "cv_score": cv_score}
    if task == "Classification":
        result.update(evaluate_classification(model, prep))
    else:
        result.update(evaluate_regression(model, prep))
    return result


def show_classification_result(r, prep):
    m = st.columns(6)
    m[0].metric("Train accuracy", f"{r['train_acc']:.4f}")
    m[1].metric("Test accuracy", f"{r['test_acc']:.4f}")
    m[2].metric("Sensitivity", f"{r['sensitivity']:.4f}")
    m[3].metric("Specificity", f"{r['specificity']:.4f}")
    m[4].metric("ROC AUC", "n/a" if np.isnan(r["auc"]) else f"{r['auc']:.4f}")
    m[5].metric("F1 (weighted)", f"{r['f1']:.4f}")

    st.write("Classification report (test data)")
    st.dataframe(r["report"].round(4))

    names = prep["class_names"]
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        show_fig(fig)
    with right:
        if r["proba"] is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            yte = prep["y_test"]
            if len(names) == 2:
                fpr, tpr, _ = roc_curve(yte, r["proba"][:, 1])
                ax.plot(fpr, tpr, label=f"AUC = {r['auc']:.3f}")
            else:
                y_bin = label_binarize(yte, classes=list(range(len(names))))
                for i, cname in enumerate(names):
                    if y_bin[:, i].sum() == 0:
                        continue
                    fpr, tpr, _ = roc_curve(y_bin[:, i], r["proba"][:, i])
                    ax.plot(fpr, tpr, label=cname)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve")
            ax.legend(fontsize=7)
            show_fig(fig)


def show_regression_result(r, prep):
    table = pd.DataFrame({"Train": r["train"], "Test": r["test"]}).T
    table.columns = ["R2", "RMSE", "MAE", "RSE", "RAE"]
    st.dataframe(table.round(4))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(prep["y_test"], r["y_pred"], alpha=0.5)
    lo, hi = float(np.min(prep["y_test"])), float(np.max(prep["y_test"]))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs predicted (test)")
    show_fig(fig)


def step_models():
    st.header("Step 7: Train models")
    need("prep", "task")
    task = st.session_state.task
    prep = st.session_state.prep

    mode = st.radio("How should the models be trained?",
                    ["Default parameters", "Best hyperparameter tuning"], horizontal=True)
    available = list(get_models(task))
    picked = st.multiselect("Choose the algorithms to run", available, default=available[:3])

    n_iter, cv = 10, 3
    if mode == "Best hyperparameter tuning":
        c1, c2 = st.columns(2)
        n_iter = c1.slider("Random search tries per model", 5, 40, 10)
        cv = c2.slider("Cross validation folds", 2, 5, 3)
        st.caption("Tuning can be slow for SVM and CatBoost on big datasets.")

    if st.button("Run selected models") and picked:
        bar = st.progress(0.0)
        for i, name in enumerate(picked):
            with st.spinner(f"Training {name}..."):
                try:
                    key = f"{name} [{'Tuned' if mode.startswith('Best') else 'Default'}]"
                    st.session_state.results[key] = train_one(name, mode, task, prep, n_iter, cv)
                    add_log(f"Trained {key}")
                except Exception as err:
                    st.error(f"{name} failed: {err}")
            bar.progress((i + 1) / len(picked))

    results = st.session_state.results
    if not results:
        return
    if st.button("Clear all results"):
        st.session_state.results = {}
        st.rerun()

    for key, r in results.items():
        with st.expander(key, expanded=False):
            if r["best_params"]:
                st.write("Best parameters:", r["best_params"], f"| CV score: {r['cv_score']:.4f}")
            if task == "Classification":
                show_classification_result(r, prep)
            else:
                show_regression_result(r, prep)


# ----------------------------------------------------------------------------
# step 8: compare
# ----------------------------------------------------------------------------
def build_comparison(task):
    rows = []
    for key, r in st.session_state.results.items():
        if task == "Classification":
            rows.append({"Model": key, "Train Accuracy": r["train_acc"], "Test Accuracy": r["test_acc"],
                         "Gap (train - test)": r["train_acc"] - r["test_acc"], "F1": r["f1"],
                         "Sensitivity": r["sensitivity"], "Specificity": r["specificity"], "ROC AUC": r["auc"]})
        else:
            rows.append({"Model": key, "Train R2": r["train"]["r2"], "Test R2": r["test"]["r2"],
                         "Test RMSE": r["test"]["rmse"], "Test MAE": r["test"]["mae"],
                         "Test RSE": r["test"]["rse"], "Test RAE": r["test"]["rae"]})
    return pd.DataFrame(rows)


def best_model_key(table, rank_by, higher_is_better):
    col = table[rank_by].fillna(-np.inf if higher_is_better else np.inf)
    return table.loc[col.idxmax() if higher_is_better else col.idxmin(), "Model"]


def step_compare():
    st.header("Step 8: Which model performed best?")
    need("prep", "task")
    task = st.session_state.task
    if not st.session_state.results:
        st.info("Train at least one model in Step 7.")
        return

    table = build_comparison(task)
    if task == "Classification":
        options = {"Test Accuracy": True, "F1": True, "ROC AUC": True}
    else:
        options = {"Test R2": True, "Test RMSE": False, "Test MAE": False}
    rank_by = st.selectbox("Rank models by", list(options))
    higher = options[rank_by]

    table = table.sort_values(rank_by, ascending=not higher).reset_index(drop=True)
    st.dataframe(table.round(4))
    best = best_model_key(table, rank_by, higher)
    st.success(f"Best model by {rank_by}: {best}")
    st.session_state["best_key"] = best

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(table) + 2))
    ax.barh(table["Model"][::-1], table[rank_by][::-1])
    ax.set_xlabel(rank_by)
    fig.tight_layout()
    show_fig(fig)

    if task == "Classification":
        big_gap = table[table["Gap (train - test)"] > 0.1]["Model"].tolist()
        if big_gap:
            st.warning(f"Possible overfitting (train is much higher than test): {big_gap}")
    # improvement idea: compare with cross validation instead of a single test split


# ----------------------------------------------------------------------------
# step 9: save
# ----------------------------------------------------------------------------
LOAD_SNIPPET = '''import pickle, pandas as pd

bundle = pickle.load(open("model_bundle.pkl", "rb"))   # only load files you trust
new = pd.DataFrame([...])                               # same columns as the original data (no target)

cat = bundle["cat_cols"]
new[cat] = new[cat].astype(str)
if bundle["encoder"] is not None and bundle["encoder_type"] == "One-Hot":
    enc = pd.DataFrame(bundle["encoder"].transform(new[cat]),
                       columns=bundle["encoder"].get_feature_names_out(cat), index=new.index)
    new = pd.concat([new.drop(columns=cat), enc], axis=1)
elif bundle["encoder"] is not None:
    new[cat] = bundle["encoder"].transform(new[cat])
new = new[bundle["feature_columns"]]
if bundle["scaler"] is not None:
    new = pd.DataFrame(bundle["scaler"].transform(new), columns=new.columns)
print(bundle["model"].predict(new))'''


def step_save():
    st.header("Step 9: Save your work")
    need("prep", "task")
    if not st.session_state.results:
        st.info("Train at least one model in Step 7.")
        return
    prep = st.session_state.prep
    results = st.session_state.results

    choice = st.radio("What do you want to save?",
                      ["Save the best model (pickle)", "Save the entire work without the model"])

    if choice.startswith("Save the best"):
        keys = list(results)
        default = keys.index(st.session_state["best_key"]) if st.session_state.get("best_key") in keys else 0
        key = st.selectbox("Model to save", keys, index=default)
        bundle = {"model": results[key]["model"], "model_name": key,
                  "task": st.session_state.task, "target": st.session_state.target,
                  "raw_columns": prep["raw_columns"], "cat_cols": prep["cat_cols"],
                  "encoder": prep["encoder"], "encoder_type": prep["encoder_type"],
                  "scaler": prep["scaler"], "label_encoder": prep["label_encoder"],
                  "class_names": prep["class_names"], "feature_columns": list(prep["X_train"].columns)}
        st.download_button("Download model_bundle.pkl", pickle.dumps(bundle), file_name="model_bundle.pkl")
        st.caption("The file has the model plus the encoder and scaler, so new data can be prepared the same way.")
        with st.expander("How to use the saved file"):
            st.code(LOAD_SNIPPET, language="python")
    else:
        table = build_comparison(st.session_state.task)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("cleaned_dataset.csv", st.session_state.df_work.to_csv(index=False))
            z.writestr("X_train.csv", prep["X_train"].to_csv(index=False))
            z.writestr("X_test.csv", prep["X_test"].to_csv(index=False))
            z.writestr("y_train.csv", prep["y_train"].to_csv(index=False))
            z.writestr("y_test.csv", prep["y_test"].to_csv(index=False))
            z.writestr("model_comparison.csv", table.to_csv(index=False))
            z.writestr("steps_log.txt", "\n".join(st.session_state.log))
        st.download_button("Download full_work.zip", buf.getvalue(), file_name="full_work.zip")
        st.caption("Contains the cleaned data, split data, results table and a log of every choice you made.")


# ----------------------------------------------------------------------------
# HTML interface (landing page + global look)
# ----------------------------------------------------------------------------
LANDING_FILE = Path(__file__).parent / "landing.html"

APP_CSS = """
<style>
.stApp{background:linear-gradient(180deg,#fdf2f8 0%,#f5f3ff 100%);}
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #fbcfe8;}
h1,h2,h3{letter-spacing:-.02em;color:#3b0a2a;}
div.stButton>button,div.stDownloadButton>button{
  border:0;border-radius:12px;padding:.6rem 1.4rem;font-weight:600;color:#fff;
  background:linear-gradient(90deg,#db2777,#8b5cf6);transition:.2s;}
div.stButton>button:hover,div.stDownloadButton>button:hover{transform:translateY(-2px);
  box-shadow:0 8px 22px rgba(219,39,119,.35);color:#fff;}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #fbcfe8;border-radius:14px;padding:14px;
  box-shadow:0 4px 14px rgba(219,39,119,.06);}
[data-testid="stExpander"]{background:#ffffff;border-radius:14px;border:1px solid #fbcfe8;}
</style>
"""


def show_landing():
    # the HTML page is shown first, the native button below moves the user into the app
    if LANDING_FILE.exists():
        components.html(LANDING_FILE.read_text(encoding="utf-8"), height=1500, scrolling=True)
    else:
        st.title("ML Workflow Helper")
        st.info("landing.html was not found next to main.py")
    _, mid, _ = st.columns([1, 1, 1])
    if mid.button("🚀 Start the workflow", use_container_width=True):
        st.session_state.started = True
        st.rerun()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ML Workflow Helper", layout="wide")
    init_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    if not st.session_state.started:   # first screen = the HTML interface
        show_landing()
        return

    st.sidebar.title("ML Workflow Helper")
    step = st.sidebar.radio("Steps", STEPS)
    if st.session_state.target:
        st.sidebar.write(f"Target: {st.session_state.target} ({st.session_state.task})")
    if st.sidebar.button("Start over"):
        st.session_state.clear()   # also returns to the landing page
        st.rerun()

    pages = [step_upload, step_overview, step_missing, step_duplicates, step_visualization,
             step_preprocess, step_models, step_compare, step_save]
    pages[STEPS.index(step)]()


if __name__ == "__main__":
    main()
