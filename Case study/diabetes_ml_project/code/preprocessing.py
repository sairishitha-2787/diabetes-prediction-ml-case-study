# =============================================================================
# PIMA DIABETES — SHARED PREPROCESSING PIPELINE
# Single source of truth for all data loading and preparation steps.
# Imported by: model_training, model_evaluation, hyperparameter_tuning,
#              feature_importance, cost_benefit_analysis
# =============================================================================

import pandas as pd
import numpy as np
from types import SimpleNamespace

from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from imblearn.over_sampling  import RandomOverSampler

# ---------------------------------------------------------------------------
# Constants — shared across all scripts
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)

ENG_FEATURES = [c for c in COLUMN_NAMES if c != 'Outcome'] + [
    'Glucose_BMI_Interaction', 'Age_Group'
]


def build_pipeline(test_size=0.20, random_state=42):
    """Load data and run the full preprocessing pipeline.

    Steps
    -----
    1. Load raw CSV from URL
    2. Replace biologically-impossible zeros with NaN
    3. Median imputation
    4. Feature engineering (Glucose_BMI_Interaction, Age_Group)
    5. Stratified 80/20 train-test split
    6. Random oversampling on training set only (no leakage)
    7. MinMaxScaler fitted on balanced training data only

    Returns
    -------
    SimpleNamespace with attributes:
        df_raw       - raw loaded DataFrame (768 x 9)
        df           - imputed + engineered DataFrame (768 x 11)
        X_train_raw  - pre-ROS, pre-scale training features
        y_train_raw  - pre-ROS training labels
        X_train_res  - post-ROS, pre-scale training features
        y_train_res  - post-ROS (balanced) training labels
        X_train_sc   - scaled training features
        X_test_raw   - pre-scale test features (use for unscaled analysis)
        X_test_sc    - scaled test features
        y_test       - test labels
        scaler       - fitted MinMaxScaler instance
        feature_names- list of feature column names (ENG_FEATURES)
    """
    # 1. Load
    df_raw = pd.read_csv(DATA_URL, names=COLUMN_NAMES)
    df = df_raw.copy()

    # 2. Replace biologically-impossible zeros with NaN
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    # 3. Median imputation (robust to outliers — proved best experimentally)
    imputer = SimpleImputer(strategy='median')
    df[ZERO_COLS] = imputer.fit_transform(df[ZERO_COLS])

    # 4. Feature engineering
    df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']
    df['Age_Group'] = pd.cut(
        df['Age'], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    X = df[ENG_FEATURES].values
    y = df['Outcome'].values

    # 5. Stratified train-test split
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Random oversampling on training set only (prevents leakage)
    ros = RandomOverSampler(random_state=random_state)
    X_train_res, y_train_res = ros.fit_resample(X_train_raw, y_train_raw)

    # 7. MinMaxScaler fitted on balanced training data only
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    return SimpleNamespace(
        df_raw=df_raw,
        df=df,
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_train_sc=X_train_sc,
        X_test_raw=X_test,
        X_test_sc=X_test_sc,
        y_test=y_test,
        scaler=scaler,
        feature_names=ENG_FEATURES,
    )
