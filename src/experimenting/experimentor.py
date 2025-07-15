import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config_utils import (
    PathConfig,
    PreprocessConfig,
    FeatureConfig,
    ExperimentConfig,
)
from lifelines.utils import concordance_index
import shap
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """封裝實驗及校正結果"""

    model_type: str
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    train_c_index: float
    test_c_index: float
    model: Any
    shap_results: Dict[str, Any]
    feature_importance: Dict[str, Dict[str, float]]
    calibrated_test_predictions: Dict[str, pd.DataFrame] = field(default_factory=dict)
    calibrated_test_c_index: Dict[str, float] = field(default_factory=dict)
    whatif_treatment_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    whatif_continuous_results: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ========================================
# 實驗函數
# ========================================
def run_single_experiment(
    args: Tuple[
        pd.DataFrame, PreprocessConfig, FeatureConfig, int, str, ExperimentConfig
    ],
) -> ExperimentResult:
    """
    執行單一實驗的包裝函數，用於並行處理

    Args:
        args: 包含所有必要參數的元組
            - processed_df: 處理後的數據
            - preprocess_config: 預處理配置
            - feature_config: 特徵配置
            - random_seed: 隨機種子
            - model_type: 模型類型
            - experiment_config: 實驗配置

    Returns:
        實驗結果
    """
    (
        processed_df,
        preprocess_config,
        feature_config,
        random_seed,
        model_type,
        experiment_config,
    ) = args

    # 重設 logging ，避免多進程的 logging 衝突
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"開始模型 {model_type} 隨機種子 {random_seed} 的實驗")

    # 執行實驗
    single_experiment_result = single_experimentor(
        processed_df,
        preprocess_config,
        feature_config,
        random_seed,
        model_type,
        experiment_config,
    )

    # 校正
    if single_experiment_result is not None:
        logger.info(f"開始對實驗結果進行校正")
        apply_calibration_to_experiment(
            single_experiment_result,
            processed_df,
            experiment_config,
        )

        logger.info(f"開始 What-if 分析")
        apply_whatif_analysis(
            single_experiment_result,
            processed_df,
            experiment_config,
        )

        logger.info(
            f"模型 {model_type} seed {random_seed} 實驗完成，"
            f"包含 {len(single_experiment_result.calibrated_test_predictions)} 種校正方法"
        )

    return single_experiment_result


def single_experimentor(
    processed_df: pd.DataFrame,
    preprocess_config: PreprocessConfig,
    feature_config: FeatureConfig,
    random_seed: int,
    model_type: str,
    experiment_config: ExperimentConfig,
):
    """
    根據模型類型進行相應的數據準備和訓練
    """
    # 模型類型與對應實驗函數的映射
    model_experiment_map = {
        "CoxPHFitter": _cox_full_experiment,
        "XGBoost_AFT": _xgboost_full_experiment,
        "CatBoost": _catboost_full_experiment,
    }

    # 檢查模型類型是否支援
    if model_type not in model_experiment_map:
        raise ValueError(f"不支援的模型類型: {model_type}")

    if (not preprocess_config.is_preprocess) and processed_df.isnull().values.any():
        if model_type == "CoxPHFitter":
            logger.warning("資料未前處理且含缺值，跳過 CoxPHFitter 模型。")
            return

    # 執行對應的實驗
    experiment_function = model_experiment_map[model_type]
    experiment_result = experiment_function(
        processed_df, feature_config, random_seed, experiment_config
    )

    # 記錄實驗結果
    logger.info(
        f"實驗完成{random_seed}，訓練C-index: {experiment_result.train_c_index:.4f}，"
        f"測試C-index: {experiment_result.test_c_index:.4f}"
    )

    return experiment_result


def _cox_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    experiment_config: ExperimentConfig,
):
    from lifelines import CoxPHFitter

    patient_id_mapping = processed_df[feature_config.patient_id].copy()
    cph_df = processed_df.copy()
    cph_df = cph_df.drop(columns=[feature_config.source, feature_config.patient_id, feature_config.BCLC_stage])

    logger.info("開始訓練模型...")
    train_data, test_data = train_test_split(
        cph_df,
        test_size=experiment_config.model_settings.test_size,
        random_state=random_seed,
        stratify=cph_df["event"],
    )

    model = CoxPHFitter()
    model.fit(train_data, duration_col="time", event_col="event")

    train_predictions = model.predict_expectation(train_data)
    test_predictions = model.predict_expectation(test_data)

    train_c_index = concordance_index(
        train_data["time"], train_predictions, train_data["event"]
    )

    test_c_index = concordance_index(
        test_data["time"], test_predictions, test_data["event"]
    )

    train_predictions_df = _build_predictions_df(
        patient_id_mapping.iloc[train_data.index], train_predictions
    )
    test_predictions_df = _build_predictions_df(
        patient_id_mapping.iloc[test_data.index], test_predictions
    )
    # ===============================================
    # SHAP
    # ===============================================
    shap_results = {}
    feature_cols = [
        col
        for col in train_data.columns
        if col
        not in list(feature_config.survival_labels)
        + [feature_config.source]
        + [feature_config.patient_id]
    ]
    X_train_features = train_data[feature_cols]
    X_test_features = test_data[feature_cols]

    def predict_risk_score(X):
        return model.predict_partial_hazard(X).values

    kernel_explainer = shap.KernelExplainer(
        predict_risk_score, shap.sample(X_train_features, 50)
    )

    kernel_shap_values = kernel_explainer.shap_values(
        shap.sample(X_test_features, 10), silent=True
    )

    shap_results = {
        "explainer": kernel_explainer,
        "shap_values": kernel_shap_values,
        "expected_value": kernel_explainer.expected_value,
        "explained_data": X_train_features,
        "feature_names": feature_cols,
    }

    # ===============================================
    # 特徵重要性
    # ===============================================
    feature_importance = {}

    # 法一: cph_coef_importance
    cox_coef_importance = np.abs(model.params_).to_dict()
    feature_importance["cox_coefficients"] = cox_coef_importance
    # 法二: SHAP 值的平均絕對值
    mean_abs_shap = np.abs(kernel_shap_values).mean(axis=0)
    feature_importance["shap_importance"] = dict(zip(feature_cols, mean_abs_shap))

    return ExperimentResult(
        model_type="CoxPHFitter",
        train_predictions=train_predictions_df,
        test_predictions=test_predictions_df,
        train_c_index=train_c_index,
        test_c_index=test_c_index,
        model=model,
        shap_results=shap_results,
        feature_importance=feature_importance,
    )


def _xgboost_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    experiment_config: ExperimentConfig,
):
    import xgboost as xgb

    patient_id_mapping = processed_df[feature_config.patient_id].copy()

    X = processed_df.drop(
        columns=list(feature_config.survival_labels)
        + [feature_config.source, feature_config.patient_id, feature_config.BCLC_stage]
    )
    y = processed_df[list(feature_config.survival_labels)]

    y_lower = y["time"].values.copy()
    y_upper = y["time"].values.copy()

    if experiment_config.model_settings.censor_limit == "inf":
        y_upper[processed_df["event"] == 0] = np.inf
    elif experiment_config.model_settings.censor_limit == "avg_life-age":
        age_diff = experiment_config.model_settings.average_age - X["Age"]
        age_diff[age_diff < 0] = 0
        censored_mask = processed_df["event"] == 0
        y_upper[censored_mask] = (
            processed_df["time"][censored_mask] + age_diff[censored_mask]
        )

    X_processed = X.copy()
    for col in list(feature_config.cat_features):
        if col in X_processed.columns:
            X_processed[col] = X_processed[col].astype("category")

    (
        X_train,
        X_test,
        y_event_train,
        y_event_test,
        y_lower_train,
        y_lower_test,
        y_upper_train,
        y_upper_test,
        patient_id_train,
        patient_id_test,
    ) = train_test_split(
        X_processed,
        processed_df["event"],
        y_lower,
        y_upper,
        patient_id_mapping,  # ← 一起分割患者ID
        test_size=experiment_config.model_settings.test_size,
        random_state=random_seed,
        stratify=processed_df["event"],
    )

    dtrain = xgb.DMatrix(X_train, enable_categorical=True)
    dtrain.set_float_info("label_lower_bound", y_lower_train)
    dtrain.set_float_info("label_upper_bound", y_upper_train)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "nthread": 1,
    }

    model = xgb.train(params, dtrain, num_boost_round=20)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_float_info("label_lower_bound", y_lower_test)
    dtest.set_float_info("label_upper_bound", y_upper_test)

    train_predictions = model.predict(dtrain)
    test_predictions = model.predict(dtest)

    train_c_index = concordance_index(y_lower_train, train_predictions, y_event_train)
    test_c_index = concordance_index(y_lower_test, test_predictions, y_event_test)

    train_predictions_df = _build_predictions_df(patient_id_train, train_predictions)

    test_predictions_df = _build_predictions_df(patient_id_test, test_predictions)

    # ===============================================
    # SHAP TreeExplainer
    # ===============================================
    shap_results = {}
    tree_explainer = shap.TreeExplainer(model)

    # 計算 SHAP 值
    tree_shap_values = tree_explainer.shap_values(dtest)

    shap_results = {
        "explainer": tree_explainer,
        "shap_values": tree_shap_values,
        "expected_value": tree_explainer.expected_value,
        "explained_data": X_test,
        "feature_names": list(X_processed.columns),
    }

    # ===============================================
    # XGBoost 原生特徵重要性
    # ===============================================
    feature_importance = {}

    # XGBoost 提供多種特徵重要性計算方式
    feature_names = list(X_processed.columns)

    # 方法1: gain (預設) - 特徵對分割的平均增益
    importance_gain = model.get_score(importance_type="gain")
    feature_importance["xgb_gain"] = importance_gain

    # 方法2: weight - 特徵被用於分割的次數
    importance_weight = model.get_score(importance_type="weight")
    feature_importance["xgb_weight"] = importance_weight

    # 方法3: cover - 特徵覆蓋的樣本數
    importance_cover = model.get_score(importance_type="cover")
    feature_importance["xgb_cover"] = importance_cover

    # 方法4: SHAP 值的平均絕對值
    mean_abs_shap = np.abs(shap_results["shap_values"]).mean(axis=0)
    feature_importance["shap_importance"] = dict(zip(feature_names, mean_abs_shap))

    return ExperimentResult(
        model_type="XGBoost_AFT",
        train_predictions=train_predictions_df,
        test_predictions=test_predictions_df,
        train_c_index=train_c_index,
        test_c_index=test_c_index,
        model=model,
        shap_results=shap_results,
        feature_importance=feature_importance,
    )


def _catboost_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    experiment_config: ExperimentConfig,
):
    from catboost import CatBoostRegressor, Pool

    patient_id_mapping = processed_df[feature_config.patient_id].copy()

    X = processed_df.drop(
        columns=list(feature_config.survival_labels)
        + [feature_config.source, feature_config.patient_id, feature_config.BCLC_stage]
    )
    y = processed_df[list(feature_config.survival_labels)]

    y_lower = y["time"].values.copy()
    y_upper = y["time"].values.copy()

    if experiment_config.model_settings.censor_limit == "inf":
        y_upper[processed_df["event"] == 0] = np.inf
    elif experiment_config.model_settings.censor_limit == "avg_life-age":
        age_diff = experiment_config.model_settings.average_age - X["Age"]
        age_diff[age_diff < 0] = 0
        censored_mask = processed_df["event"] == 0
        y_upper[censored_mask] = (
            processed_df["time"][censored_mask] + age_diff[censored_mask]
        )

    y_for_catboost = np.stack([y_lower, y_upper], axis=1)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_lower_train,
        y_lower_test,
        y_event_train,
        y_event_test,
        patient_id_train,
        patient_id_test,
    ) = train_test_split(
        X,
        y_for_catboost,
        y_lower,
        processed_df["event"],
        patient_id_mapping,
        test_size=experiment_config.model_settings.test_size,
        random_state=random_seed,
        stratify=processed_df["event"],
    )

    model = CatBoostRegressor(
        iterations=100,
        loss_function="SurvivalAft",
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train, prediction_type="Exponent")
    test_predictions = model.predict(X_test, prediction_type="Exponent")

    train_c_index = concordance_index(y_lower_train, train_predictions, y_event_train)
    test_c_index = concordance_index(y_lower_test, test_predictions, y_event_test)

    train_predictions_df = _build_predictions_df(patient_id_train, train_predictions)
    test_predictions_df = _build_predictions_df(patient_id_test, test_predictions)

    # ===============================================
    # SHAP TreeExplainer
    # ===============================================
    shap_results = {}
    tree_explainer = shap.TreeExplainer(model)

    # 計算 SHAP 值
    tree_shap_values = tree_explainer.shap_values(X_test)

    shap_results = {
        "explainer": tree_explainer,
        "shap_values": tree_shap_values,
        "expected_value": tree_explainer.expected_value,
        "explained_data": X_test,
        "feature_names": list(X.columns),
    }

    # ===============================================
    # CatBoost 原生特徵重要性
    # ===============================================
    feature_importance = {}

    # CatBoost 提供多種特徵重要性計算方式
    feature_names = list(X.columns)

    # 方法1: PredictionValuesChange (預設) - 特徵對預測值變化的影響
    importance_prediction = model.get_feature_importance()
    feature_importance["catboost_prediction"] = dict(
        zip(feature_names, importance_prediction)
    )

    # 方法2: FeatureImportance - 內部特徵重要性
    importance_internal = model.get_feature_importance(type="FeatureImportance")
    feature_importance["catboost_internal"] = dict(
        zip(feature_names, importance_internal)
    )

    # 方法3: SHAP 值的平均絕對值
    mean_abs_shap = np.abs(shap_results["shap_values"]).mean(axis=0)
    feature_importance["shap_importance"] = dict(zip(feature_names, mean_abs_shap))

    return ExperimentResult(
        model_type="CatBoost",
        train_predictions=train_predictions_df,
        test_predictions=test_predictions_df,
        train_c_index=train_c_index,
        test_c_index=test_c_index,
        model=model,
        shap_results=shap_results,
        feature_importance=feature_importance,
    )


def _build_predictions_df(patient_ids, preds):
    df = pd.DataFrame(
        {
            "patient_id": patient_ids.values,
            "predicted_survival_time": preds,
        }
    )
    return df


# TODO: 待精簡
# ========================================
# 校正函數
# ========================================


def _calibrate_predictions_knn_km(
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    k: int = 200,
) -> pd.DataFrame:
    """
    校正法一: KNN + KM curve校正
    基於訓練集建立校正模型，應用到測試集
    """
    # 合併訓練數據
    train_merged = pd.merge(
        train_predictions, train_labels, on="patient_id", how="inner"
    )
    test_merged = pd.merge(test_predictions, test_labels, on="patient_id", how="inner")

    # 以訓練集準備KNN模型
    X_train = train_merged[["predicted_survival_time"]].values
    nn_model = NearestNeighbors(n_neighbors=min(k, len(train_merged)))
    nn_model.fit(X_train)

    calibrated_predictions = []

    # 對測試集進行校正
    for idx, row in test_merged.iterrows():
        # 在訓練集中找到最近的k個鄰居
        distances, indices = nn_model.kneighbors([[row["predicted_survival_time"]]])
        neighbor_data = train_merged.iloc[indices[0]]

        # 建立KM curve
        kmf = KaplanMeierFitter()
        kmf.fit(neighbor_data["time"], neighbor_data["event"])

        # 找到生存機率為0.5的時間點
        try:
            median_survival = kmf.median_survival_time_
            if pd.isna(median_survival) or np.isinf(median_survival):
                # 使用鄰居的平均預測值作為後備方案
                median_survival = neighbor_data["predicted_survival_time"].mean()
                if pd.isna(median_survival) or np.isinf(median_survival):
                    # 如果還是有問題，使用原始預測值
                    median_survival = row["predicted_survival_time"]
        except:
            median_survival = row["predicted_survival_time"]

        calibrated_predictions.append(median_survival)

    result_df = test_merged[["patient_id"]].copy()
    result_df["calibrated_prediction"] = calibrated_predictions
    return result_df[["patient_id", "calibrated_prediction"]]


def _calibrate_predictions_regression(
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    校正法二: Censored data回歸校正
    """
    # 合併訓練數據
    train_merged = pd.merge(
        train_predictions, train_labels, on="patient_id", how="inner"
    )
    test_merged = pd.merge(test_predictions, test_labels, on="patient_id", how="inner")

    # 只使用非刪失數據訓練回歸模型
    non_censored_train = train_merged[train_merged["event"] == 1]

    if len(non_censored_train) < 2:
        logger.warning("訓練集中非刪失數據不足，跳過回歸校正")
        return test_predictions[["patient_id", "predicted_survival_time"]].rename(
            columns={"predicted_survival_time": "calibrated_prediction"}
        )

    # 建立回歸模型 (x=預測時間, y=預測-真實的差值)
    X = non_censored_train["predicted_survival_time"].values.reshape(-1, 1)
    y = (
        non_censored_train["predicted_survival_time"] - non_censored_train["time"]
    ).values

    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # 對測試數據進行校正
    X_test = test_merged["predicted_survival_time"].values.reshape(-1, 1)
    predicted_errors = reg_model.predict(X_test)

    result_df = test_merged[["patient_id"]].copy()
    result_df["calibrated_prediction"] = (
        test_merged["predicted_survival_time"] - predicted_errors
    )

    return result_df[["patient_id", "calibrated_prediction"]]


def _calibrate_predictions_segmental(
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    n_segments: int = 5,
) -> pd.DataFrame:
    """
    校正法三: 區段化校正
    """
    train_merged = pd.merge(
        train_predictions, train_labels, on="patient_id", how="inner"
    )
    test_merged = pd.merge(test_predictions, test_labels, on="patient_id", how="inner")

    # 根據預測時間分段
    time_quantiles = np.linspace(0, 1, n_segments + 1)
    time_bins = np.quantile(train_merged["predicted_survival_time"], time_quantiles)
    time_bins[-1] = np.inf  # 確保最後一個bin包含所有值

    # 計算各段的校正值
    segment_corrections = {}
    for i in range(n_segments):
        mask = (train_merged["predicted_survival_time"] >= time_bins[i]) & (
            train_merged["predicted_survival_time"] < time_bins[i + 1]
        )
        segment_data = train_merged[mask]

        if len(segment_data) > 0:
            # 只計算非刪失數據的平均誤差
            non_censored = segment_data[segment_data["event"] == 1]
            if len(non_censored) > 0:
                mean_error = np.mean(
                    non_censored["predicted_survival_time"] - non_censored["time"]
                )
                segment_corrections[i] = mean_error
            else:
                segment_corrections[i] = 0
        else:
            segment_corrections[i] = 0

    # 對測試數據應用校正
    calibrated_predictions = []

    for _, row in test_merged.iterrows():
        # 找到對應的段
        segment_idx = np.searchsorted(time_bins[1:], row["predicted_survival_time"])
        segment_idx = min(segment_idx, n_segments - 1)

        correction = segment_corrections.get(segment_idx, 0)
        calibrated_pred = row["predicted_survival_time"] - correction
        calibrated_predictions.append(calibrated_pred)

    result_df = test_merged[["patient_id"]].copy()
    result_df["calibrated_prediction"] = calibrated_predictions
    return result_df[["patient_id", "calibrated_prediction"]]


def _calibrate_predictions_curve(
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    degree: int = 3,
) -> pd.DataFrame:
    """
    校正法四: 全體訓練集curve校正
    """
    train_merged = pd.merge(
        train_predictions, train_labels, on="patient_id", how="inner"
    )
    test_merged = pd.merge(test_predictions, test_labels, on="patient_id", how="inner")

    # 只使用非刪失數據建立校正curve
    non_censored_train = train_merged[train_merged["event"] == 1]

    if len(non_censored_train) < degree + 1:
        logger.warning(f"訓練集中非刪失數據不足以擬合{degree}次多項式，跳過curve校正")
        return test_predictions[["patient_id", "predicted_survival_time"]].rename(
            columns={"predicted_survival_time": "calibrated_prediction"}
        )

    # 建立校正curve (使用多項式回歸)
    X = non_censored_train["predicted_survival_time"].values
    y = (
        non_censored_train["predicted_survival_time"] - non_censored_train["time"]
    ).values

    # 使用多項式擬合
    poly_coeffs = np.polyfit(X, y, deg=degree)

    # 對測試數據應用校正
    predicted_errors = np.polyval(
        poly_coeffs, test_merged["predicted_survival_time"].values
    )

    result_df = test_merged[["patient_id"]].copy()
    result_df["calibrated_prediction"] = (
        test_merged["predicted_survival_time"] - predicted_errors
    )

    return result_df[["patient_id", "calibrated_prediction"]]


# TODO: 待精簡
# ========================================
# 統一的校正管理器
# ========================================
def apply_calibration_to_experiment(
    experiment_result: ExperimentResult,
    processed_df: pd.DataFrame,
    experiment_config: ExperimentConfig,
) -> None:
    """
    對單個實驗結果應用多種校正方法
    直接更新 experiment_result 物件
    """
    # 準備標籤數據
    train_ids = experiment_result.train_predictions["patient_id"].unique()
    test_ids = experiment_result.test_predictions["patient_id"].unique()

    train_labels = processed_df[processed_df["patient_id"].isin(train_ids)][
        ["patient_id", "time", "event"]
    ]
    test_labels = processed_df[processed_df["patient_id"].isin(test_ids)][
        ["patient_id", "time", "event"]
    ]

    # 校正方法映射
    calibration_functions = {
        "knn_km": _calibrate_predictions_knn_km,
        "regression": _calibrate_predictions_regression,
        "segmental": _calibrate_predictions_segmental,
        "curve": _calibrate_predictions_curve,
    }

    # 應用每種校正方法
    calibration_methods = list(
        experiment_config.experiment_settings.calibration_methods
    )
    for method in calibration_methods:
        if method not in calibration_functions:
            logger.warning(f"未知的校正方法: {method}")
            continue

        try:
            # 執行校正
            calibrated_df = calibration_functions[method](
                experiment_result.train_predictions,
                experiment_result.test_predictions,
                train_labels,
                test_labels,
            )

            # 合併校正結果與原始預測
            calibrated_test = experiment_result.test_predictions.merge(
                calibrated_df[["patient_id", "calibrated_prediction"]],
                on="patient_id",
                how="left",
            )

            # 使用校正後的預測值
            calibrated_test["predicted_survival_time"] = calibrated_test[
                "calibrated_prediction"
            ].fillna(calibrated_test["predicted_survival_time"])
            calibrated_test = calibrated_test.drop(columns=["calibrated_prediction"])

            # 儲存校正結果
            experiment_result.calibrated_test_predictions[method] = calibrated_test

            # 計算校正後的C-index
            test_merged = calibrated_test.merge(
                test_labels, on="patient_id", how="inner"
            )
            if len(test_merged) > 0:
                c_index = concordance_index(
                    test_merged["time"],
                    test_merged["predicted_survival_time"],
                    test_merged["event"],
                )
                experiment_result.calibrated_test_c_index[method] = c_index

        except Exception as e:
            logger.error(f"應用 {method} 校正時發生錯誤: {str(e)}")
            continue


# TODO: 待精簡
# ========================================
# What-If 分析函數
# ========================================
def apply_whatif_analysis(
    experiment_result: ExperimentResult,
    processed_df: pd.DataFrame,
    experiment_config: ExperimentConfig,
) -> None:
    """
    對單個實驗結果應用 What-if 分析
    直接更新 experiment_result 物件
    """

    # 獲取模型和特徵
    model = experiment_result.model
    model_type = experiment_result.model_type

    # 從測試集預測中獲取患者ID
    test_patient_ids = experiment_result.test_predictions["patient_id"].unique()
    test_df = processed_df[processed_df["patient_id"].isin(test_patient_ids)].copy()

    # 獲取特徵列
    feature_cols = _get_feature_columns(experiment_result)
    if not feature_cols:
        logger.warning(f"無法獲取模型特徵列，跳過 What-if 分析")
        return

    # 1. 治療方式分析
    if experiment_config.whatif_settings.analyze_treatments:
        logger.info("執行治療方式 What-if 分析...")
        treatment_results = _analyze_treatment_modifications(
            model,
            model_type,
            test_df,
            feature_cols,
            experiment_config,
        )
        experiment_result.whatif_treatment_results = treatment_results
        logger.info(f"完成 {len(treatment_results)} 種治療分析")

    # 2. 連續特徵分析
    if experiment_config.whatif_settings.analyze_continuous:
        logger.info("執行連續特徵 What-if 分析...")
        continuous_results = _analyze_continuous_modifications(
            model,
            model_type,
            test_df,
            feature_cols,
            experiment_config,
        )
        experiment_result.whatif_continuous_results = continuous_results
        logger.info(f"完成 {len(continuous_results)} 個連續特徵分析")


def _get_feature_columns(experiment_result: ExperimentResult) -> List[str]:
    """
    從實驗結果中獲取特徵列名稱
    """
    if (
        experiment_result.shap_results
        and "feature_names" in experiment_result.shap_results
    ):
        return experiment_result.shap_results["feature_names"]

    # 如果都沒有，返回空列表
    logger.warning(f"無法從 {experiment_result.model_type} 模型中獲取特徵名稱")
    return []

def _analyze_treatment_modifications(
    model: Any,
    model_type: str,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    experiment_config: ExperimentConfig,
) -> Dict[str, pd.DataFrame]:
    """分析治療方式修改的影響：比較有治療與無治療的差異"""
    results = {}

    # 確認哪些治療在特徵中
    available_treatments = [
        t for t in experiment_config.whatif_settings.treatments if t in feature_cols
    ]
    if not available_treatments:
        logger.warning("沒有可用的治療特徵")
        return results

    # 是否按期別分層
    if (
        experiment_config.whatif_settings.stratify_by_stage
        and experiment_config.whatif_settings.stage_column in test_df.columns
    ):
        stages = test_df[experiment_config.whatif_settings.stage_column].unique()
    else:
        stages = ["all"]
        test_df["temp_stage"] = "all"
        experiment_config.whatif_settings.stage_column = "temp_stage"

    # 對每個期別分析
    for stage in stages:
        stage_df = test_df[
            test_df[experiment_config.whatif_settings.stage_column] == stage
        ]
        if stage_df.empty:
            continue

        stage_results = []

        for _, patient in stage_df.iterrows():
            patient_id = patient["patient_id"]
            X_original = patient[feature_cols].values.reshape(1, -1)

            # 原始預測（保持當前治療狀態）
            original_pred = _predict_with_model(
                model, model_type, X_original, feature_cols
            )
            if original_pred is None:
                continue

            # 記錄當前治療
            current_treatments = [t for t in available_treatments if patient[t] == 1]

            # 創建無治療的情境：將所有治療特徵設為0
            X_no_treatment = X_original.copy()
            for idx, col in enumerate(feature_cols):
                if col in available_treatments:
                    X_no_treatment[0, idx] = 0.0

            # 預測無治療的結果
            no_treatment_pred = _predict_with_model(
                model, model_type, X_no_treatment, feature_cols
            )
            if no_treatment_pred is None:
                continue

            # 計算治療效果
            stage_results.append(
                {
                    "patient_id": patient_id,
                    "stage": stage,
                    "current_treatments": current_treatments,
                    "with_treatment_prediction": original_pred,
                    "no_treatment_prediction": no_treatment_pred,
                    "treatment_benefit_months": original_pred - no_treatment_pred,
                    "treatment_benefit_percent": (
                        ((original_pred - no_treatment_pred) / no_treatment_pred * 100)
                        if no_treatment_pred > 0
                        else 0
                    ),
                }
            )

        if stage_results:
            results[f"treatment_effect_stage_{stage}"] = pd.DataFrame(stage_results)

    # 移除臨時欄位
    if "temp_stage" in test_df.columns:
        test_df.drop(columns=["temp_stage"], inplace=True)

    return results


def _analyze_continuous_modifications(
    model: Any,
    model_type: str,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    experiment_config: ExperimentConfig,
) -> Dict[str, pd.DataFrame]:
    """分析連續特徵修改的影響"""
    results = {}

    for (
        feature_name,
        feature_config,
    ) in experiment_config.whatif_settings.continuous_features.items():
        if not feature_config.get("enabled", False):
            continue

        if feature_name not in feature_cols:
            logger.warning(f"特徵 {feature_name} 不在模型中")
            continue

        feature_idx = feature_cols.index(feature_name)
        feature_results = []

        for _, patient in test_df.iterrows():
            patient_id = patient["patient_id"]
            X_original = patient[feature_cols].values.reshape(1, -1)
            original_value = X_original[0, feature_idx]

            # 原始預測
            original_pred = _predict_with_model(
                model, model_type, X_original, feature_cols
            )
            if original_pred is None:
                continue

            patient_result = {
                "patient_id": patient_id,
                f"original_{feature_name}": original_value,
                "original_prediction": original_pred,
            }

            # 測試每個修改值
            for delta in feature_config.get("modifications", []):
                new_value = original_value + delta

                # 檢查邊界
                if feature_config.get("min_value") is not None:
                    new_value = max(new_value, feature_config.get("min_value"))
                if feature_config.get("max_value") is not None:
                    new_value = min(new_value, feature_config.get("max_value"))

                X_modified = X_original.copy()
                X_modified[0, feature_idx] = new_value

                # 預測修改後的結果
                modified_pred = _predict_with_model(
                    model, model_type, X_modified, feature_cols
                )
                if modified_pred is None:
                    continue

                delta_str = f"plus_{delta}" if delta > 0 else f"minus_{abs(delta)}"
                patient_result[f"{feature_name}_{delta_str}"] = new_value
                patient_result[f"prediction_{delta_str}"] = modified_pred
                patient_result[f"change_{delta_str}"] = modified_pred - original_pred

            feature_results.append(patient_result)

        if feature_results:
            results[feature_name] = pd.DataFrame(feature_results)

    return results


def _predict_with_model(
    model: Any, model_type: str, X: np.ndarray, feature_names: List[str] = None
) -> float:
    """統一的預測介面，處理不同模型類型"""
    try:

        if model_type == "CoxPHFitter":
            X_df = pd.DataFrame(X, columns=feature_names)
            return model.predict_expectation(X_df).values[0]
        if model_type == "XGBoost_AFT":
            dmatrix = xgb.DMatrix(
                X, feature_names=feature_names, enable_categorical=True
            )
            return model.predict(dmatrix)[0]
        else:
            return model.predict(X)[0]
    except Exception as e:
        logger.warning(f"預測失敗: {e}")
        return None


# TODO: 待精簡
# ========================================
# 儲存函數
# ========================================
def save_experiment_results(
    total_experiments_result: List[ExperimentResult],
    path_config: PathConfig,
) -> Dict[str, Path]:
    """
    Args:
        total_experiments_result: 實驗結果列表
        path_config: 包含 result_save_path 的路徑設定
    Returns:
        Dict[str, Path]: 實際儲存檔案的路徑
    """
    valid_results = _filter_valid_results(total_experiments_result)
    if not valid_results:
        logger.warning("沒有有效的實驗結果可以儲存")
        return {}

    # 結果根目錄（已含 timestamp）
    result_root = path_config.result_save_dir
    # 確保根目錄存在
    result_root.mkdir(parents=True, exist_ok=True)

    return {
        "summary": _save_summary_excel(valid_results, path_config),
        "test_predictions": _save_predictions_csv(valid_results, path_config),
        "report": _save_text_report(valid_results, path_config),
        "models": _save_models_pickle(valid_results, path_config),
    }


def _filter_valid_results(results: List[ExperimentResult]) -> List[ExperimentResult]:
    """過濾有效的實驗結果"""
    valid_results = []
    none_count = 0

    for result in results:
        if result is None:
            none_count += 1
            continue

        # 檢查是否為 ExperimentResult 實例
        if isinstance(result, ExperimentResult):
            # 檢查必要欄位是否有值
            if (
                result.model_type
                and result.train_c_index is not None
                and result.test_c_index is not None
            ):
                valid_results.append(result)
        else:
            logger.warning(f"結果不是 ExperimentResult 類型: {type(result)}")

    total_count = len(results)
    valid_count = len(valid_results)

    if none_count > 0:
        logger.warning(f"跳過 {none_count} 個無效結果")

    logger.info(f"總結果: {total_count}, 有效結果: {valid_count}")
    return valid_results


def _save_summary_excel(
    results: List[ExperimentResult], path_config: PathConfig
) -> Path:
    """儲存Excel彙總報告"""
    excel_path = path_config.summary_save_path
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 工作表1: 性能概覽
        performance_data = []
        for idx, r in enumerate(results):
            performance_data.append(
                {
                    "Model_Type": r.model_type,
                    "Experiment_Index": idx,  # 使用索引代替 random_seed
                    "Train_C_Index": r.train_c_index,
                    "Test_C_Index": r.test_c_index,
                    "Performance_Gap": r.train_c_index - r.test_c_index,
                    "Prediction_Count": len(r.test_predictions),
                }
            )

            # 如果有校正結果，也加入
            for method, c_index in r.calibrated_test_c_index.items():
                performance_data.append(
                    {
                        "Model_Type": f"{r.model_type}_{method}",
                        "Experiment_Index": idx,
                        "Train_C_Index": r.train_c_index,
                        "Test_C_Index": c_index,
                        "Performance_Gap": r.train_c_index - c_index,
                        "Prediction_Count": len(r.calibrated_test_predictions[method]),
                    }
                )

        performance_df = pd.DataFrame(performance_data).round(4)
        performance_df.to_excel(writer, sheet_name="Performance", index=False)

        # 工作表2: 統計摘要
        stats_df = (
            performance_df.groupby("Model_Type")["Test_C_Index"]
            .agg(
                [
                    ("Run_Count", "count"),
                    ("Avg_Test_C_Index", "mean"),
                    ("Std_Test_C_Index", "std"),
                    ("Best_Test_C_Index", "max"),
                    ("Worst_Test_C_Index", "min"),
                ]
            )
            .round(4)
            .reset_index()
        )
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        # 工作表3: 特徵重要性摘要（如果有的話）
        importance_df = _extract_feature_importance_summary(results)
        if not importance_df.empty:
            importance_df.sort_values(
                ["Model_Type", "Experiment_Index", "Method"]
            ).to_excel(writer, sheet_name="Feature_Importance", index=False)

    return excel_path


def _extract_feature_importance_summary(
    results: List[ExperimentResult],
) -> pd.DataFrame:
    """提取所有特徵重要性方法的摘要（包含零重要性特徵）"""
    importance_data = []

    for idx, result in enumerate(results):
        # 直接使用 dataclass 屬性存取
        feature_importance = result.feature_importance
        if not feature_importance:
            continue

        model_type = result.model_type

        # 儲存所有方法的特徵重要性（包含零值）
        for method_name, method_data in feature_importance.items():
            if isinstance(method_data, dict):
                for feature, importance in method_data.items():
                    importance_data.append(
                        {
                            "Model_Type": model_type,
                            "Experiment_Index": idx,
                            "Method": method_name,
                            "Feature": feature,
                            "Importance": round(float(importance), 6),
                        }
                    )

    return pd.DataFrame(importance_data)


def _save_predictions_csv(
    results: List[ExperimentResult],
    path_config: PathConfig,
) -> Path:
    """儲存預測結果CSV（包含訓練集和測試集）"""

    csv_path = path_config.origin_predictions_save_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    for idx, result in enumerate(results):
        # 儲存訓練集原始預測
        train_predictions = result.train_predictions
        train_df = train_predictions.copy()
        train_df["model_type"] = result.model_type
        train_df["experiment_index"] = idx
        train_df["calibration_method"] = "original"
        train_df["dataset"] = "train"
        all_predictions.append(train_df)

        # 儲存測試集原始預測
        test_predictions = result.test_predictions
        test_df = test_predictions.copy()
        test_df["model_type"] = result.model_type
        test_df["experiment_index"] = idx
        test_df["calibration_method"] = "original"
        test_df["dataset"] = "test"
        all_predictions.append(test_df)

        # 儲存校正後的預測（只有測試集有校正）
        for method, calibrated_df in result.calibrated_test_predictions.items():
            cal_df = calibrated_df.copy()
            cal_df["model_type"] = result.model_type
            cal_df["experiment_index"] = idx
            cal_df["calibration_method"] = method
            cal_df["dataset"] = "test"
            all_predictions.append(cal_df)

    combined_df = pd.concat(all_predictions, ignore_index=True)
    cols = ["model_type", "experiment_index", "calibration_method", "dataset"] + [
        col
        for col in combined_df.columns
        if col
        not in ["model_type", "experiment_index", "calibration_method", "dataset"]
    ]
    combined_df = combined_df[cols]
    combined_df.to_csv(csv_path, index=False)

    return csv_path


def _save_text_report(results: List[ExperimentResult], path_config: PathConfig) -> Path:
    """儲存簡要文字報告"""
    txt_path = path_config.report_save_path

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("實驗結果報告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"實驗總數: {len(results)}\n\n")

        # 按模型分組統計
        model_groups: Dict[str, List[ExperimentResult]] = {}
        for result in results:
            model_groups.setdefault(result.model_type, []).append(result)

        for model_type, model_results in model_groups.items():
            test_scores = [r.test_c_index for r in model_results]
            f.write(f"\n{model_type}:\n")
            f.write(f"  實驗次數: {len(model_results)}\n")
            f.write(f"  平均測試C-Index: {np.mean(test_scores):.4f}\n")
            f.write(f"  最佳測試C-Index: {np.max(test_scores):.4f}\n")
            f.write(f"  標準差: {np.std(test_scores):.4f}\n")

            # 校正結果統計
            calibration_methods = set()
            for r in model_results:
                calibration_methods.update(r.calibrated_test_c_index.keys())

            for method in calibration_methods:
                cal_scores = [
                    r.calibrated_test_c_index.get(method)
                    for r in model_results
                    if method in r.calibrated_test_c_index
                ]
                cal_scores = [s for s in cal_scores if s is not None]
                if cal_scores:
                    f.write(f"  {method}校正後平均C-Index: {np.mean(cal_scores):.4f}\n")

        # 最佳結果
        best_result = max(results, key=lambda x: x.test_c_index)
        f.write(f"\n最佳表現:\n")
        f.write(f"  模型: {best_result.model_type}\n")
        f.write(f"  測試C-Index: {best_result.test_c_index:.4f}\n")

        # 最佳校正結果
        best_cal_result = None
        best_cal_method = None
        best_cal_score = 0

        for result in results:
            for method, score in result.calibrated_test_c_index.items():
                if score > best_cal_score:
                    best_cal_score = score
                    best_cal_result = result
                    best_cal_method = method

        if best_cal_result:
            f.write(f"\n最佳校正表現:\n")
            f.write(f"  模型: {best_cal_result.model_type}\n")
            f.write(f"  校正方法: {best_cal_method}\n")
            f.write(f"  校正後C-Index: {best_cal_score:.4f}\n")

    return txt_path


def _save_models_pickle(
    results: List[ExperimentResult], path_config: PathConfig
) -> Path:
    """儲存模型對象"""
    models_dir = path_config.models_save_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results):
        model = result.model
        if model is not None:
            model_type = result.model_type
            model_path = models_dir / f"{model_type}_exp_{idx}.pkl"

            try:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.warning(f"無法儲存 {model_type} 模型: {e}")

    return models_dir
