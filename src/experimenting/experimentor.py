import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config_utils import FeatureConfig, SurvivalModelConfig
from lifelines.utils import concordance_index
import shap

logger = logging.getLogger(__name__)


def single_experimentor(
    processed_df: pd.DataFrame,
    is_processed: bool,
    feature_config: FeatureConfig,
    random_seed: int,
    model_type: str,
    survival_model_config: SurvivalModelConfig,
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

    if (not is_processed) and processed_df.isnull().values.any():
        if model_type == "CoxPHFitter":
            logger.warning("資料未前處理且含缺值，跳過 CoxPHFitter 模型。")
            return

    # 執行對應的實驗
    experiment_function = model_experiment_map[model_type]
    experiment_result = experiment_function(
        processed_df, feature_config, random_seed, survival_model_config
    )

    # 記錄實驗結果
    logger.info(
        f"實驗完成{random_seed}，訓練C-index: {experiment_result['train_c_index']:.4f}，"
        f"測試C-index: {experiment_result['test_c_index']:.4f}"
    )

    return experiment_result


def _cox_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    survival_model_config: SurvivalModelConfig,
):
    from lifelines import CoxPHFitter

    cph_df = processed_df.copy()
    cph_df = cph_df.drop(columns=list(feature_config.other_labels))

    logger.info("開始訓練模型...")
    train_data, test_data = train_test_split(
        cph_df,
        test_size=survival_model_config.test_size,
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

    predictions_df = pd.DataFrame(
        {
            "patient_id": test_data.index,
            "predicted_survival_time": test_predictions.values,
        }
    )

    # ===============================================
    # SHAP
    # ===============================================
    logger.info("開始計算 SHAP 值...")

    shap_results = {}
    feature_cols = [
        col
        for col in train_data.columns
        if col
        not in list(feature_config.survival_labels) + list(feature_config.other_labels)
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
    # CPH原生的特徵重要性
    # ===============================================
    feature_importance = {}

    cox_coef_importance = np.abs(model.params_).to_dict()
    feature_importance["cox_coefficients"] = cox_coef_importance

    return {
        "model_type": "CoxPHFitter",
        "random_seed": random_seed,
        "predictions": predictions_df,
        "train_c_index": train_c_index,
        "test_c_index": test_c_index,
        "model": model,
        "shap_results": shap_results,
        "feature_importance": feature_importance,
    }


def _xgboost_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    survival_model_config: SurvivalModelConfig,
):
    import xgboost as xgb

    X = processed_df.drop(
        columns=list(feature_config.survival_labels) + list(feature_config.other_labels)
    )
    y = processed_df[
        list(feature_config.survival_labels) + list(feature_config.other_labels)
    ]

    y_lower = y["time"].values.copy()
    y_upper = y["time"].values.copy()

    if survival_model_config.censor_limit == "inf":
        y_upper[processed_df["event"] == 0] = np.inf
    elif survival_model_config.censor_limit == "avg_life-age":
        age_diff = survival_model_config.average_age - X["Age"]
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
    ) = train_test_split(
        X_processed,
        processed_df["event"],
        y_lower,
        y_upper,
        test_size=survival_model_config.test_size,
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
        "nthread": 2,
    }

    model = xgb.train(params, dtrain, num_boost_round=20)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_float_info("label_lower_bound", y_lower_test)
    dtest.set_float_info("label_upper_bound", y_upper_test)

    train_predictions = model.predict(dtrain)
    test_predictions = model.predict(dtest)

    train_c_index = concordance_index(y_lower_train, train_predictions, y_event_train)
    test_c_index = concordance_index(y_lower_test, test_predictions, y_event_test)

    predictions_df = pd.DataFrame(
        {
            "patient_id": X_test.index,
            "predicted_survival_time": test_predictions,
        }
    )

    # ===============================================
    # SHAP TreeExplainer
    # ===============================================
    logger.info("開始計算 SHAP 值...")

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

    logger.info("TreeExplainer 完成")

    # ===============================================
    # XGBoost 原生特徵重要性
    # ===============================================
    feature_importance = {}

    # XGBoost 提供多種特徵重要性計算方式
    feature_names = list(X_processed.columns)

    # 方法1: gain (預設) - 特徵對分割的平均增益
    importance_gain = model.get_score(importance_type="gain")
    feature_importance["xgb_gain"] = {
        name: importance_gain.get(f"f{i}", 0.0) for i, name in enumerate(feature_names)
    }

    # 方法2: weight - 特徵被用於分割的次數
    importance_weight = model.get_score(importance_type="weight")
    feature_importance["xgb_weight"] = {
        name: importance_weight.get(f"f{i}", 0.0)
        for i, name in enumerate(feature_names)
    }

    # 方法3: cover - 特徵覆蓋的樣本數
    importance_cover = model.get_score(importance_type="cover")
    feature_importance["xgb_cover"] = {
        name: importance_cover.get(f"f{i}", 0.0) for i, name in enumerate(feature_names)
    }

    # 方法4: SHAP 值的平均絕對值
    mean_abs_shap = np.abs(shap_results["shap_values"]).mean(axis=0)
    feature_importance["shap_importance"] = dict(zip(feature_names, mean_abs_shap))

    logger.info("特徵重要性計算完成")

    return {
        "model_type": "XGBoost_AFT",
        "random_seed": random_seed,
        "predictions": predictions_df,
        "train_c_index": train_c_index,
        "test_c_index": test_c_index,
        "model": model,
        "shap_results": shap_results,
        "feature_importance": feature_importance,
    }


def _catboost_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    survival_model_config: SurvivalModelConfig,
):
    from catboost import CatBoostRegressor, Pool

    X = processed_df.drop(
        columns=list(feature_config.survival_labels) + list(feature_config.other_labels)
    )
    y = processed_df[
        list(feature_config.survival_labels) + list(feature_config.other_labels)
    ]

    y_lower = y["time"].values.copy()
    y_upper = y["time"].values.copy()

    if survival_model_config.censor_limit == "inf":
        y_upper[processed_df["event"] == 0] = np.inf
    elif survival_model_config.censor_limit == "avg_life-age":
        age_diff = survival_model_config.average_age - X["Age"]
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
    ) = train_test_split(
        X,
        y_for_catboost,
        y_lower,
        processed_df["event"],
        test_size=survival_model_config.test_size,
        random_state=random_seed,
        stratify=processed_df["event"],
    )

    model = CatBoostRegressor(
        iterations=100,
        loss_function="SurvivalAft",
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train, prediction_type="Exponent")
    test_predictions = model.predict(X_test, prediction_type="Exponent")

    train_c_index = concordance_index(y_lower_train, train_predictions, y_event_train)
    test_c_index = concordance_index(y_lower_test, test_predictions, y_event_test)

    predictions_df = pd.DataFrame(
        {
            "patient_id": X_test.index,
            "predicted_survival_time": test_predictions,
        }
    )

    # ===============================================
    # SHAP TreeExplainer
    # ===============================================
    logger.info("開始計算 SHAP 值...")

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

    logger.info("TreeExplainer 完成")

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

    logger.info("特徵重要性計算完成")

    return {
        "model_type": "CatBoost",
        "random_seed": random_seed,
        "predictions": predictions_df,
        "train_c_index": train_c_index,
        "test_c_index": test_c_index,
        "model": model,
        "shap_results": shap_results,
        "feature_importance": feature_importance,
    }
