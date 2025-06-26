import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config_utils import FeatureConfig, SurvivalModelConfig, ExperimentConfig
from lifelines.utils import concordance_index
import shap
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    model_type: str
    random_seed: int
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    train_c_index: float
    test_c_index: float
    model: Any
    shap_results: Dict[str, Any]
    feature_importance: Dict[str, Dict[str, float]]

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
        f"實驗完成{random_seed}，訓練C-index: {experiment_result.train_c_index:.4f}，"
        f"測試C-index: {experiment_result.test_c_index:.4f}"
    )

    return experiment_result

def _cox_full_experiment(
    processed_df: pd.DataFrame,
    feature_config: FeatureConfig,
    random_seed: int,
    survival_model_config: SurvivalModelConfig,
):
    from lifelines import CoxPHFitter

    patient_id_mapping = processed_df[feature_config.patient_id].copy()
    cph_df = processed_df.copy()
    cph_df = cph_df.drop(columns=feature_config.source + feature_config.patient_id)

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



    train_predictions_df = _build_predictions_df(
        patient_id_mapping.iloc[train_data.index],
        train_predictions
    )
    test_predictions_df = _build_predictions_df(
        patient_id_mapping.iloc[test_data.index],
        test_predictions
    )
    # ===============================================
    # SHAP
    # ===============================================
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

    return ExperimentResult(
        model_type="CoxPHFitter",
        random_seed=random_seed,
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
    survival_model_config: SurvivalModelConfig,
):
    import xgboost as xgb

    patient_id_mapping = processed_df[feature_config.patient_id].copy()

    X = processed_df.drop(
        columns=list(feature_config.survival_labels) + [feature_config.source, feature_config.patient_id]
    )
    y = processed_df[list(feature_config.survival_labels)]

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
        patient_id_train,
        patient_id_test,
    ) = train_test_split(
        X_processed,
        processed_df["event"],
        y_lower,
        y_upper,
        patient_id_mapping,  # ← 一起分割患者ID
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

    train_predictions_df = _build_predictions_df(
        patient_id_train, train_predictions)

    test_predictions_df = _build_predictions_df(
        patient_id_test, test_predictions)


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
        random_seed=random_seed,
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
    survival_model_config: SurvivalModelConfig,
):
    from catboost import CatBoostRegressor, Pool

    patient_id_mapping = processed_df[feature_config.patient_id].copy()

    X = processed_df.drop(
        columns=list(feature_config.survival_labels) + [feature_config.source, feature_config.patient_id]
    )
    y = processed_df[list(feature_config.survival_labels)]

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
        patient_id_train,
        patient_id_test,
    ) = train_test_split(
        X,
        y_for_catboost,
        y_lower,
        processed_df["event"],
        patient_id_mapping,
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

    train_predictions_df = _build_predictions_df(
        patient_id_train, train_predictions)
    test_predictions_df = _build_predictions_df(
        patient_id_test, test_predictions)


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
        random_seed=random_seed,
        train_predictions=train_predictions_df,
        test_predictions=test_predictions_df,
        train_c_index=train_c_index,
        test_c_index=test_c_index,
        model=model,
        shap_results=shap_results,
        feature_importance=feature_importance,
    )

def _build_predictions_df(patient_ids, preds):
    df = pd.DataFrame({
        "patient_id": patient_ids.values,
        "predicted_survival_time": preds,
    })
    return df

# TODO: 把校正模組從analyzer移到這裡


# TODO: 儲存模組，之後要精簡該部份
def save_experiment_results(
    total_experiments_result: List[Dict[str, Any]],
    experiment_config,
    ts: str,
) -> Dict[str, Path]:
    """
    精簡版實驗結果儲存函數
    
    Args:
        total_experiments_result: 實驗結果列表
        experiment_config: 實驗配置對象
        include_models: 是否儲存模型對象
    
    Returns:
        Dict[str, Path]: 儲存文件的路徑
    """
    # 過濾有效結果
    valid_results = _filter_valid_results(total_experiments_result)
    if not valid_results:
        logger.warning("沒有有效的實驗結果可以儲存")
        return {}
    
    # 創建儲存目錄
    result_dir = _create_result_directory(experiment_config, ts)
    
    # 儲存檔案並回傳路徑
    saved_files = {}
    saved_files["summary"] = _save_summary_excel(valid_results, result_dir, ts)
    saved_files["test_predictions"] = _save_predictions_csv(valid_results, result_dir, ts)
    saved_files["report"] = _save_text_report(valid_results, result_dir, ts)
    saved_files["models"] = _save_models_pickle(valid_results, result_dir, ts)
    
    return saved_files

def _filter_valid_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """過濾有效的實驗結果"""
    valid_results = []
    none_count = 0
    
    for result in results:
        if result is None:
            none_count += 1
            continue
        
        required_fields = ['model_type', 'random_seed', 'train_c_index', 'test_c_index']
        if all(hasattr(result, field) for field in required_fields):
            valid_results.append(result)
    
    total_count = len(results)
    valid_count = len(valid_results)
    
    if none_count > 0:
        logger.warning(f"跳過 {none_count} 個無效結果")
    
    logger.info(f"總結果: {total_count}, 有效結果: {valid_count}")
    return valid_results

def _create_result_directory(experiment_config: ExperimentConfig, ts: str) -> Path:
    """創建結果儲存目錄"""
    result_path = Path(experiment_config.result_save_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_name = result_path.stem
    result_dir = result_path.parent / f"{base_name}_{ts}"
    result_dir.mkdir(exist_ok=True)
    
    return result_dir

def _save_summary_excel(
    results: List[ExperimentResult],
    result_dir: Path,
    ts: str
) -> Path:
    """儲存Excel彙總報告"""
    excel_path = result_dir / f"summary_{ts}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 工作表1: 性能概覽
        performance_df = pd.DataFrame([
            {
                'Model_Type': r.model_type,
                'Random_Seed': r.random_seed,
                'Train_C_Index': r.train_c_index,
                'Test_C_Index': r.test_c_index,
                'Performance_Gap': r.train_c_index - r.test_c_index,
                'Prediction_Count': len(r.test_predictions)
            }
            for r in results
        ]).round(4)
        performance_df.to_excel(writer, sheet_name='Performance', index=False)

        # 工作表2: 統計摘要
        stats_df = (
            performance_df
            .groupby('Model_Type')['Test_C_Index']
            .agg([
                ('Run_Count', 'count'),
                ('Avg_Test_C_Index', 'mean'),
                ('Std_Test_C_Index', 'std'),
                ('Best_Test_C_Index', 'max'),
                ('Worst_Test_C_Index', 'min')
            ])
            .round(4)
            .reset_index()
        )
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # 工作表3: 特徵重要性摘要（如果有的話）
        importance_df = _extract_feature_importance_summary(results)
        if not importance_df.empty:
            importance_df \
                .sort_values(['Model_Type', 'Random_Seed', 'Method']) \
                .to_excel(writer, sheet_name='Feature_Importance', index=False)

    return excel_path


def _extract_feature_importance_summary(results: List[ExperimentResult]) -> pd.DataFrame:
    """提取所有特徵重要性方法的摘要（包含零重要性特徵）"""
    importance_data = []
    
    for result in results:
        # 直接使用 dataclass 屬性存取
        feature_importance = result.feature_importance
        if not feature_importance:
            continue
        
        model_type = result.model_type
        seed = result.random_seed
    
        # 儲存所有方法的特徵重要性（包含零值）
        for method_name, method_data in feature_importance.items():
            if isinstance(method_data, dict):
                for feature, importance in method_data.items():
                    importance_data.append({
                        'Model_Type': model_type,
                        'Random_Seed': seed,
                        'Method': method_name,
                        'Feature': feature,
                        'Importance': round(float(importance), 6)
                    })
    
    return pd.DataFrame(importance_data)


def _save_predictions_csv(results: List[ExperimentResult], result_dir: Path, ts: str) -> Path:
    """儲存預測結果CSV"""
    csv_path = result_dir / f"predictions_{ts}.csv"
    
    all_predictions = []
    for result in results:
        test_predictions = result.test_predictions
        pred_df = test_predictions.copy()
        pred_df['model_type'] = result.model_type
        pred_df['random_seed'] = result.random_seed
        all_predictions.append(pred_df)
    
    combined_df = pd.concat(all_predictions, ignore_index=True)
    cols = ['model_type', 'random_seed'] + [
        col for col in combined_df.columns 
        if col not in ['model_type', 'random_seed']
    ]
    combined_df = combined_df[cols]
    combined_df.to_csv(csv_path, index=False)
    
    return csv_path


def _save_text_report(results: List[ExperimentResult], result_dir: Path, ts: str) -> Path:
    """儲存簡要文字報告"""
    txt_path = result_dir / f"report_{ts}.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("實驗結果報告")
        f.write("=" * 50 + "")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"實驗總數: {len(results)}\n")
        
        # 按模型分組統計
        model_groups: Dict[str, List[ExperimentResult]] = {}
        for result in results:
            model_groups.setdefault(result.model_type, []).append(result)
        
        for model_type, model_results in model_groups.items():
            test_scores = [r.test_c_index for r in model_results]
            f.write(f"{model_type}:\n")
            f.write(f"  實驗次數: {len(model_results)}\n")
            f.write(f"  平均測試C-Index: {np.mean(test_scores):.4f}\n")
            f.write(f"  最佳測試C-Index: {np.max(test_scores):.4f}\n")
            f.write(f"  標準差: {np.std(test_scores):.4f}\n")
        
        # 最佳結果
        best_result = max(results, key=lambda x: x.test_c_index)
        f.write("最佳表現:\n")
        f.write(f"  模型: {best_result.model_type}\n")
        f.write(f"  種子: {best_result.random_seed}\n")
        f.write(f"  測試C-Index: {best_result.test_c_index:.4f}\n")
    
    return txt_path


def _save_models_pickle(results: List[ExperimentResult], result_dir: Path, ts: str) -> Path:
    """儲存模型對象"""
    models_dir = result_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    for result in results:
        model = result.model
        if model is not None:
            model_type = result.model_type
            seed = result.random_seed
            model_path = models_dir / f"{model_type}_seed_{seed}.pkl"
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.warning(f"無法儲存 {model_type} 模型: {e}")
    
    return models_dir
