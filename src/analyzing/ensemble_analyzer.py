# TODO: 精簡
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.config_utils import ExperimentConfig, PathConfig
from experimenting.experimentor import ExperimentResult


logger = logging.getLogger(__name__)


# 跟experimentor.py中的ExperimentResult類似，之後再看怎麼整合
def ensemble_predictions_by_seed(
    experiment_results: List[ExperimentResult],
    path_config: PathConfig,
    include_calibrated: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    將相同模型不同 seed 的預測值平均
    支援未校正和已校正的預測結果
    """
    # 收集未校正的預測結果
    model_predictions = defaultdict(lambda: defaultdict(list))

    # 收集原始預測
    for res in experiment_results:
        if res and res.test_predictions is not None:
            df = res.test_predictions.copy()
            model_predictions[res.model_type]["original"].append(df)

            # 收集校正後的預測
            if include_calibrated:
                for method, calibrated_df in res.calibrated_test_predictions.items():
                    model_predictions[res.model_type][method].append(calibrated_df)

    # 計算 ensemble
    ensemble_results = {}

    for model_type, method_predictions in model_predictions.items():
        model_ensemble = {}

        for method, dfs in method_predictions.items():
            if not dfs:
                logger.warning(f"模型 {model_type} 方法 {method} 沒有有效的預測結果")
                continue

            # 合併並聚合
            all_df = pd.concat(dfs, ignore_index=True)
            agg = (
                all_df.groupby("patient_id")["predicted_survival_time"]
                .agg(
                    [
                        ("ensemble_prediction", "mean"),
                        ("prediction_std", "std"),
                        ("seed_count", "count"),
                        ("individual_predictions", list),
                    ]
                )
                .reset_index()
            )

            agg["model_type"] = model_type
            agg["calibration_method"] = method

            # 建立key (model_type 或 model_type_method)
            key = model_type if method == "original" else f"{model_type}_{method}"
            model_ensemble[key] = agg

        ensemble_results.update(model_ensemble)

    # 儲存結果
    d = path_config.ensemble_predictions_dir
    d.mkdir(parents=True, exist_ok=True)
    for key, df in ensemble_results.items():
        df.to_csv(d / f"{key}_ensemble_predictions.csv", index=False)
        logger.info("已儲存 Ensemble 預測: %s", d / f"{key}_ensemble_predictions.csv")

    return ensemble_results


def ensemble_feature_importance(
    experiment_results: List[ExperimentResult],
    path_config: PathConfig,
) -> Dict[str, pd.DataFrame]:
    """
    將特徵重要性平均
    注意：特徵重要性不受校正影響，所以保持原樣
    """
    # 收集所有特徵重要性數據
    rows = []
    for res in experiment_results:
        if res and res.feature_importance:
            for method, features in res.feature_importance.items():
                if isinstance(features, dict):
                    rows.extend(
                        {
                            "model_type": res.model_type,
                            "method": method,
                            "feature": feat,
                            "importance": float(imp),
                        }
                        for feat, imp in features.items()
                    )

    if not rows:
        logger.warning("沒有找到有效的特徵重要性數據")
        return {}

    # 處理數據
    df = pd.DataFrame(rows)
    ensemble_results = {}

    for model_type, group in df.groupby("model_type"):
        agg = (
            group.groupby(["feature", "method"])["importance"]
            .agg(
                [
                    ("mean_importance", "mean"),
                    ("std_importance", "std"),
                    ("seed_count", "count"),
                ]
            )
            .fillna({"std_importance": 0})
            .reset_index()
        )

        agg["model_type"] = model_type
        ensemble_results[model_type] = agg[
            [
                "model_type",
                "method",
                "feature",
                "mean_importance",
                "std_importance",
                "seed_count",
            ]
        ].sort_values(["method", "mean_importance"], ascending=[True, False])

        # 儲存各模型的 feature importance
        d = path_config.ensemble_feature_importance_dir
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{model_type}.csv", index=False)
        
    return ensemble_results


def calculate_ensemble_metrics(
    ensemble_predictions: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    original_results: List[ExperimentResult],
    path_config: PathConfig,
) -> Dict[str, Dict[str, float]]:
    """
    計算 ensemble 指標，包括校正後的結果
    """
    # 檢查必要欄位
    required = ["patient_id", "time", "event"]
    if not all(col in processed_df.columns for col in required):
        logger.error(f"processed_df 缺少必要的列: {required}")
        return {}

    true_data = processed_df[required]
    metrics = {}

    for key, pred_df in ensemble_predictions.items():
        calibration_methods = ["knn_km", "regression", "segmental", "curve"]
        model_type = key
        calibration_method = "original"

        for method in calibration_methods:
            if key.endswith(f"_{method}"):
                model_type = key[: -len(f"_{method}")]
                calibration_method = method
                break

        # 合併預測與真實數據
        merged = pred_df.merge(true_data, on="patient_id")
        if merged.empty:
            continue

        # 計算 ensemble C-index
        ens_c = concordance_index(
            merged["time"], merged["ensemble_prediction"], merged["event"]
        )

        # 獲取個別 C-index
        if calibration_method == "original":
            indiv_c_indices = [
                res.test_c_index
                for res in original_results
                if res and res.model_type == model_type
            ]
        else:
            indiv_c_indices = [
                res.calibrated_test_c_index.get(calibration_method, np.nan)
                for res in original_results
                if res
                and res.model_type == model_type
                and calibration_method in res.calibrated_test_c_index
            ]
            indiv_c_indices = [c for c in indiv_c_indices if not np.isnan(c)]

        if not indiv_c_indices:
            continue

        # 組裝指標
        indiv_array = np.array(indiv_c_indices)
        metrics[key] = {
            "model_type": model_type,
            "calibration_method": calibration_method,
            "ensemble_c_index": ens_c,
            "individual_mean_c_index": indiv_array.mean(),
            "individual_std_c_index": indiv_array.std(),
            "individual_min_c_index": indiv_array.min(),
            "individual_max_c_index": indiv_array.max(),
            "improvement": ens_c - indiv_array.mean(),
            "seed_count": len(indiv_c_indices),
            "patients_count": len(merged),
        }

        # 統一 rounding
        metrics[key] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics[key].items()
        }

    # 儲存 ensemble metrics
    d = path_config.metrics_dir
    d.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(path_config.ensemble_c_index_save_path, index=True)
    logger.info("已儲存 Ensemble C-index 指標: %s", path_config.ensemble_c_index_save_path)


    return metrics


def compare_calibration_methods(
    results: Dict[str, Dict[str, Any]],
    path_config: PathConfig,
) -> pd.DataFrame:
    """
    比較不同校正方法的效果
    """
    comparison_rows = []

    # 按模型類型分組
    model_results = defaultdict(dict)
    for key, stats in results.items():
        model_type = stats.get("model_type")
        method = stats.get("calibration_method")
        if model_type and method:
            model_results[model_type][method] = stats

    # 比較每個模型的不同校正方法
    for model_type, methods in model_results.items():
        if "original" not in methods:
            continue

        original_mae = (
            methods["original"]
            .get("non_censored_metrics", {})
            .get("mean_absolute_error")
        )
        if original_mae is None:
            continue

        for method, stats in methods.items():
            if "non_censored_metrics" not in stats:
                continue

            mae = stats["non_censored_metrics"]["mean_absolute_error"]
            improvement = (
                ((original_mae - mae) / original_mae) * 100 if original_mae > 0 else 0
            )

            comparison_rows.append(
                {
                    "model_type": model_type,
                    "calibration_method": method,
                    "mae": mae,
                    "improvement_percent": improvement,
                    "is_baseline": method == "original",
                }
            )

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        d = path_config.calibration_dir
        d.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(path_config.calibration_comparison_save_path, index=False)
        logger.info("已儲存校正比較: %s", path_config.calibration_comparison_save_path)

    return pd.DataFrame()


# ========================================
# 1. cendored和non-censored兩組數據的指標
# ========================================
def analyze_survival_predictions(
    ensemble_predictions: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    path_config: PathConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    分析生存預測結果，支援校正後的數據
    """
    calibration_methods = ["knn_km", "regression", "segmental", "curve"]
    true = processed_df[["patient_id", "time", "event"]]
    results: Dict[str, Dict[str, Any]] = {}

    for key, predictions in ensemble_predictions.items():
        merged_prediction = pd.merge(predictions, true, on="patient_id", how="inner")
        if merged_prediction.empty:
            continue

        non_censored = merged_prediction[merged_prediction["event"] == 1]
        censored = merged_prediction[merged_prediction["event"] == 0]

        stats: Dict[str, Any] = {"key": key}

        # 解析 model_type 和 calibration_method
        stats["model_type"] = key
        stats["calibration_method"] = "original"

        for method in calibration_methods:
            if key.endswith(f"_{method}"):
                stats["model_type"] = key[: -len(f"_{method}")]
                stats["calibration_method"] = method
                break

        if len(non_censored) > 0:
            errors = non_censored["ensemble_prediction"] - non_censored["time"]
            stats["non_censored_metrics"] = {
                "count": len(non_censored),
                "mean_absolute_error": float(np.abs(errors).mean()),
                "root_mean_square_error": float(np.sqrt((errors**2).mean())),
                "mean_error": float(errors.mean()),
                "std_error": float(errors.std()),
                "median_error": float(errors.median()),
            }

        if len(censored) > 0:
            correct = (censored["ensemble_prediction"] > censored["time"]).sum()
            stats["censored_metrics"] = {
                "count": len(censored),
                "correct_ratio": correct / len(censored),
                "mean_predicted_time": float(np.mean(censored["ensemble_prediction"])),
                "mean_observed_time": float(np.mean(censored["time"])),
            }

        results[key] = stats

    # 儲存分析結果
    d = path_config.metrics_dir
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, stats in results.items():
        row = {
            "key": key,
            "model_type": stats.get("model_type"),
            "calibration_method": stats.get("calibration_method"),
        }

        # 展開 nested 指標
        if "non_censored_metrics" in stats:
            for metric, value in stats["non_censored_metrics"].items():
                row[f"non_censored_{metric}"] = value

        if "censored_metrics" in stats:
            for metric, value in stats["censored_metrics"].items():
                row[f"censored_{metric}"] = value

        rows.append(row)
    analysis_df = pd.DataFrame(rows)
    analysis_df.to_csv(
        path_config.K_U_group_metrics_save_path, index=False
    )

    return results


# ========================================
# 3. 統計工具
# ========================================


def create_stage_treatment_shap_table(
    ensemble_importance: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    stage_col: str = "BCLC_stage",
    treatment_cols: List[str] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    創建stage x treatment的pivot table，每格顯示SHAP值的信心區間
    
    Returns:
        DataFrame: index為stage，columns為treatment，值為信心區間字串
    """
    if treatment_cols is None:
        treatment_cols = [
            "liver_transplantation",
            "surgical_resection", 
            "radiofrequency",
            "TACE",
            "target_therapy",
            "immunotherapy",
            "HAIC",
            "radiotherapy",
            "best_support_care",
        ]

    # 獲取SHAP數據
    shap_values = {}
    if ensemble_importance:
        # 收集所有模型的SHAP值
        all_shap_data = []
        for model_type, importance_df in ensemble_importance.items():
            shap_data = importance_df[importance_df["method"] == "shap_importance"]
            shap_data['model_type'] = model_type
            all_shap_data.append(shap_data)
        
        if all_shap_data:
            combined_shap = pd.concat(all_shap_data, ignore_index=True)
            
            # 計算每個治療的整體統計
            for treatment in treatment_cols:
                treatment_shap = combined_shap[combined_shap["feature"] == treatment]
                if not treatment_shap.empty:
                    values_array = treatment_shap["mean_importance"].values
                    mean_val = values_array.mean()
                    n = len(values_array)
                    
                    if n > 1:
                        std_val = values_array.std(ddof=1)
                        from scipy import stats
                        ci_margin = stats.t.ppf((1 + confidence_level) / 2, n - 1) * std_val / np.sqrt(n)
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    shap_values[treatment] = {
                        "mean": mean_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "ci_string": f"{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                    }
                if not treatment_shap.empty:
                    mean_val = treatment_shap["mean_importance"].mean()
                    std_val = treatment_shap["mean_importance"].std()
                    n = len(treatment_shap)
                    
                    # 計算信心區間
                    if n > 1:
                        from scipy import stats
                        ci_margin = stats.t.ppf((1 + confidence_level) / 2, n - 1) * std_val / np.sqrt(n)
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    shap_values[treatment] = {
                        "mean": mean_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "ci_string": f"{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                    }

    # 建立pivot table的數據
    stages = sorted(processed_df[stage_col].unique())
    pivot_data = []
    
    for stage in stages:
        stage_data = {"Stage": stage}
        stage_patients = processed_df[processed_df[stage_col] == stage]
        
        for treatment in treatment_cols:
            if treatment not in processed_df.columns:
                stage_data[treatment] = "N/A"
                continue
            
            # 計算該stage中接受該治療的患者數和比例
            treatment_count = (stage_patients[treatment] == 1).sum()
            stage_total = len(stage_patients)
            percentage = (treatment_count / stage_total * 100) if stage_total > 0 else 0
            
            # 組合患者統計和SHAP信心區間
            if treatment in shap_values:
                shap_ci = shap_values[treatment]["ci_string"]
                cell_value = f"{treatment_count}/{stage_total} ({percentage:.1f}%)\nSHAP: {shap_ci}"
            else:
                cell_value = f"{treatment_count}/{stage_total} ({percentage:.1f}%)\nSHAP: N/A"
            
            stage_data[treatment] = cell_value
        
        pivot_data.append(stage_data)
    
    # 轉換為DataFrame
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df = pivot_df.set_index("Stage")
    
    # 確保treatment順序一致
    pivot_df = pivot_df[treatment_cols]
    
    return pivot_df


def create_stage_treatment_shap_summary(
    pivot_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    創建更易讀的Stage x Treatment摘要表
    分別儲存患者統計和SHAP值
    """
    # 分離患者統計和SHAP值
    patient_stats = pivot_table.copy()
    shap_stats = pivot_table.copy()
    
    for col in pivot_table.columns:
        for idx in pivot_table.index:
            cell_value = str(pivot_table.loc[idx, col])
            if "\n" in cell_value:
                parts = cell_value.split("\n")
                patient_stats.loc[idx, col] = parts[0]
                shap_stats.loc[idx, col] = parts[1].replace("SHAP: ", "") if len(parts) > 1 else "N/A"
            else:
                patient_stats.loc[idx, col] = cell_value
                shap_stats.loc[idx, col] = "N/A"
    
    # 儲存兩個表格
    patient_stats.to_csv(output_path.parent / "stage_treatment_patient_stats.csv")
    shap_stats.to_csv(output_path.parent / "stage_treatment_shap_stats.csv")
    
    # 儲存合併表格
    pivot_table.to_csv(output_path)
# ========================================
# 4. 治療方式調整器 (修正版)
# ========================================


def analyze_treatment_modifications(
    experiment_results: List[ExperimentResult],  # 改為ExperimentResult類型
    processed_df: pd.DataFrame,
    stage_col: str = "BCLC_stage",
    treatment_cols: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    分析治療方式調整對預測的影響（使用所有訓練的模型）
    對每個模型的測試集進行what-if分析
    """
    logger.info("分析治療方式調整影響...")

    if treatment_cols is None:
        treatment_cols = [
            "liver_transplantation",
            "surgical_resection",
            "radiofrequency",
            "TACE",
            "target_therapy",
            "immunotherapy",
            "HAIC",
            "radiotherapy",
            "best_support_care",
        ]

    all_modification_results = {}
    
    # 對每個實驗結果進行分析
    for exp_idx, exp_result in enumerate(experiment_results):
        if not exp_result or not exp_result.model:
            continue
            
        logger.info(f"分析模型 {exp_result.model_type} (實驗 {exp_idx})")
        
        model = exp_result.model
        
        # 從該模型的測試集預測中獲取患者ID
        test_patient_ids = exp_result.test_predictions['patient_id'].unique()
        test_df = processed_df[processed_df['patient_id'].isin(test_patient_ids)].copy()
        
        # 獲取特徵列
        if hasattr(model, 'feature_names_'):
            feature_cols = list(model.feature_names_)
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            # XGBoost
            feature_cols = model.get_booster().feature_names
        elif hasattr(model, 'feature_names'):
            # XGBoost Booster 物件
            feature_cols = model.feature_names
        else:
            # 從feature_importance獲取特徵名稱
            # 優先使用 model_importance，如果沒有就用其他方法
            for method in ['xgb_gain', 'xgb_weight', 'catboost_prediction', 'cox_coefficients', 'shap_importance']:
                if method in exp_result.feature_importance:
                    feature_cols = list(exp_result.feature_importance[method].keys())
                    break
            else:
                feature_cols = []
        
        feature_cols = [col for col in feature_cols if col in test_df.columns]
        
        if not feature_cols:
            logger.warning(f"無法獲取模型 {exp_idx} 的特徵列")
            continue

        # 按stage分析
        model_modifications = {}
        
        for stage in test_df[stage_col].unique():
            stage_patients = test_df[test_df[stage_col] == stage]
            stage_results = []

            for idx, patient in stage_patients.iterrows():
                patient_id = patient["patient_id"]
                current_treatments = {
                    col: patient[col] for col in treatment_cols 
                    if col in patient.index and col in feature_cols
                }

                # 原始預測
                try:
                    X_original = patient[feature_cols].values.reshape(1, -1)
                    
                    if exp_result.model_type == "XGBoost_AFT":
                        import xgboost as xgb
                        dmatrix_original = xgb.DMatrix(X_original)
                        original_pred = model.predict(dmatrix_original)[0]
                    else:
                        original_pred = model.predict(X_original)[0]
                except Exception as e:
                    logger.warning(f"預測患者 {patient_id} 失敗: {e}")
                    continue

                # 嘗試每種治療調整
                for treatment in treatment_cols:
                    if treatment not in feature_cols:
                        continue

                    # 創建調整後的特徵
                    X_modified = X_original.copy()
                    
                    # 關閉所有當前治療，只開啟目標治療
                    for i, col in enumerate(feature_cols):
                        if col in treatment_cols:
                            X_modified[0, i] = 1.0 if col == treatment else 0.0

                    # 預測調整後的結果
                    try:
                        modified_pred = model.predict(X_modified)[0]

                        # 記錄結果
                        stage_results.append({
                            "patient_id": patient_id,
                            "stage": stage,
                            "model_type": exp_result.model_type,
                            "experiment_idx": exp_idx,
                            "original_treatment": [
                                k for k, v in current_treatments.items() if v == 1
                            ],
                            "new_treatment": treatment,
                            "original_prediction": original_pred,
                            "modified_prediction": modified_pred,
                            "prediction_change": modified_pred - original_pred,
                            "relative_change": (
                                (modified_pred - original_pred) / original_pred * 100
                                if original_pred != 0
                                else 0
                            ),
                        })
                    except Exception as e:
                        logger.warning(f"調整預測失敗: {e}")
                        continue

            if stage_results:
                stage_df = pd.DataFrame(stage_results)
                stage_df = stage_df.sort_values("prediction_change", ascending=False)
                model_modifications[f"{exp_result.model_type}_{exp_idx}_stage_{stage}"] = stage_df

        all_modification_results.update(model_modifications)
    
    # 彙總分析結果
    summary_results = {}
    
    # 按stage彙總所有模型的結果
    all_stages = set()
    for key in all_modification_results:
        if "_stage_" in key:
            stage = key.split("_stage_")[-1]
            all_stages.add(stage)
    
    for stage in sorted(all_stages):
        stage_dfs = []
        for key, df in all_modification_results.items():
            if f"_stage_{stage}" in key:
                stage_dfs.append(df)
        
        if stage_dfs:
            # 合併所有模型對該stage的分析
            combined_df = pd.concat(stage_dfs, ignore_index=True)
            
            # 計算每種治療調整的平均影響
            treatment_summary = combined_df.groupby('new_treatment').agg({
                'prediction_change': ['mean', 'std', 'count'],
                'relative_change': ['mean', 'std']
            }).round(2)
            
            summary_results[f"stage_{stage}_summary"] = treatment_summary
            summary_results[f"stage_{stage}_detailed"] = combined_df
    
    return summary_results


# ========================================
# 5. 變數調整器 (修正版)
# ========================================

def analyze_bmi_modifications(
    experiment_results: List[ExperimentResult],  # 改為ExperimentResult類型
    processed_df: pd.DataFrame,
    bmi_col: str = "BMI",
) -> Dict[str, Any]:
    """
    分析BMI調整對預測的影響（使用已訓練的模型）
    """
    logger.info("分析BMI調整影響...")

    if bmi_col not in processed_df.columns:
        logger.warning(f"找不到BMI列: {bmi_col}")
        return {"message": f"找不到BMI列: {bmi_col}"}

    # 獲取第一個可用的模型
    model_info = None
    for result in experiment_results:
        if result and result.model:
            model_info = result
            break

    if not model_info:
        logger.warning("找不到可用的模型進行BMI調整分析")
        return {"message": "找不到可用的模型"}

    model = model_info.model
    
    # 從測試集預測中獲取患者ID
    test_patient_ids = model_info.test_predictions['patient_id'].unique()
    test_df = processed_df[processed_df['patient_id'].isin(test_patient_ids)].copy()
    
    # 獲取特徵列
    if hasattr(model, 'feature_names_'):
        feature_cols = list(model.feature_names_)
    elif hasattr(model, 'feature_names'):
        # XGBoost Booster 物件
        feature_cols = model.feature_names
    else:
        # 從feature_importance獲取特徵名稱
        for method in ['xgb_gain', 'xgb_weight', 'catboost_prediction', 'cox_coefficients', 'shap_importance']:
            if method in model_info.feature_importance:
                feature_cols = list(model_info.feature_importance[method].keys())
                break
        else:
            feature_cols = []
    
    feature_cols = [col for col in feature_cols if col in test_df.columns]

    if bmi_col not in feature_cols:
        logger.warning(f"BMI不在模型特徵中")
        return {"message": "BMI不在模型特徵中"}

    # BMI分組分析（以下邏輯保持不變）
    bmi_values = test_df[bmi_col].dropna()
    min_bmi = int(np.floor(bmi_values.min()))
    max_bmi = int(np.ceil(bmi_values.max()))

    bmi_results = []

    # 對每個BMI區間進行分析
    for bmi_start in range(min_bmi, max_bmi):
        bmi_end = bmi_start + 1
        mask = (test_df[bmi_col] >= bmi_start) & (test_df[bmi_col] < bmi_end)
        group_patients = test_df[mask]

        if len(group_patients) == 0:
            continue

        group_analysis = {
            "bmi_range": f"{bmi_start}-{bmi_end}",
            "patient_count": len(group_patients),
            "mean_bmi": group_patients[bmi_col].mean(),
            "predictions": [],
        }

        # 分析每個患者的BMI調整影響
        for idx, patient in group_patients.iterrows():
            patient_id = patient["patient_id"]
            original_bmi = patient[bmi_col]

            # 準備特徵
            X_original = patient[feature_cols].values.reshape(1, -1)
            bmi_idx = feature_cols.index(bmi_col)

            try:
                # 原始預測
                original_pred = model.predict(X_original)[0]

                # BMI +1 預測
                X_plus1 = X_original.copy()
                X_plus1[0, bmi_idx] = original_bmi + 1
                pred_plus1 = model.predict(X_plus1)[0]

                # BMI -1 預測
                X_minus1 = X_original.copy()
                X_minus1[0, bmi_idx] = max(original_bmi - 1, 10)  # 避免BMI過低
                pred_minus1 = model.predict(X_minus1)[0]

                patient_result = {
                    "patient_id": patient_id,
                    "original_bmi": original_bmi,
                    "original_prediction": original_pred,
                    "bmi_plus1_prediction": pred_plus1,
                    "bmi_minus1_prediction": pred_minus1,
                    "change_plus1": pred_plus1 - original_pred,
                    "change_minus1": pred_minus1 - original_pred,
                }

                group_analysis["predictions"].append(patient_result)

            except Exception as e:
                logger.warning(f"BMI調整預測失敗 (患者 {patient_id}): {e}")
                continue

        if group_analysis["predictions"]:
            # 計算該組的平均影響
            changes_plus1 = [p["change_plus1"] for p in group_analysis["predictions"]]
            changes_minus1 = [p["change_minus1"] for p in group_analysis["predictions"]]

            group_analysis["avg_change_plus1"] = np.mean(changes_plus1)
            group_analysis["avg_change_minus1"] = np.mean(changes_minus1)
            group_analysis["std_change_plus1"] = np.std(changes_plus1)
            group_analysis["std_change_minus1"] = np.std(changes_minus1)

            bmi_results.append(group_analysis)

    # 準備結果
    if bmi_results:
        # 轉換為DataFrame以便分析
        results_df = pd.DataFrame(
            [
                {
                    "bmi_range": r["bmi_range"],
                    "patient_count": r["patient_count"],
                    "mean_bmi": r["mean_bmi"],
                    "avg_change_plus1": r["avg_change_plus1"],
                    "avg_change_minus1": r["avg_change_minus1"],
                }
                for r in bmi_results
            ]
        )

        # 繪製圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # BMI+1影響
        ax1.bar(results_df["bmi_range"], results_df["avg_change_plus1"])
        ax1.set_xlabel("BMI Range")
        ax1.set_ylabel("Average Survival Change (months)")
        ax1.set_title("Impact of BMI +1 on Survival Prediction")
        ax1.tick_params(axis="x", rotation=45)

        # BMI-1影響
        ax2.bar(results_df["bmi_range"], results_df["avg_change_minus1"])
        ax2.set_xlabel("BMI Range")
        ax2.set_ylabel("Average Survival Change (months)")
        ax2.set_title("Impact of BMI -1 on Survival Prediction")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("bmi_modification_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 計算總體統計
        all_predictions = []
        for r in bmi_results:
            all_predictions.extend(r["predictions"])

        summary = {
            "total_groups": len(bmi_results),
            "total_patients_analyzed": len(all_predictions),
            "avg_survival_change_plus1": np.mean(
                [p["change_plus1"] for p in all_predictions]
            ),
            "avg_survival_change_minus1": np.mean(
                [p["change_minus1"] for p in all_predictions]
            ),
            "max_positive_change": max([p["change_plus1"] for p in all_predictions]),
            "max_negative_change": min([p["change_minus1"] for p in all_predictions]),
        }

        logger.info(f"BMI調整分析完成:")
        logger.info(f"  分析組別數: {summary['total_groups']}")
        logger.info(f"  分析患者數: {summary['total_patients_analyzed']}")
        logger.info(
            f"  BMI+1平均影響: {summary['avg_survival_change_plus1']:.2f} months"
        )
        logger.info(
            f"  BMI-1平均影響: {summary['avg_survival_change_minus1']:.2f} months"
        )

        return {
            "results_df": results_df,
            "detailed_results": bmi_results,
            "summary": summary,
            "plot_saved": "bmi_modification_analysis.png",
        }
    else:
        return {"message": "沒有足夠的數據進行BMI分析"}

