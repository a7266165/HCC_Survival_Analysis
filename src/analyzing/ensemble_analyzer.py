# TODO: 精簡
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from lifelines.utils import concordance_index
from utils.config_utils import PathConfig
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

    d = path_config.ensemble_feature_importance_dir
    d.mkdir(parents=True, exist_ok=True)
    all_results = pd.concat(ensemble_results.values(), ignore_index=True)
    all_results.to_csv(d / "all_models_feature_importance.csv", index=False)

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
    logger.info(
        "已儲存 Ensemble C-index 指標: %s", path_config.ensemble_c_index_save_path
    )

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
    analysis_df.to_csv(path_config.K_U_group_metrics_save_path, index=False)

    return results
