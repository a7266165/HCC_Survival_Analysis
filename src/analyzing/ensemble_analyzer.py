import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict
from lifelines.utils import concordance_index
from dataclasses import dataclass

from utils.config_utils import PathConfig
from experimenting.experimentor import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleAnalysisResults:
    """封裝Ensemble分析結果"""

    ensemble_predictions: Dict[str, pd.DataFrame]
    feature_importance: Dict[str, pd.DataFrame]
    ensemble_metrics: Dict[str, Dict[str, float]]
    survival_analysis: Dict[str, Dict[str, Any]]
    calibration_comparison: pd.DataFrame


class EnsembleAnalyzer:
    """
    Ensemble分析器

    使用示例：
        analyzer = EnsembleAnalyzer(path_config)
        results = analyzer.run_complete_analysis(
            experiment_results=total_experiments_result,
            processed_df=processed_df,
            include_calibrated=True
        )
    """

    def __init__(self, path_config: PathConfig):
        """
        初始化分析器

        Args:
            path_config: 路徑配置
        """
        self.path_config = path_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # 確保必要目錄存在
        self._ensure_directories()

        # 中間結果緩存
        self._ensemble_predictions: Optional[Dict[str, pd.DataFrame]] = None
        self._feature_importance: Optional[Dict[str, pd.DataFrame]] = None
        self._ensemble_metrics: Optional[Dict[str, Dict[str, float]]] = None
        self._survival_analysis: Optional[Dict[str, Dict[str, Any]]] = None
        self._calibration_comparison: Optional[pd.DataFrame] = None

    def _ensure_directories(self) -> None:
        """確保目錄存在"""
        directories = [
            self.path_config.ensemble_predictions_dir,
            self.path_config.ensemble_feature_importance_dir,
            self.path_config.metrics_dir,
            self.path_config.calibration_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def run_complete_analysis(
        self,
        experiment_results: List[ExperimentResult],
        processed_df: pd.DataFrame,
        include_calibrated: bool = True,
    ) -> EnsembleAnalysisResults:
        """
        執行完整ensemble分析

        Args:
            experiment_results: 實驗結果列表
            processed_df: 處理後的原始數據
            include_calibrated: 是否包含校正結果

        Returns:
            完整的分析結果封裝
        """
        self.logger.info("開始完整的Ensemble分析...")

        try:
            # 1. Ensemble預測
            self.logger.info("1/5 生成Ensemble預測...")
            self._ensemble_predictions = self._generate_ensemble_predictions(
                experiment_results, include_calibrated
            )

            # 2. 特徵重要性分析
            self.logger.info("2/5 分析特徵重要性...")
            self._feature_importance = self._analyze_feature_importance(
                experiment_results
            )

            # 3. 計算Ensemble指標
            self.logger.info("3/5 計算Ensemble指標...")
            self._ensemble_metrics = self._calculate_ensemble_metrics(
                processed_df, experiment_results
            )

            # 4. 生存數據分析
            self.logger.info("4/5 執行生存數據分析...")
            self._survival_analysis = self._analyze_survival_predictions(processed_df)

            # 5. 校正方法比較
            self.logger.info("5/5 比較校正方法...")
            self._calibration_comparison = self._compare_calibration_methods()

            self.logger.info("✅ Ensemble分析完成！")

            return EnsembleAnalysisResults(
                ensemble_predictions=self._ensemble_predictions,
                feature_importance=self._feature_importance,
                ensemble_metrics=self._ensemble_metrics,
                survival_analysis=self._survival_analysis,
                calibration_comparison=self._calibration_comparison,
            )

        except Exception as e:
            self.logger.error(f"❌ Ensemble分析過程中發生錯誤: {e}")
            raise

    def _generate_ensemble_predictions(
        self, experiment_results: List[ExperimentResult], include_calibrated: bool
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
                    for (
                        method,
                        calibrated_df,
                    ) in res.calibrated_test_predictions.items():
                        model_predictions[res.model_type][method].append(calibrated_df)

        # 計算 ensemble
        ensemble_results = {}

        for model_type, method_predictions in model_predictions.items():
            model_ensemble = {}

            for method, dfs in method_predictions.items():
                if not dfs:
                    self.logger.warning(
                        f"模型 {model_type} 方法 {method} 沒有有效的預測結果"
                    )
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

        # 儲存結果 - 使用 self.path_config
        for key, df in ensemble_results.items():
            save_path = (
                self.path_config.ensemble_predictions_dir
                / f"{key}_ensemble_predictions.csv"
            )
            df.to_csv(save_path, index=False)
            self.logger.info(f"已儲存 Ensemble 預測: {save_path}")

        return ensemble_results

    def _analyze_feature_importance(
        self, experiment_results: List[ExperimentResult]
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
            self.logger.warning("沒有找到有效的特徵重要性數據")
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

        # 儲存結果 - 使用 self.path_config
        all_results = pd.concat(ensemble_results.values(), ignore_index=True)
        save_path = (
            self.path_config.ensemble_feature_importance_dir
            / "all_models_feature_importance.csv"
        )
        all_results.to_csv(save_path, index=False)
        self.logger.info(f"已儲存特徵重要性: {save_path}")

        return ensemble_results

    def _calculate_ensemble_metrics(
        self, processed_df: pd.DataFrame, experiment_results: List[ExperimentResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        計算 ensemble 指標，包括校正後的結果
        """
        if self._ensemble_predictions is None:
            raise ValueError("必須先生成ensemble預測")

        # 檢查必要欄位
        required = ["patient_id", "time", "event"]
        if not all(col in processed_df.columns for col in required):
            self.logger.error(f"processed_df 缺少必要的列: {required}")
            return {}

        true_data = processed_df[required]
        metrics = {}

        for key, pred_df in self._ensemble_predictions.items():
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
                    for res in experiment_results
                    if res and res.model_type == model_type
                ]
            else:
                indiv_c_indices = [
                    res.calibrated_test_c_index.get(calibration_method, np.nan)
                    for res in experiment_results
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
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        metrics_df.to_csv(self.path_config.ensemble_c_index_save_path, index=True)
        self.logger.info(
            f"已儲存 Ensemble C-index 指標: {self.path_config.ensemble_c_index_save_path}"
        )

        return metrics

    def _analyze_survival_predictions(
        self, processed_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        分析生存預測結果，支援校正後的數據
        """
        if self._ensemble_predictions is None:
            raise ValueError("必須先生成ensemble預測")

        calibration_methods = ["knn_km", "regression", "segmental", "curve"]
        true = processed_df[["patient_id", "time", "event"]]
        results: Dict[str, Dict[str, Any]] = {}

        for key, predictions in self._ensemble_predictions.items():
            merged_prediction = pd.merge(
                predictions, true, on="patient_id", how="inner"
            )
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
                    "mean_predicted_time": float(
                        np.mean(censored["ensemble_prediction"])
                    ),
                    "mean_observed_time": float(np.mean(censored["time"])),
                }

            results[key] = stats

        # 儲存分析結果 - 使用 self.path_config
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
        analysis_df.to_csv(self.path_config.K_U_group_metrics_save_path, index=False)
        self.logger.info(
            f"已儲存K/U群組指標: {self.path_config.K_U_group_metrics_save_path}"
        )

        return results

    def _compare_calibration_methods(self) -> pd.DataFrame:
        """
        比較不同校正方法的效果
        """
        if self._survival_analysis is None:
            raise ValueError("必須先完成生存分析")

        comparison_rows = []

        # 按模型類型分組
        model_results = defaultdict(dict)
        for key, stats in self._survival_analysis.items():
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
                    ((original_mae - mae) / original_mae) * 100
                    if original_mae > 0
                    else 0
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
            comparison_df.to_csv(
                self.path_config.calibration_comparison_save_path, index=False
            )
            self.logger.info(
                f"已儲存校正比較: {self.path_config.calibration_comparison_save_path}"
            )
            return comparison_df

        return pd.DataFrame()

    # 便捷的訪問屬性 - 提供更友好的API
    @property
    def ensemble_predictions(self) -> Dict[str, pd.DataFrame]:
        """獲取ensemble預測結果"""
        if self._ensemble_predictions is None:
            raise ValueError(
                "尚未執行ensemble預測分析，請先調用 run_complete_analysis()"
            )
        return self._ensemble_predictions

    @property
    def feature_importance(self) -> Dict[str, pd.DataFrame]:
        """獲取特徵重要性結果"""
        if self._feature_importance is None:
            raise ValueError("尚未執行特徵重要性分析，請先調用 run_complete_analysis()")
        return self._feature_importance

    @property
    def ensemble_metrics(self) -> Dict[str, Dict[str, float]]:
        """獲取ensemble指標"""
        if self._ensemble_metrics is None:
            raise ValueError(
                "尚未執行ensemble指標計算，請先調用 run_complete_analysis()"
            )
        return self._ensemble_metrics

    @property
    def survival_analysis(self) -> Dict[str, Dict[str, Any]]:
        """獲取生存分析結果"""
        if self._survival_analysis is None:
            raise ValueError("尚未執行生存分析，請先調用 run_complete_analysis()")
        return self._survival_analysis

    @property
    def calibration_comparison(self) -> pd.DataFrame:
        """獲取校正比較結果"""
        if self._calibration_comparison is None:
            raise ValueError("尚未執行校正比較，請先調用 run_complete_analysis()")
        return self._calibration_comparison

    def get_analysis_summary(self) -> Dict[str, Any]:
        """獲取分析結果摘要"""
        if self._ensemble_predictions is None:
            return {"status": "未開始分析"}

        summary = {
            "status": "分析完成",
            "ensemble_predictions_count": (
                len(self._ensemble_predictions) if self._ensemble_predictions else 0
            ),
            "feature_importance_models": (
                len(self._feature_importance) if self._feature_importance else 0
            ),
            "ensemble_metrics_count": (
                len(self._ensemble_metrics) if self._ensemble_metrics else 0
            ),
            "survival_analysis_methods": (
                len(self._survival_analysis) if self._survival_analysis else 0
            ),
            "calibration_comparison_size": (
                len(self._calibration_comparison)
                if self._calibration_comparison is not None
                else 0
            ),
        }

        return summary
