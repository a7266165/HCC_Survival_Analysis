# src/analyzing/visualizer.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from utils.config_utils import PathConfig
from lifelines import KaplanMeierFitter
from experimenting.experimentor import ExperimentResult

logger = logging.getLogger(__name__)


class SurvivalVisualizer:
    """存活分析視覺化工具"""

    def __init__(
        self,
        path_config: PathConfig,
        processed_df: pd.DataFrame,
        experiment_results: List[ExperimentResult],
    ):
        """
        初始化視覺化工具

        Args:
            path_config: 路徑配置物件
            processed_df: 處理後的資料集（包含生存時間和事件）
            experiment_results: 實驗結果列表
        """
        self.path_config = path_config
        self.figures_dir = path_config.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 儲存資料參考
        self.processed_df = processed_df
        self.experiment_results = experiment_results

        # 資料驗證
        self._validate_data()

        # 設定視覺化風格
        self._setup_plot_style()

    def _validate_data(self) -> None:
        """驗證輸入資料的完整性"""
        if self.processed_df is not None:
            required_cols = ["patient_id", "time", "event"]
            missing_cols = [
                col for col in required_cols if col not in self.processed_df.columns
            ]
            if missing_cols:
                logger.warning(f"processed_df 缺少必要欄位: {missing_cols}")

    def _setup_plot_style(self):
        """設定統一的繪圖風格"""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

    def plot_feature_importance_heatmap(
        self, top_n: int = 20, figsize: Optional[tuple] = None
    ) -> None:
        """
        從 ensemble_feature_importance 資料夾讀取資料並繪製熱圖

        Args:
            top_n: 顯示前 N 個重要特徵
            figsize: 圖片大小
        """
        logger.info(f"繪製前 {top_n} 個特徵重要性熱圖...")

        # 讀取特徵重要性資料（現在只有一個檔案）
        importance_file = (
            self.path_config.ensemble_feature_importance_dir
            / "all_models_feature_importance.csv"
        )

        if not importance_file.exists():
            logger.warning(f"找不到特徵重要性資料: {importance_file}")
            return

        df = pd.read_csv(importance_file)

        # 使用單一檔案的資料
        # 假設使用 shap_importance 方法
        shap_data = df[df["method"] == "shap_importance"]

        if shap_data.empty:
            logger.warning("找不到 shap_importance 方法的資料")
            return

        # 計算所有模型的平均重要性來選擇 top N
        all_features = {}

        # 按 feature 分組，計算跨模型的平均重要性
        for feature in shap_data["feature"].unique():
            feature_data = shap_data[shap_data["feature"] == feature]
            avg_importance = feature_data["mean_importance"].mean()
            all_features[feature] = avg_importance

        # 排序並選擇 top N
        top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        top_feature_names = [f[0] for f in top_features]

        # 建立熱圖矩陣
        models = sorted(shap_data["model_type"].unique())
        heatmap_matrix = np.zeros((len(top_feature_names), len(models)))

        for j, model in enumerate(models):
            model_data = shap_data[shap_data["model_type"] == model]

            for i, feature in enumerate(top_feature_names):
                feat_data = model_data[model_data["feature"] == feature]
                if not feat_data.empty:
                    heatmap_matrix[i, j] = feat_data["mean_importance"].values[0]

        # 繪製熱圖
        if figsize is None:
            figsize = (10, max(8, len(top_feature_names) * 0.3))

        plt.figure(figsize=figsize)

        # 建立 DataFrame 以便使用 seaborn
        heatmap_df = pd.DataFrame(
            heatmap_matrix, index=top_feature_names, columns=models
        )

        # 繪製
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".4f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Mean SHAP Importance"},
            linewidths=0.5,
            square=False,
        )

        plt.title(f"Top {top_n} Feature Importance Heatmap (SHAP)", fontsize=16, pad=20)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / f"feature_importance_top{top_n}_heatmap.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"特徵重要性熱圖已儲存至: {save_path}")

    def plot_ku_group_metrics(self, figsize: Optional[tuple] = None) -> None:
        """
        讀取 K_U_group_metrics.csv 並繪製分析圖

        Args:
            figsize: 圖片大小
        """
        logger.info("繪製 K/U 群組指標分析圖...")

        # 讀取資料
        metrics_path = self.path_config.K_U_group_metrics_save_path
        if not metrics_path.exists():
            logger.warning(f"找不到 K/U 群組指標資料: {metrics_path}")
            return

        data = pd.read_csv(metrics_path)

        # 建立圖表
        if figsize is None:
            figsize = (16, 10)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        # 1. Non-censored MAE 比較
        self._plot_mae_comparison(data, axes[0])

        # 2. Censored 預測正確率
        self._plot_censored_accuracy(data, axes[1])

        # 3. 預測誤差分布（Mean ± Std）
        self._plot_error_distribution(data, axes[2])

        # 4. 校正方法效果比較
        self._plot_calibration_effect(data, axes[3])

        plt.suptitle(
            "Survival Prediction Analysis: Censored vs Non-censored Groups",
            fontsize=16,
            y=0.995,
        )
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / "ku_group_metrics_analysis.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"K/U 群組分析圖已儲存至: {save_path}")

    def _plot_mae_comparison(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """繪製 MAE 比較圖"""
        # 篩選有 MAE 的資料
        mae_col = "non_censored_mean_absolute_error"
        mae_data = data[data[mae_col].notna()].copy()

        if mae_data.empty:
            ax.text(
                0.5,
                0.5,
                "No MAE data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Pivot 資料
        pivot = mae_data.pivot_table(
            values=mae_col,
            index="model_type",
            columns="calibration_method",
            aggfunc="first",
        )

        # 重新排序 columns，讓 original 在最前面
        if "original" in pivot.columns:
            cols = ["original"] + [c for c in pivot.columns if c != "original"]
            pivot = pivot[cols]

        # 繪製
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Mean Absolute Error Comparison", fontsize=13)
        ax.set_xlabel("Model Type")
        ax.set_ylabel("MAE (months)")
        ax.legend(title="Calibration Method", frameon=True)
        ax.grid(axis="y", alpha=0.3)

        # 旋轉 x 標籤
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_censored_accuracy(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """繪製 Censored 組預測正確率"""
        # 篩選有正確率的資料
        acc_col = "censored_correct_ratio"
        acc_data = data[data[acc_col].notna()].copy()

        if acc_data.empty:
            ax.text(
                0.5,
                0.5,
                "No censored accuracy data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Pivot 資料
        pivot = acc_data.pivot_table(
            values=acc_col,
            index="model_type",
            columns="calibration_method",
            aggfunc="first",
        )

        # 重新排序 columns
        if "original" in pivot.columns:
            cols = ["original"] + [c for c in pivot.columns if c != "original"]
            pivot = pivot[cols]

        # 繪製
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Prediction Accuracy for Censored Patients", fontsize=13)
        ax.set_xlabel("Model Type")
        ax.set_ylabel("Correct Prediction Ratio")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Calibration Method", frameon=True)
        ax.grid(axis="y", alpha=0.3)

        # 加上百分比標籤
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_error_distribution(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """繪製預測誤差分布"""
        # 準備資料
        mean_col = "non_censored_mean_error"
        std_col = "non_censored_std_error"

        error_data = data[data[mean_col].notna()].copy()

        if error_data.empty:
            ax.text(
                0.5,
                0.5,
                "No error distribution data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # 準備繪圖資料
        models = error_data["model_type"].unique()
        methods = error_data["calibration_method"].unique()

        # 重新排序 methods
        if "original" in methods:
            methods = ["original"] + [m for m in methods if m != "original"]

        x = np.arange(len(models))
        width = 0.8 / len(methods)

        # 繪製每個方法
        for i, method in enumerate(methods):
            method_data = error_data[error_data["calibration_method"] == method]

            means = []
            stds = []

            for model in models:
                model_method_data = method_data[method_data["model_type"] == model]
                if not model_method_data.empty:
                    means.append(model_method_data[mean_col].values[0])
                    stds.append(model_method_data[std_col].values[0])
                else:
                    means.append(0)
                    stds.append(0)

            # 繪製 bar 圖 with error bars
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(
                x + offset, means, width, yerr=stds, label=method, alpha=0.8, capsize=5
            )

        ax.set_xlabel("Model Type")
        ax.set_ylabel("Prediction Error (months)")
        ax.set_title("Prediction Error Distribution (Mean ± Std)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(title="Calibration Method", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    def _plot_calibration_effect(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """繪製校正方法的改善效果"""
        # 計算相對於 original 的改善百分比
        mae_col = "non_censored_mean_absolute_error"

        # 取得 original 的 MAE
        original_data = data[
            (data["calibration_method"] == "original") & (data[mae_col].notna())
        ]

        if original_data.empty:
            ax.text(
                0.5,
                0.5,
                "No original data for comparison",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # 計算改善百分比
        improvements = []

        for model in original_data["model_type"].unique():
            orig_mae = original_data[original_data["model_type"] == model][
                mae_col
            ].values[0]

            # 計算每種校正方法的改善
            model_data = data[(data["model_type"] == model) & (data[mae_col].notna())]

            for _, row in model_data.iterrows():
                if row["calibration_method"] != "original":
                    improvement = (orig_mae - row[mae_col]) / orig_mae * 100
                    improvements.append(
                        {
                            "model": model,
                            "method": row["calibration_method"],
                            "improvement": improvement,
                        }
                    )

        if not improvements:
            ax.text(
                0.5,
                0.5,
                "No calibration data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # 轉換為 DataFrame 並繪圖
        imp_df = pd.DataFrame(improvements)
        pivot = imp_df.pivot(index="model", columns="method", values="improvement")

        # 繪製
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Calibration Improvement (% MAE Reduction)", fontsize=13)
        ax.set_xlabel("Model Type")
        ax.set_ylabel("Improvement (%)")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.legend(title="Calibration Method", frameon=True)
        ax.grid(axis="y", alpha=0.3)

        # 加上百分比標籤
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", fontsize=9)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def plot_feature_importance_bars(
        self, top_n: int = 15, figsize: Optional[tuple] = None
    ) -> None:
        """
        讀取特徵重要性資料並繪製分組柱狀圖

        Args:
            top_n: 顯示前 N 個重要特徵
            figsize: 圖片大小
        """
        logger.info(f"繪製前 {top_n} 個特徵重要性柱狀圖...")

        # 讀取特徵重要性資料
        importance_file = (
            self.path_config.ensemble_feature_importance_dir
            / "all_models_feature_importance.csv"
        )

        if not importance_file.exists():
            logger.warning(f"找不到特徵重要性資料: {importance_file}")
            return

        df = pd.read_csv(importance_file)

        # 使用 shap_importance 方法
        shap_data = df[df["method"] == "shap_importance"]

        if shap_data.empty:
            logger.warning("找不到 shap_importance 方法的資料")
            return

        # 取得所有模型
        models = sorted(shap_data["model_type"].unique())
        n_models = len(models)

        # 設定圖片大小
        if figsize is None:
            figsize = (8 * n_models, 6)

        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        # 為每個模型繪製柱狀圖
        for idx, model in enumerate(models):
            ax = axes[idx]

            # 取得該模型的資料並排序
            model_data = shap_data[shap_data["model_type"] == model].copy()
            model_data = model_data.sort_values("mean_importance", ascending=True).tail(
                top_n
            )

            # 繪製橫向柱狀圖
            bars = ax.barh(model_data["feature"], model_data["mean_importance"])

            # 設定顏色（重要性越高顏色越深）
            colors = plt.cm.RdBu_r(
                model_data["mean_importance"] / model_data["mean_importance"].max()
            )
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # 加上數值標籤
            for i, (_, row) in enumerate(model_data.iterrows()):
                ax.text(
                    row["mean_importance"] + 0.001,
                    i,
                    f"{row['mean_importance']:.4f}",
                    va="center",
                    fontsize=9,
                )

            ax.set_xlabel("Mean SHAP Importance")
            ax.set_title(f"{model} - Top {top_n} Features", fontsize=12)
            ax.grid(axis="x", alpha=0.3)

            # 調整邊界
            ax.set_xlim(0, model_data["mean_importance"].max() * 1.15)

        plt.suptitle(f"Feature Importance Analysis (SHAP)", fontsize=16, y=0.98)
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / f"feature_importance_top{top_n}_bars.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"特徵重要性柱狀圖已儲存至: {save_path}")

    def plot_calibration_scatter(self, figsize: Optional[tuple] = None) -> None:
        """
        繪製校正前後的散點圖，展示預測誤差分布（包含訓練集和測試集）

        Args:
            figsize: 圖片大小
        """
        if self.processed_df is None:
            logger.warning("需要 processed_df 才能繪製校正散點圖")
            return

        logger.info("繪製校正前後的預測誤差散點圖（訓練集+測試集）...")

        # 讀取原始預測結果
        predictions_dir = self.path_config.result_save_dir / "original_predictions"
        predictions_file = predictions_dir / "original_predictions_result.csv"

        if not predictions_file.exists():
            logger.warning(f"找不到預測結果檔案: {predictions_file}")
            return

        # 讀取所有預測結果
        all_predictions = pd.read_csv(predictions_file)

        # 檢查是否有 dataset 欄位（新格式）
        has_dataset_column = "dataset" in all_predictions.columns

        # 合併真實的生存時間資料
        true_data = self.processed_df[["patient_id", "time", "event"]]
        all_predictions = all_predictions.merge(true_data, on="patient_id", how="left")

        # 取得模型列表（排除 CoxPHFitter，因為沒有校正）
        models = [m for m in all_predictions["model_type"].unique()]

        # 校正方法
        calibration_methods = ["knn_km", "regression", "segmental", "curve", "leaf_similarity"]

        # 設定圖片大小 - 現在需要顯示訓練集和測試集
        if figsize is None:
            figsize = (16, 8 * len(models))  # 每個模型需要兩行

        fig, axes = plt.subplots(len(models) * 2, len(calibration_methods), figsize=figsize)
        if len(models) == 1:
            axes = axes.reshape(2, -1)

        # 為每個模型和校正方法繪製散點圖
        for i, model in enumerate(models):
            # 取得訓練集和測試集的原始預測
            if has_dataset_column:
                train_original = all_predictions[
                    (all_predictions["model_type"] == model)
                    & (all_predictions["calibration_method"] == "original")
                    & (all_predictions["dataset"] == "train")
                ]

                test_original = all_predictions[
                    (all_predictions["model_type"] == model)
                    & (all_predictions["calibration_method"] == "original")
                    & (all_predictions["dataset"] == "test")
                ]
            else:
                # 舊格式：只有測試集資料
                train_original = pd.DataFrame()  # 空的 DataFrame
                test_original = all_predictions[
                    (all_predictions["model_type"] == model)
                    & (all_predictions["calibration_method"] == "original")
                ]

            if test_original.empty:
                continue

            # 計算預測誤差
            if not train_original.empty:
                train_original_error = (
                    train_original["predicted_survival_time"] - train_original["time"]
                )
            test_original_error = (
                test_original["predicted_survival_time"] - test_original["time"]
            )

            for j, method in enumerate(calibration_methods):
                # 訓練集圖
                ax_train = axes[i * 2, j]

                if not train_original.empty:
                    # 繪製訓練集原始預測（藍點）
                    ax_train.scatter(
                        train_original["time"],
                        train_original_error,
                        alpha=0.5,
                        s=20,
                        c="blue",
                        label="Original (Train)",
                        edgecolors="none",
                    )

                    # 加上統計資訊
                    mae = np.abs(train_original_error).mean()
                    bias = train_original_error.mean()
                    ax_train.text(
                        0.02,
                        0.98,
                        f"MAE: {mae:.2f}\nBias: {bias:.2f}",
                        transform=ax_train.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                        fontsize=9,
                    )
                else:
                    ax_train.text(
                        0.5,
                        0.5,
                        "Training Set Data\nNot Available\n\n(Re-run to generate)",
                        ha="center",
                        va="center",
                        transform=ax_train.transAxes,
                        fontsize=12,
                        color="gray",
                    )

                # 加上零線
                ax_train.axhline(
                    y=0, color="black", linestyle="--", alpha=0.5, linewidth=1
                )
                ax_train.set_xlabel("Actual Survival Time (months)")
                ax_train.set_ylabel("Prediction Error (months)")
                ax_train.set_title(f"{model} - {method} (Train)")
                ax_train.grid(True, alpha=0.3)
                ax_train.set_ylim(-50, 50)
                if not train_original.empty:
                    ax_train.legend(frameon=True, loc="upper right")

                # 測試集圖
                ax_test = axes[i * 2 + 1, j]

                # 繪製原始預測（藍點）
                ax_test.scatter(
                    test_original["time"],
                    test_original_error,
                    alpha=0.5,
                    s=20,
                    c="blue",
                    label="Original",
                    edgecolors="none",
                )

                # 取得校正後的預測（測試集）
                if has_dataset_column:
                    test_calibrated = all_predictions[
                        (all_predictions["model_type"] == model)
                        & (all_predictions["calibration_method"] == method)
                        & (all_predictions["dataset"] == "test")
                    ]
                else:
                    test_calibrated = all_predictions[
                        (all_predictions["model_type"] == model)
                        & (all_predictions["calibration_method"] == method)
                    ]

                if not test_calibrated.empty:
                    # 計算校正後的預測誤差
                    test_calibrated_error = (
                        test_calibrated["predicted_survival_time"]
                        - test_calibrated["time"]
                    )

                    # 繪製校正後的預測（紅點）
                    ax_test.scatter(
                        test_calibrated["time"],
                        test_calibrated_error,
                        alpha=0.5,
                        s=20,
                        c="red",
                        label=method,
                        edgecolors="none",
                    )

                    # 加上統計資訊比較
                    orig_mae = np.abs(test_original_error).mean()
                    cal_mae = np.abs(test_calibrated_error).mean()
                    improvement = (
                        ((orig_mae - cal_mae) / orig_mae * 100) if orig_mae > 0 else 0
                    )

                    ax_test.text(
                        0.02,
                        0.98,
                        f"Original MAE: {orig_mae:.2f}\n{method} MAE: {cal_mae:.2f}\nImprove: {improvement:.1f}%",
                        transform=ax_test.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                        fontsize=9,
                    )

                # 加上零線
                ax_test.axhline(
                    y=0, color="black", linestyle="--", alpha=0.5, linewidth=1
                )

                # 設定標籤和標題
                ax_test.set_xlabel("Actual Survival Time (months)")
                ax_test.set_ylabel("Prediction Error (months)")
                ax_test.set_title(f"{model} - {method} (Test)")
                ax_test.grid(True, alpha=0.3)
                ax_test.legend(frameon=True, loc="upper right")

                # 設定 y 軸範圍
                ax_test.set_ylim(-50, 50)

        plt.suptitle(
            "Calibration Effect: Prediction Error Distribution (Train & Test Sets)",
            fontsize=16,
            y=0.995,
        )
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / "calibration_scatter_plots_train_test.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"校正散點圖（訓練集+測試集）已儲存至: {save_path}")

    def plot_survival_curves(
        self,
        risk_groups: Optional[int] = 3,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        繪製生存曲線（Kaplan-Meier曲線）按風險分組

        Args:
            risk_groups: 風險分組數量（預設為3：低、中、高風險）
            figsize: 圖片大小
        """
        if self.processed_df is None:
            logger.warning("需要 processed_df 才能繪製生存曲線")
            return

        logger.info(f"繪製按 {risk_groups} 個風險組別分類的生存曲線...")

        # 讀取ensemble預測結果
        ensemble_dir = self.path_config.ensemble_predictions_dir

        # 尋找可用的ensemble預測檔案
        ensemble_files = list(ensemble_dir.glob("*_ensemble_predictions.csv"))
        if not ensemble_files:
            logger.warning("找不到ensemble預測結果")
            return

        # 使用第一個找到的檔案（或可以讓使用者選擇）
        ensemble_df = pd.read_csv(ensemble_files[0])

        # 合併預測與真實資料
        merged_df = ensemble_df.merge(
            self.processed_df[["patient_id", "time", "event"]], on="patient_id"
        )

        # 根據預測時間將患者分組
        merged_df["risk_group"] = pd.qcut(
            merged_df["ensemble_prediction"],
            q=risk_groups,
            labels=[f"Group {i+1}" for i in range(risk_groups)],
        )

        # 設定圖片大小
        if figsize is None:
            figsize = (12, 8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左圖：Kaplan-Meier生存曲線
        kmf = KaplanMeierFitter()
        colors = plt.cm.coolwarm(np.linspace(0, 1, risk_groups))

        for i, (group_name, group_df) in enumerate(
            merged_df.groupby("risk_group", observed=False)
        ):
            kmf.fit(group_df["time"], group_df["event"], label=group_name)
            kmf.plot_survival_function(ax=ax1, color=colors[i], ci_show=True)

        ax1.set_xlabel("Time (months)")
        ax1.set_ylabel("Survival Probability")
        ax1.set_title("Kaplan-Meier Survival Curves by Risk Groups")
        ax1.legend(title="Risk Groups", loc="best")
        ax1.grid(True, alpha=0.3)

        # 右圖：預測時間分布
        for i, (group_name, group_df) in enumerate(
            merged_df.groupby("risk_group", observed=False)
        ):
            ax2.hist(
                group_df["ensemble_prediction"],
                bins=20,
                alpha=0.6,
                label=group_name,
                color=colors[i],
                density=True,
            )

        ax2.set_xlabel("Predicted Survival Time (months)")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of Predicted Survival Times")
        ax2.legend(title="Risk Groups")
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Survival Analysis by Risk Groups", fontsize=16)
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / f"survival_curves_{risk_groups}_groups.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"生存曲線已儲存至: {save_path}")

    def plot_whatif_treatment_analysis(
            self,
            figsize: Optional[tuple] = None,
        ) -> None:
            """
            視覺化What-if治療分析結果 - 表格形式
            將多次實驗的同一patient_id結果平均後，按治療手段和分期顯示
            根據整體平均改良幅度排序（改良幅度大的在上面）
            """
            if self.experiment_results is None:
                logger.warning("需要 experiment_results 才能繪製What-if治療分析")
                return

            logger.info("生成What-if治療分析表格...")

            # 收集所有治療分析結果
            all_treatment_results = []

            for result in self.experiment_results:
                if result and result.whatif_treatment_results:
                    for analysis_name, df in result.whatif_treatment_results.items():
                        df_copy = df.copy()
                        df_copy["model_type"] = result.model_type
                        all_treatment_results.append(df_copy)

            if not all_treatment_results:
                logger.warning("沒有找到What-if治療分析結果")
                return

            # 合併所有結果
            combined_df = pd.concat(all_treatment_results, ignore_index=True)
            
            # 先對同一個patient_id的多次實驗結果取平均
            patient_avg = combined_df.groupby(['patient_id', 'stage']).agg({
                'treatment_benefit_months': 'mean',
                'current_treatments': 'first'  # 同一個病人的治療應該是一樣的
            }).reset_index()

            # 獲取所有治療手段
            all_treatments = set()
            for treatments in patient_avg["current_treatments"]:
                if treatments:  # 確保不是空列表
                    all_treatments.update(treatments)
            all_treatments = list(all_treatments)

            # 獲取所有分期
            stages = sorted(patient_avg["stage"].unique())

            # 計算每個治療的整體平均改良幅度（用於排序）
            treatment_overall_benefit = {}
            
            for treatment in all_treatments:
                all_benefits = []
                
                for stage in stages:
                    stage_patients = patient_avg[patient_avg["stage"] == stage]
                    with_treatment = stage_patients[
                        stage_patients["current_treatments"].apply(lambda x: treatment in x if x else False)
                    ]
                    
                    if len(with_treatment) > 0:
                        # 收集該治療在該分期的所有效益值
                        all_benefits.extend(with_treatment["treatment_benefit_months"].tolist())
                
                # 計算該治療的整體平均效益
                if all_benefits:
                    treatment_overall_benefit[treatment] = np.mean(all_benefits)
                else:
                    treatment_overall_benefit[treatment] = 0
            
            # 根據整體平均效益排序（降序，效益大的在前）
            all_treatments = sorted(all_treatments, 
                                key=lambda x: treatment_overall_benefit[x], 
                                reverse=True)

            # 創建結果字典
            results_dict = {}
            
            for treatment in all_treatments:
                results_dict[treatment] = {}
                
                for stage in stages:
                    # 篩選該分期且有使用該治療的病人
                    stage_patients = patient_avg[patient_avg["stage"] == stage]
                    with_treatment = stage_patients[
                        stage_patients["current_treatments"].apply(lambda x: treatment in x if x else False)
                    ]
                    
                    if len(with_treatment) > 0:
                        mean_benefit = with_treatment["treatment_benefit_months"].mean()
                        std_benefit = with_treatment["treatment_benefit_months"].std()
                        n_patients = len(with_treatment)
                        
                        # 格式化顯示：平均值 ± 標準差 (n=樣本數)
                        results_dict[treatment][f"Stage_{stage}"] = f"{mean_benefit:.1f}±{std_benefit:.1f} (n={n_patients})"
                    else:
                        results_dict[treatment][f"Stage_{stage}"] = "N/A"
            
            # 創建DataFrame
            results_df = pd.DataFrame.from_dict(results_dict, orient='index')
            
            # 確保欄位順序正確
            column_order = [f"Stage_{s}" for s in stages]
            results_df = results_df[column_order]
            
            # 設定圖片大小
            if figsize is None:
                figsize = (14, max(8, len(all_treatments) * 0.5))
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('tight')
            ax.axis('off')
            
            # 準備表格數據
            table_data = []
            
            # 添加標題行
            header = ['Treatment', 'Overall Avg'] + [f'Stage {s}' for s in stages]
            
            # 添加數據行（按照排序後的順序）
            for treatment in all_treatments:
                row = [treatment]
                # 添加整體平均效益
                overall_avg = treatment_overall_benefit[treatment]
                row.append(f"{overall_avg:.1f}")
                # 添加各分期數據
                for stage in stages:
                    row.append(results_df.loc[treatment, f"Stage_{stage}"])
                table_data.append(row)
            
            # 創建表格
            table = ax.table(cellText=table_data, 
                            colLabels=header,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.2, 0.12] + [0.136] * len(stages))
            
            # 設置表格樣式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # 設置標題顏色
            for i in range(len(header)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 設置行顏色（交替）並根據效益大小設置漸變色
            max_benefit = max(treatment_overall_benefit.values()) if treatment_overall_benefit.values() else 1
            min_benefit = min(treatment_overall_benefit.values()) if treatment_overall_benefit.values() else 0
            
            for i in range(1, len(table_data) + 1):
                # 獲取該行治療的整體效益
                treatment = all_treatments[i-1]
                benefit = treatment_overall_benefit[treatment]
                
                # 計算顏色強度（效益越高，顏色越深）
                if max_benefit > min_benefit:
                    intensity = (benefit - min_benefit) / (max_benefit - min_benefit)
                else:
                    intensity = 0.5
                
                # 使用綠色系，效益越高越深
                base_color = (0.9 - 0.3 * intensity, 0.95, 0.9 - 0.2 * intensity)
                
                for j in range(len(header)):
                    if j == 1:  # Overall Avg 欄位
                        # 特別標記整體平均欄位
                        table[(i, j)].set_facecolor((0.85, 0.85, 0.95))
                        table[(i, j)].set_text_props(weight='bold')
                    else:
                        table[(i, j)].set_facecolor(base_color)
            
            # 設置標題
            plt.title("Treatment Effect by Stage (Sorted by Overall Average Benefit)\n"
                    "(Average Benefit in Months after Multiple Experiments)", 
                    fontsize=14, pad=20, weight='bold')
            
            # 添加註解
            plt.figtext(0.5, 0.02, 
                        "Values show: mean±std (n=number of patients)\n"
                        "Positive values indicate treatment benefit compared to no treatment\n"
                        "Treatments are sorted by overall average benefit (descending)",
                        ha='center', fontsize=10, style='italic')
            
            # 儲存圖片
            save_path = self.figures_dir / "whatif_treatment_table_sorted.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # 同時儲存為CSV（包含平均值±標準差和整體平均）
            csv_data = {}
            csv_data_mean = {}  # 單獨儲存平均值
            csv_data_std = {}   # 單獨儲存標準差

            for treatment in all_treatments:  # 使用排序後的順序
                csv_data[treatment] = {'Overall_Average': f"{treatment_overall_benefit[treatment]:.1f}"}
                csv_data_mean[treatment] = {'Overall_Average': treatment_overall_benefit[treatment]}
                csv_data_std[treatment] = {'Overall_Average': None}
                
                for stage in stages:
                    stage_patients = patient_avg[patient_avg["stage"] == stage]
                    with_treatment = stage_patients[
                        stage_patients["current_treatments"].apply(lambda x: treatment in x if x else False)
                    ]
                    
                    if len(with_treatment) > 0:
                        mean_benefit = with_treatment["treatment_benefit_months"].mean()
                        std_benefit = with_treatment["treatment_benefit_months"].std()
                        n_patients = len(with_treatment)
                        
                        # 主要CSV：平均值±標準差 (n=樣本數)
                        csv_data[treatment][f"Stage_{stage}"] = f"{mean_benefit:.1f}±{std_benefit:.1f} (n={n_patients})"
                        
                        # 額外CSV：分別儲存平均值和標準差
                        csv_data_mean[treatment][f"Stage_{stage}"] = mean_benefit
                        csv_data_std[treatment][f"Stage_{stage}"] = std_benefit
                    else:
                        csv_data[treatment][f"Stage_{stage}"] = "N/A"
                        csv_data_mean[treatment][f"Stage_{stage}"] = None
                        csv_data_std[treatment][f"Stage_{stage}"] = None

            # 儲存主要CSV（包含完整統計資訊）
            csv_df = pd.DataFrame.from_dict(csv_data, orient='index')
            csv_path = self.figures_dir / "whatif_treatment_table_sorted.csv"
            csv_df.to_csv(csv_path)

            # 儲存純數值的CSV檔案（方便後續分析）
            csv_df_mean = pd.DataFrame.from_dict(csv_data_mean, orient='index')
            csv_path_mean = self.figures_dir / "whatif_treatment_table_sorted_mean.csv"
            csv_df_mean.to_csv(csv_path_mean)

            csv_df_std = pd.DataFrame.from_dict(csv_data_std, orient='index')
            csv_path_std = self.figures_dir / "whatif_treatment_table_sorted_std.csv"
            csv_df_std.to_csv(csv_path_std)

            logger.info(f"What-if治療分析表格（已排序）已儲存至: {save_path}")
            logger.info(f"CSV檔案（完整統計）已儲存至: {csv_path}")
            logger.info(f"CSV檔案（平均值）已儲存至: {csv_path_mean}")
            logger.info(f"CSV檔案（標準差）已儲存至: {csv_path_std}")

    def plot_whatif_continuous_analysis(
        self,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        視覺化What-if連續特徵分析結果

        Args:
            figsize: 圖片大小
        """
        if self.experiment_results is None:
            logger.warning("需要 experiment_results 才能繪製What-if連續特徵分析")
            return

        logger.info("繪製What-if連續特徵分析視覺化...")

        # 收集所有連續特徵分析結果
        continuous_results = {}

        for result in self.experiment_results:
            if result and result.whatif_continuous_results:
                for feature_name, df in result.whatif_continuous_results.items():
                    if feature_name not in continuous_results:
                        continuous_results[feature_name] = []

                    df_copy = df.copy()
                    df_copy["model_type"] = result.model_type
                    continuous_results[feature_name].append(df_copy)

        if not continuous_results:
            logger.warning("沒有找到What-if連續特徵分析結果")
            return

        # 為每個特徵建立一個圖
        n_features = len(continuous_results)
        if figsize is None:
            figsize = (16, 6 * n_features)

        fig, axes = plt.subplots(n_features, 2, figsize=figsize)
        if n_features == 1:
            axes = axes.reshape(1, -1)

        for idx, (feature_name, feature_dfs) in enumerate(continuous_results.items()):
            # 合併該特徵的所有結果
            combined_feature_df = pd.concat(feature_dfs, ignore_index=True)

            # 左圖：特徵變化對生存時間的影響
            self._plot_feature_effect_curve(
                combined_feature_df, feature_name, axes[idx, 0]
            )

            # 右圖：特徵變化的患者分布
            self._plot_feature_change_distribution(
                combined_feature_df, feature_name, axes[idx, 1]
            )

        plt.suptitle("What-if Continuous Feature Analysis", fontsize=16, y=0.995)
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / "whatif_continuous_analysis.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"What-if連續特徵分析圖已儲存至: {save_path}")

    def plot_whatif_categorical_analysis(
        self,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        視覺化What-if類別特徵分析結果 - 每個特徵獨立一張表
        """
        if self.experiment_results is None:
            logger.warning("需要 experiment_results 才能繪製What-if類別特徵分析")
            return

        logger.info("生成What-if類別特徵分析表格...")

        # 收集所有類別特徵分析結果
        categorical_results = {}

        for result in self.experiment_results:
            if result and hasattr(result, 'whatif_categorical_results') and result.whatif_categorical_results:
                for analysis_name, df in result.whatif_categorical_results.items():
                    feature_name = analysis_name.replace("categorical_", "")
                    
                    if feature_name not in categorical_results:
                        categorical_results[feature_name] = []
                    
                    df_copy = df.copy()
                    df_copy["model_type"] = result.model_type
                    categorical_results[feature_name].append(df_copy)

        if not categorical_results:
            logger.warning("沒有找到What-if類別特徵分析結果")
            return

        # 為每個特徵生成獨立的表格
        for feature_name, feature_dfs in categorical_results.items():
            self._plot_single_categorical_feature(feature_name, feature_dfs, figsize)

    def _plot_single_categorical_feature(
        self,
        feature_name: str,
        feature_dfs: List[pd.DataFrame],
        figsize: Optional[tuple] = None,
    ) -> None:
        """為單個類別特徵生成分析表格"""
        # 合併所有結果
        combined_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 獲取所有轉換類型
        change_columns = [col for col in combined_df.columns if col.startswith("from_") and col.endswith("_change_months")]
        transitions = []
        
        for col in change_columns:
            # 從 "from_X_to_Y_change_months" 提取 "X→Y"
            parts = col.replace("_change_months", "").split("_")
            if len(parts) == 4:  # from_X_to_Y
                try:
                    # 確保 X 和 Y 都是有效的數值（不是 'nan' 字串）
                    from_val = parts[1]
                    to_val = parts[3]
                    # 檢查是否為有效的數值（不是 'nan' 字串）
                    if from_val.lower() != 'nan' and to_val.lower() != 'nan':
                        transition = f"{from_val}→{to_val}"
                        if transition not in transitions:
                            transitions.append(transition)
                except:
                    continue
        
        if not transitions:
            logger.warning(f"特徵 {feature_name} 沒有找到有效的轉換")
            return
        
        # 獲取所有分期
        stages = sorted(combined_df["stage"].unique())
        
        # 對同一個patient_id的多次實驗結果取平均
        # 構建用於聚合的列
        agg_dict = {}
        for transition in transitions:
            col_name = f"from_{transition.split('→')[0]}_to_{transition.split('→')[1]}_change_months"
            if col_name in combined_df.columns:
                agg_dict[col_name] = 'mean'
        
        patient_avg = combined_df.groupby(['patient_id', 'stage', 'original_value']).agg(agg_dict).reset_index()
        
        # 計算每個轉換的整體平均改良幅度（用於排序）
        transition_overall_benefit = {}
        
        for transition in transitions:
            col_name = f"from_{transition.split('→')[0]}_to_{transition.split('→')[1]}_change_months"
            if col_name in patient_avg.columns:
                all_benefits = patient_avg[col_name].dropna().values
                if len(all_benefits) > 0:
                    transition_overall_benefit[transition] = np.mean(all_benefits)
                else:
                    transition_overall_benefit[transition] = 0
            else:
                transition_overall_benefit[transition] = 0
        
        # 根據整體平均效益排序（降序）
        transitions = sorted(transitions, 
                           key=lambda x: transition_overall_benefit.get(x, 0), 
                           reverse=True)
        
        # 創建結果字典
        results_dict = {}
        
        for transition in transitions:
            results_dict[transition] = {}
            from_val, to_val = transition.split('→')
            col_name = f"from_{from_val}_to_{to_val}_change_months"
            
            for stage in stages:
                # 篩選該分期且原始值符合的病人
                try:
                    from_val_int = int(from_val)
                    stage_patients = patient_avg[
                        (patient_avg["stage"] == stage) & 
                        (patient_avg["original_value"] == from_val_int)
                    ]
                except ValueError:
                    # 如果無法轉換為整數，跳過
                    results_dict[transition][f"Stage_{stage}"] = "N/A"
                    continue
                
                if len(stage_patients) > 0 and col_name in stage_patients.columns:
                    values = stage_patients[col_name].dropna()
                    if len(values) > 0:
                        mean_benefit = values.mean()
                        std_benefit = values.std()
                        n_patients = len(values)
                        
                        # 格式化顯示
                        results_dict[transition][f"Stage_{stage}"] = f"{mean_benefit:.1f}±{std_benefit:.1f} (n={n_patients})"
                    else:
                        results_dict[transition][f"Stage_{stage}"] = "N/A"
                else:
                    results_dict[transition][f"Stage_{stage}"] = "N/A"
        
        # 創建DataFrame
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        
        # 確保欄位順序正確
        column_order = [f"Stage_{s}" for s in stages]
        results_df = results_df[column_order]
        
        # 設定圖片大小
        if figsize is None:
            figsize = (14, max(6, len(transitions) * 0.8))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # 準備表格數據
        table_data = []
        
        # 添加標題行
        if stages == ["all"]:
            header = ['Transition', 'Overall Avg', 'All Patients']
        else:
            header = ['Transition', 'Overall Avg'] + [f'Stage {s}' for s in stages]
        
        # 添加數據行
        for transition in transitions:
            row = [transition]
            # 添加整體平均效益
            overall_avg = transition_overall_benefit.get(transition, 0)
            row.append(f"{overall_avg:.1f}")
            # 添加各分期數據
            if stages == ["all"]:
                row.append(results_df.loc[transition, "Stage_all"])
            else:
                for stage in stages:
                    row.append(results_df.loc[transition, f"Stage_{stage}"])
            table_data.append(row)
        
        # 創建表格
        table = ax.table(cellText=table_data, 
                        colLabels=header,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.12] + [0.146] * len(stages))
        
        # 設置表格樣式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # 設置標題顏色
        for i in range(len(header)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 設置行顏色（根據效益大小）
        max_benefit = max(transition_overall_benefit.values()) if transition_overall_benefit.values() else 1
        min_benefit = min(transition_overall_benefit.values()) if transition_overall_benefit.values() else 0
        
        for i in range(1, len(table_data) + 1):
            transition = transitions[i-1]
            benefit = transition_overall_benefit.get(transition, 0)
            
            # 計算顏色強度
            if max_benefit > min_benefit:
                intensity = (benefit - min_benefit) / (max_benefit - min_benefit)
            else:
                intensity = 0.5
            
            # 根據效益正負使用不同顏色
            if benefit > 0:
                # 正效益：綠色系
                base_color = (0.9 - 0.3 * intensity, 0.95, 0.9 - 0.2 * intensity)
            else:
                # 負效益：紅色系
                base_color = (0.95, 0.9 - 0.3 * abs(intensity), 0.9 - 0.2 * abs(intensity))
            
            for j in range(len(header)):
                if j == 1:  # Overall Avg 欄位
                    table[(i, j)].set_facecolor((0.85, 0.85, 0.95))
                    table[(i, j)].set_text_props(weight='bold')
                else:
                    table[(i, j)].set_facecolor(base_color)
        
        # 設置標題
        feature_display_name = feature_name.replace('_', ' ').upper()
        plt.title(f"Categorical Feature Analysis: {feature_display_name}\n"
                 "(Sorted by Overall Average Benefit)", 
                 fontsize=14, pad=20, weight='bold')
        
        # 添加註解
        plt.figtext(0.5, 0.02, 
                    "Values show: mean±std (n=number of patients)\n"
                    "Positive values indicate improvement after category change\n"
                    "Transitions are sorted by overall average benefit (descending)",
                    ha='center', fontsize=10, style='italic')
        
        # 儲存圖片
        save_path = self.figures_dir / f"whatif_categorical_{feature_name}_table.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 同時儲存為CSV
        csv_data = {}
        for transition in transitions:
            csv_data[transition] = {'Overall_Average': f"{transition_overall_benefit.get(transition, 0):.1f}"}
            
            for stage in stages:
                csv_data[transition][f"Stage_{stage}"] = results_df.loc[transition, f"Stage_{stage}"]
        
        csv_df = pd.DataFrame.from_dict(csv_data, orient='index')
        csv_path = self.figures_dir / f"whatif_categorical_{feature_name}_table.csv"
        csv_df.to_csv(csv_path)
        
        logger.info(f"類別特徵 {feature_name} 分析表格已儲存至: {save_path}")
        logger.info(f"CSV檔案已儲存至: {csv_path}")

    def _plot_feature_effect_curve(
        self, df: pd.DataFrame, feature_name: str, ax: plt.Axes
    ) -> None:
        """繪製特徵變化對生存時間的影響曲線"""
        # 找出所有的變化columns
        change_cols = [col for col in df.columns if col.startswith("change_")]

        if not change_cols:
            ax.text(0.5, 0.5, "No change data available", ha="center", va="center")
            return

        # 計算每個變化水平的平均影響
        changes = []
        effects = []
        stds = []

        for col in change_cols:
            if "_" in col:
                delta_str = col.split("_", 1)[1]
                if "plus" in delta_str:
                    delta = float(delta_str.replace("plus_", ""))
                else:
                    delta = -float(delta_str.replace("minus_", ""))

                effect_mean = df[col].mean()
                effect_std = df[col].std()

                changes.append(delta)
                effects.append(effect_mean)
                stds.append(effect_std)

        # 排序
        sorted_idx = np.argsort(changes)
        changes = np.array(changes)[sorted_idx]
        effects = np.array(effects)[sorted_idx]
        stds = np.array(stds)[sorted_idx]

        # 繪製曲線
        ax.errorbar(changes, effects, yerr=stds, fmt="o-", capsize=5, markersize=8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel(f"{feature_name} Change")
        ax.set_ylabel("Survival Time Change (months)")
        ax.set_title(f"Effect of {feature_name} Changes on Survival")
        ax.grid(True, alpha=0.3)

    def _plot_feature_change_distribution(
        self, df: pd.DataFrame, feature_name: str, ax: plt.Axes
    ) -> None:
        """繪製特徵變化的效果分布"""
        # 收集所有變化效果
        change_cols = [col for col in df.columns if col.startswith("change_")]

        all_changes = []
        labels = []

        for col in change_cols:
            if df[col].notna().any():
                all_changes.append(df[col].dropna().values)
                delta_str = col.split("_", 1)[1]
                labels.append(delta_str.replace("_", " "))

        if not all_changes:
            ax.text(0.5, 0.5, "No change data available", ha="center", va="center")
            return

        # 繪製箱型圖
        bp = ax.boxplot(all_changes, labels=labels, patch_artist=True)

        # 設定顏色
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(bp["boxes"])))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"{feature_name} Modification")
        ax.set_ylabel("Survival Time Change (months)")
        ax.set_title(f"Distribution of Effects for {feature_name} Changes")
        ax.grid(True, axis="y", alpha=0.3)

    def plot_temporal_prediction_analysis(
        self,
        time_bins: Optional[int] = 10,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        時間序列分析：預測誤差隨時間的變化

        Args:
            time_bins: 時間分組數量
            figsize: 圖片大小
        """
        if self.processed_df is None:
            logger.warning("需要 processed_df 才能繪製時間序列預測分析")
            return

        logger.info("繪製時間序列預測分析...")

        # 讀取ensemble預測結果
        ensemble_dir = self.path_config.ensemble_predictions_dir
        ensemble_files = list(ensemble_dir.glob("*_ensemble_predictions.csv"))

        if not ensemble_files:
            logger.warning("找不到ensemble預測結果")
            return

        # 設定圖片大小
        if figsize is None:
            figsize = (16, 10)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        # 為每個找到的模型創建分析
        for file_idx, ensemble_file in enumerate(ensemble_files[:4]):  # 最多顯示4個
            if file_idx >= 4:
                break

            ax = axes[file_idx]

            # 讀取預測結果
            predictions = pd.read_csv(ensemble_file)
            model_name = ensemble_file.stem.replace("_ensemble_predictions", "")

            # 合併預測與真實資料
            merged = predictions.merge(
                self.processed_df[["patient_id", "time", "event"]], on="patient_id"
            )

            # 只分析非刪失數據
            non_censored = merged[merged["event"] == 1].copy()

            if non_censored.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No non-censored data for {model_name}",
                    ha="center",
                    va="center",
                )
                continue

            # 計算預測誤差
            non_censored["prediction_error"] = (
                non_censored["ensemble_prediction"] - non_censored["time"]
            )

            # 將時間分組
            non_censored["time_bin"] = pd.qcut(
                non_censored["time"], q=time_bins, duplicates="drop"
            )

            # 計算每個時間段的誤差統計
            time_stats = (
                non_censored.groupby("time_bin", observed=False)["prediction_error"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            # 取得時間段的中點
            time_stats["time_mid"] = time_stats["time_bin"].apply(
                lambda x: x.mid if hasattr(x, "mid") else 0
            )

            # 繪製誤差隨時間的變化
            ax.errorbar(
                time_stats["time_mid"],
                time_stats["mean"],
                yerr=time_stats["std"],
                fmt="o-",
                capsize=5,
                markersize=6,
                label="Mean ± Std",
            )

            # 加上零線
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            # 加上樣本數資訊
            for _, row in time_stats.iterrows():
                ax.annotate(
                    f"n={row['count']}",
                    (row["time_mid"], row["mean"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    alpha=0.7,
                )

            ax.set_xlabel("Actual Survival Time (months)")
            ax.set_ylabel("Prediction Error (months)")
            ax.set_title(f"{model_name} - Prediction Error Over Time")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle("Temporal Analysis of Prediction Accuracy", fontsize=16, y=0.995)
        plt.tight_layout()

        # 儲存
        save_path = self.figures_dir / "temporal_prediction_analysis.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"時間序列預測分析圖已儲存至: {save_path}")

    def plot_prediction_accuracy_by_time(
            self,
            time_points: Optional[List[int]] = None,
            figsize: Optional[tuple] = None,
        ) -> None:
            """
            繪製不同時間點的預測準確度指標

            Args:
                time_points: 要評估的時間點列表（月）
                figsize: 圖片大小
            """
            if self.processed_df is None:
                logger.warning("需要 processed_df 才能繪製時間點預測準確度")
                return

            logger.info("繪製不同時間點的預測準確度指標...")

            if time_points is None:
                time_points = [6, 12, 24, 36, 60]  # 預設時間點

            # 讀取ensemble預測結果
            ensemble_dir = self.path_config.ensemble_predictions_dir
            ensemble_files = list(ensemble_dir.glob("*_ensemble_predictions.csv"))

            if not ensemble_files:
                logger.warning("找不到ensemble預測結果")
                return

            # 收集所有模型的準確度資料
            accuracy_data = []

            for ensemble_file in ensemble_files:
                predictions = pd.read_csv(ensemble_file)
                model_name = ensemble_file.stem.replace("_ensemble_predictions", "")

                # 合併預測與真實資料
                merged = predictions.merge(
                    self.processed_df[["patient_id", "time", "event"]], on="patient_id"
                )

                # 計算每個時間點的準確度
                for t in time_points:
                    # 實際生存超過t的患者
                    actual_survived = merged["time"] >= t
                    # 預測生存超過t的患者
                    predicted_survived = merged["ensemble_prediction"] >= t

                    # 計算混淆矩陣元素
                    tp = ((actual_survived) & (predicted_survived)).sum()
                    tn = ((~actual_survived) & (~predicted_survived)).sum()
                    fp = ((~actual_survived) & (predicted_survived)).sum()
                    fn = ((actual_survived) & (~predicted_survived)).sum()

                    # 計算各項指標
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    
                    # 計算MCC
                    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    if denominator > 0:
                        mcc = (tp * tn - fp * fn) / denominator
                    else:
                        mcc = 0

                    accuracy_data.append(
                        {
                            "model": model_name,
                            "time_point": t,
                            "mcc": mcc,
                            "ppv": ppv,
                            "npv": npv,
                            "sensitivity": sensitivity,
                            "specificity": specificity,
                            "n_patients": len(merged),
                        }
                    )

            if not accuracy_data:
                logger.warning("無法計算準確度資料")
                return

            # 轉換為DataFrame
            acc_df = pd.DataFrame(accuracy_data)

            # 設定圖片大小
            if figsize is None:
                figsize = (20, 8)  # 增加寬度以容納5個子圖

            # 創建2x3的子圖布局（最後一個位置留空）
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.flatten()

            # 指標列表和對應的軸
            metrics = ["mcc", "ppv", "npv", "sensitivity", "specificity"]
            metric_names = ["MCC", "PPV", "NPV", "Sensitivity", "Specificity"]
            
            # 為每個指標繪製圖表
            for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[idx]
                
                # 為每個模型繪製曲線
                for model in acc_df["model"].unique():
                    model_data = acc_df[acc_df["model"] == model]
                    
                    ax.plot(
                        model_data["time_point"],
                        model_data[metric],
                        marker="o",
                        label=model,
                        linewidth=2,
                        markersize=8
                    )
                
                # 設定圖表屬性
                ax.set_xlabel("Time Point (months)")
                ax.set_ylabel(metric_name)
                ax.set_title(f"{metric_name} at Different Time Points")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # 根據指標類型設定y軸範圍
                if metric == "mcc":
                    ax.set_ylim(-0.1, 1.05)  # MCC可能是負值
                else:
                    ax.set_ylim(0, 1.05)
                    
                ax.set_xticks(time_points)

            # 隱藏最後一個空的子圖
            axes[-1].set_visible(False)

            # 設定總標題
            plt.suptitle("Prediction Performance Metrics at Specific Time Points", fontsize=16)
            plt.tight_layout()

            # 儲存
            save_path = self.figures_dir / "prediction_metrics_by_time.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"時間點預測準確度指標圖已儲存至: {save_path}")
            
            # 額外儲存數據表格以供參考
            summary_path = self.figures_dir / "prediction_metrics_summary.csv"
            acc_df.to_csv(summary_path, index=False)
            logger.info(f"預測指標摘要已儲存至: {summary_path}")
