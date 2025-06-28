# src/analyzing/visualizer.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path
from utils.config_utils import PathConfig

logger = logging.getLogger(__name__)


class SurvivalVisualizer:
    """存活分析視覺化工具"""

    def __init__(self, path_config: PathConfig):
        """
        初始化視覺化工具

        Args:
            path_config: 路徑配置物件
        """
        self.path_config = path_config
        self.figures_dir = path_config.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 設定視覺化風格
        self._setup_plot_style()

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

    def plot_calibration_scatter(
        self, processed_df: pd.DataFrame, figsize: Optional[tuple] = None
    ) -> None:
        """
        繪製校正前後的散點圖，展示預測誤差分布（包含訓練集和測試集）

        Args:
            processed_df: 包含真實生存時間的原始資料
            figsize: 圖片大小
        """
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
        true_data = processed_df[["patient_id", "time", "event"]]
        all_predictions = all_predictions.merge(true_data, on="patient_id", how="left")

        # 取得模型列表（排除 CoxPHFitter，因為沒有校正）
        models = [
            m for m in all_predictions["model_type"].unique() if m != "CoxPHFitter"
        ]

        # 校正方法
        calibration_methods = ["knn_km", "regression", "segmental", "curve"]

        # 設定圖片大小 - 現在需要顯示訓練集和測試集
        if figsize is None:
            figsize = (16, 8 * len(models))  # 每個模型需要兩行

        fig, axes = plt.subplots(len(models) * 2, 4, figsize=figsize)
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
