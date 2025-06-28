import logging
from pathlib import Path
import pandas as pd
from utils.config_utils import load_config
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import (
    single_experimentor,
    save_experiment_results,
    apply_calibration_to_experiment,
    apply_whatif_analysis,
)
from analyzing.ensemble_analyzer import (
    ensemble_predictions_by_seed,
    ensemble_feature_importance,
    calculate_ensemble_metrics,
    analyze_survival_predictions,
    compare_calibration_methods,
)


logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)


def setup_logging():
    fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def main():

    # ========================================
    # step 0: 啟動logger(不用動)
    # ========================================
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("程式啟動，開始讀取設定檔")

    # ========================================
    # Step 1: 讀取csv檔(不用動)
    # ========================================
    path_config = load_config("path_config")
    df = pd.read_csv(path_config.raw_dataset_path)
    logger.info("成功讀取原始資料，共 %d 筆", len(df))

    # ========================================
    # Step 2: 資料處理(不用動)
    # ========================================
    feature_config = load_config("feature_config")
    preprocess_config = load_config("preprocess_config")
    processed_df = data_preprocessor(df, feature_config, preprocess_config)
    logger.info("資料處理完成，共 %d 筆", len(processed_df))

    # ========================================
    # Step 3: 進行實驗
    # ========================================
    experiment_config = load_config("experiment_config")

    total_experiments_result = []

    for model_type in experiment_config.experiment_settings.models_to_train:
        for random_seed in range(experiment_config.experiment_settings.num_experiments):
            logger.info(f"開始模型 {model_type} 隨機種子 {random_seed} 的實驗")
            single_experiment_result = single_experimentor(
                processed_df,
                preprocess_config,
                feature_config,
                random_seed,
                model_type,
                experiment_config.model_settings,
            )

            logger.info(f"開始對實驗結果進行校正")
            if single_experiment_result is not None:
                apply_calibration_to_experiment(
                    single_experiment_result,
                    processed_df,
                    list(experiment_config.experiment_settings.calibration_methods),
                )

                logger.info(f"開始 What-if 分析")
                apply_whatif_analysis(
                    single_experiment_result,
                    processed_df,
                    experiment_config.whatif_settings,
                )

                logger.info(
                    f"模型 {model_type} seed {random_seed} 實驗完成，"
                    f"包含 {len(single_experiment_result.calibrated_test_predictions)} 種校正方法"
                )

            total_experiments_result.append(single_experiment_result)
    logger.info("所有實驗完成，共 %d 筆資料", len(total_experiments_result))
    saved_files = save_experiment_results(total_experiments_result, path_config)

    # ========================================
    # Step 4: 進階Ensemble分析
    # ========================================
    logger.info("開始Ensemble分析...")
    # 4.1 Ensemble預測（包含校正結果）
    ensemble_preds = ensemble_predictions_by_seed(
        total_experiments_result,
        path_config,
        include_calibrated=True,  # 包含校正結果
    )

    # 4.2 特徵重要性分析（不受校正影響）
    ensemble_importance = ensemble_feature_importance(
        total_experiments_result,
        path_config,
    )

    # 4.3 計算Ensemble指標（包含校正結果）
    ensemble_metrics = calculate_ensemble_metrics(
        ensemble_preds,
        processed_df,
        total_experiments_result,
        path_config,
    )

    # 4.4 生存數據分析（包含校正結果）
    survival_analysis = analyze_survival_predictions(
        ensemble_preds, processed_df, path_config
    )

    # 4.5 比較不同校正方法的效果
    calibration_comparison = compare_calibration_methods(
        survival_analysis,
        path_config,
    )
    if not calibration_comparison.empty:
        logger.info("完成校正方法比較分析")


if __name__ == "__main__":
    main()
