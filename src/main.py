import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from utils.config_utils import load_config, FeatureConfig
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import (
    single_experimentor,
    save_experiment_results,
    apply_calibration_to_experiment,
)
from analyzing.ensemble_analyzer import (
    ensemble_predictions_by_seed,
    ensemble_feature_importance,
    calculate_ensemble_metrics,
    analyze_survival_predictions,
    compare_calibration_methods,
)


ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)


def setup_logging():
    fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def main():

    # step 0: 啟動logger
    setup_logging()
    logger = logging.getLogger(__name__)  # 創建logger實例
    logger.info("程式啟動，開始讀取設定檔")

    # Step 1: 讀取csv檔
    """
    輸入:
    - csv檔路徑
    輸出:
    - DataFrame: 包含csv檔內容的DataFrame
    """
    dataset_config = load_config("dataset_config")
    df = pd.read_csv(dataset_config.raw_dataset_path)
    logger.info("成功讀取原始資料，共 %d 筆", len(df))

    # Step 2: 資料處理
    """
    輸入:
    - DataFrame: 包含csv檔內容的DataFrame
    
    輸出:
    - DataFrame: 處理後的DataFrame
    """
    feature_config: FeatureConfig = load_config("feature_config")
    preprocess_config = load_config("preprocess_config")
    processed_df = data_preprocessor(df, feature_config, preprocess_config)
    logger.info("資料處理完成，共 %d 筆", len(processed_df))

    # Step 3: 進行實驗
    """
    輸入:
    - DataFrame: 處理後的DataFrame
    
    輸出:?
    
    實驗內容:
    將處理後的DataFrame分成訓練集和測試集，使用訓練集訓練模型，並在測試集上進行評估。
    """
    experiment_config = load_config("experiment_config")
    survival_model_config = load_config("survival_model_config")

    calibration_methods = ["knn_km", "regression", "segmental", "curve"]
    total_experiments_result = []

    for model_type in experiment_config.models_to_train:
        logger.info(f"開始實驗模型: {model_type}")

        for random_seed in range(experiment_config.num_experiments):
            # 執行單次實驗
            single_experiment_result = single_experimentor(
                processed_df,
                preprocess_config.is_preprocess,
                feature_config,
                random_seed,
                model_type,
                survival_model_config,
            )

            # 立即對實驗結果進行校正
            if single_experiment_result is not None:
                apply_calibration_to_experiment(
                    single_experiment_result,
                    processed_df,
                    calibration_methods,
                    random_seed,
                )

                logger.info(
                    f"模型 {model_type} seed {random_seed} 實驗完成，"
                    f"包含 {len(single_experiment_result.calibrated_test_predictions)} 種校正方法"
                )

            total_experiments_result.append(single_experiment_result)
            # 對實驗校正
            # apply_all_calibraters(single_experiment_result, processed_df, experiment_config, random_seed)
    logger.info("所有實驗完成，共 %d 筆資料", len(total_experiments_result))
    saved_files = save_experiment_results(
        total_experiments_result, experiment_config, ts
    )

    # ========================================
    # Step 4: 進階Ensemble分析
    # ========================================
    logger.info("開始Ensemble分析...")
    # 4.1 Ensemble預測（包含校正結果）
    ensemble_preds = ensemble_predictions_by_seed(
        total_experiments_result,
        experiment_config,
        ts,
        include_calibrated=True,  # 包含校正結果
    )

    # 4.2 特徵重要性分析（不受校正影響）
    ensemble_importance = ensemble_feature_importance(
        total_experiments_result, experiment_config, ts
    )

    # 4.3 計算Ensemble指標（包含校正結果）
    ensemble_metrics = calculate_ensemble_metrics(
        ensemble_preds,
        processed_df,
        total_experiments_result,
        experiment_config,
        ts,
        include_calibrated=True,
    )

    # 4.4 生存數據分析（包含校正結果）
    survival_analysis = analyze_survival_predictions(
        ensemble_preds, processed_df, experiment_config, ts
    )

    # 4.5 比較不同校正方法的效果
    calibration_comparison = compare_calibration_methods(
        survival_analysis, experiment_config, ts
    )
    if not calibration_comparison.empty:
        logger.info("完成校正方法比較分析")


if __name__ == "__main__":
    main()
