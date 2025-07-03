import logging
import pandas as pd
from utils.config_utils import load_config
from utils.multiprocess_utils import MultiprocessConfig, run_parallel_tasks
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import (
    run_single_experiment,
    save_experiment_results,
)
from analyzing.ensemble_analyzer import (
    ensemble_predictions_by_seed,
    ensemble_feature_importance,
    calculate_ensemble_metrics,
    analyze_survival_predictions,
    compare_calibration_methods,
)
from analyzing.visualizer import SurvivalVisualizer

logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)

def main():

    # ========================================
    # step 0: 啟動logger
    # ========================================
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.info("程式啟動，開始讀取設定檔")

    # ========================================
    # Step 1: 讀取csv檔
    # ========================================
    path_config = load_config("path_config")
    df = pd.read_csv(path_config.raw_dataset_path)
    logger.info("成功讀取原始資料，共 %d 筆", len(df))

    # ========================================
    # Step 2: 資料處理
    # ========================================
    feature_config = load_config("feature_config")
    preprocess_config = load_config("preprocess_config")
    processed_df = data_preprocessor(df, feature_config, preprocess_config)
    logger.info("資料處理完成，共 %d 筆", len(processed_df))

    # ========================================
    # Step 3: 進行實驗
    # ========================================
    experiment_config = load_config("experiment_config")
    multiprocess_config = MultiprocessConfig()

    # 準備所有實驗的參數
    experiment_args = []

    for model_type in experiment_config.experiment_settings.models_to_train:
        for random_seed in range(experiment_config.experiment_settings.num_experiments):
            args = (
                processed_df,
                preprocess_config,
                feature_config,
                random_seed,
                model_type,
                experiment_config.model_settings,
                experiment_config.whatif_settings,
                list(experiment_config.experiment_settings.calibration_methods),
            )
            experiment_args.append(args)

    # 使用多進程工具執行實驗
    logger.info(f"準備執行 {len(experiment_args)} 個實驗")

    # 執行所有實驗
    total_experiments_result = run_parallel_tasks(
        task_function=run_single_experiment,
        task_args_list=experiment_args,
        config=multiprocess_config,
    )

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

    # ========================================
    # Step 5: 視覺化分析結果
    # ========================================
    logger.info("開始生成視覺化圖表...")

    # 建立視覺化工具
    visualizer = SurvivalVisualizer(path_config)

    # 生成特徵重要性熱圖
    visualizer.plot_feature_importance_heatmap(top_n=20)

    # 生成特徵重要性柱狀圖
    visualizer.plot_feature_importance_bars(top_n=15)

    # 生成 K/U 群組分析圖
    visualizer.plot_ku_group_metrics()

    # 生成校正前後散點圖
    visualizer.plot_calibration_scatter(processed_df)

    logger.info("視覺化完成！")


if __name__ == "__main__":
    import multiprocessing  # 避免 Windows 系統多進程問題

    multiprocessing.freeze_support()
    main()
