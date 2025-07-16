import logging
import pandas as pd
from utils.config_utils import load_config
from utils.multiprocess_utils import run_parallel_tasks
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import (
    run_single_experiment,
    save_experiment_results,
)
from analyzing.ensemble_analyzer import EnsembleAnalyzer
from analyzing.visualizer import SurvivalVisualizer

logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)


def main():

    # ========================================
    # step 0: 啟動logger
    # ========================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
                experiment_config,
            )
            experiment_args.append(args)

    # 使用多進程工具執行實驗
    logger.info(f"準備執行 {len(experiment_args)} 個實驗")
    multiprocess_config = load_config("multiprocess_config")
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
    ensemble_analyzer = EnsembleAnalyzer(path_config)

    analysis_results = ensemble_analyzer.run_complete_analysis(
        experiment_results=total_experiments_result,
        processed_df=processed_df,
        include_calibrated=True,
    )

    summary = ensemble_analyzer.get_analysis_summary()
    logger.info(f"分析摘要: {summary}")

    # ========================================
    # Step 5: 視覺化分析結果
    # ========================================
    logger.info("開始生成視覺化圖表...")

    # 建立視覺化工具
    visualizer = SurvivalVisualizer(path_config, processed_df, total_experiments_result)

    # 生成特徵重要性熱圖
    visualizer.plot_feature_importance_heatmap(top_n=20)

    # 生成特徵重要性柱狀圖
    visualizer.plot_feature_importance_bars(top_n=15)

    # 生成 K/U 群組分析圖
    visualizer.plot_ku_group_metrics()

    # 校正前後的散點圖
    visualizer.plot_calibration_scatter()

    # 存活曲線
    visualizer.plot_survival_curves(risk_groups=3)  # 分成低、中、高風險三組

    # What-if 治療分析結果視覺化
    visualizer.plot_whatif_treatment_analysis()

    # What-if 連續特徵分析視覺化
    visualizer.plot_whatif_continuous_analysis()

    # What-if 類別特徵分析視覺化
    visualizer.plot_whatif_categorical_analysis()

    # 時間序列預測分析
    visualizer.plot_temporal_prediction_analysis(time_bins=10)  # 將時間分成10個區間

    # 不同時間點的預測準確度
    visualizer.plot_prediction_accuracy_by_time(
        time_points=[6, 12, 24, 36, 60]  # 6個月、1年、2年、3年、5年
    )
    logger.info("實驗完成！")


if __name__ == "__main__":
    import multiprocessing  # 避免 Windows 系統多進程問題

    multiprocessing.freeze_support()
    main()
