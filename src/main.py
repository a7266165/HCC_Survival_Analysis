import logging
from pathlib import Path
import pandas as pd
from utils.config_utils import load_config
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import (
    run_single_experiment,
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

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import multiprocessing

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
    # Step 3: 進行實驗（可選擇使用並行處理）
    # ========================================
    experiment_config = load_config("experiment_config")

    # 可以透過環境變數或配置檔控制是否使用並行處理
    use_parallel = True  # 可以改成從配置檔讀取

    if (
        use_parallel
        and len(experiment_config.experiment_settings.models_to_train)
        * experiment_config.experiment_settings.num_experiments
        > 1
    ):
        # 使用並行處理
        logger.info("使用並行處理模式")

        # 決定使用的 worker 數量
        cpu_count = multiprocessing.cpu_count()
        total_experiments = (
            len(experiment_config.experiment_settings.models_to_train)
            * experiment_config.experiment_settings.num_experiments
        )
        # 使用較少的核心數：CPU核心數-1、總實驗數、或4，取最小值
        max_workers = min(max(cpu_count - 1, 1), total_experiments, 4)
        logger.info(
            f"使用 {max_workers} 個 CPU 核心進行並行處理（系統共有 {cpu_count} 個核心）"
        )

        # 準備所有實驗的參數，並記錄原始順序
        experiment_args = []
        experiment_index_map = {}
        index = 0

        for model_type in experiment_config.experiment_settings.models_to_train:
            for random_seed in range(
                experiment_config.experiment_settings.num_experiments
            ):
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
                experiment_index_map[(model_type, random_seed)] = index
                index += 1

        # 使用並行處理執行實驗
        results_dict = {}  # 用於保持結果順序
        completed = 0
        total = len(experiment_args)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_info = {
                executor.submit(run_single_experiment, args): (
                    args,
                    experiment_index_map[(args[4], args[3])],
                )
                for args in experiment_args
            }

            # 處理完成的任務
            for future in as_completed(future_to_info):
                args, original_index = future_to_info[future]
                model_type = args[4]
                random_seed = args[3]

                try:
                    result = future.result()
                    results_dict[original_index] = result
                    completed += 1
                    logger.info(
                        f"進度: {completed}/{total} ({completed/total*100:.1f}%) - "
                        f"完成 {model_type} seed {random_seed}"
                    )
                except Exception as exc:
                    logger.error(
                        f"模型 {model_type} seed {random_seed} 執行失敗: {exc}"
                    )
                    results_dict[original_index] = None
                    completed += 1

        # 按原始順序重建結果列表
        total_experiments_result = [
            results_dict[i] for i in range(len(experiment_args))
        ]

    else:
        # 使用原本的順序執行
        logger.info("使用順序處理模式")
        total_experiments_result = []

        for model_type in experiment_config.experiment_settings.models_to_train:
            for random_seed in range(
                experiment_config.experiment_settings.num_experiments
            ):
                logger.info(f"開始模型 {model_type} 隨機種子 {random_seed} 的實驗")
                single_experiment_result = single_experimentor(
                    processed_df,
                    preprocess_config,
                    feature_config,
                    random_seed,
                    model_type,
                    experiment_config.model_settings,
                )

                if single_experiment_result is not None:
                    logger.info(f"開始對實驗結果進行校正")
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

    # ========================================
    # Step 5: 視覺化分析結果
    # ========================================
    logger.info("開始生成視覺化圖表...")
    from analyzing.visualizer import SurvivalVisualizer

    # 建立視覺化工具（暫時手動設定路徑）
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
    multiprocessing.freeze_support()
    main()
