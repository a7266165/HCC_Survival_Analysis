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
    _find_result_directory,
    ensemble_predictions_by_seed,
    ensemble_feature_importance,
    calculate_ensemble_metrics,
    analyze_survival_predictions,
    compare_calibration_methods,
    create_stage_treatment_shap_table,
    create_stage_treatment_shap_summary,
    analyze_treatment_modifications,
    analyze_bmi_modifications,
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

            # 對實驗結果進行校正
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

    # # ========================================
    # # Step 5: 進階臨床分析
    # # ========================================
    # logger.info("開始進階臨床分析...")
    
    # # 5.1 創建Stage x Treatment統計表（含SHAP值）
    # logger.info("創建Stage-Treatment統計表...")
    # stage_treatment_pivot = create_stage_treatment_shap_table(
    #     ensemble_importance,
    #     processed_df,
    #     stage_col="BCLC_stage",
    #     treatment_cols=None,  # 使用預設治療欄位
    # )
    
    # if not stage_treatment_pivot.empty:
    #     # 儲存結果
    #     result_dir = _find_result_directory(experiment_config, ts)
    #     save_path = result_dir / "stage_treatment_pivot_table.csv"
        
    #     # 創建詳細摘要
    #     create_stage_treatment_shap_summary(stage_treatment_pivot, save_path)
        
    #     logger.info(f"Stage-Treatment Pivot Table已儲存至: {save_path}")
    #     logger.info(f"分析了 {len(stage_treatment_pivot.index)} 個期別")
    #     logger.info(f"包含 {len(stage_treatment_pivot.columns)} 種治療方式")
    
    # # 5.2 分析治療方式調整的影響
    # logger.info("分析治療方式調整影響...")
    # treatment_modification_results = analyze_treatment_modifications(
    #     total_experiments_result,
    #     processed_df,
    #     stage_col="BCLC_stage",
    #     treatment_cols=None
    # )
    
    # if treatment_modification_results:
    #     # 儲存結果
    #     result_dir = _find_result_directory(experiment_config, ts)
    #     treatment_mod_dir = result_dir / "treatment_modifications"
    #     treatment_mod_dir.mkdir(exist_ok=True)
        
    #     # 儲存詳細結果和摘要
    #     for key, data in treatment_modification_results.items():
    #         if "summary" in key:
    #             # 儲存摘要統計
    #             save_path = treatment_mod_dir / f"{key}.csv"
    #             data.to_csv(save_path)
    #             logger.info(f"治療調整摘要已儲存: {key}")
                
    #             # 顯示最有影響的治療
    #             if not data.empty:
    #                 top_treatments = data.sort_values(
    #                     ('prediction_change', 'mean'), 
    #                     ascending=False
    #                 ).head(3)
    #                 stage = key.replace("stage_", "").replace("_summary", "")
    #                 logger.info(f"Stage {stage} - Top 3 最有效治療調整:")
    #                 for treatment, row in top_treatments.iterrows():
    #                     mean_change = row[('prediction_change', 'mean')]
    #                     logger.info(f"  {treatment}: 平均延長 {mean_change:.1f} 個月")
    #         elif "detailed" in key:
    #             # 儲存詳細資料
    #             save_path = treatment_mod_dir / f"{key}.csv"
    #             data.to_csv(save_path, index=False)
    
    # # 5.3 分析BMI調整的影響
    # logger.info("分析BMI調整影響...")
    # bmi_modification_results = analyze_bmi_modifications(
    #     total_experiments_result,
    #     processed_df,
    #     bmi_col="BMI"
    # )
    
    # if "results_df" in bmi_modification_results:
    #     # 儲存BMI分析結果
    #     result_dir = _find_result_directory(experiment_config, ts)
    #     bmi_results_df = bmi_modification_results["results_df"]
    #     bmi_results_df.to_csv(result_dir / "bmi_modification_analysis.csv", index=False)
        
    #     # 儲存圖片位置
    #     if "plot_saved" in bmi_modification_results:
    #         plot_source = Path(bmi_modification_results["plot_saved"])
    #         plot_dest = result_dir / plot_source.name
    #         if plot_source.exists():
    #             import shutil
    #             shutil.move(str(plot_source), str(plot_dest))
    #             logger.info(f"BMI分析圖表已儲存至: {plot_dest}")
        
    #     # 顯示摘要
    #     summary = bmi_modification_results.get("summary", {})
    #     logger.info("BMI調整分析摘要:")
    #     logger.info(f"  分析患者數: {summary.get('total_patients_analyzed', 0)}")
    #     logger.info(f"  BMI+1平均影響: {summary.get('avg_survival_change_plus1', 0):.2f} months")
    #     logger.info(f"  BMI-1平均影響: {summary.get('avg_survival_change_minus1', 0):.2f} months")
    
    # ========================================
    # Step 6: 生成最終報告
    # ========================================
    # logger.info("生成最終報告...")
    
    # # 建立報告摘要
    # report_summary = {
    #     'experiment_date': ts,
    #     'total_experiments': len(total_experiments_result),
    #     'models_tested': list(experiment_config.models_to_train),
    #     'calibration_methods': calibration_methods,
    #     'clinical_analyses': {
    #         'stage_treatment_analysis': stage_treatment_pivot.shape[0] if not stage_treatment_pivot.empty else 0,
    #         'treatment_modifications_analyzed': len(treatment_modification_results),
    #         'bmi_analysis_completed': 'results_df' in bmi_modification_results
    #     }
    # }
    
    # # 儲存摘要報告
    # result_dir = _find_result_directory(experiment_config, ts)
    # with open(result_dir / "experiment_summary.json", 'w', encoding='utf-8') as f:
    #     import json
    #     json.dump(report_summary, f, indent=2, ensure_ascii=False)
    
    # logger.info("實驗完成！所有結果已儲存至: %s", result_dir)
    
    # return {
    #     'experiment_results': total_experiments_result,
    #     'ensemble_predictions': ensemble_preds,
    #     'ensemble_metrics': ensemble_metrics,
    #     'survival_analysis': survival_analysis,
    #     'calibration_comparison': calibration_comparison,
    #     'stage_treatment_pivot': stage_treatment_pivot,
    #     'treatment_modifications': treatment_modification_results,
    #     'bmi_modifications': bmi_modification_results,
    #     'summary': report_summary
    # }


if __name__ == "__main__":
    main()
