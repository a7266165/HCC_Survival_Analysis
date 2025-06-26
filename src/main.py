import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from utils.config_utils import load_config, FeatureConfig
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import single_experimentor, save_experiment_results
from analyzing.ensemble_analyzer import (
    ensemble_predictions_by_seed, 
    ensemble_feature_importance, 
    calculate_ensemble_metrics,
    analyze_survival_predictions,
    apply_all_calibrations,
    create_stage_treatment_shap_table,
    analyze_treatment_modifications,
    analyze_bmi_modifications
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

    total_experiments_result = []

    for model_type in experiment_config.models_to_train:
        logger.info(f"開始實驗模型: {model_type}")
        for random_seed in range(experiment_config.num_experiments):
            # 實驗
            single_experiment_result = single_experimentor(
                processed_df,
                preprocess_config.is_preprocess,
                feature_config,
                random_seed,
                model_type,
                survival_model_config,
            )
            total_experiments_result.append(single_experiment_result)
            # 對實驗校正
            # apply_all_calibraters(single_experiment_result, processed_df, experiment_config, random_seed)
    logger.info("所有實驗完成，共 %d 筆資料", len(total_experiments_result))
    saved_files = save_experiment_results(
        total_experiments_result, 
        experiment_config,
        ts
    )

    # ========================================
    # Step 4: 進階Ensemble分析
    # ========================================
    logger.info("開始Ensemble分析...")
    ensemble_preds = ensemble_predictions_by_seed(total_experiments_result, experiment_config, ts)
    ensemble_importance = ensemble_feature_importance(total_experiments_result, experiment_config, ts)
    ensemble_metrics = calculate_ensemble_metrics(
        ensemble_preds, 
        processed_df,
        total_experiments_result,
        experiment_config, ts
    )
    # 4.2 生存數據分析
    survival_analysis = analyze_survival_predictions(ensemble_preds, processed_df, experiment_config, ts)

    # 4.3 校準分析
    calibration_results = apply_all_calibrations(total_experiments_result,processed_df, experiment_config, ts)
        
    # # 4.4 Stage-Treatment SHAP統計表 (修正版)
    # logger.info("生成Stage-Treatment統計表...")

    # shap_analysis = create_stage_treatment_shap_table(
    #     ensemble_importance, 
    #     processed_df,
    #     stage_col='BCLC_stage',
    #     treatment_cols=feature_config.treatments
    # )


    # # 4.5 治療方式調整分析 (修正版 - 使用已訓練模型)
    # logger.info("執行治療方式調整分析...")
    # try:
    #     treatment_results = analyze_treatment_modifications(
    #         total_experiments_result,
    #         processed_df,
    #         stage_col='BCLC_stage'
    #     )
        
    #     if treatment_results:
    #         logger.info(f"治療調整分析完成: {len(treatment_results)} 個stage")
            
    #         # 顯示每個stage的結果摘要
    #         for stage, stage_df in treatment_results.items():
    #             logger.info(f"\nStage {stage} 治療調整分析:")
    #             logger.info(f"  分析患者數: {len(stage_df['patient_id'].unique())}")
                
    #             # 找出最有效的治療調整
    #             if not stage_df.empty:
    #                 # 正向變化Top 5
    #                 top_positive = stage_df.nlargest(5, 'prediction_change')
    #                 logger.info("  預測改善最大的治療調整 (Top 5):")
    #                 for _, row in top_positive.iterrows():
    #                     logger.info(f"    患者{row['patient_id']}: {row['new_treatment']} "
    #                             f"(+{row['prediction_change']:.2f} months)")
                    
    #                 # 負向變化Top 5
    #                 top_negative = stage_df.nsmallest(5, 'prediction_change')
    #                 logger.info("  預測下降最大的治療調整 (Top 5):")
    #                 for _, row in top_negative.iterrows():
    #                     logger.info(f"    患者{row['patient_id']}: {row['new_treatment']} "
    #                             f"({row['prediction_change']:.2f} months)")
                    
    #                 # 按治療方式統計平均影響
    #                 treatment_summary = stage_df.groupby('new_treatment').agg({
    #                     'prediction_change': ['mean', 'std', 'count']
    #                 }).round(2)
                    
    #                 logger.info(f"\n  各治療方式的平均影響:")
    #                 logger.info(treatment_summary)
                    
    #                 # 儲存結果
    #                 stage_df.to_excel(f'treatment_modifications_stage_{stage}.xlsx', index=False)
    #                 logger.info(f"  結果已儲存為 treatment_modifications_stage_{stage}.xlsx")
    #     else:
    #         logger.info("治療調整分析未產生結果")
    # except Exception as e:
    #     logger.warning(f"治療調整分析失敗: {e}")

    # # 4.6 BMI變數調整分析 (修正版 - 使用已訓練模型)
    # logger.info("\n執行BMI變數調整分析...")
    # try:
    #     bmi_results = analyze_bmi_modifications(
    #         total_experiments_result,
    #         processed_df,
    #         bmi_col='BMI'  # 根據你的數據調整
    #     )
        
    #     if 'results_df' in bmi_results:
    #         logger.info("BMI調整分析完成")
    #         logger.info(f"  分析組別數: {bmi_results['summary']['total_groups']}")
    #         logger.info(f"  分析患者數: {bmi_results['summary']['total_patients_analyzed']}")
    #         logger.info(f"  BMI+1平均影響: {bmi_results['summary']['avg_survival_change_plus1']:.2f} months")
    #         logger.info(f"  BMI-1平均影響: {bmi_results['summary']['avg_survival_change_minus1']:.2f} months")
    #         logger.info(f"  最大正向變化: +{bmi_results['summary']['max_positive_change']:.2f} months")
    #         logger.info(f"  最大負向變化: {bmi_results['summary']['max_negative_change']:.2f} months")
    #         logger.info(f"  圖表已儲存: {bmi_results['plot_saved']}")
            
    #         # 顯示各BMI區間的詳細結果
    #         results_df = bmi_results['results_df']
    #         logger.info("\n各BMI區間的平均影響:")
    #         for _, row in results_df.iterrows():
    #             logger.info(f"  BMI {row['bmi_range']}: "
    #                     f"患者數={row['patient_count']}, "
    #                     f"BMI+1={row['avg_change_plus1']:+.2f}, "
    #                     f"BMI-1={row['avg_change_minus1']:+.2f}")
            
    #         # 儲存詳細結果
    #         results_df.to_excel('bmi_modification_analysis.xlsx', index=False)
    #         logger.info("\n詳細BMI分析結果已儲存為 bmi_modification_analysis.xlsx")
            
    #         # 儲存每個患者的詳細預測變化
    #         all_patient_results = []
    #         for group in bmi_results['detailed_results']:
    #             for pred in group['predictions']:
    #                 pred['bmi_group'] = group['bmi_range']
    #                 all_patient_results.append(pred)
            
    #         if all_patient_results:
    #             patient_df = pd.DataFrame(all_patient_results)
    #             patient_df.to_excel('bmi_patient_level_predictions.xlsx', index=False)
    #             logger.info("患者層級預測變化已儲存為 bmi_patient_level_predictions.xlsx")
    #     else:
    #         logger.info("BMI分析結果: " + bmi_results.get('message', '無結果'))
    # except Exception as e:
    #     logger.warning(f"BMI調整分析失敗: {e}")

    # # ========================================
    # # Step 5: 結果總結
    # # ========================================
    # logger.info("\n" + "="*50)
    # logger.info("分析總結:")
    # logger.info("="*50)

    # # Ensemble分析總結
    # if ensemble_metrics:
    #     logger.info("\nEnsemble模型表現:")
    #     for model_type, metrics in ensemble_metrics.items():
    #         logger.info(f"\n{model_type}:")
    #         logger.info(f"  Ensemble C-index: {metrics['ensemble_c_index']:.6f}")
    #         logger.info(f"  改善幅度: {metrics['improvement']:+.6f}")
    #         logger.info(f"  種子數: {metrics['seed_count']}")
        
    #     best_model = max(ensemble_metrics.items(), key=lambda x: x[1]['ensemble_c_index'])
    #     logger.info(f"\n最佳ensemble模型: {best_model[0]} (C-index: {best_model[1]['ensemble_c_index']:.6f})")

    # # 生存分析總結
    # if survival_analysis:
    #     logger.info("\n生存預測分析:")
    #     for model_type, analysis in survival_analysis.items():
    #         if 'censored_metrics' in analysis:
    #             logger.info(f"\n{model_type} - 死亡事件預測:")
    #             logger.info(f"  MAE: {analysis['censored_metrics']['mae']:.2f} months")
    #             logger.info(f"  RMSE: {analysis['censored_metrics']['rmse']:.2f} months")

    # # 治療調整分析總結
    # if 'treatment_results' in locals() and treatment_results:
    #     logger.info("\n治療調整分析:")
    #     total_adjustments = sum(len(df) for df in treatment_results.values())
    #     logger.info(f"  總共分析了 {total_adjustments} 個治療調整方案")
    #     logger.info(f"  涵蓋 {len(treatment_results)} 個疾病分期")

    # # BMI調整分析總結
    # if 'bmi_results' in locals() and 'summary' in bmi_results:
    #     logger.info("\nBMI調整分析:")
    #     logger.info(f"  BMI每增加1單位，平均生存時間變化: {bmi_results['summary']['avg_survival_change_plus1']:+.2f} months")
    #     logger.info(f"  BMI每減少1單位，平均生存時間變化: {bmi_results['summary']['avg_survival_change_minus1']:+.2f} months")

    # # 儲存所有分析結果
    # analysis_results = {
    #     'ensemble_predictions': ensemble_preds,
    #     'ensemble_importance': ensemble_importance,
    #     'ensemble_metrics': ensemble_metrics,
    #     'survival_analysis': survival_analysis,
    #     'calibration_results': locals().get('calibration_results', {}),
    #     'treatment_results': locals().get('treatment_results', {}),
    #     'bmi_results': locals().get('bmi_results', {}),
    #     'shap_analysis': locals().get('shap_analysis', {})
    # }

    # # 儲存總結報告
    # with open('advanced_analysis_summary.txt', 'w', encoding='utf-8') as f:
    #     f.write("進階分析總結報告\n")
    #     f.write("="*50 + "\n\n")
        
    #     f.write("1. Ensemble模型表現\n")
    #     for model_type, metrics in ensemble_metrics.items():
    #         f.write(f"\n{model_type}:\n")
    #         f.write(f"  Ensemble C-index: {metrics['ensemble_c_index']:.6f}\n")
    #         f.write(f"  改善幅度: {metrics['improvement']:+.6f}\n")
        
    #     f.write("\n2. 檔案清單\n")
    #     f.write("  - stage_treatment_patient_counts.xlsx: 各期別治療方式患者數統計\n")
    #     f.write("  - treatment_shap_summary.xlsx: 治療方式SHAP值摘要\n")
    #     f.write("  - treatment_modifications_stage_*.xlsx: 各期別治療調整分析\n")
    #     f.write("  - bmi_modification_analysis.xlsx: BMI調整影響分析\n")
    #     f.write("  - bmi_modification_analysis.png: BMI調整影響圖表\n")

    # logger.info("\n總結報告已儲存為 advanced_analysis_summary.txt")
    # logger.info("所有分析完成！")



if __name__ == "__main__":
    main()
