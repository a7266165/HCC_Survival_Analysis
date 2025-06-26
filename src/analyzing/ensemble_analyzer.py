# TODO: 精簡
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.config_utils import analysisConfig, ExperimentConfig
from experimenting.experimentor import ExperimentResult



logger = logging.getLogger(__name__)

# 跟experimentor.py中的ExperimentResult類似，之後再看怎麼整合
def _find_result_directory(experiment_config: ExperimentConfig, ts: str) -> Path:
    """尋找結果儲存目錄"""
    result_path = Path(experiment_config.result_save_path)
    base_name = result_path.stem
    result_dir = result_path.parent / f"{base_name}_{ts}"
    return result_dir

def ensemble_predictions_by_seed(experiment_results: List[ExperimentResult], experiment_config: ExperimentConfig, ts:str) -> Dict[str, pd.DataFrame]:
    """將相同模型不同 seed 的預測值平均"""
    model_predictions = defaultdict(list)
    
    # 收集預測結果
    for res in experiment_results:
        if res and res.test_predictions is not None:
            df = res.test_predictions.copy()
            df['seed'] = res.random_seed
            model_predictions[res.model_type].append(df)
    
    # 計算 ensemble
    ensemble_results = {}
    for model_type, dfs in model_predictions.items():
        if not dfs:
            logger.warning(f"模型 {model_type} 沒有有效的預測結果")
            continue
            
        # 合併並聚合
        all_df = pd.concat(dfs, ignore_index=True)
        agg = all_df.groupby('patient_id')['predicted_survival_time'].agg([
            ('ensemble_prediction', 'mean'),
            ('prediction_std', 'std'),
            ('seed_count', 'count'),
            ('individual_predictions', list)
        ]).reset_index()
        
        agg['model_type'] = model_type
        ensemble_results[model_type] = agg[[
            'model_type', 'patient_id', 'ensemble_prediction',
            'prediction_std', 'seed_count', 'individual_predictions'
        ]]

    result_dir = _find_result_directory(experiment_config, ts)

    for model_type, df in ensemble_results.items():
        # 儲存到CSV
        save_path = result_dir / f"{model_type}_ensemble_predictions.csv"
        df.to_csv(save_path, index=False)

    return ensemble_results

def ensemble_feature_importance(experiment_results: List[ExperimentResult], experiment_config: ExperimentConfig, ts:str) -> Dict[str, pd.DataFrame]:
    """將特徵重要性平均"""
    logger.info("開始計算 ensemble 特徵重要性...")
    
    # 收集所有特徵重要性數據
    rows = []
    for res in experiment_results:
        if res and res.feature_importance:
            for method, features in res.feature_importance.items():
                if isinstance(features, dict):
                    rows.extend({
                        'model_type': res.model_type,
                        'seed': res.random_seed,
                        'method': method,
                        'feature': feat,
                        'importance': float(imp)
                    } for feat, imp in features.items())
    
    if not rows:
        logger.warning("沒有找到有效的特徵重要性數據")
        return {}
    
    # 處理數據
    df = pd.DataFrame(rows)
    ensemble_results = {}
    
    for model_type, group in df.groupby('model_type'):
        agg = group.groupby(['feature', 'method'])['importance'].agg([
            ('mean_importance', 'mean'),
            ('std_importance', 'std'),
            ('seed_count', 'count')
        ]).fillna({'std_importance': 0}).reset_index()
        
        agg['model_type'] = model_type
        ensemble_results[model_type] = agg[[
            'model_type', 'method', 'feature',
            'mean_importance', 'std_importance', 'seed_count'
        ]].sort_values(['method', 'mean_importance'], ascending=[True, False])

        # 儲存到CSV
        result_dir = _find_result_directory(experiment_config, ts)
        save_path = result_dir / f"{model_type}_ensemble_feature_importance.csv"

    return ensemble_results

def calculate_ensemble_metrics(
    ensemble_predictions: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    original_results: List[ExperimentResult],
    experiment_config: ExperimentConfig,
    ts: str
) -> Dict[str, Dict[str, float]]:
    # 檢查必要欄位
    required = ['patient_id', 'time', 'event']
    if not all(col in processed_df.columns for col in required):
        logger.error(f"processed_df 缺少必要的列: {required}")
        return {}
    
    true_data = processed_df[required]
    metrics = {}
    
    for model_type, pred_df in ensemble_predictions.items():
        # 合併預測與真實數據
        merged = pred_df.merge(true_data, on='patient_id')
        if merged.empty:
            continue
        
        # 計算 ensemble C-index
        ens_c = concordance_index(
            merged['time'], merged['ensemble_prediction'], merged['event']
        )
        
        # 直接獲取已計算的個別 C-index
        indiv_c_indices = [
            res.test_c_index 
            for res in original_results 
            if res and res.model_type == model_type
        ]
        
        if not indiv_c_indices:
            continue
        
        # 組裝指標
        indiv_array = np.array(indiv_c_indices)
        metrics[model_type] = {
            'ensemble_c_index': ens_c,
            'individual_mean_c_index': indiv_array.mean(),
            'individual_std_c_index': indiv_array.std(),
            'individual_min_c_index': indiv_array.min(),
            'individual_max_c_index': indiv_array.max(),
            'improvement': ens_c - indiv_array.mean(),
            'seed_count': len(indiv_c_indices),
            'patients_count': len(merged)
        }
        # 統一 rounding
        metrics[model_type] = {
            k: round(v, 6) if isinstance(v, float) else v 
            for k, v in metrics[model_type].items()
        }

    # 儲存 ensemble metrics
    result_dir = _find_result_directory(experiment_config, ts)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(result_dir / "ensemble_metrics.csv", index=True)

    return metrics

# ========================================
# 1. cendored和non-censored兩組數據的指標
# ========================================

def analyze_survival_predictions(
    ensemble_predictions: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    ts: str
) -> Dict[str, Dict[str, float]]:
    true = processed_df[['patient_id','time','event']]
    results: Dict[str, Dict[str,float]] = {}
    for model_type, predictions in ensemble_predictions.items():
        merged_prediction = pd.merge(predictions, true, on='patient_id', how='inner')
        if merged_prediction.empty:
            continue
        non_censored = merged_prediction[merged_prediction['event']==1]
        censored = merged_prediction[merged_prediction['event']==0]
        stats: Dict[str, Any] = {}
        if len(non_censored) > 0:
            errors = non_censored['ensemble_prediction'] - non_censored['time']
            stats['non_censored_metrics'] = {
                'count': len(non_censored),
                'mean_absolute_error': float(np.abs(errors).mean()),
                'root_mean_square_error': float(np.sqrt((errors ** 2).mean())),
                'mean_error': float(errors.mean()),
                'std_error': float(errors.std()),
                'median_error': float(errors.median())
            }
        
        if len(censored):
            correct = (censored['ensemble_prediction']>censored['time']).sum()
            stats['censored_metrics'] = {
                'count':len(censored), 'correct_ratio': correct/len(censored),
                'mean_predicted_time':np.mean(censored['ensemble_prediction']),
                'mean_observed_time':np.mean(censored['time'])
            }
        results[model_type] = stats

        # 儲存分析結果
        result_dir = _find_result_directory(experiment_config, ts)
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df.to_csv(result_dir / f"{model_type}_survival_analysis.csv", index=True)

    return results


# ========================================
# 2. 預測校正功能
# ========================================

def calibrate_predictions_knn_km(
    predictions: pd.DataFrame,
    true_labels: pd.DataFrame,
    k: int = 200
) -> pd.DataFrame:
    """
    校正法一: KNN + KM curve校正
    """    
    merged_df = pd.merge(predictions, true_labels, on='patient_id', how='inner')
    calibrated_df = merged_df.copy()
    
    # 準備KNN
    X = merged_df[['ensemble_prediction']].values
    nn_model = NearestNeighbors(n_neighbors=min(k, len(merged_df)))
    nn_model.fit(X)
    
    calibrated_predictions = []
    
    for idx, row in merged_df.iterrows():
        # 找到最近的k個鄰居
        distances, indices = nn_model.kneighbors([[row['ensemble_prediction']]])
        neighbor_data = merged_df.iloc[indices[0]]
        
        # 建立KM curve
        kmf = KaplanMeierFitter()
        kmf.fit(neighbor_data['time'], neighbor_data['event'])
        
        # 找到生存機率為0.5的時間點
        try:
            median_survival = kmf.median_survival_time_
            if pd.isna(median_survival):
                median_survival = row['ensemble_prediction']
        except:
            median_survival = row['ensemble_prediction']
            
        calibrated_predictions.append(median_survival)
    
    calibrated_df['calibrated_prediction'] = calibrated_predictions
    return calibrated_df[['patient_id', 'calibrated_prediction']]


def calibrate_predictions_regression(
    predictions: pd.DataFrame,
    true_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    校正法二: Censored data回歸校正
    """
    
    # 合併訓練數據
    train_merged = pd.merge(train_predictions, train_labels, on='patient_id', how='inner')
    censored_train = train_merged[train_merged['event'] == 1]
    
    if len(censored_train) < 2:
        logger.warning("訓練集中censored數據不足，跳過回歸校正")
        return test_predictions
    
    # 建立回歸模型 (x=真實時間, y=預測-真實的差值)
    X = censored_train['time'].values.reshape(-1, 1)
    y = (censored_train['ensemble_prediction'] - censored_train['time']).values
    
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    
    # 對測試數據進行校正
    test_merged = pd.merge(test_predictions, test_labels, on='patient_id', how='inner')
    predicted_errors = reg_model.predict(test_merged['time'].values.reshape(-1, 1))
    
    calibrated_df = test_merged.copy()
    calibrated_df['calibrated_prediction'] = calibrated_df['ensemble_prediction'] - predicted_errors
    
    return calibrated_df[['patient_id', 'calibrated_prediction']]


def calibrate_predictions_segmental(
    predictions: pd.DataFrame,
    true_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    校正法三: 區段化校正
    """
    
    train_merged = pd.merge(train_predictions, train_labels, on='patient_id', how='inner')
    test_merged = pd.merge(test_predictions, test_labels, on='patient_id', how='inner')
    
    # 根據時間點分段
    time_quantiles = np.linspace(0, 1, n_segments + 1)
    time_bins = np.quantile(train_merged['time'], time_quantiles)
    
    # 計算各段的校正值
    segment_corrections = {}
    for i in range(n_segments):
        mask = (train_merged['time'] >= time_bins[i]) & (train_merged['time'] < time_bins[i+1])
        segment_data = train_merged[mask]
        
        if len(segment_data) > 0:
            mean_error = np.mean(segment_data['ensemble_prediction'] - segment_data['time'])
            segment_corrections[i] = mean_error
        else:
            segment_corrections[i] = 0
    
    # 對測試數據應用校正
    calibrated_df = test_merged.copy()
    calibrated_predictions = []
    
    for _, row in test_merged.iterrows():
        # 找到對應的段
        segment_idx = np.searchsorted(time_bins[1:], row['time'])
        segment_idx = min(segment_idx, n_segments - 1)
        
        correction = segment_corrections.get(segment_idx, 0)
        calibrated_pred = row['ensemble_prediction'] - correction
        calibrated_predictions.append(calibrated_pred)
    
    calibrated_df['calibrated_prediction'] = calibrated_predictions
    return calibrated_df[['patient_id', 'calibrated_prediction']]


def calibrate_predictions_curve(
    predictions: pd.DataFrame,
    true_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    校正法四: 全體訓練集curve校正
    """
    
    train_merged = pd.merge(train_predictions, train_labels, on='patient_id', how='inner')
    test_merged = pd.merge(test_predictions, test_labels, on='patient_id', how='inner')
    
    # 建立校正curve (使用多項式回歸)
    X = train_merged['time'].values
    y = (train_merged['ensemble_prediction'] - train_merged['time']).values
    
    # 使用3次多項式擬合
    poly_coeffs = np.polyfit(X, y, deg=3)
    
    # 對測試數據應用校正
    calibrated_df = test_merged.copy()
    predicted_errors = np.polyval(poly_coeffs, test_merged['time'].values)
    calibrated_df['calibrated_prediction'] = calibrated_df['ensemble_prediction'] - predicted_errors
    
    return calibrated_df[['patient_id', 'calibrated_prediction']]


def apply_all_calibrations(
    ensemble_preds: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    random_seed: int,
    ts: str,
) -> Dict[str, Dict[str, Any]]:
    true_labels = processed_df[['patient_id','time','event']]
    results: Dict[str, Dict[str, Any]] = {}
    for model_type, pred in ensemble_preds.items():
        mr: Dict[str, Any] = {}
        orig = analyze_survival_predictions({model_type: pred}, processed_df, experiment_config, ts).get(model_type, {})
        mr['original'] = orig
        # 校正方法一: KNN + KM curve校正
        knn_cal = calibrate_predictions_knn_km(pred, true_labels)
        merged = pd.merge(pred, knn_cal, on='patient_id', how='left')
        merged['ensemble_prediction'] = merged['calibrated_prediction'].fillna(merged['ensemble_prediction'])
        mr['knn_km'] = analyze_survival_predictions({model_type: merged}, processed_df, experiment_config, ts)[model_type]
        # 校正方法二: Censored data回歸校正
        reg_cal = calibrate_predictions_regression(pred, true_labels)
        merged = pd.merge(pred, reg_cal, on='patient_id', how='left')
        merged['ensemble_prediction'] = merged['calibrated_prediction'].fillna(merged['ensemble_prediction'])
        mr['regression'] = analyze_survival_predictions({model_type: merged}, processed_df, experiment_config, ts)[model_type]
        # 校正方法三: 區段化校正
        seg_cal = calibrate_predictions_segmental(pred, true_labels)
        merged = pd.merge(pred, seg_cal, on='patient_id', how='left')
        merged['ensemble_prediction'] = merged['calibrated_prediction'].fillna(merged['ensemble_prediction'])
        mr['segmental'] = analyze_survival_predictions({model_type: merged}, processed_df, experiment_config, ts)[model_type]
        # 校正方法四: 全體訓練集curve校正
        curve_cal = calibrate_predictions_curve(pred, true_labels)
        merged = pd.merge(pred, curve_cal, on='patient_id', how='left')
        merged['ensemble_prediction'] = merged['calibrated_prediction'].fillna(merged['ensemble_prediction'])
        mr['curve'] = analyze_survival_predictions({model_type: merged}, processed_df, experiment_config, ts)[model_type]

        results[model_type] = mr
    # 儲存校正結果
    result_dir = _find_result_directory(experiment_config, ts)
    for model_type, metrics in results.items():
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.to_csv(result_dir / f"{model_type}_calibration_metrics.csv", index=True)


    return results


# ========================================
# 3. 統計工具
# ========================================

def create_stage_treatment_shap_table(
    ensemble_importance: Dict[str, pd.DataFrame],
    processed_df: pd.DataFrame,
    stage_col: str,
    treatment_cols: List[str] 
) -> pd.DataFrame:
    """
    創建stage x treatment的患者數統計表，並附加整體SHAP值
    注意：SHAP值是對整體特徵計算的，不是按stage分別計算
    """
    
    # 獲取SHAP數據
    shap_values = {}
    if ensemble_importance:
        model_type = list(ensemble_importance.keys())[0]
        shap_data = ensemble_importance[model_type]
        shap_data = shap_data[shap_data['method'] == 'shap_importance']
        
        # 建立治療方式到SHAP值的映射
        for _, row in shap_data.iterrows():
            treatment = row['feature']
            if treatment in treatment_cols:
                shap_values[treatment] = {
                    'mean': row['mean_importance'],
                    'std': row['std_importance']
                }
    
    # 統計各stage的治療患者數
    stages = sorted(processed_df[stage_col].unique())
    
    for stage in stages:
        stage_patients = processed_df[processed_df[stage_col] == stage]
        
        for treatment in treatment_cols:
            if treatment not in processed_df.columns:
                continue
                
            # 計算該stage中接受該治療的患者數
            treatment_patients = stage_patients[stage_patients[treatment] == 1]
            count = len(treatment_patients)
            
            # 建立統計記錄
            stat_record = {
                'Stage': stage,
                'Treatment': treatment,
                'Patient_Count': count,
                'Stage_Total': len(stage_patients),
                'Percentage': f"{count/len(stage_patients)*100:.1f}%" if len(stage_patients) > 0 else "0.0%"
            }
        
    return {}


# ========================================
# 4. 治療方式調整器 (修正版)
# ========================================

def analyze_treatment_modifications(
    experiment_results: List[Dict[str, Any]],
    processed_df: pd.DataFrame,
    stage_col: str = 'BCLC_stage',
    treatment_cols: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    分析治療方式調整對預測的影響（使用已訓練的模型）
    """
    logger.info("分析治療方式調整影響...")
    
    if treatment_cols is None:
        treatment_cols = [
            'liver_transplantation', 'surgical_resection', 'radiofrequency', 
            'TACE', 'target_therapy', 'immunotherapy', 'HAIC', 
            'radiotherapy', 'best_support_care'
        ]
    
    # 獲取第一個可用的模型和測試集
    model_info = None
    for result in experiment_results:
        if result and 'model' in result and 'test_indices' in result:
            model_info = result
            break
    
    if not model_info:
        logger.warning("找不到可用的模型進行治療調整分析")
        return {}
    
    model = model_info['model']
    test_indices = model_info['test_indices']
    test_df = processed_df.iloc[test_indices].copy()
    
    # 確保特徵列存在
    feature_cols = [col for col in model_info.get('feature_columns', []) if col in test_df.columns]
    
    modification_results = {}
    
    # 按stage分析
    for stage in test_df[stage_col].unique():
        stage_patients = test_df[test_df[stage_col] == stage]
        stage_results = []
        
        for idx, patient in stage_patients.iterrows():
            patient_id = patient['patient_id']
            current_treatments = {col: patient[col] for col in treatment_cols if col in patient.index}
            
            # 原始預測
            try:
                X_original = patient[feature_cols].values.reshape(1, -1)
                original_pred = model.predict(X_original)[0]
            except Exception as e:
                logger.warning(f"預測患者 {patient_id} 失敗: {e}")
                continue
            
            # 嘗試每種治療調整
            for treatment in treatment_cols:
                if treatment not in feature_cols:
                    continue
                
                # 創建調整後的特徵
                X_modified = X_original.copy()
                treatment_idx = feature_cols.index(treatment)
                
                # 關閉當前治療，開啟新治療
                for i, col in enumerate(feature_cols):
                    if col in treatment_cols:
                        X_modified[0, i] = 1 if col == treatment else 0
                
                # 預測調整後的結果
                try:
                    modified_pred = model.predict(X_modified)[0]
                    
                    # 記錄結果
                    stage_results.append({
                        'patient_id': patient_id,
                        'stage': stage,
                        'original_treatment': [k for k, v in current_treatments.items() if v == 1],
                        'new_treatment': treatment,
                        'original_prediction': original_pred,
                        'modified_prediction': modified_pred,
                        'prediction_change': modified_pred - original_pred,
                        'relative_change': (modified_pred - original_pred) / original_pred if original_pred != 0 else 0
                    })
                except Exception as e:
                    logger.warning(f"調整預測失敗: {e}")
                    continue
        
        if stage_results:
            stage_df = pd.DataFrame(stage_results)
            # 按預測變化排序
            stage_df = stage_df.sort_values('prediction_change', ascending=False)
            modification_results[stage] = stage_df
            
            logger.info(f"Stage {stage}: 分析了 {len(stage_df)} 個治療調整")
            
            # 顯示最大影響的調整
            if len(stage_df) > 0:
                max_change = stage_df.iloc[0]
                logger.info(f"  最大正向變化: {max_change['new_treatment']} "
                          f"(+{max_change['prediction_change']:.2f} months)")
                
                min_change = stage_df.iloc[-1]
                logger.info(f"  最大負向變化: {min_change['new_treatment']} "
                          f"({min_change['prediction_change']:.2f} months)")
    
    return modification_results


# ========================================
# 5. 變數調整器 (修正版)
# ========================================

def analyze_bmi_modifications(
    experiment_results: List[Dict[str, Any]],
    processed_df: pd.DataFrame,
    bmi_col: str = 'BMI'
) -> Dict[str, Any]:
    """
    分析BMI調整對預測的影響（使用已訓練的模型）
    """
    logger.info("分析BMI調整影響...")
    
    if bmi_col not in processed_df.columns:
        logger.warning(f"找不到BMI列: {bmi_col}")
        return {'message': f'找不到BMI列: {bmi_col}'}
    
    # 獲取第一個可用的模型和測試集
    model_info = None
    for result in experiment_results:
        if result and 'model' in result and 'test_indices' in result:
            model_info = result
            break
    
    if not model_info:
        logger.warning("找不到可用的模型進行BMI調整分析")
        return {'message': '找不到可用的模型'}
    
    model = model_info['model']
    test_indices = model_info['test_indices']
    test_df = processed_df.iloc[test_indices].copy()
    feature_cols = [col for col in model_info.get('feature_columns', []) if col in test_df.columns]
    
    if bmi_col not in feature_cols:
        logger.warning(f"BMI不在模型特徵中")
        return {'message': 'BMI不在模型特徵中'}
    
    # BMI分組分析
    bmi_values = test_df[bmi_col].dropna()
    min_bmi = int(np.floor(bmi_values.min()))
    max_bmi = int(np.ceil(bmi_values.max()))
    
    bmi_results = []
    
    # 對每個BMI區間進行分析
    for bmi_start in range(min_bmi, max_bmi):
        bmi_end = bmi_start + 1
        mask = (test_df[bmi_col] >= bmi_start) & (test_df[bmi_col] < bmi_end)
        group_patients = test_df[mask]
        
        if len(group_patients) == 0:
            continue
        
        group_analysis = {
            'bmi_range': f"{bmi_start}-{bmi_end}",
            'patient_count': len(group_patients),
            'mean_bmi': group_patients[bmi_col].mean(),
            'predictions': []
        }
        
        # 分析每個患者的BMI調整影響
        for idx, patient in group_patients.iterrows():
            patient_id = patient['patient_id']
            original_bmi = patient[bmi_col]
            
            # 準備特徵
            X_original = patient[feature_cols].values.reshape(1, -1)
            bmi_idx = feature_cols.index(bmi_col)
            
            try:
                # 原始預測
                original_pred = model.predict(X_original)[0]
                
                # BMI +1 預測
                X_plus1 = X_original.copy()
                X_plus1[0, bmi_idx] = original_bmi + 1
                pred_plus1 = model.predict(X_plus1)[0]
                
                # BMI -1 預測
                X_minus1 = X_original.copy()
                X_minus1[0, bmi_idx] = max(original_bmi - 1, 10)  # 避免BMI過低
                pred_minus1 = model.predict(X_minus1)[0]
                
                patient_result = {
                    'patient_id': patient_id,
                    'original_bmi': original_bmi,
                    'original_prediction': original_pred,
                    'bmi_plus1_prediction': pred_plus1,
                    'bmi_minus1_prediction': pred_minus1,
                    'change_plus1': pred_plus1 - original_pred,
                    'change_minus1': pred_minus1 - original_pred
                }
                
                group_analysis['predictions'].append(patient_result)
                
            except Exception as e:
                logger.warning(f"BMI調整預測失敗 (患者 {patient_id}): {e}")
                continue
        
        if group_analysis['predictions']:
            # 計算該組的平均影響
            changes_plus1 = [p['change_plus1'] for p in group_analysis['predictions']]
            changes_minus1 = [p['change_minus1'] for p in group_analysis['predictions']]
            
            group_analysis['avg_change_plus1'] = np.mean(changes_plus1)
            group_analysis['avg_change_minus1'] = np.mean(changes_minus1)
            group_analysis['std_change_plus1'] = np.std(changes_plus1)
            group_analysis['std_change_minus1'] = np.std(changes_minus1)
            
            bmi_results.append(group_analysis)
    
    # 準備結果
    if bmi_results:
        # 轉換為DataFrame以便分析
        results_df = pd.DataFrame([{
            'bmi_range': r['bmi_range'],
            'patient_count': r['patient_count'],
            'mean_bmi': r['mean_bmi'],
            'avg_change_plus1': r['avg_change_plus1'],
            'avg_change_minus1': r['avg_change_minus1']
        } for r in bmi_results])
        
        # 繪製圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # BMI+1影響
        ax1.bar(results_df['bmi_range'], results_df['avg_change_plus1'])
        ax1.set_xlabel('BMI Range')
        ax1.set_ylabel('Average Survival Change (months)')
        ax1.set_title('Impact of BMI +1 on Survival Prediction')
        ax1.tick_params(axis='x', rotation=45)
        
        # BMI-1影響  
        ax2.bar(results_df['bmi_range'], results_df['avg_change_minus1'])
        ax2.set_xlabel('BMI Range')
        ax2.set_ylabel('Average Survival Change (months)')
        ax2.set_title('Impact of BMI -1 on Survival Prediction')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('bmi_modification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 計算總體統計
        all_predictions = []
        for r in bmi_results:
            all_predictions.extend(r['predictions'])
        
        summary = {
            'total_groups': len(bmi_results),
            'total_patients_analyzed': len(all_predictions),
            'avg_survival_change_plus1': np.mean([p['change_plus1'] for p in all_predictions]),
            'avg_survival_change_minus1': np.mean([p['change_minus1'] for p in all_predictions]),
            'max_positive_change': max([p['change_plus1'] for p in all_predictions]),
            'max_negative_change': min([p['change_minus1'] for p in all_predictions])
        }
        
        logger.info(f"BMI調整分析完成:")
        logger.info(f"  分析組別數: {summary['total_groups']}")
        logger.info(f"  分析患者數: {summary['total_patients_analyzed']}")
        logger.info(f"  BMI+1平均影響: {summary['avg_survival_change_plus1']:.2f} months")
        logger.info(f"  BMI-1平均影響: {summary['avg_survival_change_minus1']:.2f} months")
        
        return {
            'results_df': results_df,
            'detailed_results': bmi_results,
            'summary': summary,
            'plot_saved': 'bmi_modification_analysis.png'
        }
    else:
        return {
            'message': '沒有足夠的數據進行BMI分析'
        }