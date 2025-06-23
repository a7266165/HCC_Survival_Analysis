import pandas as pd
from utils.config_utils import PreprocessConfig
import logging

logger = logging.getLogger(__name__)

def data_preprocessor(df: pd.DataFrame, preprocess_config: PreprocessConfig) -> pd.DataFrame:
    """
    處理DataFrame中的缺失值，並根據設定進行擴充。

    處理方式:
    (1) 檢查是否要處理資料
    (2) 若要處理資料，採用哪種方式填補特徵
        法一：填零
        法二：填眾數
        法三：填中位數
        法四：填平均數(需針對類別特徵寫特殊處理)
        法五：使用MIDA模型填補
    (3) 若要擴充資料，則進行N倍擴充
        訓練MIDA模型，並使用該模型擴充資料N倍

    :param df: 原始DataFrame
    :param preprocess_config: 包含處理設定的配置物件
    :return: 處理後的DataFrame
    """

    # step 1: 檢查是否需要處理資料
    if not preprocess_config.is_preprocess:
        return df

    # step 2: 根據設定進行資料處理
    if preprocess_config.impute_method == "zero":
        df.fillna(0, inplace=True)
    elif preprocess_config.impute_method == "mode":
        feats = preprocess_config.num_feats + preprocess_config.cat_feats
        mode_map = {feat: df[feat].mode().iloc[0] for feat in feats}
        df.fillna(value=mode_map, inplace=True)
    elif preprocess_config.impute_method == "median":
        feats = preprocess_config.num_feats + preprocess_config.cat_feats
        median_map = {feat: df[feat].median() for feat in feats}
        df.fillna(value=median_map, inplace=True)
    # TODO: 實作填補平均數與訓練MIDA模型填補特徵
    elif preprocess_config.impute_method == "mean":
        pass
    elif preprocess_config.impute_method == "mida":
        pass
    else:
        logger.error(f"只支援以下填補方法: zero, mode, median, mean, mida，但收到: {preprocess_config.impute_method}")
        raise ValueError(f"Unsupported impute method: {preprocess_config.impute_method}")

    # step 3: 檢查是否需要擴充資料

    # step 4: 根據設定進行資料擴充

    return df