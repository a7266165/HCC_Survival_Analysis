import pandas as pd
import logging
from typing import Callable, Dict
from utils.config_utils import PreprocessConfig

logger = logging.getLogger(__name__)

# 定義各種 impute 策略
def _impute_zero(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return df.fillna(0)

def _impute_mode(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    mode_map = {f: df[f].mode().iloc[0] for f in feats}
    return df.fillna(value=mode_map)

def _impute_median(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    median_map = {f: df[f].median() for f in feats}
    return df.fillna(value=median_map)

IMPUTE_STRATEGIES: Dict[str, Callable[[pd.DataFrame, list[str]], pd.DataFrame]] = {
    "zero":   _impute_zero,
    "mode":   _impute_mode,
    "median": _impute_median,
    # "mean": _impute_mean,
    # "mida": _impute_mida,
}

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
    # step 0: 複製原始DataFrame以避免修改原始資料
    preprocess_df = df.copy()

    # step 1: 檢查是否需要處理資料
    if not preprocess_config.is_preprocess:
        return preprocess_df

    # step 2: 根據設定進行資料處理
    # TODO: 實作填補平均數與訓練MIDA模型填補特徵
    feats = preprocess_config.num_feats + preprocess_config.cat_feats
    method = preprocess_config.impute_method
    try:
        imputer = IMPUTE_STRATEGIES[method]
    except KeyError:
        logger.error("不支援%s，只支援zero, mode, median, mean, mida", method)
        raise ValueError(f"Unsupported impute method: {method}, only support zero, mode, median, mean, mida") from None
    preprocess_df = imputer(preprocess_df, feats)

    # step 3: 檢查是否需要擴充資料

    # step 4: 根據設定進行資料擴充

    return preprocess_df