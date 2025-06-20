import logging
from pathlib import Path
import pandas as pd

from utils.config_utils import load_data_config, DataConfig


def setup_logging():
    fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("程式啟動，開始讀取設定檔")
    # Step 1: 讀取csv檔
    """
    輸入:
    - csv檔路徑
    輸出:
    - DataFrame: 包含csv檔內容的DataFrame
    """
    config: DataConfig = load_data_config()
    logger.debug("DataConfig 內容：%s", config)

    # 檢查原始資料檔案
    if not config.raw_data_path.exists():
        logger.error("原始 CSV 不存在：%s", config.raw_data_path)
        return

    # 讀入資料
    df = pd.read_csv(config.raw_data_path)
    logger.info("成功讀取原始資料，共 %d 筆", len(df))

    # Step 2: 資料處理
    """
    輸入:
    - DataFrame: 包含csv檔內容的DataFrame
    
    輸出:
    - DataFrame: 處理後的DataFrame
    
    處理方式:
    (1) 不做任何處理
    (2) 將DataFrame中的所有缺失值填充為0，不做擴充
    (3) 將DataFrame中的所有缺失值填充為0，擴充N倍
    (4) 訓練MIDA填充模型，並使用該模型填充DataFrame中的所有缺失值，不做擴充
    (5) 訓練MIDA填充模型，並使用該模型填充DataFrame中的所有缺失值，擴充N倍    
    """
    # Step 3: 進行實驗
    """
    輸入:
    - DataFrame: 處理後的DataFrame
    
    輸出:?
    
    實驗內容:
    將處理後的DataFrame分成訓練集和測試集，使用訓練集訓練模型，並在測試集上進行評估。
    """

    # Step 4: 整合實驗結果
    """
    將多次實驗的結果整合成一個DataFrame，綜合評估

    
    評估指標包括
    (1) C-Index
    (2) 模型預測的存活時間
    (3) SHAP值
    (4) right-censored的正確比例
    (5) non-censored的正確比例
    (6) 調整BMI
    (7) 調整治療手段
    """


if __name__ == "__main__":
    main()
