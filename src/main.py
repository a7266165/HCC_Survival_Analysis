import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from utils.config_utils import load_config
from preprocessing.preprocessor import data_preprocessor
from experimenting.experimentor import single_experimentor

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
    feature_config = load_config("feature_config")
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
            single_experiment_result = single_experimentor(
                processed_df,
                preprocess_config.is_preprocess,
                feature_config,
                random_seed,
                model_type,
                survival_model_config,
            )
            total_experiments_result.append(single_experiment_result)
    logger.info("所有實驗完成，共 %d 筆資料", len(total_experiments_result))
    # Step 4: 儲存實驗結果
    """
    輸入:total_experiments_result
    輸出:實驗結果儲存到指定路徑
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_df = pd.DataFrame(total_experiments_result)
    result_path = Path(experiment_config.result_save_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = result_path.with_name(f"{result_path.stem}_{ts}{result_path.suffix}")
    result_df.to_csv(save_path, index=False)
    logger.info(f"結果已儲存到 {save_path}")


if __name__ == "__main__":
    main()
