import json
from pathlib import Path


def load_data_config() -> dict:
    """
    讀取 config/data_config.json，回傳一個 dict，
    裡面 key 對應到已解析的絕對路徑：
      - base_dir
      - raw_data_path
      - imputed_data_dir
      - augmented_data_dir
    """
    # 1. 定位到專案根目錄（HCC_Survival_Analysis/）
    project_root = Path(__file__).resolve().parent.parent.parent

    # 2. config.json 的路徑
    config_path = project_root / "config" / "data_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} 不存在，請確認檔案位置")

    # 3. 載入 JSON
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    # 4. 轉成絕對路徑
    base_dir = project_root / raw_cfg.get("base_dir", ".")
    paths = {
        "base_dir": base_dir,
        "raw_data_path": base_dir / raw_cfg["raw_data_path"],
        "imputed_data_dir": base_dir / raw_cfg["imputed_data_dir"],
        "augmented_data_dir": base_dir / raw_cfg["augmented_data_dir"],
    }
    return paths
