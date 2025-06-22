# src/utils/config_utils.py
import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    raw_dataset_path: Path
    imputed_dataset_dir: Path
    augmented_dataset_dir: Path


def _get_project_root() -> Path:
    """回傳專案根目錄 (HCC_Survival_Analysis/)."""
    return Path(__file__).resolve().parents[2]


def _get_config_path(project_root, config_name) -> Path:
    """回傳 config/{config_name}.json 的完整路徑。"""
    return project_root / "config" / f"{config_name}.json"


def _load_json(path: Path) -> dict:
    """載入並解析 JSON; 若檔案不存在則記錄錯誤並拋例外。"""
    if not path.exists():
        logger.error("找不到設定檔：%s", path)
        raise FileNotFoundError(f"{path} 不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_config() -> DatasetConfig:
    """
    讀取 config/dataset_config.json，回傳 DatasetConfig 物件，欄位包含：
      - base_dir
      - raw_data_path
      - imputed_data_dir
      - augmented_data_dir
    """
    # name setting
    cfg_name = "dataset_config"

    project_root = _get_project_root()
    cfg_path = _get_config_path(project_root, cfg_name)
    cfg = _load_json(cfg_path)

    return DatasetConfig(
        raw_dataset_path=project_root / cfg["raw_dataset_path"],
        imputed_dataset_dir=project_root / cfg["imputed_dataset_dir"],
        augmented_dataset_dir=project_root / cfg["augmented_dataset_dir"],
    )
