# src/utils/config_utils.py

import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    base_dir: Path
    raw_data_path: Path
    imputed_data_dir: Path
    augmented_data_dir: Path


def _get_project_root() -> Path:
    """回傳專案根目錄 (HCC_Survival_Analysis/)."""
    return Path(__file__).resolve().parents[2]


def _get_config_path() -> Path:
    """回傳 config/data_config.json 的完整路徑."""
    return _get_project_root() / "config" / "data_config.json"


def _load_json(path: Path) -> dict:
    """載入並解析 JSON; 若檔案不存在則記錄錯誤並拋例外。"""
    if not path.exists():
        logger.error("找不到設定檔：%s", path)
        raise FileNotFoundError(f"{path} 不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def load_data_config() -> DataConfig:
    """
    讀取 config/data_config.json，回傳 DataConfig 物件，欄位包含：
      - base_dir
      - raw_data_path
      - imputed_data_dir
      - augmented_data_dir
    """
    cfg = _load_json(_get_config_path())
    project_root = _get_project_root()
    base = project_root / cfg.get("base_dir", ".")
    return DataConfig(
        base_dir=base,
        raw_data_path=base / cfg["raw_data_path"],
        imputed_data_dir=base / cfg["imputed_data_dir"],
        augmented_data_dir=base / cfg["augmented_data_dir"],
    )
