# src/utils/config_utils.py
import json
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    raw_dataset_path: Path
    imputed_dataset_dir: Path
    augmented_dataset_dir: Path

    @staticmethod
    def from_dict(cfg: dict, project_root: Path) -> "DatasetConfig":
        return DatasetConfig(
            raw_dataset_path     = project_root / cfg["raw_dataset_path"],
            imputed_dataset_dir  = project_root / cfg["imputed_dataset_dir"],
            augmented_dataset_dir= project_root / cfg["augmented_dataset_dir"],
        )

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

    return DatasetConfig.from_dict(cfg, project_root)

@dataclass
class PreprocessConfig:
    num_feats: List[str]
    cat_feats: List[str]
    keep_feats: List[str]
    treatments: List[str]
    labels: List[str]
    is_preprocess: bool
    impute_method: str
    is_augment: bool
    augment_times: int

    @staticmethod
    def from_dict(cfg: dict) -> "PreprocessConfig":
        return PreprocessConfig(
            num_feats      = cfg["num_feats"],
            cat_feats      = cfg["cat_feats"],
            keep_feats     = cfg["keep_feats"],
            treatments     = cfg["treatments"],
            labels         = cfg["labels"],
            is_preprocess  = cfg["is_preprocess"],
            impute_method  = cfg["impute_method"],
            is_augment     = cfg["is_augment"],
            augment_times  = cfg["augment_times"]
        )

def load_preprocess_config() -> PreprocessConfig:
    """
    讀取 config/preprocess_config.json，回傳 PreprocessConfig 物件，欄位包含：
      - num_feats
      - cat_feats
      - keep_feats
      - treatments
      - labels
      - is_preprocess
      - impute_method
      - is_augment
      - augment_times
    """
    # name setting
    cfg_name = "preprocess_config"

    project_root = _get_project_root()
    cfg_path = _get_config_path(project_root, cfg_name)
    cfg = _load_json(cfg_path)

    return PreprocessConfig.from_dict(cfg)