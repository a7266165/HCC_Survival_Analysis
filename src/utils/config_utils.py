# src/utils/config_utils.py
import json
import logging
from pathlib import Path
from typing import Tuple, Union, overload, Literal
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


@dataclass(frozen=True)
class DatasetConfig:
    raw_dataset_path: Path
    imputed_dataset_dir: Path
    augmented_dataset_dir: Path

    @classmethod
    def from_dict(cls, cfg: dict) -> "DatasetConfig":
        return DatasetConfig(
            raw_dataset_path=_PROJECT_ROOT / cfg["raw_dataset_path"],
            imputed_dataset_dir=_PROJECT_ROOT / cfg["imputed_dataset_dir"],
            augmented_dataset_dir=_PROJECT_ROOT / cfg["augmented_dataset_dir"],
        )


@dataclass(frozen=True)
class PreprocessConfig:
    num_feats: Tuple[str, ...]
    cat_feats: Tuple[str, ...]
    keep_feats: Tuple[str, ...]
    treatments: Tuple[str, ...]
    labels: Tuple[str, ...]
    is_preprocess: bool
    impute_method: str
    is_augment: bool
    augment_times: int

    @classmethod
    def from_dict(cls, cfg: dict) -> "PreprocessConfig":
        return PreprocessConfig(
            num_feats=tuple(cfg["num_feats"]),
            cat_feats=tuple(cfg["cat_feats"]),
            keep_feats=tuple(cfg["keep_feats"]),
            treatments=tuple(cfg["treatments"]),
            labels=tuple(cfg["labels"]),
            is_preprocess=cfg["is_preprocess"],
            impute_method=cfg["impute_method"],
            is_augment=cfg["is_augment"],
            augment_times=cfg["augment_times"],
        )


@dataclass(frozen=True)
class ExperimentConfig:
    num_experiments: int

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExperimentConfig":
        return ExperimentConfig(
            num_experiments=cfg["num_experiments"],
        )


_CONFIG_CLASSES = {
    "dataset_config": DatasetConfig,
    "preprocess_config": PreprocessConfig,
    "experiment_config": ExperimentConfig,
}
ConfigName = Literal["dataset_config", "preprocess_config", "experiment_config"]


@overload
def load_config(cfg_name: Literal["dataset_config"]) -> DatasetConfig: ...
@overload
def load_config(cfg_name: Literal["preprocess_config"]) -> PreprocessConfig: ...
@overload
def load_config(cfg_name: Literal["experiment_config"]) -> ExperimentConfig: ...


@lru_cache(maxsize=None)
def load_config(
    cfg_name: ConfigName,
) -> Union[DatasetConfig, PreprocessConfig, ExperimentConfig]:

    if cfg_name not in _CONFIG_CLASSES:
        raise ValueError(f"未知的配置類型: {cfg_name}")

    cfg_path = _CONFIG_DIR / f"{cfg_name}.json"
    cfg = _load_json(cfg_path)

    config_class = _CONFIG_CLASSES[cfg_name]
    return config_class.from_dict(cfg)


def _load_json(path: Path) -> dict:
    """載入並解析 JSON; 若檔案不存在則記錄錯誤並拋例外。"""
    if not path.exists():
        logger.error("找不到設定檔：%s", path)
        raise FileNotFoundError(f"{path} 不存在")
    return json.loads(path.read_text(encoding="utf-8"))
