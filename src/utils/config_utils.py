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
class FeatureConfig:
    num_features: Tuple[str, ...]
    cat_features: Tuple[str, ...]
    keep_features: Tuple[str, ...]
    treatments: Tuple[str, ...]
    survival_labels: Tuple[str, ...]
    source:str
    patient_id: str

    @classmethod
    def from_dict(cls, cfg: dict) -> "FeatureConfig":
        return FeatureConfig(
            num_features=tuple(cfg["num_features"]),
            cat_features=tuple(cfg["cat_features"]),
            keep_features=tuple(cfg["keep_features"]),
            treatments=tuple(cfg["treatments"]),
            survival_labels=tuple(cfg["survival_labels"]),
            source=cfg["source"],
            patient_id=cfg["patient_id"],
        )


@dataclass(frozen=True)
class PreprocessConfig:
    is_preprocess: bool
    impute_method: str
    is_augment: bool
    augment_times: int

    @classmethod
    def from_dict(cls, cfg: dict) -> "PreprocessConfig":
        return PreprocessConfig(
            is_preprocess=cfg["is_preprocess"],
            impute_method=cfg["impute_method"],
            is_augment=cfg["is_augment"],
            augment_times=cfg["augment_times"],
        )


@dataclass(frozen=True)
class ExperimentConfig:
    num_experiments: int
    models_to_train: Tuple[str, ...]
    result_save_path: Path

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExperimentConfig":
        return ExperimentConfig(
            num_experiments=cfg["num_experiments"],
            models_to_train=tuple(cfg["models_to_train"]),
            result_save_path=_PROJECT_ROOT / cfg.get("result_save_path"),
        )

@dataclass(frozen=True)
class SurvivalModelConfig:
    test_size: float
    censor_limit: str
    average_age: float

    @classmethod
    def from_dict(cls, cfg: dict) -> "SurvivalModelConfig":
        return SurvivalModelConfig(
            test_size=cfg["test_size"],
            censor_limit=cfg["censor_limit"],
            average_age=cfg["average_age"],
        )

@dataclass(frozen=True)
class analysisConfig:
    ensemble_predictions_save_path: Path
    K_U_metrics_save_path: Path
    ensemble_feature_importance_save_path: Path
    ensemble_c_index_save_path: Path
    @classmethod
    def from_dict(cls, cfg: dict) -> "analysisConfig":
        return analysisConfig(
            ensemble_predictions_save_path=_PROJECT_ROOT / cfg["ensemble_predictions_save_path"],
            K_U_metrics_save_path=_PROJECT_ROOT / cfg["K_U_metrics_save_path"],
            ensemble_feature_importance_save_path=_PROJECT_ROOT / cfg["ensemble_feature_importance_save_path"],
            ensemble_c_index_save_path=_PROJECT_ROOT / cfg["ensemble_c_index_save_path"],
        )


_CONFIG_CLASSES = {
    "dataset_config": DatasetConfig,
    "feature_config": FeatureConfig,
    "preprocess_config": PreprocessConfig,
    "experiment_config": ExperimentConfig,
    "survival_model_config": SurvivalModelConfig,
    "analysis_config": analysisConfig,
}
ConfigName = Literal[
    "dataset_config",
    "feature_config",
    "preprocess_config",
    "experiment_config",
    "survival_model_config",
    "analysis_config",
]


@overload
def load_config(cfg_name: Literal["dataset_config"]) -> DatasetConfig: ...
@overload
def load_config(cfg_name: Literal["feature_config"]) -> FeatureConfig: ...
@overload
def load_config(cfg_name: Literal["preprocess_config"]) -> PreprocessConfig: ...
@overload
def load_config(cfg_name: Literal["experiment_config"]) -> ExperimentConfig: ...
@overload
def load_config(cfg_name: Literal["survival_model_config"]) -> SurvivalModelConfig: ...
@overload
def load_config(cfg_name: Literal["analysis_config"]) -> analysisConfig: ...


@lru_cache(maxsize=None)
def load_config(
    cfg_name: ConfigName,
) -> Union[
    DatasetConfig,
    FeatureConfig,
    PreprocessConfig,
    ExperimentConfig,
    SurvivalModelConfig,
    analysisConfig,
]:
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
