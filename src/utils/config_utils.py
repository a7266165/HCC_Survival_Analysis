# src/utils/config_utils.py
import json
import logging
from pathlib import Path
from typing import Tuple, Union, overload, Literal, Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger(__name__)

# 根目錄、各子目錄
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATASET_DIR = _PROJECT_ROOT / "dataset"
_RESULTS_DIR = _PROJECT_ROOT / "results" / f"{ts}"


@dataclass(frozen=True)
class PathConfig:
    raw_dataset_path: Path
    imputed_dataset_dir: Path
    augmented_dataset_dir: Path

    result_save_dir: Path
    summary_save_path: Path
    origin_predictions_save_path: Path
    report_save_path: Path
    models_save_dir: Path

    ensemble_predictions_dir: Path
    ensemble_feature_importance_dir: Path
    metrics_dir: Path
    calibration_dir: Path
    K_U_group_metrics_save_path: Path
    ensemble_c_index_save_path: Path
    calibration_comparison_save_path: Path

    @classmethod
    def from_dict(cls, cfg: dict) -> "PathConfig":
        return PathConfig(
            raw_dataset_path=_DATASET_DIR / cfg["raw_dataset_name"],
            imputed_dataset_dir=_DATASET_DIR / cfg["imputed_dataset_name"],
            augmented_dataset_dir=_DATASET_DIR / cfg["augmented_dataset_name"],
            result_save_dir=_RESULTS_DIR,
            summary_save_path=_RESULTS_DIR / cfg["summary_save_name"],
            origin_predictions_save_path=_RESULTS_DIR
            / cfg["origin_predictions_save_name"],
            report_save_path=_RESULTS_DIR / cfg["report_save_name"],
            models_save_dir=_RESULTS_DIR / cfg["models_save_dir_name"],
            ensemble_predictions_dir=_RESULTS_DIR
            / cfg["ensemble_predictions_dir_name"],
            ensemble_feature_importance_dir=_RESULTS_DIR
            / cfg["ensemble_feature_importance_dir_name"],
            metrics_dir=_RESULTS_DIR / cfg["metrics_dir_name"],
            calibration_dir=_RESULTS_DIR / cfg["calibration_dir_name"],
            K_U_group_metrics_save_path=_RESULTS_DIR
            / cfg["metrics_dir_name"]
            / cfg["K_U_group_metrics_save_name"],
            ensemble_c_index_save_path=_RESULTS_DIR
            / cfg["metrics_dir_name"]
            / cfg["ensemble_c_index_save_name"],
            calibration_comparison_save_path=_RESULTS_DIR
            / cfg["calibration_dir_name"]
            / cfg["calibration_comparison_save_name"],
        )


@dataclass(frozen=True)
class FeatureConfig:
    num_features: Tuple[str, ...]
    cat_features: Tuple[str, ...]
    keep_features: Tuple[str, ...]
    treatments: Tuple[str, ...]
    survival_labels: Tuple[str, ...]
    source: str
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


# ====================================================
# # TODO: 待精簡
# @dataclass(frozen=True)
# class SurvivalModelConfig:
#     test_size: float
#     censor_limit: str
#     average_age: float

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "SurvivalModelConfig":
#         return SurvivalModelConfig(
#             test_size=cfg["test_size"],
#             censor_limit=cfg["censor_limit"],
#             average_age=cfg["average_age"],
#         )

# # TODO: 待精簡
# # What-If 分析配置
# @dataclass(frozen=True)
# class TreatmentAnalysisConfig:
#     """治療方式分析配置"""
#     enabled: bool
#     treatments_to_test: List[str]
#     analysis_mode: str  # "single_treatment" or "combination"
#     stratify_by_stage: bool
#     stage_column: str

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "TreatmentAnalysisConfig":
#         return cls(
#             enabled=cfg.get("enabled", True),
#             treatments_to_test=cfg.get("treatments_to_test", []),
#             analysis_mode=cfg.get("analysis_mode", "single_treatment"),
#             stratify_by_stage=cfg.get("stratify_by_stage", True),
#             stage_column=cfg.get("stage_column", "BCLC_stage")
#         )


# @dataclass(frozen=True)
# class FeatureModificationConfig:
#     """單一特徵修改配置"""
#     enabled: bool
#     modifications: List[float]
#     min_value: Optional[float]
#     max_value: Optional[float]

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "FeatureModificationConfig":
#         return cls(
#             enabled=cfg.get("enabled", False),
#             modifications=cfg.get("modifications", []),
#             min_value=cfg.get("min_value"),
#             max_value=cfg.get("max_value")
#         )


# @dataclass(frozen=True)
# class ContinuousFeatureAnalysisConfig:
#     """連續特徵分析配置"""
#     enabled: bool
#     features: Dict[str, FeatureModificationConfig]

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "ContinuousFeatureAnalysisConfig":
#         features = {}
#         for feature_name, feature_cfg in cfg.get("features", {}).items():
#             features[feature_name] = FeatureModificationConfig.from_dict(feature_cfg)

#         return cls(
#             enabled=cfg.get("enabled", True),
#             features=features
#         )


# @dataclass(frozen=True)
# class OutputSettingsConfig:
#     """輸出設定配置"""
#     save_individual_results: bool
#     save_summary_only: bool
#     create_visualizations: bool

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "OutputSettingsConfig":
#         return cls(
#             save_individual_results=cfg.get("save_individual_results", False),
#             save_summary_only=cfg.get("save_summary_only", True),
#             create_visualizations=cfg.get("create_visualizations", True)
#         )


# @dataclass(frozen=True)
# class WhatIfConfig:
#     """What-if 分析總配置"""
#     treatment_analysis: TreatmentAnalysisConfig
#     continuous_feature_analysis: ContinuousFeatureAnalysisConfig
#     output_settings: OutputSettingsConfig

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "WhatIfConfig":
#         return cls(
#             treatment_analysis=TreatmentAnalysisConfig.from_dict(
#                 cfg.get("treatment_analysis", {})
#             ),
#             continuous_feature_analysis=ContinuousFeatureAnalysisConfig.from_dict(
#                 cfg.get("continuous_feature_analysis", {})
#             ),
#             output_settings=OutputSettingsConfig.from_dict(
#                 cfg.get("output_settings", {})
#             )
#         )

# @dataclass(frozen=True)
# class ExperimentConfig:
#     num_experiments: int
#     models_to_train: Tuple[str, ...]
#     calibration_methods: Tuple[str, ...]
#     survival_model_config: SurvivalModelConfig
#     whatif_config: WhatIfConfig

#     @classmethod
#     def from_dict(cls, cfg: dict) -> "ExperimentConfig":
#         return ExperimentConfig(
#             num_experiments=cfg["num_experiments"],
#             models_to_train=tuple(cfg["models_to_train"]),
#             calibration_methods=tuple(cfg["calibration_methods"]),
#             survival_model_config=SurvivalModelConfig.from_dict(cfg["survival_model_config"]),
#             whatif_config=WhatIfConfig.from_dict(cfg["whatif_config"]),
#         )

# 把原本的 104-203 行替換成：


@dataclass(frozen=True)
class ExperimentSettings:
    num_experiments: int
    models_to_train: Tuple[str, ...]
    calibration_methods: Tuple[str, ...]

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExperimentSettings":
        return cls(
            num_experiments=cfg["num_experiments"],
            models_to_train=tuple(cfg["models_to_train"]),
            calibration_methods=tuple(cfg["calibration_methods"]),
        )


@dataclass(frozen=True)
class ModelSettings:
    test_size: float
    censor_limit: str
    average_age: float

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelSettings":
        return cls(
            test_size=cfg["test_size"],
            censor_limit=cfg["censor_limit"],
            average_age=cfg["average_age"],
        )


@dataclass(frozen=True)
class WhatIfSettings:
    # 治療分析
    analyze_treatments: bool
    treatments: Tuple[str, ...]
    treatment_mode: str
    stratify_by_stage: bool
    stage_column: str

    # 連續特徵分析
    analyze_continuous: bool
    continuous_features: Dict[str, Dict[str, Any]]

    # 輸出設定
    save_individual: bool
    save_summary: bool
    create_plots: bool

    @classmethod
    def from_dict(cls, cfg: dict) -> "WhatIfSettings":
        return cls(
            analyze_treatments=cfg.get("analyze_treatments", True),
            treatments=tuple(cfg.get("treatments", [])),
            treatment_mode=cfg.get("treatment_mode", "single"),
            stratify_by_stage=cfg.get("stratify_by_stage", True),
            stage_column=cfg.get("stage_column", "BCLC_stage"),
            analyze_continuous=cfg.get("analyze_continuous", True),
            continuous_features=cfg.get("continuous_features", {}),
            save_individual=cfg.get("save_individual", False),
            save_summary=cfg.get("save_summary", True),
            create_plots=cfg.get("create_plots", True),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_settings: ExperimentSettings
    model_settings: ModelSettings
    whatif_settings: WhatIfSettings

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExperimentConfig":
        return cls(
            experiment_settings=ExperimentSettings.from_dict(
                cfg["experiment_settings"]
            ),
            model_settings=ModelSettings.from_dict(cfg["model_settings"]),
            whatif_settings=WhatIfSettings.from_dict(cfg["whatif_settings"]),
        )


# ====================================================

_CONFIG_CLASSES = {
    "path_config": PathConfig,
    "feature_config": FeatureConfig,
    "preprocess_config": PreprocessConfig,
    "experiment_config": ExperimentConfig,
}
ConfigName = Literal[
    "path_config",
    "feature_config",
    "preprocess_config",
    "experiment_config",
    "survival_model_config",
    "analysis_config",
    "whatif_config",
]


@overload
def load_config(cfg_name: Literal["path_config"]) -> PathConfig: ...
@overload
def load_config(cfg_name: Literal["feature_config"]) -> FeatureConfig: ...
@overload
def load_config(cfg_name: Literal["preprocess_config"]) -> PreprocessConfig: ...
@overload
def load_config(cfg_name: Literal["experiment_config"]) -> ExperimentConfig: ...


@lru_cache(maxsize=None)
def load_config(
    cfg_name: ConfigName,
) -> Union[
    PathConfig,
    FeatureConfig,
    PreprocessConfig,
    ExperimentConfig,
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
