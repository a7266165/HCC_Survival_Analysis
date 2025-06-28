# HCC Survival Analysis

A comprehensive survival analysis system for Hepatocellular Carcinoma (HCC) patients using machine learning approaches.

## Features

### 🏥 Clinical Survival Analysis
- Multiple survival models: Cox Proportional Hazards, XGBoost AFT, CatBoost AFT
- Ensemble predictions across multiple random seeds for robust results
- Comprehensive evaluation metrics including C-index

### 📊 Advanced Calibration Methods
- KNN + Kaplan-Meier curve calibration
- Regression-based calibration for censored data
- Segmental calibration
- Polynomial curve calibration

### 🔍 What-if Analysis
- Treatment modification analysis: Evaluate impact of different treatment combinations
- Continuous feature analysis: Assess effects of BMI, Age, AFP changes
- Stage-stratified analysis for personalized insights

### 📈 Visualization Suite
- Feature importance heatmaps and bar charts
- Calibration effect scatter plots (train/test comparison)
- K/U group metrics analysis
- Comprehensive survival prediction error analysis

### ⚡ Performance Optimization
- Multiprocessing support with configurable worker count
- Automatic CPU core allocation
- Progress tracking for long-running experiments

## Installation

### Prerequisites
- Python 3.11+
- Anaconda or Miniconda
- Poetry for dependency management

### Setup Instructions

1. **Create Conda Environment**
```bash
# Create a new conda environment with Python 3.11
conda create -n hcc_survival python=3.11
conda activate hcc_survival
```

2. **Clone Repository and Install Dependencies**
```bash
# Clone the repository
git clone https://github.com/yourusername/HCC_Survival_Analysis.git
cd HCC_Survival_Analysis

# Install dependencies using poetry
poetry install
```

3. **Verify Installation**
```bash
# Run a test to ensure everything is working
python src/main.py
```

## Project Structure

```
HCC_Survival_Analysis/
├── config/                         # Configuration files
│   ├── experiment_config.json      # Model and experiment settings
│   ├── feature_config.json         # Feature definitions
│   ├── multiprocess_config.json    # Multiprocessing settings
│   ├── path_config.json            # File path configurations
│   └── preprocess_config.json      # Data preprocessing settings
├── dataset/                        # Data directory
│   ├── processed/                  # Processed datasets
│   │   ├── augmented/              # Augmented data
│   │   └── imputed/                # Imputed data
│   └── raw/                        # Raw data files
│       └── hospitals/              # Hospital-specific data
├── results/                        # Experiment results (auto-generated)
│   └── {timestamp}/                # Results for each run
│       ├── calibration/            # Calibration analysis results
│       ├── ensemble_feature_importance/
│       ├── ensemble_predictions/
│       ├── figures/                # Generated visualizations
│       ├── metrics/                # Performance metrics
│       ├── models/                 # Saved model files
│       ├── original_predictions/
│       ├── report.txt          
│       └── summary.xlsx            # Comprehensive summary
├── src/                            # Source code
│   ├── analyzing/                  # Analysis modules
│   │   ├── ensemble_analyzer.py
│   │   └── visualizer.py
│   ├── experimenting/              # Experiment execution
│   │   └── experimentor.py
│   ├── preprocessing/              # Data preprocessing
│   │   └── preprocessor.py
│   ├── utils/                      # Utility modules
│   │   ├── config_utils.py
│   │   └── multiprocess_utils.py
│   └── main.py                     # Main entry point
├── LICENSE                         # MIT License
├── README.md                       # This file
├── poetry.lock                     # Poetry lock file
└── pyproject.toml                  # Project dependencies
```

## Usage

### Basic Usage

1. **Prepare your data**
   - Place your dataset in `dataset/raw/`
   - Update `config/path_config.json` with your file names

2. **Configure experiment**
   - Edit `config/experiment_config.json` to set:
     - Number of experiments (random seeds)
     - Models to train
     - Calibration methods to apply

3. **Run analysis**
```bash
python src/main.py
```

### Configuration Options

#### Multiprocessing Settings (`config/multiprocess_config.json`)
```json
{
  "enabled": true,           # Enable/disable multiprocessing
  "max_workers": 4,          # Maximum number of CPU cores
  "use_cpu_count": true,     # Auto-detect optimal core count
  "reserve_cpus": 1          # CPUs to reserve for system
}
```

#### Model Settings (`config/experiment_config.json`)
- `num_experiments`: Number of random seeds for ensemble
- `models_to_train`: List of models to use
- `test_size`: Train/test split ratio
- `calibration_methods`: Calibration techniques to apply

### Output

Results are saved in `results/{timestamp}/` including:
- **summary.xlsx**: Comprehensive performance summary
- **figures/**: All generated visualizations
- **models/**: Trained model pickle files
- **ensemble_predictions/**: Aggregated predictions
- **metrics/**: Detailed performance metrics

## Key Components

### Models
- **CoxPHFitter**: Traditional Cox Proportional Hazards model
- **XGBoost_AFT**: Accelerated Failure Time model with gradient boosting
- **CatBoost_AFT**: CatBoost implementation for survival analysis

### Calibration Methods
- **knn_km**: K-nearest neighbors with Kaplan-Meier estimation
- **regression**: Linear regression on non-censored data
- **segmental**: Segment-based bias correction
- **curve**: Polynomial curve fitting for calibration

### Analysis Features
- Feature importance analysis using SHAP values
- Ensemble predictions across multiple seeds
- Comprehensive error analysis for censored/non-censored groups
- What-if scenario analysis for treatment planning

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hcc_survival_analysis,
  title = {HCC Survival Analysis: Machine Learning for Hepatocellular Carcinoma Prognosis},
  author = {Yu-Ren Jhuang},
  year = {2025},
  url = {https://github.com/a7266165/HCC_Survival_Analysis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and collaborators
- Special thanks to the medical teams providing domain expertise
- Built with love for improving patient outcomes 💝