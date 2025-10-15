# MolGBN-OPR: Multi-modal Fusion Model Based on Gradient Boosting Neural Network for Predicting Organic Micropollutant Rejection Efficiency

## Project Introduction

This project addresses the bottlenecks of insufficient prediction accuracy and unclear mechanism解析 in the rejection efficiency of organic micropollutants by nanofiltration/reverse osmosis polyamide membranes. It proposes a multi-modal fusion machine learning model based on the gradient boosting neural network framework. By systematically integrating traditional tabular data (such as membrane structure parameters, operating conditions, and physicochemical properties of pollutants) with multi-modal representations of molecules (such as graph structures, images, and fingerprints), three fusion prediction schemes are constructed to achieve high-precision prediction and provide molecular-level mechanism explanation.

## Research Background

Nanofiltration/reverse osmosis polyamide membranes play an important role in water treatment. However, traditional empirical models and single-modal machine learning methods have difficulty accurately predicting the rejection efficiency of organic micropollutants and cannot reveal the rejection mechanism at the molecular level. This research breaks through the limitations of traditional methods through a multi-modal fusion strategy, providing a theoretical basis and data-driven new path for the directed design of membrane materials.

## Main Contributions

- Proposed a multi-modal fusion machine learning model based on the gradient boosting neural network framework
- Systematically integrated multi-source data such as tabular data, molecular graph structures, molecular images, and molecular fingerprints
- Constructed three fusion prediction schemes, among which the fusion model of tabular data and molecular graph features (GB+Graph) achieved the best prediction performance
- Achieved high-precision prediction (R² reached 0.9014), significantly better than the baseline model with pure tabular data (R² = 0.8494)
- Used SHAP method for interpretability analysis, providing mechanism explanations beyond traditional descriptors from the perspective of microstructure
- Developed atomic-level attribution analysis to automatically identify the core role of key functional groups in molecule-membrane interactions

## Project Structure

```
MolGBN-OPR/
├── checkpoint/          # Trained model weight files
├── data/                # Datasets
│   ├── processed/       # Processed datasets
│   └── raw/             # Raw datasets
├── dataset/             # Data loading and preprocessing modules
├── models/              # Model definitions
│   ├── GNNModels.py     # Graph neural network models
│   ├── ensemblemodel.py # Ensemble model framework
│   ├── gbnnModel.py     # Gradient boosting neural network models
│   ├── splinear.py      # Non-linear modules
│   └── weaklearner.py   # Weak learner modules
├── src/                 # Source code
│   ├── compare/         # Model comparison related code
│   ├── interpret_shap/  # SHAP interpretability analysis
│   ├── unimodal/        # Single-modal models (e.g., GBR, XGBoost, etc.)
│   └── utils/           # Utility functions
├── main.py              # Main entry file
└── requirements.txt     # Project dependencies
```

## Installation Guide

### Environment Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)
- scikit-learn
- RDKit
- pandas, numpy, matplotlib, seaborn, etc.

### Installation Steps

1. Clone the repository


2. Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset Introduction

The dataset used in this project is MemTrOC-Dataset, which contains experimental data on organic micropollutant filtration by nanofiltration/reverse osmosis membranes, mainly including:

- Membrane structure parameters: pore size, thickness, etc.
- Operating conditions: pressure, temperature, pH, etc.
- Pollutant physicochemical properties: molecular weight, molecular radius, charge, etc.
- Molecular structure information: SMILES string representation
- Target variable: rejection rate

The dataset is located at `data/processed/MemTrOC-Dataset.csv`.

## Model Architecture

This project implements three main multi-modal fusion schemes:

1. **Tabular data + molecular graph feature fusion model (GB+Graph)**
   - Uses graph neural network to extract molecular structure features
   - Performs deep fusion with traditional tabular features
   - Achieves the best prediction performance with R² of 0.9014

2. **Tabular data + molecular image feature fusion model (GB+Image)**
   - Converts molecular structure to images
   - Uses CNN to extract image features
   - Fuses with tabular features for prediction

3. **Tabular data + molecular fingerprint feature fusion model (GB+MACCS)**
   - Extracts MACCS fingerprints of molecules
   - Fuses with tabular features for prediction

The core of the model architecture lies in the dynamic integration network (DynamicNet), which combines multiple weak learners through gradient boosting strategy to achieve significant improvement in prediction accuracy.

## Usage

### 1. Single Model Training and Testing

#### Tabular data + molecular graph feature model (best performance)
```bash
python src/tableGraphTrainGPU.py --cuda
```

#### Tabular data + molecular image feature model
```bash
python src/tableImageTrain.py
```

#### Tabular data + molecular fingerprint feature model
```bash
python src/tableMACCSkeysTrain.py
```

### 2. Model Comparison

Generate comparison scatter plots of prediction results from different models:
```bash
python src/compare_models_scatter.py
```

### 3. Feature Importance Analysis

Perform global feature importance analysis using SHAP method:
```bash
cd src/interpret_shap
python featureShap.py
```

### 4. Molecular Structure Importance Visualization

Analyze and visualize the importance of each atom/functional group in specific molecular structures:
```bash
python src/molecule_feature_importance.py
```

## Result Analysis

### Key Findings

1. **Importance of molecular topological information**: Research shows that the graph structure information of molecules plays a key role in improving prediction accuracy, because it can capture the three-dimensional spatial structure and functional group distribution of molecules.

2. **Dominance of steric hindrance effect**: Global feature importance ranking reveals the dominant influence of spatial hindrance parameters such as molecular radius, molecular weight, and membrane pore size on rejection rate.

3. **Functional group recognition**: Atomic-level attribution analysis can automatically identify the core role of key functional groups (such as hydroxyl, carboxyl, amino groups, etc.) in molecule-membrane interactions.

## Interpretability Analysis

This project achieves model interpretability through two main methods:

1. **SHAP method**: Used for global feature importance ranking and feature dependency analysis, revealing the key mechanisms of membrane-pollutant interactions.

2. **Molecular structure importance visualization**: Intuitively shows the contribution of each atom/functional group in the molecule to the prediction result through color depth, helping to understand the rejection mechanism at the microscopic level.

## Typical Application Scenarios

1. **Membrane material design**: Guiding the directed design of new high-efficiency membrane materials based on the key features revealed by the model.

2. **Pollution control strategy optimization**: Predicting the pollutant rejection efficiency under different operating conditions and optimizing operating parameters.

3. **Environmental risk assessment**: Rapidly evaluating the membrane filtration behavior of new organic pollutants to provide support for environmental risk assessment.
```