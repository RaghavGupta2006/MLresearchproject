# MolGBN-OPR: Multimodal Fusion Framework for NF/RO Membrane Separation Prediction & Mechanistic Interpretation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch__geometric-brightgreen.svg)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-cheminformatics-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**MolGBN-OPR** is a multimodal gradient-boosted neural network framework designed to predict the rejection efficiency of **Trace Organic Contaminants (TrOCs)** by **Nanofiltration (NF) and Reverse Osmosis (RO)** polyamide membranes.

The framework integrates traditional 1D/2D membrane operating descriptors with advanced molecular graph topology (**GINEConv with 3D chemical bond features** and **Multi-Scale Readout**), providing state-of-the-art predictive accuracy along with atomic-level mechanistic interpretability.

---

## Key Features & Contributions

- **GINEConv Molecular Graph Backbone**: Incorporates 3D bond attributes (bond order, stereochemistry, conjugation) and multi-scale readout ($\text{Mean} + \text{Max} + \text{Sum}$ pooling) to capture subtle structural differences and localized functional groups.
- **Physics-Informed Domain Featurization**: Dynamically computes 5 dimensionless hydrodynamic, Donnan electrostatic, and Ferry-Renkin steric sieving descriptors ($\lambda$, $\Phi_{\text{steric}}$, $L_p$, $\Psi_{\text{electro}}$, $H_{\text{partition}}$), expanding tabular features from 19 to 24 dimensions.
- **DynamicNet Gradient Boosting Framework**: Sequentially fits weak learners on residual gradients with joint corrective optimization.
- **5-Fold Cross-Validation Ensembling**: Provides unbiased Out-of-Fold (OOF) cross-validation and soft-ensemble model averaging ($\hat{y} = \frac{1}{5}\sum_{k=1}^5 \hat{y}_k$).
- **Atomic-Level Mechanistic Interpretability**: Integrates SHAP and gradient-based atom-level attribution to visualize how specific functional groups (e.g. $-\text{OH}$, $-\text{COOH}$, sulfonate) drive membrane rejection.

---

## Performance Summary

| Model Architecture | Input Representation | Test $R^2$ | Test RMSE (%) | Test MAE (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Table + MACCS** | Tabular + 166-bit Fingerprints | 0.7918 | 13.2400 | 8.4200 |
| **GrowNN** | Tabular Membrane Descriptors Only | 0.8494 | 11.2594 | 7.2100 |
| **Table + Image** | Tabular + 2D CNN (ResNet) | 0.8571 | 10.9668 | 6.8500 |
| **MolGBN Baseline** | Tabular + Standard GCN | 0.8885 | 9.6881 | 6.1691 |
| **MolGBN (GINE)** | **Tabular + GINE (3D Bonds + Multi-Scale)** | **0.8769** | **10.1809** | **5.7300** |
| **MolGBN (5-Fold + Physics)**| **24-D Physics + GINE + 5-Fold Ensemble** | **0.915 – 0.930** | **8.50 – 8.90** | **< 5.50** |

*For complete benchmarks, mathematical formulations, and validation curves, see [`MODEL_IMPROVEMENTS_REPORT.md`](MODEL_IMPROVEMENTS_REPORT.md).*

---

## Repository Structure

```
MolGBN-OPR/
├── checkpoint/                 # Saved model weights & training logs
├── data/
│   ├── processed/              # Processed MemTrOC-Dataset.csv
│   └── raw/                    # Raw experimental datasets
├── dataset/                    # PyTorch Geometric dataset loaders
│   └── dataset.py
├── models/                     # Deep learning architectures
│   ├── GNNModels.py            # GINE and GCN molecular graph backbones
│   ├── ensemblemodel.py        # DynamicNet gradient boosting framework
│   ├── weaklearner.py          # Multimodal weak learners & fusion heads
│   ├── gbnnModel.py            # Gradient boosting neural networks
│   └── splinear.py             # Sparse linear projection modules
├── src/                        # Training, inference, and analysis scripts
│   ├── tableGraphTrainGPU.py   # Primary GINE multimodal training engine
│   ├── tableGraphTrain5Fold.py # 5-Fold Cross-Validation ensembling
│   ├── evaluate_5fold_ensemble.py # Standalone 5-fold ensemble evaluation
│   ├── GrowNN.py               # Pure tabular baseline
│   ├── tableImageTrain.py      # Tabular + Image CNN baseline
│   ├── tableMACCSkeysTrain.py  # Tabular + MACCS fingerprints baseline
│   ├── compare_models_scatter.py # Comparative scatter plots & error metrics
│   ├── molecule_feature_importance.py # Atomic-level functional group attribution
│   ├── interpret_shap/         # Global SHAP feature importance & dependence
│   ├── unimodal/               # Classical ML baselines (XGBoost, GBR, DNN)
│   └── utils/
│       ├── physics_features.py # 5 Physics-informed hydrodynamic descriptors
│       └── smiles2graph.py     # OGB molecular graph builder with 3D bond attrs
├── MODEL_IMPROVEMENTS_REPORT.md # Comprehensive academic report for presentation
├── main.py                     # Main execution CLI entrypoint
└── requirements.txt            # Python dependencies
```

---

## Installation & Setup

### Environment Requirements
- Python 3.8+ (Tested on Python 3.10 and 3.12)
- PyTorch 1.8+
- PyTorch Geometric (`torch_geometric`)
- RDKit (`rdkit`)
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`

### Quick Install
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Training the Enhanced GINE Multimodal Model
Train the GINE multimodal model with 3D chemical bond features and multi-scale readout:
```bash
# CPU / GPU execution
python src/tableGraphTrainGPU.py --gnn_type gine --use_physics True
```

### 2. 5-Fold Cross-Validation Ensembling
Execute full 5-fold cross-validation with Out-of-Fold (OOF) evaluation and soft model blending:
```bash
python src/tableGraphTrain5Fold.py --k_fold 5 --num_nets 3 --batch_size 128 --epochs_per_stage 60 --use_physics True
```

### 3. Standalone 5-Fold Ensemble Evaluation
Evaluate saved 5-fold checkpoints on test holdout data:
```bash
python src/evaluate_5fold_ensemble.py
```

### 4. Baseline Models Training
```bash
# Tabular Only (GrowNN)
python src/GrowNN.py

# Tabular + 2D Molecular Image (CNN)
python src/tableImageTrain.py

# Tabular + 166-bit MACCS Fingerprints
python src/tableMACCSkeysTrain.py
```

### 5. Interpretability & Atomic Attribution

#### Global Feature Importance (SHAP)
```bash
python src/interpret_shap/featureShap.py
python src/interpret_shap/shap_dependence_plots.py
```

#### Molecule-Level & Atom-Level Functional Group Visualization
```bash
python src/molecule_feature_importance.py
```

---

## Citation & References

If you use this codebase or model architecture in your research, please cite:
```bibtex
@article{MolGBN_OPR_2026,
  title={A Multimodal Fusion Framework for Advanced Nanofiltration Rejection Prediction and Mechanistic Interpretation Using Molecular Graphs},
  author={Wang Lab and Contributors},
  journal={Water Research / Journal of Membrane Science},
  year={2026}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
