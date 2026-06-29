# 🦅 Alpha Hunter — Neuroevolution Trading SDK

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**Alpha Hunter** is an academic reference SDK and quantitative library designed to implement neuroevolutionary strategies in financial markets. This repository contains the structural modules, data engineering pipelines, mathematical consensus models, and statistical validation engines described in the Work.

*Note: The core execution orchester scripts (`motor_ga.py`, `produccion.py`), search history databases (`optuna_trading.db`), and trained neural weights (`modelos_produccion/`) are excluded from this public repository to safeguard the strategy's operational edge in live markets.*

---

## 🚀 Architecture and Modules Included

The library provides the structural components of the 4-phase optimization pipeline:

### 🔬 Feature Engineering & Ingestion (`aux/`, `eda/`)
Implements dynamic data loading, rolling Z-score scaling, and technical/institutional signal processing, including Eigenportfolios (PCA) and Regime Analysis (Rolling Mahalanobis).

### ⚙️ Neural Consensus (`modelo/`)
Defines the **Triunvirato de Expertos** architecture (3 specialized MLPs mapping short, medium, and long-term features) and the L1-Norm / Sparsity consensus mechanism.

### 🏆 Walk-Forward & Bootstrap (`analisis/`)
Implements the deslizante Walk-Forward evaluation framework and the vectorised **Circular Block Bootstrap** Monte Carlo algorithm (P-Value attribution).

---

## 🛠️ Core Technology Stack

- **Engine**: PyTorch (Vectorized evaluation with full MPS/Metal & CUDA support).
- **Consensus**: Democracy-based L1-norm portfolio weights allocation.
- **Features**: 85 technical and institutional features (VIX, ^TNX, Momentum, Risk).

---

## 📦 Project Structure

```bash
├── analisis/           # Pipeline phases, Walk-Forward, and Bootstrap Monte Carlo
├── aux/                # Data management, dynamic slippage, and feature engineering
├── eda/                # Exploratory Data Analysis, PCA, and regime characterization
├── modelo/             # Neural architectures, L1 consensus, and fitness logic
├── tests/              # Integrity and anti-leakage test suite
└── config_example.json # Baseline configuration parameters template
```

---

## 🚦 Integration & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup configuration**:
   Clone the template configuration file to customize parameters:
   ```bash
   cp config_example.json config.json
   ```

3. **Verify structure**:
   Run the integrity and feature extraction tests to verify environment compatibility:
   ```bash
   pytest tests/
   ```

---

**Developed with 💙 for Advanced Agentic Trading.**
