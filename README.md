# 🦅 Alpha Hunter — Neuroevolution Trading System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**Alpha Hunter** is a state-of-the-art trading framework designed to discover and exploit persistent alpha in financial markets using **Neuroevolution**. The system evolves specialized neural networks through Genetic Algorithms (GA) to generate non-correlated returns against the SP500 benchmark.

---

## 🚀 Pipeline Architecture

The system operates through a cascaded 4-phase optimization pipeline to ensure statistical significance and prevent over-fitting:

### 🔬 Phase 1: Metric Meta-Optimization
Optimizes the **Fitness Function** weights by maximizing the Spearman correlation between In-Sample (IS) ranking and Out-of-Sample (OOS) alpha performance.
- *Goal*: Find a metric that actually predicts future performance.

### ⚙️ Phase 2: Bayesian Hyperparameter Tuning
Uses **Optuna** to calibrate model architecture (hidden layers), GA evolution parameters (mutation rate, elite ratio), and execution policies (conviction thresholds).
- *Goal*: Stabilize the evolutionary process and risk management.

### 🏆 Phase 3: Walk-Forward (The Final Exam)
Executes a multi-year historical backtest with strict non-overlapping windows. Implements **Monte Carlo Bootstrap** to calculate the **P-Value** (Probability of Alpha being Zero).
- *Goal*: Statistical certification of the strategy.

### 📡 Phase 4: Production
Assembles the winning committee of models and generates real-time trading signals for the current market regime.

---

## 🛠️ Core Technology Stack

- **Engine**: PyTorch (Vectorized GA evaluation with full MPS/Metal & CUDA support).
- **Optimization**: Bayesian Search (Optuna) + Evolutionary Strategies.
- **Alpha v5.0 Features**: 94 technical and institutional features including:
  - **Rolling Mahalanobis Distance** (Regime detection).
  - **Information Ratio Distribution**.
  - **Eigenportfolio Exposure** (PCA-based factor decomposition).
- **Risk Management**: Dynamic Slippage modeling, Hybrid Fear (VIX + Trend) detection, and Rebalancing Drift control.

---

## 📦 Project Structure

```bash
├── analisis/           # Pipeline phases (F1-F4) and Master Runner
├── aux/                # Data management, slippage, and feature engineering
├── modelo/             # Neural architectures and fitness logic
├── tests/              # Integrity and anti-leakage test suite
├── config_example.json # Template config with baseline parameters
├── motor_ga.py         # The core evolutionary engine
└── master_runner.py    # Orchestrator of the full pipeline
```

---

## 🚦 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup configuration**:
   Clone the template configuration file:
   ```bash
   cp config_example.json config.json
   ```
   *(Note: The optimal production configurations and the trained weights within `modelos_produccion/` are omitted from the public repository to safeguard the final execution edge).*

3. **Run the baseline pipeline**:
   ```bash
   python analisis/master_runner.py
   ```

4. **Monitor Progress**:
   The system generates detailed reports in `ejecuciones/` including equity curves, diagnostic JSONs, and P-Value attribution charts.

---

## 📊 Performance & Diagnostics

The framework is designed to prioritize **Statistical Sovereignty**. Every execution is tracked with a deterministic `config.json` state, ensuring that results are reproducible and free from data leakage.

> [!IMPORTANT]
> The system targets a **P-Value < 0.05**. If the Walk-Forward results in a higher P-Value, the system identifies the failure regime (e.g., Bear market non-adaptation) to allow for iterative refinement of the institutional features.

---

**Developed with 💙 for Advanced Agentic Trading.**
