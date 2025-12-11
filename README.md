<div align="center">

# 🎬 From Buzz to Blockbuster
### Predicting Movie Revenue via Hybrid Machine Learning & Sentiment Analysis

[![Paper](https://img.shields.io/badge/IEEE-ICMSCI%202025-blue?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/10894031)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-BERT%20%2B%20CatBoost-yellow?style=for-the-badge&logo=huggingface&logoColor=black)]()

> **"Data-Driven Decisions for the Box Office."**
> A hybrid framework integrating **BERT embeddings**, **CatBoost**, and **LSTM** to forecast financial success with **97.8% Classification Accuracy**.

[View Paper](#-citation) • [Architecture](#-system-architecture) • [Benchmarks](#-performance--benchmarks) • [Installation](#-getting-started)

</div>

---

## 🧐 Abstract

The entertainment industry is volatile. Traditional forecasting relies on static historical data (Cast, Budget, Genre), often missing the "pulse" of the audience. **This project bridges that gap.**

We propose a **Hybrid Predictive System** that fuses quantitative metrics with qualitative real-time data. By leveraging **BERT embeddings** for context-aware sentiment analysis of social media (Twitter, YouTube) and **LSTM** for time-series trend forecasting, we provide actionable insights for studios and investors to mitigate financial risk.

### 🌟 Key Capabilities
* **🧠 Context-Aware Sentiment:** Uses **BERT** and **VADER** to analyze social buzz and **LDA** for topic modeling.
* **📈 Precision Regression:** **CatBoost Regressor** predicts opening weekend revenue with **96% R²**.
* **🏷️ Success Classification:** Categorizes films as *Hit*, *Average*, or *Flop* using **Random Forest**.
* **⏳ Trend Forecasting:** **LSTM** networks analyze temporal engagement patterns.

---

## 🏗️ System Architecture

Our pipeline processes unstructured social data alongside structured movie metadata to feed a multi-model ensemble.

```mermaid
graph TD
    subgraph Data Collection
    A[Social Media] -->|Twitter/YouTube| C(Unstructured Text)
    B[IMDb / TMDb] -->|Metadata| D(Structured Data)
    end

    subgraph Preprocessing
    C --> E[Cleaning & Normalization]
    E --> F[BERT Embeddings]
    E --> G[LDA Topic Modeling]
    end

    subgraph Hybrid Modeling
    D & F & G --> H{Feature Fusion}
    H --> I[CatBoost Regressor]
    H --> J[Random Forest Classifier]
    H --> K[LSTM Forecaster]
    end

    subgraph Output
    I --> L[Revenue Prediction]
    J --> M[Hit/Flop Verdict]
    K --> N[Trend Analysis]
    end
