#  AI Stock Advisor using Explainable AI

An end-to-end AI-powered stock forecasting platform that combines **Hybrid Deep Learning, Explainable AI (XAI), Algorithmic Trading Strategies, and Natural Language Explanations** through an interactive dashboard.

Unlike traditional black-box forecasting systems, this project not only predicts future stock prices but also explains why predictions were made and converts them into practical Buy, Hold and Sell recommendations.

---

# Project Overview

Stock market forecasting is a challenging problem due to the dynamic and non-linear behaviour of financial markets. Traditional forecasting models often provide predictions without explaining the reasoning behind them.

This project addresses these limitations by integrating:

 Deep Learning based Forecasting

 Explainable AI

 Trading Signal Generation

 Natural Language Explanations

 Interactive Dashboard

 AI Chatbot

The project was developed as a **B.Tech Final Year Project at Jaypee Institute of Information Technology.**

---

# 🎥 Demo

A complete demonstration of the application is available in the demo video: https://github.com/devag2004/AI-stock-Advisor-using-eXplainable-AI/tree/main/AI stock advisor demo.mp4



---

#  System Architecture

<img width="674" height="581" alt="image" src="https://github.com/user-attachments/assets/ab124821-0cce-42c3-bb85-9f5a5d43b8ea" />

The overall architecture integrates multiple independent modules that work together to generate forecasting results, explain model behaviour and provide trading recommendations.

### Pipeline:

Stock Selection

↓

Historical Data Collection

↓

Technical Indicator Generation

↓

Informer Forecasting Model

↓

Explainable AI Analysis

↓

Trading Agent

↓

Natural Language Explanation

↓

Dashboard Visualisation

↓

AI Chatbot Interaction

---

#  Application Workflow

<img width="1025" height="908" alt="image" src="https://github.com/user-attachments/assets/bc10b35a-367f-47d8-a7ab-1fdc61b6601d" />

The workflow consists of the following stages:

### Step 1

User selects a stock ticker.

### Step 2

Historical OHLCV data is fetched using Yahoo Finance.

### Step 3

Technical indicators are generated.

### Step 4

Informer model predicts future prices.

### Step 5

XAI techniques analyse prediction behaviour.

### Step 6

Trading agent generates Buy/Hold/Sell signals.

### Step 7

LLM produces natural language explanations.

### Step 8

Dashboard displays results.

### Step 9

Chatbot answers user queries.

---

#  User Interaction

<img width="616" height="706" alt="image" src="https://github.com/user-attachments/assets/e8d6742f-a0e4-41d2-aca3-cda3692e2ca6" />

The system allows users to:

- Register/Login
- Select Stock Tickers
- Fetch Historical Data
- Generate Forecasts
- View Explainable AI Insights
- Generate Trading Signals
- View AI Explanations
- Interact with Chatbot

---

#  Data Model

<img width="443" height="1165" alt="image" src="https://github.com/user-attachments/assets/ccfaf68d-0b9b-4476-a243-0e6fa33d20b0" />

The data model connects:

- Stock Data
- Technical Indicators
- Forecasting Module
- XAI Module
- Trading Agent
- LLM Explanation Module
- Dashboard
- User

---

#  Interactive Dashboard
<img width="552" height="373" alt="image" src="https://github.com/user-attachments/assets/d8bdb22e-c28f-4c9a-b00d-7bf9892d46ce" />
<img width="944" height="672" alt="image" src="https://github.com/user-attachments/assets/770fe8e0-259c-4ad6-bdb3-7b83a539b377" />
<img width="905" height="648" alt="image" src="https://github.com/user-attachments/assets/37762e55-6c07-4d0f-8eeb-0bb2ca87771a" />


The dashboard provides an intuitive interface for stock analysis.

### Features:

- Stock Selection
- Historical OHLCV Data
- Company Information
- Forecasting
- Explainable AI
- Trading Signals
- Future Price Prediction

---

# Stock Forecasting

<img width="556" height="412" alt="image" src="https://github.com/user-attachments/assets/f8190146-9927-417f-b9c6-b81660b97124" />


Historical technical indicators are processed by the Informer Transformer model to generate future stock price predictions.

The dashboard visualises both actual and predicted prices for performance evaluation.

---

# Explainable AI

<img width="493" height="305" alt="image" src="https://github.com/user-attachments/assets/c43c0b54-1442-4989-b900-0bf72d47eeb7" />
<img width="463" height="337" alt="image" src="https://github.com/user-attachments/assets/eae1a64f-1927-46d5-86a1-274a1090d458" />
<img width="1432" height="896" alt="Screenshot 2026-04-07 195500" src="https://github.com/user-attachments/assets/37a2bf8e-d81b-4f87-a939-a438298bdb2c" />
<img width="465" height="304" alt="image" src="https://github.com/user-attachments/assets/497b98b2-8219-4c48-a12f-c600b1b6aa1c" />
<img width="758" height="268" alt="image" src="https://github.com/user-attachments/assets/6f2f9d36-d73b-4d6f-a472-7241c496d9e7" />


Four Explainable AI techniques are integrated:

## Integrated Gradients

Analyses feature contributions over the input sequence.

## Saliency Maps

Identifies sensitive input regions affecting predictions.

## Permutation Importance

Measures performance degradation when features are shuffled.

## SHAP

Provides feature-level contributions to model outputs.

The outputs of all methods are combined and converted into simple natural language explanations.

---

# Trading Signal Generation

Three trading strategies were investigated.

## AB=CD Harmonic Pattern Strategy

Pattern based trading approach.

## Evolution Strategy Agent

Population based optimisation.

## Neuro Evolution Agent

Neural network combined with evolutionary optimisation.

The trading modules convert forecasting outputs into practical Buy, Hold and Sell recommendations.

The results of these can be viewed from the results excel sheet.

---

# Financial Chatbot

<img width="1500" height="647" alt="Screenshot 2026-04-07 195639" src="https://github.com/user-attachments/assets/ba9ac48b-0d07-4ab6-9228-333805736be0" />
<img width="1455" height="928" alt="Screenshot 2026-04-07 195649" src="https://github.com/user-attachments/assets/e96e2906-c901-447e-bd79-908fd416b6a4" />
<img width="1443" height="1013" alt="Screenshot 2026-04-07 195702" src="https://github.com/user-attachments/assets/48204e0a-46dc-4b0b-80f9-54b86f4503b4" />


The chatbot enables users to ask questions regarding:

- Forecasts
- Technical Indicators
- Feature Importance
- Trading Signals
- Model Behaviour

Responses are generated using Ollama and a local Gemma2 language model.

---

# Research Notebook

The repository includes a complete research notebook:

```
model_xai_agent notebook.ipynb
```

The notebook documents the entire experimentation pipeline including:

- Historical Data Collection
- Technical Indicator Generation
- Six Forecasting Models
- Four XAI Techniques
- Three Trading Agents
- Model Comparison
- Final Model Selection

---

# 🏆 Forecasting Models

The following architectures were evaluated:

| Model |
|-------|
| LSTM |
| LSTM-GAN |
| Informer Transformer |
| ConvTransformer |
| Hierarchical CNN-LSTM with Attention |

The Informer Transformer demonstrated the best overall performance and was selected for deployment.

---

# 🔍 Explainable AI Methods

| Method |
|----------|
| Integrated Gradients |
| Saliency Maps |
| Permutation Importance |
| SHAP |

These methods improve transparency by identifying the indicators responsible for stock price predictions.

---

# Technologies Used

## Machine Learning

- PyTorch
- NumPy
- Pandas

## Financial Data

- Yahoo Finance
- pandas_ta

## Explainable AI

- SHAP
- Captum

## Dashboard

- Streamlit

## Language Model

- Ollama
- Gemma2

## Database

- PostgreSQL

---

# 📁 Repository Structure

```
AI-stock-Advisor-using-eXplainable-AI

│

├── app.py

├── chatbot_app.py

├── init_db.py

├── model_xai_agent notebook.ipynb

├── results forecasting agents xai (3).xlsx

├── stockapp_env.yml

├── stockapp_pip_req.txt

├── LICENSE

└── README.md
```

---

# 🚀 Installation

Clone repository:

```
git clone https://github.com/devag2004/AI-stock-Advisor-using-eXplainable-AI
```

Install dependencies:

```
pip install -r stockapp_pip_req.txt
```

or

```
conda env create -f stockapp_env.yml
```

Run dashboard:

```
streamlit run app.py
```

Run chatbot:

```
streamlit run chatbot_app.py
```

---

# 📈 Results

The proposed framework successfully combines:

 Stock Forecasting

 Explainable AI

 Trading Signal Generation

 Natural Language Explanations

 Interactive Dashboard

 AI Chatbot

into a single unified decision support platform.

---

# 🔮 Future Scope

Potential future extensions include:

- Multi-stock Portfolio Analysis
- Real-time Market Integration
- News Sentiment Analysis
- Reinforcement Learning Trading Agents
- Cloud Deployment

---


# License

This project is released under the MIT License.

See the LICENSE file for additional details.
