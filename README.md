# 🌬️ Air Quality Forecasting - PM2.5 Prediction in Beijing
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview
This project implements a **Long Short-Term Memory (LSTM) neural network** to forecast PM2.5 air pollution levels in Beijing using historical air quality and meteorological data.  
The model predicts **hourly PM2.5 concentrations** based on 24 hours of historical data, achieving a validation RMSE of **64.15 μg/m³**.  
- **Competition**: ALU Machine Learning Techniques I Kaggle Challenge  
- **Target**: RMSE < 4000 on Kaggle Leaderboard  

We experimented with various LSTM architectures, regularization techniques, and temporal validation strategies to address distribution shift challenges.

---

## 🎯 Key Features
- **Time Series Forecasting** using LSTM networks with sequence-to-one prediction  
- **Comprehensive Data Preprocessing** with missing value imputation and feature engineering  
- **Hyperparameter Optimization** with 15+ experimental configurations  
- **Temporal Validation** strategies to address distribution shift between training (2010-2013) and test (2013-2014)  
- **Production-ready Pipeline** from data loading to Kaggle submission generation  

---

## 📊 Dataset
**Source**: Beijing PM2.5 air quality dataset with meteorological features.  
- **Training Data**: January 1, 2010 - July 2, 2013 (30,675 samples)  
- **Test Data**: July 2, 2013 - December 31, 2014 (13,148 samples)  

**Features**:  
- **Meteorological**: `DEWP` (dew point), `TEMP`, `PRES`, `Iws` (wind speed)  
- **Precipitation**: `Is` (snow), `Ir` (rain)  
- **Wind Direction**: One-hot encoded categorical variables  
- **Temporal**: `hour`, `dayofweek`, `month` (engineered features)  
- **Target**: `PM2.5` concentration (μg/m³)  

---

## 🏗️ Model Architecture
### Best LSTM Configuration
```python
Sequential([
    LSTM(64, return_sequences=True, input_shape=(24, 9), 
         kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    LSTM(32, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1)
])
```

**Hyperparameters**:  
- **Optimizer**: Adam (learning_rate=0.0005)  
- **Loss Function**: Mean Squared Error (MSE)  
- **Batch Size**: 64, **Sequence Length**: 24 hours  
- **Regularization**: L2 (λ=0.01) + Dropout (50%)  
- **Callbacks**: EarlyStopping, ReduceLROnPlateau  

---

## 📈 Experimental Results

| **Exp. #** | **Model Architecture & Key Changes** | **Parameters (LR, BS, Epochs, Reg.)** | **Validation RMSE** | **Kaggle Score** | **Notes & Key Findings** |
|------------|---------------------------------------|---------------------------------------|---------------------|------------------|--------------------------|
| 1 | Baseline LSTM (50 units) | LR=0.001, BS=32, Epochs=50, Drop=0.3 | 85.24 | - | Established baseline |
| 2 | Increased Capacity (100 units) | LR=0.001, BS=32, Epochs=50, Drop=0.3 | 82.15 | - | Slight improvement |
| 3 | Stacked LSTMs (64→32) | LR=0.001, BS=64, Epochs=50 | 78.93 | - | Captured better features |
| 4 | Higher Dropout (0.5) | LR=0.0005, BS=64 | 75.43 | - | Reduced overfitting |
| 5 | **Stacked + L2** | LR=0.0005, BS=64, Epochs=100, Drop=0.5, L2=0.01 | **64.15** | - | **Best local RMSE** |
| 6 | Simplified Model (32 units) | LR=0.0001, BS=128 | 72.18 | - | Too simple → worse |
| 9 | Time-Based Split (realistic) | Same as Exp 5 | 112.50 | - | Revealed temporal shift |
| 10 | Stacked + L2 | LR=0.0005, BS=64 | 64.15 | ~4000+ | Negative preds hurt score |
| 11 | Simplified LSTM 32 | LR=0.0005, BS=64 | 64.15 | ~3800+ | Better Kaggle performance |
| 12 | **Final Submission** | LR=0.0005, BS=64, Drop=0.5, L2=0.01 | 64.15 | **<4000** | **Target achieved** |

---

## 🎯 Key Challenges & Solutions
**1. Temporal Distribution Shift**  
- **Problem**: Test period (2013-2014) different from training (2010-2013)  
- **Solution**: Time-based validation split instead of random split  

**2. Overfitting**  
- **Problem**: Model memorized training patterns  
- **Solution**: Aggressive regularization (L2 + Dropout)  

**3. Negative Predictions**  
- **Problem**: StandardScaler produced impossible negative PM2.5 values  
- **Solution**: Clip predictions at 0 using `np.clip(predictions, 0, None)`  

**4. Data Leakage**  
- **Problem**: Traditional feature engineering used future information  
- **Solution**: Strict time-based sequence creation without look-ahead  

---

## 🚀 Quick Start
### Prerequisites
```bash
Python 3.8+
TensorFlow 2.12+
scikit-learn, pandas, numpy, matplotlib
```

### Installation & Run
```bash
# Clone repository
git clone https://github.com/Gakwaya011/Time-Series-Forecasting.git
cd Time-Series-Forecasting

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

# Run the project
jupyter notebook air_quality_forecasting.ipynb
# Or: python main.py
```

---

## 📊 File Structure
```
Time-Series-Forecasting/
├── Data/
│   ├── train.csv                 # Training data (35K+ samples)
│   ├── test.csv                  # Test data (13K+ samples)
│   └── sample_submission.csv     # Submission format
├── air_quality_forecasting.ipynb # Main analysis notebook
├── main.py                       # Main training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔮 Future Work
- **Advanced Architectures**: Bidirectional LSTM, Transformer models  
- **Feature Engineering**: Weather forecasts, holiday indicators, traffic data  
- **Ensemble Methods**: Combine multiple models for improved robustness  
- **Transfer Learning**: Pre-train on data from other cities with similar climate  
- **Real-time Deployment**: API for live air quality predictions  

---

## 👥 Author
**Christophe Gakwaya**  
- GitHub: [@Gakwaya011](https://github.com/Gakwaya011)  
- Course: Machine Learning Techniques I  
- Institution: African Leadership University