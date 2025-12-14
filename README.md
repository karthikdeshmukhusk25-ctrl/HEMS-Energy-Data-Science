# Home Energy Management System (HEMS)
**Course:** ITS8080 | **Topic:** Digital Transformation of Energy

A data-driven workflow for a smart Home Energy Management System that analyzes historical electricity demand, solar generation, and market prices to build forecasting models and optimize battery storage control.

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)




## Executive Summary

This project develops a complete data science pipeline for predicting household electricity demand and optimizing a home battery system. By leveraging machine learning forecasting models, the system aims to minimize electricity costs and maximize renewable energy self-consumption.

The workflow follows the **Data Science Lifecycle**:
1. **Exploration:** Understanding data structure and quality
2. **Cleaning:** Handling missing values in solar generation data
3. **Engineering:** Creating time-based and weather-based features
4. **Modelling:** Comparing Statistical (ARIMA) vs. Machine Learning (XGBoost) approaches
5. **Deployment:** Using the best model to optimize battery storage system

## Key Findings

- **Best Model Performance:** XGBoost achieved **15.24% Normalized RMSE**, outperforming ARIMA (16.27%) and Naïve baseline (35.23%)
- **Data Quality:** Successfully handled Missing Not At Random (MNAR) solar data using Multivariate KNN Imputation
- **Economic Impact:** Battery optimization demonstrated:
  - Daily cost: **€0.28** (low solar scenario)
  - Daily profit: **€-0.08** (high solar scenario)

## Dataset

The project uses hourly observations from `train_test.csv` containing:
- **Demand** (kW): Electricity consumption
- **Price** (€/kWh): Market electricity price
- **PV Generation** (kW): Solar panel output (including pv_mod1, pv_mod2, pv_mod3)
- **Weather Parameters:**
  - Temperature
  - Pressure
  - Cloud cover (total, low, mid, high)
  - Wind speed
  - Solar radiation (shortwave, direct, diffuse, direct normal irradiance)

**Dataset Period:** July 1, 2013 to June 30, 2014 (8,759 hourly observations)

## Methodology

### 1. Data Exploration & Cleaning
- Converted timestamps to datetime objects
- Visualized time series patterns (1-week sample)
- Identified missing values in solar modules using heatmaps
- Applied **KNN Imputation** (n_neighbors=5) using temperature and radiation features

### 2. Feature Engineering
- **Transformations:** Log transformation (`np.log1p`) to normalize right-skewed demand data
- **Time-based Features:**
  - Month, DayOfWeek, Hour
  - Is_Weekend flag
- **Weather-based Features:**
  - Heating_Intensity: `max(0, 18°C - Temperature)` to model heating demand
- **Lag Features:** Created temporal lags (1h, 2h, 3h, 24h, 48h, 168h) for ML model

### 3. Time Series Analysis
- Performed additive seasonal decomposition
- Identified strong daily (24-hour) seasonality patterns
- Created visualizations:
  - Observed data, trend, seasonality, and residuals
  - Daily heatmap showing hour-of-day patterns
  - Typical daily profile (aggregated average)

### 4. Forecasting Models

#### ARIMA (2,1,2)
- Applied first-order differencing to achieve stationarity (ADF test: p-value = 0.0000)
- Walk-forward validation on last 7 days (168 hours)
- **Performance:** 16.27% Normalized RMSE

#### XGBoost Regressor
- Hyperparameters:
  - n_estimators: 1000
  - learning_rate: 0.05
  - max_depth: 5
  - early_stopping_rounds: 50
- Utilizes lag features, time features, and exogenous variables (price, weather)
- **Performance:** 15.24% Normalized RMSE (Winner)

### 5. Battery Optimization Algorithm
- Simulated 10 kWh battery system with 5 kW charge/discharge rate
- **Control Strategy:**
  - Charge when: Solar surplus exists OR prices are below average
  - Discharge when: Demand exceeds solar AND prices are above average
  - Buy from grid when: Prices are low (no discharge)
- Initial State of Charge (SOC): 5 kWh (50% capacity)

## Results

### Model Comparison
| Model | Normalized RMSE |
|-------|----------------|
| Naïve Baseline (Persistence) | 35.23% |
| ARIMA (2,1,2) | 16.27% |
| **XGBoost** | **15.24%** |

### Economic Optimization
- **Low PV Scenario:** Total daily cost = €0.28
- **High PV Scenario:** Total daily profit = €-0.08

## Project Structure

```
.
├── energyds.ipynb          # Main Jupyter notebook with complete analysis
├── README.md               # This file
└── DataSet_ToUSE/         # Dataset directory (not included in repo)
    ├── train_test.csv      # Training and test data
    ├── forecast.csv        # Future forecast data
    └── optimisation.csv    # Battery optimization scenario data
```

##  Dependencies

```python
# Data Manipulation
pandas
numpy

# Visualization
matplotlib
seaborn

# Statistical Modelling
statsmodels

# Machine Learning
xgboost
scikit-learn
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn statsmodels xgboost scikit-learn
```

### Data Dictionary
The dataset consists of hourly smart meter readings with the following key features:

| Column | Description | Unit |
| :--- | :--- | :--- |
| `timestamp` | Date and time of observation | DateTime |
| `Demand` | Household electricity consumption | kW |
| `Price` | Electricity market price | €/kWh |
| `pv` | Solar power generation | kW |
| `Temperature` | Ambient temperature | °C |
| `direct_radiation` | Solar irradiance | W/m² |

## Usage

1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt  # If you create one
   ```

2. **Prepare Data:**
   - Place your dataset in `DataSet_ToUSE/` directory
   - Ensure files: `train_test.csv`, `forecast.csv`, `optimisation.csv`

3. **Run Analysis:**
   - Open `energyds.ipynb` in Jupyter Notebook/Lab
   - Execute cells sequentially

4. **Key Outputs:**
   - Statistical summaries and visualizations
   - Model performance metrics
   - 7-day future demand forecast
   - Battery state-of-charge profile

## Tasks Completed

- ✅ Task 1: Data Inspection & Loading
- ✅ Task 3: Visualization & Statistical Summaries
- ✅ Task 4: Data Cleaning (KNN Imputation)
- ✅ Task 5: Feature Engineering
- ✅ Task 6: Time Series Decomposition
- ✅ Task 7: Statistical Modelling (ARIMA)
- ✅ Task 8: Machine Learning (XGBoost)
- ✅ Task 9: Model Comparison
- ✅ Task 10: Forecasting Pipeline
- ✅ Task 11: Optimal Battery Control
- ✅ Task 12: Conclusions

## Insights

1. **Feature Importance:** Lag features (especially 24h and 168h) are the strongest predictors, capturing daily and weekly patterns.

2. **Model Selection:** XGBoost outperforms ARIMA due to its ability to capture non-linear relationships between demand, weather, and price variables.

3. **Economic Value:** Intelligent battery control can turn a high solar scenario into a profitable operation, demonstrating the real-world value of accurate demand forecasting.

4. **Data Quality:** Multivariate imputation using correlated features (temperature, radiation) successfully reconstructed missing solar data without significant information loss.

## Future Works
- **Deep Learning:** Implement LSTM or Transformer-based models to capture longer temporal dependencies.

- **Real-Time Deployment:** Integrate the model with a live Home Assistant API.

- **Grid Constraint Awareness:** Add battery degradation costs and grid export limits to the optimization algorithm.

## Technical Notes

- **Stationarity:** Demand data required first-order differencing for ARIMA modeling
- **Missing Data:** Solar module data showed contiguous missing blocks (MNAR), handled with KNN imputation
- **Validation:** Walk-forward validation used to simulate real-world forecasting conditions
- **Feature Scaling:** Log transformation applied to normalize demand distribution

## Acknowledgments
- **TalTech (Tallinn University of Technology):** Energy Data Science Program  

- **Dataset:** Provided by course instructors

- **Inspiration:** Smart grid and renewable energy research community

- **Tools:** Open-source Python ecosystem


EuroTeq: Exchange Program provided in collaboration with Technical University of Denmark.

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Statsmodels Time Series Analysis: https://www.statsmodels.org/
- Seasonal Decomposition: Additive model with 24-hour period
- Course Materials, ITS8080 Digital Transformation of Energy, 2025.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.

## Author

Karthik Deshmukh | S250174

## License

This project is part of an academic course assignment (ITS8080).

---

**Note:** This project demonstrates a complete data science workflow from raw data to actionable insights, showcasing both statistical and machine learning approaches for time series forecasting in energy management applications.


