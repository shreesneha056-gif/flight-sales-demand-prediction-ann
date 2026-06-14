# Flight Price Prediction using Artificial Neural Networks (ANN)

An end-to-end Deep Learning regression project that predicts airline ticket prices using an Artificial Neural Network implemented in Keras/TensorFlow. The pipeline processes raw structured flight logs, resolves multi-collinearity via Variance Inflation Factor (VIF), handles high-cardinality feature sets, and scales variables for neural network optimization.

## 🚀 Project Overview
Dynamic ticket pricing models depend heavily on multiple factors such as route nodes, service class, and specific carrier assignments. This project treats price prediction as a pure regression challenge, building a robust dense neural network sequence to minimize continuous target error distributions over large-scale aviation datasets.

## 📂 Repository Structure
* **`python---ANN.ipynb`**: The primary Jupyter Notebook containing data integration pipelines, multicollinearity analysis, feature engineering, and the full Keras sequential neural network implementation.

## 📊 Dataset & Pipeline Overview
The notebook handles structured data combinations from two source spaces (`business.csv` and `economy.csv`) mapping out over 300,000 distinct flight records.

### 1. Data Cleaning & Engineering
* **Temporal Transformations**: Converts raw string entries into standard datetime properties (`date`, `dep_time`).
* **Duration Formatting**: Uses regular expressions (`regex`) to parse compound string time representations (e.g., `02h 00m`) into quantitative numerical intervals (Total Minutes).
* **Target Refinement**: Strips regional formatting and currency markers from price string properties before mapping them into integer types.

### 2. Feature Filtering via Variance Inflation Factor (VIF)
* To prevent feature inflation errors, categorical columns (`airline`, `ch_code`, `from`, `stop`, `to`, `class`) are parsed via tracking dummy variables (`pd.get_dummies`).
* A structural **VIF Matrix** is generated to audit multicollinearity across all 66 transformed columns.
* Highly collinear parameters are filtered to focus on standard low-variance indicators (such as `class_economy`, selective source/destination hubs, and structural flight identifiers).

### 3. Deep Learning Architecture
The model relies on a dense `Sequential` layout structured as follows:
* **Input / Hidden Layer 1**: 50 Units, Rectified Linear Unit (`ReLU`) activation.
* **Hidden Layer 2**: 30 Units, Rectified Linear Unit (`ReLU`) activation.
* **Output Layer**: 1 Unit mapping the regression prediction bounds.
* **Compilation**: Optimized using the **Adam** optimizer, tracked using Mean Absolute Error (`mae`), and minimized via Mean Squared Error (`loss='mse'`).

## 🛠️ Tech Stack & Environment
* **Core Language:** Python
* **Deep Learning Framework:** Keras / TensorFlow
* **Data Diagnostics:** Statsmodels (Variance Inflation Factor)
* **Data Wrangling & Processing:** Pandas, NumPy, Scikit-Learn (StandardScaler, train_test_split)
* **Visualization:** Seaborn, Matplotlib

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME