# Flight Price Prediction using Deep Learning

An end-to-end Deep Learning project that predicts flight ticket prices by processing multi-class features (airlines, cities, times, stops) and utilizing advanced Neural Network architectures. This project explores predictive modeling using **Artificial Neural Networks (ANNs)** implemented across **PyTorch** and **TensorFlow / Keras**.

## 🚀 Project Overview
Predicting airline ticket prices is a complex regression problem due to highly dynamic pricing strategies, categorical variations (e.g., business vs. economy class), and temporal dependencies. 

This repository implements deep learning regression pipelines to preprocess comprehensive flight datasets, manage high-cardinality categorical features, scale multi-class distributions, and train neural networks optimized via Root Mean Squared Error (RMSE).

## 📂 Repository Structure
* **`python---ANN.ipynb`**: Focuses on structured data engineering, feature pipeline design, and building baseline Sequential Artificial Neural Networks (ANN).
* **`python---PyTorch, TensorFlow.ipynb`**: A comparative implementation expanding the architectures across both major deep learning frameworks (PyTorch and TensorFlow) to evaluate training behavior and optimization differences.

## 📊 Dataset Features
The models are trained on structured datasets (`business.csv` and `economy.csv`) consisting of over 300,000 flight records. Key variables include:
* **Categorical**: `airline`, `ch_code`, `from` (Source City), `to` (Destination City), `stop` (Number of stops), `class` (Seat Class).
* **Temporal / Continuous**: `date`, `dep_time` (Departure), `arr_time` (Arrival), `time_taken` (Duration).
* **Target Variable**: `price` (Flight ticket price).

## 🛠️ Tech Stack & Libraries
* **Core Language:** Python
* **Deep Learning Frameworks:** PyTorch, TensorFlow / Keras
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Preprocessing:** Scikit-Learn (OneHotEncoder, StandardScaler, MinMaxScaler)

## ⚙️ Model Architecture & Pipeline
1. **Data Preprocessing:**
   * Cleansing and structural parsing of string metrics (e.g., duration formatting).
   * Feature encoding using tracking matrix transformations for high-cardinality values.
   * Feature scaling using standard distribution transformations to ensure stable gradient updates.
2. **Neural Network Framework:**
   * Dense, fully connected layer stacks (Linear/Dense).
   * Non-linear activation routing via **ReLU** to prevent vanishing gradients.
   * Target output tracking via a single linear neuron optimized for continuous value regression.
   * Model evaluation using performance metrics including **Root Mean Squared Error (RMSE)** and **R² (Coefficient of Determination)** score checks.

## 🚀 How to Run locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME