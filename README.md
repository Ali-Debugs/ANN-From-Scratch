# Binary Classification of Diabetes using ANN

## 📌 Project Overview
This project implements a **Binary Classification** model to predict diabetes using an **Artificial Neural Network (ANN)** built entirely from scratch. Unlike standard implementations using high-level frameworks like TensorFlow or PyTorch, this project utilizes **NumPy** to construct the neural network logic—including forward propagation, backpropagation, and gradient descent—from first principles.

The goal is to analyze health determinants such as BMI, blood pressure, and age to classify whether a patient is diabetic or not.

## 📂 Dataset
The project utilizes the **Diabetes Health Indicators Dataset (BRFSS2015)**.
* **Source:** Behavioral Risk Factor Surveillance System (BRFSS) 2015, curated by Alex Teboul on Kaggle.
* **Description:** A clean subset of the CDC's annual health survey containing responses regarding health behaviors and chronic conditions.
* **Input Features (8 selected):**
    * `HighBP`: High Blood Pressure (0 = no, 1 = yes)
    * `HighChol`: High Cholesterol (0 = no, 1 = yes)
    * `BMI`: Body Mass Index (Continuous)
    * `Smoker`: History of smoking (0 = no, 1 = yes)
    * `Stroke`: History of stroke (0 = no, 1 = yes)
    * `HeartDiseaseorAttack`: Coronary heart disease (0 = no, 1 = yes)
    * `PhysActivity`: Physical activity in past 30 days (0 = no, 1 = yes)
    * `Age`: 13-level age category (1 = 18-24 ... 13 = 80+)
* **Target Variable:**
    * `Diabetes_binary`: 0 = No Diabetes, 1 = Diabetes/Pre-diabetes

## 🛠️ Methodology & Technical Implementation

### 1. Data Preprocessing
* **Standardization:** Applied **Z-score Normalization** to continuous variables (e.g., BMI) to ensure a mean of 0 and standard deviation of 1.
* **Bias Injection:** Added a bias unit (column of 1s) to the input matrix to allow the decision boundary to shift freely.

### 2. Neural Network Architecture
* **Input Layer:** 9 Neurons (8 Features + 1 Bias unit).
* **Hidden Layer:** 4 Neurons using **Sigmoid** activation.
* **Output Layer:** 1 Neuron using **Sigmoid** activation for binary classification.

### 3. Optimization
* **Algorithm:** Gradient Descent.
* **Loss Function:** Mean Squared Error (MSE) monitoring, with a simplified gradient update `(y_hat - y)` effectively mimicking Cross-Entropy for stability.
* **Hyperparameters:**
    * Epochs: 300
    * Learning Rate: 0.1

## 📊 Results and Discussion
The model was trained over 300 epochs. The training dynamics showed:

* **Loss Curve:** The Mean Squared Error (MSE) consistently decreased, confirming that the Gradient Descent algorithm successfully optimized the weights.
* **Accuracy Curve:** The accuracy improved rapidly in the initial epochs and stabilized.

> "The results demonstrate that a simple ANN with a single hidden layer (4 neurons) is capable of learning from the standardized health indicators."

## 💻 Tech Stack
* **Language:** Python 3.13.7
* **Environment:** Jupyter Notebook
* **Libraries:**
    * `numpy`: For matrix operations and mathematical functions.
    * `pandas`: For data loading and manipulation.
    * `matplotlib`: For plotting Loss and Accuracy graphs.

## 🚀 How to Run
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Ali-Debugs/ANN-From-Scratch.git](https://github.com/Ali-Debugs/ANN-From-Scratch.git)
    ```
2.  **Navigate to the Directory:**
    ```bash
    cd ANN-From-Scratch
    ```
3.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib
    ```
4.  **Run the Notebook:**
    Open `Code.ipynb` in Jupyter Notebook or VS Code and run all cells.
