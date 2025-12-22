# Decision Tree - Crop Recommendation System

## 📋 Project Overview
This project implements a **Decision Tree Classifier** to recommend the most suitable crop based on soil and environmental conditions. The model uses both traditional Decision Tree and XGBoost algorithms to predict crop types with high accuracy.

## 🎯 Problem Statement
Given soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall data, predict which crop will be most suitable for cultivation. This helps farmers make data-driven decisions for optimal crop selection.

## 📊 Dataset
**File:** `Crop_recommendation.csv`

**Features:**
- `N` - Nitrogen content in soil
- `P` - Phosphorus content in soil  
- `K` - Potassium content in soil
- `temperature` - Temperature in Celsius
- `humidity` - Relative humidity in percentage
- `ph` - pH value of soil
- `rainfall` - Rainfall in mm
- `label` - Target variable (Crop type)

The dataset contains multiple crop types with balanced distribution across all classes.

---

## 🧠 Theory: Decision Tree Algorithm

### What is a Decision Tree?
A **Decision Tree** is a supervised learning algorithm used for both classification and regression tasks. It creates a tree-like model of decisions based on feature values, where:
- **Root Node** represents the entire dataset
- **Internal Nodes** represent feature tests
- **Branches** represent decision rules
- **Leaf Nodes** represent final predictions

### How Does it Work?

1. **Splitting:** The algorithm selects the best feature to split the data using impurity measures
2. **Decision Making:** At each node, it asks a yes/no question about a feature
3. **Recursive Process:** This continues until a stopping criterion is met
4. **Prediction:** New data follows the tree path to reach a leaf node for classification

### Key Concepts

#### 1. **Splitting Criteria**

**Gini Index (Gini Impurity):**
- Measures the probability of incorrectly classifying a randomly chosen element
- Formula: $Gini = 1 - \sum_{i=1}^{n} p_i^2$
- Where $p_i$ is the probability of class $i$
- **Range:** 0 (pure) to 0.5 (maximum impurity for binary classification)
- **Lower is better:** Perfect classification = 0

**Entropy (Information Gain):**
- Measures the randomness or uncertainty in the dataset
- Formula: $Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i)$
- **Range:** 0 (pure) to 1 (maximum uncertainty for binary classification)
- **Lower is better:** Pure node = 0

**Information Gain:**
- Reduction in entropy after splitting
- Formula: $IG = Entropy_{parent} - \sum \frac{n_j}{n} \times Entropy_{child_j}$
- **Higher is better:** More information gained from the split

#### 2. **Hyperparameters**

- **`max_depth`:** Maximum depth of the tree (prevents overfitting)
- **`min_samples_split`:** Minimum samples required to split a node
- **`min_samples_leaf`:** Minimum samples required at leaf nodes
- **`criterion`:** Splitting criterion ('gini' or 'entropy')

### Advantages ✅
- Easy to understand and interpret
- Handles both numerical and categorical data
- Requires minimal data preprocessing
- Can capture non-linear relationships
- Feature importance is easily extracted

### Disadvantages ❌
- Prone to overfitting (especially deep trees)
- Sensitive to small variations in data
- Can create biased trees with imbalanced datasets
- Not optimal for continuous/smooth decision boundaries

---

## 🚀 XGBoost (eXtreme Gradient Boosting)

### What is XGBoost?
**XGBoost** is an advanced implementation of gradient boosted decision trees designed for speed and performance. It builds multiple weak learners (decision trees) sequentially, where each new tree corrects errors made by previous trees.

### How Does it Work?

1. **Initial Prediction:** Start with a simple prediction (often the mean)
2. **Calculate Residuals:** Find the errors of current predictions
3. **Build Tree:** Create a new tree to predict these residuals
4. **Update Model:** Add the new tree to the ensemble with a learning rate
5. **Repeat:** Continue until no improvement or max iterations reached

### Key Concepts

**Gradient Boosting:**
- Uses gradient descent to minimize loss function
- Each tree is built to reduce the loss of the previous ensemble
- Combines weak learners to create a strong learner

**Regularization:**
- L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting
- Penalizes complex models

**Key Hyperparameters:**
- **`n_estimators`:** Number of trees to build
- **`max_depth`:** Maximum depth of each tree
- **`learning_rate`:** Step size for weight updates (smaller = more conservative)

### Why XGBoost?
- **High Performance:** Often wins Kaggle competitions
- **Handles Missing Values:** Automatically learns best direction for missing data
- **Regularization:** Built-in L1 and L2 regularization
- **Parallel Processing:** Fast training through parallelization
- **Cross-Validation:** Built-in cross-validation at each iteration

---

## 🔧 Implementation Details

### 1. Data Preprocessing
```python
# Load dataset
data = pd.read_csv("../../dataset/Crop_recommendation.csv")

# Feature Selection
X = data.iloc[:,:-1]  # All features except target
Y = data["label"]      # Target variable

# Handle Multicollinearity
# Removed 'P' (Phosphorus) due to high correlation with 'N' (Nitrogen)
X = X.drop(columns=["P"])

# Feature Scaling (Standardization)
# Converts features to have mean=0 and std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
```

**Why Standardization?**
- Makes features comparable in scale
- Improves convergence speed
- Some algorithms (like XGBoost) benefit from normalized features

### 2. Model Training with GridSearchCV

**Grid Search:** Exhaustive search over specified parameter values to find the best combination.

```python
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

**Cross-Validation (K-Fold, k=5):**
- Splits data into 5 equal parts
- Trains on 4 parts, validates on 1 part
- Repeats 5 times with different validation sets
- Averages performance metrics

### 3. Model Evaluation

**Metrics Used:**

**Accuracy:**
- Formula: $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- Percentage of correct predictions

**F1 Score:**
- Formula: $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- Harmonic mean of precision and recall
- Good for imbalanced datasets
- **Weighted F1:** Averages F1 scores weighted by class support

---

## 📈 Results

### Decision Tree Classifier
- **Best Parameters:** Found through GridSearchCV
- **Accuracy:** ~99% (typical for this dataset)
- **F1 Score:** High weighted F1 score indicating excellent performance

### XGBoost Classifier
- **Best Parameters:** Optimized max_depth, n_estimators, and learning_rate
- **Accuracy:** ~99.5% (typically outperforms basic Decision Tree)
- **Performance:** More robust and generalizable

---

## 💻 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Execution
1. Ensure dataset is in `../../dataset/Crop_recommendation.csv`
2. Open `decision_tree.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. View results and model performance metrics

---

## 🗂️ Project Structure
```
DecisionTree/
├── decision_tree.ipynb      # Main implementation notebook
└── README.md                # This file
```

---

## 📚 Key Learnings

1. **Feature Engineering:** Identifying and removing multicollinear features improves model performance
2. **Hyperparameter Tuning:** GridSearchCV helps find optimal parameters systematically
3. **Algorithm Comparison:** XGBoost generally outperforms vanilla Decision Trees
4. **Cross-Validation:** Essential for reliable performance estimates
5. **Data Scaling:** StandardScaler normalizes features for better convergence

---

## 🔮 Future Improvements

- Implement Random Forest for comparison
- Add feature importance visualization
- Create confusion matrix for detailed error analysis
- Deploy model as web application for farmer accessibility
- Include cost-benefit analysis for different crops
- Add real-time soil testing integration

---

## 📖 References

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/grid_search.html)

---

## 👨‍💻 Author
Machine Learning Practitioner | Agricultural Analytics

**Note:** This project demonstrates the practical application of decision tree algorithms in agricultural decision-making systems.
