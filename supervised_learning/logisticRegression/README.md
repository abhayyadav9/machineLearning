# Logistic Regression Projects

## 📋 Project Overview
This folder contains two binary classification projects using Logistic Regression:
1. **Customer Churn Prediction** - Predicting whether a bank customer will leave
2. **Rock vs Mine Prediction** - Classifying sonar signals to detect rocks or mines

Both projects demonstrate the effectiveness of logistic regression in solving real-world classification problems.

---

# Project 1: Customer Churn Prediction

## 🎯 Problem Statement
Predict whether a bank customer will churn (leave the bank) based on their demographic information, account details, and banking behavior. This helps banks identify at-risk customers and take proactive retention measures.

## 📊 Dataset
**File:** `Churn_Modelling.csv`

**Key Features:**
- `CreditScore` - Customer's credit score
- `Age` - Customer's age
- `Tenure` - Number of years with the bank
- `Balance` - Account balance
- `NumOfProducts` - Number of bank products used
- `HasCrCard` - Whether customer has a credit card (0/1)
- `IsActiveMember` - Whether customer is an active member (0/1)
- `EstimatedSalary` - Customer's estimated salary
- `Exited` - Target variable (1 = Churned, 0 = Retained)

**Additional Features (not used):**
- Customer name, geography, gender

---

## 🧠 Theory: Logistic Regression

### What is Logistic Regression?
**Logistic Regression** is a supervised learning algorithm used for **binary classification** problems. Despite its name containing "regression," it's a classification algorithm that predicts the probability of an instance belonging to a particular class.

### Mathematical Foundation

**Logistic Function (Sigmoid):**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$

**Properties of Sigmoid:**
- **Range:** (0, 1) - Perfect for probabilities!
- **S-shaped curve:** Smooth transition from 0 to 1
- **Threshold:** Typically 0.5 for binary classification

**Decision Boundary:**
$$\text{Predicted Class} = \begin{cases} 
1 & \text{if } \sigma(z) \geq 0.5 \\
0 & \text{if } \sigma(z) < 0.5
\end{cases}$$

### How Does it Work?

1. **Linear Combination:** Compute weighted sum of features
   $$z = w^Tx + b$$

2. **Sigmoid Transformation:** Convert to probability
   $$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

3. **Classification:** Apply threshold (default 0.5)

4. **Training:** Minimize log loss (binary cross-entropy)
   $$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

### Log Loss (Binary Cross-Entropy)

**Intuition:**
- Penalizes confident wrong predictions heavily
- Rewards confident correct predictions
- Measures difference between predicted and actual probabilities

### Regularization

**L2 Regularization (Ridge):**
$$J(\theta) = \text{Original Loss} + \lambda\sum_{j=1}^{n}\theta_j^2$$

**Parameter C in sklearn:**
- $C = \frac{1}{\lambda}$ (inverse of regularization strength)
- **Small C:** Strong regularization (simpler model)
- **Large C:** Weak regularization (more complex model)

**Why Regularization?**
- Prevents overfitting
- Reduces coefficient magnitudes
- Improves generalization to unseen data

### Assumptions of Logistic Regression

1. **Binary Target:** Dependent variable is binary (0/1)
2. **Independence:** Observations are independent
3. **No Multicollinearity:** Features are not highly correlated
4. **Linear Relationship:** Log-odds are linearly related to features
5. **Large Sample Size:** Sufficient data for reliable estimates

### Advantages ✅
- Probabilistic interpretation (outputs probabilities)
- Efficient and fast to train
- Works well with linearly separable classes
- Less prone to overfitting with regularization
- Coefficients show feature importance and direction

### Disadvantages ❌
- Assumes linear decision boundary
- Sensitive to outliers
- Requires feature scaling
- Cannot handle non-linear relationships without feature engineering
- Struggles with imbalanced datasets

---

## 🔧 Implementation Details - Churn Prediction

### 1. Exploratory Data Analysis (EDA)

#### **Correlation Heatmap**
```python
cor = numeric_df.corr()
sns.heatmap(cor, annot=True, fmt='.1f', cmap='coolwarm', center=0)
```
**Purpose:** Identify relationships between features and target

#### **Box Plots for Feature Analysis**
```python
sns.boxplot(x='Exited', y='Age', data=df)
sns.boxplot(x='Exited', y='Balance', data=df)
sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
```

**Key Insights:**
- **Age:** Older customers tend to churn more
- **Balance:** Customers with very high or very low balance may churn
- **Salary:** Less correlation with churn

### 2. Feature Selection
Selected 8 most relevant features:
```python
features = ['Age', 'Balance', 'EstimatedSalary', 'CreditScore', 
            'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
```

### 3. Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(Y, X, test_size=0.2)
```
- 80% training data
- 20% testing data

### 4. Model Training (Basic)
```python
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

### 5. Hyperparameter Tuning with Pipeline

**Pipeline Benefits:**
- Combines preprocessing and modeling
- Prevents data leakage
- Simplifies workflow

```python
for lambda_val in np.arange(0.1, 100, 0.1):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1/lambda_val)
    )
    pipe.fit(X_train, y_train)
    test_score.append(pipe.score(X_test, y_test))
```

**Finding Optimal Lambda:**
```python
best_lambda = 0.1 * np.argmax(test_score) * 0.1
```

### 6. Model Interpretation

**Coefficients:**
```python
model.coef_
# Shows weight of each feature:
# Positive: Increases probability of churn
# Negative: Decreases probability of churn
# Large magnitude: Strong influence
```

**Feature Importance Order:**
Features with largest absolute coefficient values are most influential.

---

# Project 2: Rock vs Mine Prediction (Sonar Dataset)

## 🎯 Problem Statement
Classify sonar signals bouncing off different objects to determine whether the object is a rock (R) or a mine (M). This has applications in naval defense and underwater exploration.

## 📊 Dataset
**File:** `sonar_data.csv`

**Structure:**
- **60 Features:** Sonar signal frequencies at different angles
- **1 Target:** 'R' (Rock) or 'M' (Mine)
- **208 Samples:** Balanced dataset with roughly equal R and M

**Feature Description:**
Each feature represents the energy within a particular frequency band, integrated over a certain period of time. The transmitted sonar signal is a frequency-modulated chirp, and the reflected signals are processed to extract these 60 features.

---

## 🔧 Implementation Details - Rock vs Mine

### 1. Data Exploration
```python
df.shape  # (208, 61)
df.describe()  # Statistical summary
df[60].value_counts()  # Class distribution: R vs M
df.groupby(60).mean()  # Average feature values per class
```

### 2. Data Preparation
```python
# Separate features and target
X = df.drop(columns=60, axis=0)  # 60 sonar features
Y = df[60]                       # Target labels (R/M)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.1,      # 10% test, 90% train
    stratify=Y,          # Maintains class distribution
    random_state=1
)
```

**Why Stratify?**
- Ensures both train and test sets have similar class distributions
- Critical for small or imbalanced datasets
- Provides reliable performance estimates

### 3. Model Training
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 4. Model Evaluation

**Training Accuracy:**
```python
training_accuracy = accuracy_score(y_train, model.predict(X_train))
# Typically: ~83%
```

**Test Accuracy:**
```python
test_accuracy = accuracy_score(y_test, model.predict(X_test))
# Typically: ~76%
```

**Analysis:**
- Slight gap between train and test accuracy is normal
- ~76% test accuracy is reasonable for this challenging dataset
- No severe overfitting (gap is small)

### 5. Making Predictions

**Example Prediction:**
```python
input_data = (0.0210, 0.0121, 0.0203, ..., 0.0161, 0.0133)  # 60 features
input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1, -1)  # (1, 60) shape
prediction = model.predict(input_reshaped)

if prediction[0] == "R":
    print("Rock detected")
else:
    print("Mine detected")
```

**Why Reshape?**
- Model expects 2D array: (n_samples, n_features)
- Single prediction needs shape (1, 60)
- `.reshape(1, -1)` converts (60,) to (1, 60)

---

## 📈 Model Comparison

| Project | Dataset Size | Features | Accuracy | Challenge |
|---------|--------------|----------|----------|-----------|
| **Churn Prediction** | ~10,000 | 8 | ~80-85% | Class imbalance, multiple features |
| **Rock vs Mine** | 208 | 60 | ~76% | Small dataset, high dimensionality |

---

## 💻 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Churn Prediction
1. Ensure dataset is in `../../dataset/Churn_Modelling.csv`
2. Open `churnPredictionLogistic.ipynb`
3. Run all cells sequentially
4. View EDA plots and model performance
5. Observe hyperparameter tuning results

### Rock vs Mine Prediction
1. Ensure dataset is in `../../dataset/sonar_data.csv`
2. Open `Rock_mine_prediction.ipynb`
3. Run all cells sequentially
4. View training and test accuracy
5. Test predictions with sample data

---

## 🗂️ Project Structure
```
logisticRegression/
├── churnPredictionLogistic.ipynb    # Bank churn prediction
├── Rock_mine_prediction.ipynb       # Sonar classification
└── README.md                        # This file
```

---

## 📚 Key Learnings

### Churn Prediction
1. **EDA Importance:** Visualizations reveal important patterns (age vs churn)
2. **Feature Selection:** Choosing relevant features improves model performance
3. **Hyperparameter Tuning:** Finding optimal regularization strength is crucial
4. **Pipeline Usage:** Ensures consistent preprocessing across train/test
5. **Business Value:** Identifying churn helps target retention efforts

### Rock vs Mine
1. **Stratified Splitting:** Essential for maintaining class balance
2. **High Dimensionality:** 60 features with 208 samples (ratio ~1:3.5)
3. **Reshape Importance:** Understanding array shapes for predictions
4. **Real-World Application:** Military/defense classification problem
5. **Performance Expectations:** ~76% is good given small dataset size

---

## 🔮 Future Improvements

### Churn Prediction
- Handle class imbalance (SMOTE, class weights)
- Feature engineering (tenure/age ratio, balance/salary ratio)
- Try ensemble methods (Random Forest, XGBoost)
- Cost-sensitive learning (different costs for false positives/negatives)
- ROC curve and AUC score analysis
- Confusion matrix for detailed error analysis

### Rock vs Mine
- Cross-validation for robust performance estimates
- Feature selection (PCA, recursive feature elimination)
- Try non-linear models (SVM with RBF kernel, Neural Networks)
- Ensemble methods (Voting classifier, stacking)
- Data augmentation techniques
- Real-time classification system

---

## 📊 Key Metrics Explained

### Accuracy
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
- Simple, but misleading for imbalanced datasets

### Confusion Matrix
```
                Predicted
                R    M
Actual    R   [TN   FP]
          M   [FN   TP]
```

### Precision & Recall
$$Precision = \frac{TP}{TP + FP}$$
$$Recall = \frac{TP}{TP + FN}$$

### F1 Score
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

---

## 📖 References

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- [Pipeline and GridSearchCV](https://scikit-learn.org/stable/modules/compose.html)

---

## 👨‍💻 Author
Machine Learning Practitioner | Classification Expert

**Note:** These projects demonstrate practical applications of logistic regression in business analytics (churn) and defense systems (sonar classification).
