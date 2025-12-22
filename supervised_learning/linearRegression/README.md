# Linear Regression Projects

## 📋 Project Overview
This folder contains two comprehensive Linear Regression projects:
1. **House Price Prediction** - Predicting rental prices based on property features
2. **Pattern Plotting with Linear Regression** - Understanding polynomial regression through synthetic data

Both projects demonstrate the power and versatility of linear regression in solving real-world and theoretical problems.

---

# Project 1: House Rent Price Prediction

## 🎯 Problem Statement
Predict the rental price of houses based on various features like location, size, furnishing status, number of bedrooms, bathrooms, floor level, and tenant preferences.

## 📊 Dataset
**File:** `House_Rent_Dataset.csv`

**Key Features:**
- `BHK` - Number of bedrooms
- `Rent` - Monthly rent (Target variable)
- `Size` - Size of the house in square feet
- `Floor` - Floor number and total floors (e.g., "Ground out of 2")
- `Area Type` - Type of area (Super Area, Carpet Area, Built Area)
- `Area Locality` - Specific location within the city
- `City` - City name
- `Furnishing Status` - Furnished, Semi-Furnished, or Unfurnished
- `Tenant Preferred` - Preferred tenant type (Bachelors, Family, etc.)
- `Bathroom` - Number of bathrooms
- `Point of Contact` - Contact type (Owner, Agent, etc.)
- `Posted On` - Date when the listing was posted

---

## 🧠 Theory: Linear Regression

### What is Linear Regression?
**Linear Regression** is a supervised learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.

### Mathematical Foundation

**Simple Linear Regression:**
$$y = \beta_0 + \beta_1x + \epsilon$$

**Multiple Linear Regression:**
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

Where:
- $y$ = Dependent variable (target)
- $x_i$ = Independent variables (features)
- $\beta_0$ = Intercept (bias term)
- $\beta_i$ = Coefficients (weights)
- $\epsilon$ = Error term (residuals)

### How Does it Work?

1. **Hypothesis Function:** Predicts output as a linear combination of inputs
2. **Cost Function (MSE):** Measures prediction error
   $$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^{(i)}) - y^{(i)})^2$$
3. **Optimization:** Uses gradient descent or normal equation to minimize cost
4. **Model Parameters:** Learns optimal $\beta$ values that minimize error

### Assumptions of Linear Regression

1. **Linearity:** Linear relationship between features and target
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of residuals
4. **Normality:** Residuals are normally distributed
5. **No Multicollinearity:** Features are not highly correlated

### Advantages ✅
- Simple and interpretable
- Fast to train and predict
- Works well with linearly separable data
- Provides feature importance through coefficients
- Good baseline model

### Disadvantages ❌
- Assumes linear relationships
- Sensitive to outliers
- Can underfit complex patterns
- Affected by multicollinearity
- Requires feature scaling for best performance

---

## 🔧 Implementation Details - House Price Prediction

### 1. Data Preprocessing

#### **Column Cleaning**
```python
df.columns = df.columns.str.strip()  # Remove extra spaces
```

#### **Floor Processing**
Complex parsing of floor information:
- Extracts floor number (0 for Ground, -1 for Basement)
- Extracts total floors in building
- Creates two new features: `floor_number` and `total_floors`

Example transformations:
- "Ground out of 3" → floor_number=0, total_floors=3
- "5 out of 10" → floor_number=5, total_floors=10
- "Basement" → floor_number=-1

#### **Date Feature Engineering**
```python
df[date_col] = pd.to_datetime(df[date_col])
df['posted_year'] = df[date_col].dt.year
df['posted_month'] = df[date_col].dt.month
df['posted_day'] = df[date_col].dt.day
```

**Why?** Temporal features can capture market trends and seasonal variations.

#### **One-Hot Encoding**
Converts categorical variables into binary columns:
```python
categorical_columns = ['Area Type', 'Area Locality', 'City', 
                       'Furnishing Status', 'Tenant Preferred', 
                       'Point of Contact']
df_final = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
```

**Example:**
```
City: Mumbai → City_Mumbai=1, City_Delhi=0, City_Bangalore=0
```

### 2. Feature Scaling

**StandardScaler:** Standardizes features to have mean=0 and variance=1
$$z = \frac{x - \mu}{\sigma}$$

**Why Scaling?**
- Features have different ranges (e.g., BHK: 1-5, Size: 100-5000)
- Ensures all features contribute equally
- Improves gradient descent convergence
- Required for algorithms sensitive to feature magnitudes

### 3. Model Training
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

### 4. Model Evaluation

**R² Score (Coefficient of Determination):**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- **Range:** 0 to 1 (can be negative for very poor models)
- **Interpretation:** Proportion of variance explained by the model
- **0.7-0.8:** Good model
- **0.8-0.9:** Very good model
- **>0.9:** Excellent model (or possible overfitting)

**Visualization:**
- Scatter plot of Actual vs Predicted values
- Perfect predictions would lie on the diagonal line

---

# Project 2: Pattern Plotting with Polynomial Regression

## 🎯 Problem Statement
Understand how polynomial features can help linear regression capture non-linear patterns in data. Generate synthetic data with complex polynomial relationships and fit models of increasing complexity.

---

## 🧠 Theory: Polynomial Regression

### What is Polynomial Regression?
**Polynomial Regression** extends linear regression by adding polynomial features, allowing the model to capture non-linear relationships while still using linear regression techniques.

### Mathematical Foundation

**Polynomial of Degree n:**
$$y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + ... + \beta_nx^n + \epsilon$$

**Key Insight:** This is still linear in the parameters ($\beta$), so we use Linear Regression after feature transformation!

### How Does it Work?

1. **Feature Transformation:** Create polynomial features from original features
   - Original: $x$
   - Degree 2: $x, x^2$
   - Degree 3: $x, x^2, x^3$
   
2. **Apply Linear Regression:** Fit linear model on transformed features

3. **Prediction:** Use the learned coefficients on polynomial features

### Polynomial Features

**Example with degree=2 and 2 original features:**
```
Original: [x₁, x₂]
Transformed: [1, x₁, x₂, x₁², x₁x₂, x₂²]
```

### Bias-Variance Tradeoff

**Underfitting (High Bias):**
- Model too simple (low degree polynomial)
- Cannot capture data patterns
- High training and test error

**Optimal Fit:**
- Model complexity matches data complexity
- Low training and test error
- Good generalization

**Overfitting (High Variance):**
- Model too complex (very high degree polynomial)
- Fits training data perfectly, including noise
- Low training error, high test error
- Poor generalization

### Choosing the Right Degree

**Methods:**
1. **Visual Inspection:** Plot predictions for different degrees
2. **Cross-Validation:** Use validation set performance
3. **Regularization:** Add L1/L2 penalties to prevent overfitting
4. **Information Criteria:** AIC, BIC for model selection

---

## 🔧 Implementation Details - Pattern Plotting

### 1. Synthetic Data Generation

```python
X = np.random.rand(50, 1)
Y = 1.3*(X**8) - 1.4*(X**7) - 0.5*(X**6) + 0.4*(X**5) - 
    0.3*(X**4) + 0.05*(X**3) + 1.2*(X**2) - 0.6*X + 
    0.1*np.random.rand(50, 1)
```

**Complex 8th-degree polynomial** with added noise to simulate real-world data.

### 2. Model Training with Different Degrees

The code iterates through polynomial degrees 1 to 4:

```python
for degree in range(1, 5):
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    # Standardize
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    # Train
    model = LinearRegression()
    model.fit(X_poly_scaled, Y)
    
    # Predict and visualize
    predictions = model.predict(X_poly_scaled)
    # Plot results
```

### 3. Visualization Analysis

**What to Observe:**
- **Degree 1 (Linear):** Straight line, cannot capture curve - **UNDERFITTING**
- **Degree 2 (Quadratic):** Simple curve, still underfitting
- **Degree 3-4:** Better fit, captures more complexity
- **Higher degrees (8+):** Would perfectly fit, risk of **OVERFITTING**

**Best Practice:** The true function is degree 8, but degrees 6-8 would fit best. Lower degrees show clear underfitting.

---

## 📈 Results

### House Price Prediction
- **R² Score:** ~0.75-0.85 (typical for real estate data)
- **Model:** Successfully predicts rent prices with reasonable accuracy
- **Key Predictors:** Size, BHK, City, Furnishing Status, Floor

### Pattern Plotting
- **Observation:** Visual demonstration of bias-variance tradeoff
- **Learning:** Understanding when to use polynomial features
- **Insight:** Model complexity should match data complexity

---

## 💻 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### House Price Prediction
1. Ensure dataset is in `../../dataset/House_Rent_Dataset.csv`
2. Open `housePricePrediction.ipynb`
3. Run all cells sequentially
4. View predictions and R² score

### Pattern Plotting
1. Open `PatternPlottingLinearRegressina.ipynb`
2. Run all cells to see polynomial fits of different degrees
3. Observe how model complexity affects fit

---

## 🗂️ Project Structure
```
linearRegression/
├── housePricePrediction.ipynb              # Real-world application
├── PatternPlottingLinearRegressina.ipynb   # Theoretical demonstration
└── README.md                                # This file
```

---

## 📚 Key Learnings

### House Price Prediction
1. **Feature Engineering:** Parsing and creating meaningful features from raw data
2. **Categorical Encoding:** One-hot encoding for categorical variables
3. **Feature Scaling:** Essential for optimal linear regression performance
4. **Real-World Data:** Handling messy, real-world datasets with complex preprocessing

### Pattern Plotting
1. **Polynomial Features:** Transforming linear models to capture non-linear patterns
2. **Model Complexity:** Visual understanding of underfitting and overfitting
3. **Standardization:** Important for polynomial features (values grow exponentially)
4. **Degree Selection:** Balancing model complexity with generalization

---

## 🔮 Future Improvements

### House Price Prediction
- Feature selection using correlation analysis
- Handle outliers using IQR method
- Implement Ridge/Lasso regression for regularization
- Add geospatial features (latitude, longitude)
- Ensemble methods (Random Forest, XGBoost)
- Deploy as web API for real-time predictions

### Pattern Plotting
- Add cross-validation for degree selection
- Implement regularized polynomial regression
- Compare with other non-linear models (Decision Trees, Neural Networks)
- Interactive visualization with sliders for degree selection

---

## 📖 References

- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Polynomial Features Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [StandardScaler Guide](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

## 👨‍💻 Author
Machine Learning Practitioner | Data Science Enthusiast

**Note:** These projects demonstrate both practical application (house prices) and theoretical understanding (polynomial regression) of linear regression algorithms.
