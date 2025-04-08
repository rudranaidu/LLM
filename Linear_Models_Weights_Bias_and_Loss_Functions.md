Sure! Here's the full **Markdown version** of the Level 1 running notes:

---

```markdown
# 📝 Running Notes: Level 1 – Weights, Bias, Linear Models, and Training Basics

---

## ✅ 1. What is a Model?
- A **model** is a mathematical function.
- Takes **input (X)** and gives an **output (Y)**.
- Goal: Predict Y accurately from X.

---

## ✅ 2. Inputs and Outputs in a Simple Problem
Example: Predict exam scores from hours studied.

| X (Hours) | Y (Score) |
|-----------|-----------|
| 1         | 40        |
| 2         | 50        |
| 3         | 60        |

---

## ✅ 3. Weights and Biases
Model equation:

```
Y = W * X + b
```

- **W (Weight)**: Influence of input on output.
- **b (Bias)**: Base shift to adjust predictions.
- These are **learnable parameters**.

Example:  
If `W = 10`, `b = 30`, then  
`Y = 10 * 2 + 30 = 50`

---

## ✅ 4. Linear Regression – Simple Predictive Model
- Learns the best line that fits the data.
- Objective: Find optimal **W and b** to minimize the prediction error.

Python Function:
```python
def predict(x, w, b):
    return w * x + b
```

---

## ✅ 5. Loss Function – Measure of Error

Use **Mean Squared Error (MSE)**:
```
Loss = (1/N) * Σ(actual - predicted)²
```

Purpose:
- Tells how far predictions are from actual values.
- The model tries to **minimize** this loss.

Example:

| Actual | Predicted |
|--------|-----------|
| 40     | 28        |
| 50     | 36        |
| 60     | 44        |

Loss:
```
Loss = ((40-28)² + (50-36)² + (60-44)²) / 3 = 198.7
```

---

## ✅ 6. Gradient Descent – Learning Weights

Process to **optimize weights and bias** by minimizing the loss.

Steps:
1. Compute predictions
2. Calculate loss
3. Calculate gradients (slope of loss curve)
4. Update weights & bias in the opposite direction of gradient

Formula:
```
w = w - learning_rate * dw
b = b - learning_rate * db
```

---

## ✅ 7. Backpropagation – Weight Adjustment

The process of:
1. Calculating prediction error.
2. Propagating the error **backward** through the model.
3. Computing **partial derivatives** of loss w.r.t each weight and bias.
4. Updating them based on how much they influenced the loss.

Example (Python):
```python
x = 2
y_true = 50
w = 5
b = 20
lr = 0.01

# Prediction
y_pred = w * x + b  # 30

# Loss
loss = (y_true - y_pred) ** 2  # 400

# Gradients
dw = -2 * x * (y_true - y_pred)  # -80
db = -2 * (y_true - y_pred)      # -40

# Update
w = w - lr * dw  # 5.8
b = b - lr * db  # 20.4
```

---

## ✅ Summary Table

| Concept           | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| Weight (W)        | Learns how important each input is                                      |
| Bias (b)          | Shifts the prediction to fit the data                                   |
| Prediction (ŷ)    | Estimated output from the model (ŷ = W*X + b)                           |
| Loss              | Error between actual and predicted values                               |
| Gradient          | Direction of steepest increase of loss (used to minimize it)            |
| Gradient Descent  | Technique to adjust weights to reduce loss                              |
| Backpropagation   | Calculates how much each parameter contributed to error                 |

---
```

---
