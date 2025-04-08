## 🔍 3. Flow of Information – Deep Dive

In a **neural network**, data flows **forward** through layers during prediction. This is called the **forward pass**.

For each **neuron** in a layer:

```
Z = W * X + b
A = activation(Z)
```

### 👉 What Does Each Term Mean?

- **X** = Input vector to the layer (can come from previous layer)
- **W** = Weight matrix for this layer
- **b** = Bias vector for this layer
- **Z** = Linear transformation (raw score before activation)
- **activation(Z)** = Adds non-linearity; output is **A**

---

### 🎯 Why Use Activation Functions?

If we **don't use activation**, stacking multiple layers is just a **giant linear equation**, no better than a simple linear model.

**Activation functions break linearity**, enabling the network to learn complex patterns and decision boundaries.

---

### ⚙️ Common Activation Functions

#### 1. **ReLU (Rectified Linear Unit)**
- Formula: `ReLU(x) = max(0, x)`
- Output: Same as input if input is positive; otherwise zero.
- Used heavily in **hidden layers**.
- **Fast & effective**.

Example:
```
Input: [-2, 0, 3]
ReLU Output: [0, 0, 3]
```

#### 2. **Sigmoid**
- Formula: `Sigmoid(x) = 1 / (1 + exp(-x))`
- Output range: (0, 1)
- Great for **binary classification** (like spam or not spam)

Example:
```
Input: 0
Sigmoid Output: 0.5

Input: 10
Sigmoid Output: ~1

Input: -10
Sigmoid Output: ~0
```

#### 3. **Tanh**
- Formula: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- Output range: (-1, 1)
- Zero-centered; used in some cases.

---

### 🧠 Layer-wise Data Transformation (Example)

Suppose we have a **2-layer network**:

1. **Input Layer:** X = [0.5, 0.8] (2 features)
2. **Hidden Layer 1 (3 neurons):**
   - Z1 = W1·X + b1
   - A1 = ReLU(Z1)
3. **Output Layer (1 neuron):**
   - Z2 = W2·A1 + b2
   - A2 = Output (no activation for regression or use softmax/sigmoid for classification)

---

## 📝 Running Notes in Markdown

Here's the Markdown version of the complete running notes, including the expanded Section 3.

---

### 🔖 Suggested File Name:
```
Level2_NeuralNetworks_MLP_Activation_Backprop.md
```

---

```markdown
# 🧠 Level 2: Neural Networks & Multi-Layer Perceptrons (MLPs)

---

## ✅ 1. What is a Neural Network?
- A neural network is a system of **layers of neurons**.
- Each layer transforms input data to a new representation.
- Goal: Learn weights that can **map input to output correctly**.

---

## ✅ 2. What is a Multi-Layer Perceptron (MLP)?
- A type of feedforward neural network.
- Contains:
  - Input layer
  - One or more hidden layers
  - Output layer
- Each neuron in a layer is connected to every neuron in the next (fully connected).

---

## ✅ 3. Flow of Information (Forward Pass)

### 🔄 Step-by-step:

For each neuron:
```
Z = W * X + b
A = activation(Z)
```

- `Z`: Linear transformation (raw value)
- `A`: Activated output (non-linear)

---

### 🔧 Why Activation Functions?
- Without activation, a neural network behaves like a single-layer linear model.
- Activation functions allow the network to **learn complex, non-linear relationships**.

---

## ⚙️ Common Activation Functions

### 🔹 ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- Output is 0 if x < 0, else x.
- Fast, simple, widely used in hidden layers.

**Example**:
Input: `[-2, 0, 3]` → Output: `[0, 0, 3]`

---

### 🔹 Sigmoid
```
Sigmoid(x) = 1 / (1 + exp(-x))
```
- Output range: (0, 1)
- Great for binary classification.

**Example**:
- Input: `0` → Output: `0.5`
- Input: `10` → Output: `~1`
- Input: `-10` → Output: `~0`

---

### 🔹 Tanh
```
Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```
- Output range: (-1, 1)
- Zero-centered version of sigmoid.

---

## ✅ 4. Loss Calculation
```
Loss = Mean Squared Error or Cross-Entropy
```

---

## ✅ 5. Backpropagation – Training the Network
- Compute loss at output.
- Propagate gradients backward layer-by-layer.
- Use **chain rule** to compute gradient for each weight and bias.

For layer L:
```
dZ = dA * activation_derivative(Z)
dW = dZ * A_prev.T
db = dZ
```

Update step:
```
W = W - learning_rate * dW
b = b - learning_rate * db
```

---

## ✅ 6. Example: 2-layer MLP in NumPy
```python
import numpy as np

# Input
X = np.array([[0.5], [0.8]])

# Weights & Biases
W1 = np.random.randn(3, 2)
b1 = np.random.randn(3, 1)
W2 = np.random.randn(1, 3)
b2 = np.random.randn(1, 1)

# Forward Pass
Z1 = np.dot(W1, X) + b1
A1 = np.maximum(0, Z1)  # ReLU activation
Z2 = np.dot(W2, A1) + b2
Y_hat = Z2  # Output
```

---

## ✅ Summary Table

| Concept             | Explanation                                                       |
|--------------------|-------------------------------------------------------------------|
| Z = WX + b         | Linear combination                                                |
| A = activation(Z)  | Adds non-linearity                                                |
| ReLU               | max(0, x), used in hidden layers                                  |
| Sigmoid            | 1 / (1 + exp(-x)), used in output for binary classification       |
| Forward Pass       | Input flows layer-by-layer with transformations                   |
| Backpropagation    | Error moves backward to update weights and biases                 |
| Weight Update      | Gradient descent adjusts weights to reduce error                  |

---
```

---
