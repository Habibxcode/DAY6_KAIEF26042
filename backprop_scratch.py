"""
backprop_scratch.py — Part A: Neural Network from Scratch
==========================================================
Implements a 2-layer neural network (input → hidden[ReLU] → output[Sigmoid])
using only NumPy. No autograd, no Keras/TensorFlow/PyTorch.

Demonstrates the XOR problem, which is the classic test for a network that
requires a hidden layer (XOR is not linearly separable).

Author  : Day 6 Mini-Project
Course  : Knots AI Engineering Foundation — Cohort 1
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions & their derivatives
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-z)).
    Clipped to avoid numerical overflow.
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid in terms of its output a: a * (1 - a)."""
    return a * (1.0 - a)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss.

    Args:
        y_true: Ground-truth labels, shape (n, 1).
        y_pred: Network predictions (sigmoid outputs), shape (n, 1).

    Returns:
        Scalar loss value.
    """
    # Clip predictions to prevent log(0)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_true.shape[0]
    loss = -np.mean(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    )
    return float(loss)


# ---------------------------------------------------------------------------
# TwoLayerNet class
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """A 2-layer fully-connected neural network trained with backpropagation.

    Architecture:
        Input (input_size)
          ↓  W1, b1
        Hidden (hidden_size)  — ReLU activation
          ↓  W2, b2
        Output (output_size)  — Sigmoid activation

    Weights are initialised with Xavier / Glorot uniform initialisation:
        W ~ Uniform(-sqrt(6 / (fan_in + fan_out)),
                    +sqrt(6 / (fan_in + fan_out)))

    Parameters
    ----------
    input_size  : int   — number of input features
    hidden_size : int   — number of hidden neurons
    output_size : int   — number of output neurons
    lr          : float — learning rate (default 0.01)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = 0.01,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # ---- Xavier (Glorot) uniform initialisation ----
        # Scales weights so variance is consistent across layers.
        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))

        self.W1 = _xavier(input_size, hidden_size)   # shape (input,  hidden)
        self.b1 = np.zeros((1, hidden_size))          # shape (1,      hidden)
        self.W2 = _xavier(hidden_size, output_size)  # shape (hidden, output)
        self.b2 = np.zeros((1, output_size))          # shape (1,      output)

        # Cache for forward-pass values needed during backprop
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the forward pass.

        Args:
            X: Input data, shape (n_samples, input_size).

        Returns:
            a2: Output activations (predictions), shape (n_samples, output_size).
        """
        # Layer 1: linear → ReLU
        z1 = X @ self.W1 + self.b1    # (n, hidden)
        a1 = relu(z1)                  # (n, hidden)

        # Layer 2: linear → Sigmoid
        z2 = a1 @ self.W2 + self.b2   # (n, output)
        a2 = sigmoid(z2)               # (n, output)

        # Store intermediate values for backward pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

        return a2

    # ------------------------------------------------------------------
    # Backward pass (backpropagation)
    # ------------------------------------------------------------------

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute gradients via backpropagation and update weights.

        Uses the chain rule to propagate the gradient of the loss
        with respect to each parameter.

        Args:
            X: Input data, shape (n_samples, input_size).
            y: True labels, shape (n_samples, output_size).
        """
        n = X.shape[0]

        # Retrieve cached values from forward pass
        a1 = self._cache["a1"]
        a2 = self._cache["a2"]
        z1 = self._cache["z1"]

        # ---- Output layer gradients ----
        # dL/dz2  =  dL/da2 * da2/dz2
        #          =  (a2 - y) * sigmoid'(a2)          [BCE + sigmoid simplifies]
        # For BCE loss with sigmoid output: dL/dz2 = a2 - y  (elegant form)
        dz2 = a2 - y                              # (n, output)
        dW2 = (a1.T @ dz2) / n                   # (hidden, output)
        db2 = np.mean(dz2, axis=0, keepdims=True) # (1,      output)

        # ---- Hidden layer gradients ----
        # Propagate gradient back through W2
        da1 = dz2 @ self.W2.T                    # (n, hidden)
        # Apply ReLU derivative
        dz1 = da1 * relu_derivative(z1)           # (n, hidden)
        dW1 = (X.T @ dz1) / n                    # (input,  hidden)
        db1 = np.mean(dz1, axis=0, keepdims=True) # (1,      hidden)

        # ---- Gradient descent parameter update ----
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        print_every: int = 1000,
    ) -> list[float]:
        """Train the network for a given number of epochs.

        Args:
            X          : Input data, shape (n_samples, input_size).
            y          : True labels, shape (n_samples, output_size).
            epochs     : Number of full passes over the training data.
            print_every: Log loss every N epochs (set to 0 to suppress).

        Returns:
            loss_history: List of loss values recorded at each epoch.
        """
        loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            # Forward pass → compute predictions
            predictions = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy(y, predictions)
            loss_history.append(loss)

            # Backward pass → update weights
            self.backward(X, y)

            # Optional logging
            if print_every and epoch % print_every == 0:
                print(f"  Epoch {epoch:>6d}/{epochs}  |  Loss: {loss:.6f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and return raw sigmoid probabilities."""
        return self.forward(X)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions (0 or 1) based on a threshold."""
        return (self.predict(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Test block — XOR problem
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Part A: Backpropagation from Scratch — XOR Demo")
    print("=" * 55)

    # ---- XOR truth table ----
    # XOR cannot be solved by a single-layer (linear) model;
    # a hidden layer is required to learn the non-linear boundary.
    X_xor = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]], dtype=float)

    y_xor = np.array([[0],
                       [1],
                       [1],
                       [0]], dtype=float)

    # ---- Reproducibility ----
    np.random.seed(42)

    # ---- Build network ----
    # 2 inputs → 4 hidden neurons → 1 output
    net = TwoLayerNet(
        input_size=2,
        hidden_size=4,
        output_size=1,
        lr=0.1,          # slightly higher LR to converge on XOR in 5000 epochs
    )

    print("\nTraining for 5000 epochs …\n")

    # ---- Train ----
    history = net.train(X_xor, y_xor, epochs=5000, print_every=1000)

    # ---- Results ----
    raw_preds = net.predict(X_xor)
    rounded_preds = np.round(raw_preds).astype(int)

    print("\n" + "-" * 40)
    print("  Results after training:")
    print("-" * 40)
    print(f"  {'Input':<12} {'Raw Prob':>10}  {'Predicted':>10}  {'Target':>8}")
    print(f"  {'-'*12} {'-'*10}  {'-'*10}  {'-'*8}")
    for i, (x, raw, pred, tgt) in enumerate(
        zip(X_xor, raw_preds, rounded_preds, y_xor)
    ):
        inp = f"[{int(x[0])}, {int(x[1])}]"
        print(
            f"  {inp:<12} {raw[0]:>10.4f}  {str(pred):>10}  {str(tgt.astype(int)):>8}"
        )

    print("-" * 40)
    print(f"\n  Rounded predictions : {rounded_preds.flatten().tolist()}")
    print(f"  Target              : [0, 1, 1, 0]")
    print(f"\n  Final loss          : {history[-1]:.6f}")

    correct = np.all(rounded_preds == y_xor.astype(int))
    print(f"  XOR solved          : {'✓ YES' if correct else '✗ NO'}")
    print("=" * 55)
