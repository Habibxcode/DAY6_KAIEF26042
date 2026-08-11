"""
mnist_classifier.py — Part B: MNIST Digit Classifier with Keras
================================================================
Builds a fully-connected (Dense) neural network to classify handwritten
digits from the MNIST dataset.

Target accuracy: > 97 % on the test set.

Features
--------
* Batch-normalised Dense architecture with Dropout regularisation
* Adam optimiser with sparse categorical cross-entropy loss
* EarlyStopping callback to prevent over-fitting
* Matplotlib visualisation of 5 misclassified examples

Author  : Day 6 Mini-Project
Course  : Knots AI Engineering Foundation — Cohort 1
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- TensorFlow / Keras imports ----
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    sys.exit(
        f"[ERROR] TensorFlow is required but could not be imported: {e}\n"
        "Install it with:  pip install tensorflow"
    )

# Suppress TensorFlow info / warning logs (optional — remove to see full logs)
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 10        # digits 0–9
EPOCHS = 10             # maximum training epochs (EarlyStopping may stop early)
BATCH_SIZE = 128        # mini-batch size
VALIDATION_SPLIT = 0.1  # 10 % of training data used for validation
DROPOUT_RATE = 0.3      # dropout probability
PATIENCE = 3            # EarlyStopping patience (epochs without improvement)


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_mnist():
    """Load the MNIST dataset and normalise pixel values to [0, 1].

    MNIST contains:
        - 60 000 training images  (28 × 28 grayscale)
        -  10 000 test images

    Returns:
        (X_train, y_train), (X_test, y_test): Preprocessed arrays.
    """
    print("[INFO] Loading MNIST dataset …")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # ---- Flatten 28×28 images into 784-dim vectors ----
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(-1, 784).astype("float32")  / 255.0

    print(f"  Training samples : {X_train.shape[0]:,}  |  shape: {X_train.shape}")
    print(f"  Test samples     : {X_test.shape[0]:,}   |  shape: {X_test.shape}")
    print(f"  Label range      : {y_train.min()} – {y_train.max()}")

    return (X_train, y_train), (X_test, y_test)


# ---------------------------------------------------------------------------
# 2. Model definition
# ---------------------------------------------------------------------------

def build_model(input_dim: int = 784, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Build and return a fully-connected Keras Sequential model.

    Architecture:
        Input(784)
          → Dense(512, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(128, ReLU) → Dropout(0.3)
          → Dense(10,  Softmax)   ← output probabilities for each digit

    Args:
        input_dim  : Flattened input dimension (default 784 for MNIST).
        num_classes: Number of output classes (default 10).

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            # ---- Layer 1 ----
            layers.Dense(512, activation="relu", input_shape=(input_dim,),
                         name="dense_1"),
            layers.BatchNormalization(name="bn_1"),
            layers.Dropout(DROPOUT_RATE, name="dropout_1"),

            # ---- Layer 2 ----
            layers.Dense(256, activation="relu", name="dense_2"),
            layers.BatchNormalization(name="bn_2"),
            layers.Dropout(DROPOUT_RATE, name="dropout_2"),

            # ---- Layer 3 ----
            layers.Dense(128, activation="relu", name="dense_3"),
            layers.Dropout(DROPOUT_RATE, name="dropout_3"),

            # ---- Output layer ----
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="MNIST_Classifier",
    )

    # ---- Compile ----
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # labels are integers, not one-hot
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> keras.callbacks.History:
    """Train the model with EarlyStopping on validation accuracy.

    Args:
        model  : Compiled Keras model.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Keras History object containing per-epoch metrics.
    """
    # EarlyStopping monitors validation accuracy and restores best weights
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n[INFO] Training the model …")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the trained model on the test set.

    Args:
        model : Trained Keras model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        (test_loss, test_accuracy) as floats.
    """
    print("\n[INFO] Evaluating on the test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f} %")

    if test_acc >= 0.97:
        print("  ✓ Target accuracy of >97 % achieved!")
    else:
        print("  ✗ Did not reach target accuracy of 97 %. Consider more epochs.")

    return test_loss, test_acc


# ---------------------------------------------------------------------------
# 5. Misclassification analysis & visualisation
# ---------------------------------------------------------------------------

def plot_misclassified(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_examples: int = 5,
) -> None:
    """Find and plot n misclassified examples from the test set.

    Each subplot shows:
        - The 28×28 digit image
        - True label vs. Predicted label (with confidence %)

    Args:
        model      : Trained Keras model.
        X_test     : Test features (flattened).
        y_test     : True test labels.
        n_examples : Number of misclassified examples to display (default 5).
    """
    print(f"\n[INFO] Finding {n_examples} misclassified examples …")

    # ---- Get predictions ----
    y_probs = model.predict(X_test, verbose=0)          # (n_test, 10)
    y_pred  = np.argmax(y_probs, axis=1)                # predicted class index

    # ---- Find indices where prediction ≠ true label ----
    misclassified_idx = np.where(y_pred != y_test)[0]

    if len(misclassified_idx) == 0:
        print("  ✓ No misclassified examples found — perfect classifier!")
        return

    print(f"  Total misclassified: {len(misclassified_idx):,} out of {len(y_test):,}")

    # Select the first n_examples misclassified samples
    sample_idx = misclassified_idx[:n_examples]

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(
        "MNIST — Misclassified Examples",
        fontsize=14, fontweight="bold", y=1.02,
    )
    gs = gridspec.GridSpec(1, n_examples, figure=fig, wspace=0.4)

    for col, idx in enumerate(sample_idx):
        ax = fig.add_subplot(gs[0, col])

        # Reshape flattened vector back to 28×28 for display
        image = X_test[idx].reshape(28, 28)
        confidence = y_probs[idx, y_pred[idx]] * 100

        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"True: {y_test[idx]}\nPred: {y_pred[idx]}\n({confidence:.1f} %)",
            fontsize=11,
            color="red",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mnist_misclassified.png", dpi=120, bbox_inches="tight")
    print("  Plot saved to: mnist_misclassified.png")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Training curve visualisation (bonus)
# ---------------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History) -> None:
    """Plot training vs. validation accuracy and loss curves.

    Args:
        history: Keras History object returned by model.fit().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MNIST Classifier — Training History", fontsize=13, fontweight="bold")

    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    # ---- Accuracy subplot ----
    axes[0].plot(epochs_ran, history.history["accuracy"],     label="Train Accuracy", marker="o")
    axes[0].plot(epochs_ran, history.history["val_accuracy"], label="Val Accuracy",   marker="s")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Loss subplot ----
    axes[1].plot(epochs_ran, history.history["loss"],     label="Train Loss", marker="o")
    axes[1].plot(epochs_ran, history.history["val_loss"], label="Val Loss",   marker="s")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mnist_training_history.png", dpi=120, bbox_inches="tight")
    print("\n[INFO] Training history plot saved to: mnist_training_history.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates data loading, training, evaluation, and visualisation."""
    print("=" * 60)
    print("  Part B: MNIST Digit Classifier — Keras Fully-Connected")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()

    # 2. Build model
    model = build_model()
    print("\n[INFO] Model summary:")
    model.summary()

    # 3. Train
    history = train_model(model, X_train, y_train)

    # 4. Evaluate
    evaluate_model(model, X_test, y_test)

    # 5. Plot misclassified examples
    plot_misclassified(model, X_test, y_test, n_examples=5)

    # 6. Training curves
    plot_training_history(history)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
