"""
traffic_cnn.py — Part C: CNN Classifier on CIFAR-10 (Traffic Sign Substitute)
==============================================================================
Builds a Convolutional Neural Network (CNN) to classify images.

The CIFAR-10 dataset is used as a convenient substitute for traffic-sign
data because it shares the same 32×32×3 colour-image format. CIFAR-10
contains 60 000 images across 10 classes (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck).

Target: ≥ 70 % accuracy within 15 epochs (CIFAR-10 is significantly harder
than traffic-sign datasets; state-of-the-art simple CNNs achieve ~80-85 %).

Exact Architecture (as specified)
----------------------------------
    Conv2D(32, 3×3, ReLU)
    MaxPool2D(2×2)
    Conv2D(64, 3×3, ReLU)
    MaxPool2D(2×2)
    Flatten
    Dense(128, ReLU)
    Dropout(0.5)
    Dense(num_classes, Softmax)

Model is saved to: traffic_model.keras

Author  : Day 6 Mini-Project
Course  : Knots AI Engineering Foundation — Cohort 1
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

# ---- TensorFlow / Keras imports ----
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError as e:
    sys.exit(
        f"[ERROR] TensorFlow is required but could not be imported: {e}\n"
        "Install it with:  pip install tensorflow"
    )

# Suppress TF info messages
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_CLASSES   = 10      # CIFAR-10 has 10 classes
EPOCHS        = 15      # maximum training epochs
BATCH_SIZE    = 64
MODEL_PATH    = "traffic_model.keras"
DROPOUT_RATE  = 0.5

# Human-readable class labels for CIFAR-10
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_cifar10():
    """Load CIFAR-10 and normalise pixel values to [0, 1].

    CIFAR-10:
        - 50 000 training images  (32 × 32 × 3 RGB)
        - 10 000 test images

    Returns:
        (X_train, y_train), (X_test, y_test): Preprocessed arrays.
        y labels are integer class indices (0–9).
    """
    print("[INFO] Loading CIFAR-10 dataset …")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # ---- Normalise pixel values from [0, 255] → [0.0, 1.0] ----
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Labels from CIFAR-10 are shaped (n, 1); squeeze to (n,)
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    print(f"  Training images  : {X_train.shape}   dtype: {X_train.dtype}")
    print(f"  Test images      : {X_test.shape}    dtype: {X_test.dtype}")
    print(f"  Training labels  : {y_train.shape}")
    print(f"  Classes          : {NUM_CLASSES}  ({', '.join(CIFAR10_LABELS)})")

    return (X_train, y_train), (X_test, y_test)


# ---------------------------------------------------------------------------
# 2. Data augmentation (optional layer — improves generalisation)
# ---------------------------------------------------------------------------

def build_augmentation_layer() -> keras.Sequential:
    """Return a Keras preprocessing model that applies random augmentations.

    Applied only during training; skipped during inference automatically.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),          # random horizontal flip
            layers.RandomRotation(0.1),               # ±10° rotation
            layers.RandomZoom(0.1),                   # ±10% zoom
            layers.RandomTranslation(0.1, 0.1),      # ±10% shift in x/y
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------------------
# 3. Model definition (exact specified architecture)
# ---------------------------------------------------------------------------

def build_cnn(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
) -> keras.Model:
    """Build and compile the CNN model.

    Exact architecture as specified:
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool →
        Flatten → Dense(128) → Dropout(0.5) → Dense(num_classes)

    Plus optional data-augmentation and BatchNormalisation for improved
    generalisation on CIFAR-10.

    Args:
        input_shape     : Image shape (H, W, C).
        num_classes     : Number of output classes.
        use_augmentation: Whether to prepend the augmentation layer.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")
    x = inputs

    # ---- Optional augmentation (training only) ----
    if use_augmentation:
        aug_layer = build_augmentation_layer()
        x = aug_layer(x)   # augmentation layer is automatically no-op at inference

    # ---- Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool ----
    x = layers.Conv2D(
        filters=32, kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_1",
    )(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1")(x)

    # ---- Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool ----
    x = layers.Conv2D(
        filters=64, kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2")(x)

    # ---- Flatten ----
    x = layers.Flatten(name="flatten")(x)

    # ---- Dense(128) → Dropout(0.5) ----
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    # ---- Output: Dense(num_classes, Softmax) ----
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Traffic_CNN")

    # ---- Compile ----
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_cnn(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> keras.callbacks.History:
    """Train the CNN with callbacks for adaptive learning rate and early stopping.

    Args:
        model  : Compiled Keras model.
        X_train: Training images.
        y_train: Training labels.

    Returns:
        Keras History object.
    """
    # ReduceLROnPlateau: halve the LR if val_loss plateaus for 3 epochs
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # EarlyStopping: stop if val_accuracy doesn't improve for 5 epochs
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n[INFO] Training the CNN …")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[reduce_lr, early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_cnn(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the trained CNN on the test set.

    Args:
        model : Trained Keras model.
        X_test: Test images.
        y_test: Test labels.

    Returns:
        (test_loss, test_accuracy).
    """
    print("\n[INFO] Evaluating on the test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f} %")
    return test_loss, test_acc


# ---------------------------------------------------------------------------
# 6. Visualisation helpers
# ---------------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History) -> None:
    """Plot training and validation accuracy / loss curves.

    Args:
        history: Keras History object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CIFAR-10 CNN — Training History", fontsize=13, fontweight="bold")

    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs_ran, history.history["accuracy"],     "b-o", label="Train")
    axes[0].plot(epochs_ran, history.history["val_accuracy"], "r-s", label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs_ran, history.history["loss"],     "b-o", label="Train")
    axes[1].plot(epochs_ran, history.history["val_loss"], "r-s", label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("traffic_training_history.png", dpi=120, bbox_inches="tight")
    print("[INFO] Training history saved to: traffic_training_history.png")
    plt.show()


def plot_sample_predictions(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n: int = 10,
) -> None:
    """Display a grid of n sample test images with their predictions.

    Args:
        model : Trained Keras model.
        X_test: Test images.
        y_test: True labels.
        n     : Number of samples to display.
    """
    indices = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    y_probs = model.predict(X_sample, verbose=0)
    y_preds = np.argmax(y_probs, axis=1)

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    fig.suptitle("CIFAR-10 CNN — Sample Predictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_sample[i])
        true_label = CIFAR10_LABELS[y_sample[i]]
        pred_label = CIFAR10_LABELS[y_preds[i]]
        color = "green" if y_preds[i] == y_sample[i] else "red"
        ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("traffic_sample_predictions.png", dpi=120, bbox_inches="tight")
    print("[INFO] Sample predictions saved to: traffic_sample_predictions.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates data loading, training, evaluation, and model saving."""
    print("=" * 65)
    print("  Part C: CNN Traffic Classifier (CIFAR-10 as substitute)")
    print("=" * 65)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_cifar10()

    # 2. Build CNN
    model = build_cnn(use_augmentation=True)
    print("\n[INFO] Model summary:")
    model.summary()

    # 3. Train
    history = train_cnn(model, X_train, y_train)

    # 4. Evaluate
    test_loss, test_acc = evaluate_cnn(model, X_test, y_test)

    # 5. Save the model
    print(f"\n[INFO] Saving model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    print(f"  ✓ Model saved successfully.")

    # 6. Visualise training history
    plot_training_history(history)

    # 7. Show sample predictions
    plot_sample_predictions(model, X_test, y_test, n=10)

    print("\n[INFO] Done.")
    print(f"  Final test accuracy: {test_acc * 100:.2f} %")


if __name__ == "__main__":
    main()
