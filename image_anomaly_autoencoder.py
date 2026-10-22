
import numpy as np
import tensorflow as tf  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model  # type: ignore[reportMissingModuleSource]

tf.keras.utils.set_random_seed(42)

def make_normal_images(n):
    images = np.zeros((n, 28, 28, 1), dtype=np.float32)

    for i in range(n):
        x = np.random.randint(6, 15)
        y = np.random.randint(6, 15)
        images[i, y:y+8, x:x+8, 0] = 1.0

    images += np.random.normal(0, 0.025, images.shape)
    return np.clip(images, 0, 1)

def build_autoencoder():
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(latent)
    x = layers.Dense(784, activation="sigmoid")(x)
    outputs = layers.Reshape((28, 28, 1))(x)
    return Model(inputs, outputs)

if __name__ == "__main__":
    normal_train = make_normal_images(1500)
    normal_validation = make_normal_images(200)

    model = build_autoencoder()
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        normal_train, normal_train,
        epochs=8,
        batch_size=64,
        validation_data=(normal_validation, normal_validation),
        verbose=1
    )

    validation_recon = model.predict(normal_validation, verbose=0)
    errors = np.mean(
        (normal_validation - validation_recon)**2,
        axis=(1, 2, 3)
    )

    threshold = np.percentile(errors, 99)

    # Demonstration anomaly: a large diagonal pattern.
    test = make_normal_images(1)
    for i in range(28):
        test[0, i, i, 0] = 1.0

    reconstruction = model.predict(test, verbose=0)
    error = np.mean((test - reconstruction)**2)

    print("Anomaly threshold:", round(float(threshold), 6))
    print("Test reconstruction error:", round(float(error), 6))
    print("Result:", "ANOMALY" if error > threshold else "NORMAL")

    plt.figure(figsize=(8, 3))
    for i, (image, title) in enumerate([
        (test[0, :, :, 0], "Input"),
        (reconstruction[0, :, :, 0], "Reconstruction"),
        (np.abs(test[0] - reconstruction[0])[:, :, 0],
         "Error map")
    ]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
