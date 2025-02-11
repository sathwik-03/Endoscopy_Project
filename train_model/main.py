import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ✅ Enable GPU & Prevent Memory Issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is set for TensorFlow!")
    except RuntimeError as e:
        print(e)

print("Num GPUs Available:", len(gpus))

# ✅ Enable Mixed Precision for Speed (Apple M1 Optimized)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ✅ Load Data (Replace with actual dataset)
IMG_SIZE = 128  # Ensure this matches final output size
X_train = np.random.rand(12001, IMG_SIZE, IMG_SIZE, 1).astype("float32")
y_train = np.random.randint(0, 2, (12001, IMG_SIZE, IMG_SIZE, 1)).astype("float32")

# ✅ Normalize y_train if needed
if y_train.max() > 1:
    y_train = y_train / 255.0  # Convert to 0-1 range

# ✅ Check Shapes Before Training
print(f"X_train shape: {X_train.shape}")  # (12001, 128, 128, 1)
print(f"y_train shape: {y_train.shape}")  # (12001, 128, 128, 1)
print(f"y_train unique values: {np.unique(y_train)}")  # Should be [0, 1]

# ✅ Define U-Net Model (Fixed Output Shape)
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Encoder (Downsampling)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 128 → 64

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 64 → 32

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 32 → 16

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)  # 16 → 32
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)  # 32 → 64
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)  # 64 → 128
    x = layers.BatchNormalization()(x)

    # Output Layer (Must Match Input Size)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", dtype="float32")(x)  # Ensure float32

    model = keras.Model(inputs, outputs, name="segmentation_model")
    return model

# ✅ Compile Model
model = build_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# ✅ Save Model
model.save("endoscopy_segmentation_model.h5")
print("✅ Model training complete and saved as 'endoscopy_segmentation_model.h5'")

print('change 2')