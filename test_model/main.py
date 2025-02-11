import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model('nerve_detection_unet.h5')  # Ensure correct model filename

# Load and preprocess test image
def predict_image(image_path):
    IMG_SIZE = 256
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=(0, -1))  # Add batch & channel dimension

    # Predict
    prediction = model.predict(img)
    
    # If using classification (binary output)
    if prediction.shape[-1] == 1:
        confidence = prediction[0][0]  # Confidence score
        label = "Damaged" if confidence > 0.5 else "Healthy"
        print(f"🔍 Prediction: {label} ({confidence:.2%} confidence)")

    # If using segmentation (U-Net output)
    else:
        print("🧠 Nerve segmentation completed!")
        predicted_mask = (prediction[0] > 0.5).astype(np.uint8)  # Convert to binary mask

        # Show original & segmented images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(predicted_mask.squeeze(), cmap="gray")
        ax[1].set_title("Predicted Nerve Mask")
        ax[1].axis("off")

        plt.show()

# Test on an image (Change path as needed)
test_image_path = './pre_processed/healthy/test_image.jpg'  # Update path if needed
predict_image(test_image_path)
