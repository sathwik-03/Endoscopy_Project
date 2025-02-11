import os
import numpy as np
import cv2

# Define paths to preprocessed images
healthy_path = r'./pre_processed/healthy'
damaged_path = r'./pre_processed/damaged'

IMG_SIZE = 256  # Image size for resizing
X_list = []
y_list = []
batch_size = 1000  # Process 1000 images at a time

def remove_corrupt_images():
    """Removes corrupt JPEG images."""
    print("\U0001F6E0 Removing corrupt images...")
    for folder in [healthy_path, damaged_path]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        print(f"⚠️ Removing corrupt image: {img_path}")
                        os.remove(img_path)
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")
                    os.remove(img_path)
    print("✅ Corrupt images removed!")

def load_images_from_folder(folder, label):
    """Loads images from the specified folder and assigns a label."""
    global X_list, y_list

    if not os.path.exists(folder):
        print(f"❌ Warning: Folder '{folder}' does not exist. Skipping...")
        return

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Skipping corrupt image: {img_path}")
            continue  # Skip unreadable images

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_list.append(img)
        y_list.append(label)

        # Save in batches
        if len(X_list) >= batch_size:
            save_batch()

def save_batch():
    global X_list, y_list
    if X_list:
        X = np.array(X_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float16) / 255.0  # Use float16 to reduce size
        y = np.array(y_list)

        # Check if .npy files exist before loading
        if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
            X_old = np.load("X_train.npy", allow_pickle=True)
            y_old = np.load("y_train.npy", allow_pickle=True)
            
            # Check shape consistency
            if X_old.shape[1:] == X.shape[1:]:
                X = np.concatenate((X_old, X))
                y = np.concatenate((y_old, y))
            else:
                print("❌ Error: Inconsistent image size detected! Skipping append.")
                return
        
        np.save("X_train.npy", X)
        np.save("y_train.npy", y)
        print(f"💾 Saved {len(X_list)} more images! Total: {X.shape[0]}")

    X_list = []
    y_list = []

# Step 1: Remove corrupt images
remove_corrupt_images()

# Step 2: Load images from both categories
print("📥 Loading images...")
load_images_from_folder(healthy_path, label=0)  # Healthy
load_images_from_folder(damaged_path, label=1)  # Damaged

# Step 3: Save any remaining images
save_batch()

print(f"✅ Dataset saved successfully!")
