import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = (128, 128)

# Load your trained model (optional if already in memory)
model = tf.keras.models.load_model("leaf_disease_model.h5")

# GUI to select image
Tk().withdraw()  # Hide the root window
file_path = askopenfilename(title="Select an image to classify")

# Load and preprocess image
img = image.load_img(file_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)[0][0]
class_label = "Healthy" if prediction > 0.5 else "Diseased"

# Show result
plt.imshow(img)
plt.title(f"Predicted: {class_label} ({prediction:.2f})")
plt.axis('off')
plt.show()
