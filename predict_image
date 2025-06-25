# predict_image.py
import json
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# CONFIG
IMG_SIZE = (224, 224)

# Load model dan label
model = load_model("banana_ripeness_model.keras")
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
label_map = dict((v, k) for k, v in class_indices.items())

# Fungsi prediksi berulang
def predict_uploaded_image():
    while True:
        print("\nPilih gambar (atau Cancel untuk keluar)...")
        Tk().withdraw()
        file_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

        if not file_path:
            print("Tidak ada file yang dipilih. Selesai.")
            break

        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = label_map[pred_idx]

        print(f"Prediksi: {label} ({confidence:.2f}%)")
        plt.imshow(img)
        plt.title(f"{label.upper()} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

# Jalankan
predict_uploaded_image()
