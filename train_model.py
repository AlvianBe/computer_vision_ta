import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Link Dataset = https://drive.google.com/drive/folders/10rHEGwRWLVqg2fJ_AB3Z7yW0cj-8xbgA?usp=sharing

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATASET_PATH = "dataset_banana"

# Data generators
train_gen = ImageDataGenerator(rescale=1./255,
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_data = val_gen.flow_from_directory(os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Class weight
y_train = train_data.classes
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights = dict(enumerate(class_weights))

# Model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Training awal
callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
             ReduceLROnPlateau(factor=0.5, patience=3)]
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
                    callbacks=callbacks, class_weight=class_weights)

# Fine-tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10,
          callbacks=callbacks, class_weight=class_weights)

# Save model
model.save("banana_ripeness_model.keras")
print("✅ Model saved as 'banana_ripeness_model.keras'")

# Optional: plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Training Accuracy")
plt.show()