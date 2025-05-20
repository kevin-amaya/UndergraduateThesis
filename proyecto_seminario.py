import os
import zipfile


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

# Cambia la ruta a la ubicación en tu computadora
local_zip = 'D:/Users/PC/Documents/Universidad/colon11.zip'

# Descomprimir el archivo ZIP
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/Users/PC/Documents/Universidad/colon11.zip')
zip_ref.close()

# Rutas locales a las carpetas de entrenamiento y prueba
train_dir = 'D:/Users/PC/Documents/Universidad/colon11.zip/Colon1/training_set'
test_dir = 'D:/Users/PC/Documents/Universidad/colon11.zip/Colon1/test_set'

# Generador de datos con aumento para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Separar el 20% para validación
)

# Generador de datos de validación y prueba (solo rescale)
test_datagen = ImageDataGenerator(rescale=1/255)

# Cargar datos de validación
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Cargar datos de prueba
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical'
)


# Cargar el modelo preentrenado VGG16 sin las capas superiores (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Congelar las capas del modelo preentrenado para que no se actualicen durante el entrenamiento
base_model.trainable = False

from tensorflow.keras.layers import GlobalAveragePooling2D # import the GlobalAveragePooling2D class from tensorflow.keras.layers

# Crear el modelo final añadiendo capas densas a la salida de VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularización para evitar el sobreajuste
    Dense(2, activation='softmax')  # Capa de salida para 2 clases (cáncer o no)
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,  # Puedes ajustar este número según los resultados
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)





