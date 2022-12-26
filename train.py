import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
import numpy as np

def create_unet_model(input_shape, num_classes):
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling path
    x = inputs
    skip_connections = []
    for i in range(5):
        filters = 2**(8+i)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)
        skip_connections.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    # Upsampling path
    for i in range(5):
        filters = 2**(7-i)
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_connections.pop()])
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)

    # Output Layer
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

input_shape = (512,512,1)
num_classes = 2
model = create_unet_model(input_shape, num_classes)
model.summary()

# Load the data
image_path = "dem.tif"
labels_path = "labels.tif"

with rasterio.open(image_path, "r") as dataset:
    image = dataset.read(1)

with rasterio.open(labels_path, "r") as dataset:
    labels = dataset.read(1)

# Preprocess the data
image = tf.cast(image, tf.float32) / tf.cast(tf.reduce_max(image), tf.float32)
labels = tf.cast(labels, tf.uint8)

print(image.shape)
print(labels.shape)

# One-hot encode the labels tensor
labels = tf.one_hot(labels, depth=2)

print(image.shape)
print(labels.shape)

image = tf.reshape(image, shape=[1, image.shape[0], image.shape[1], 1])
labels = tf.reshape(labels, shape=[1, labels.shape[0], labels.shape[1], 2])

print(image.shape)
print(labels.shape)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(image, labels, epochs=5, batch_size=32)

# Save the model
model.save("model.h5")

# Use the model to make predictions
model = tf.keras.models.load_model("model.h5")

# Use the trained model to make predictions
predictions = model.predict(image)
print(predictions[0,:,:,1])

# Save the predictions using rasterio
with rasterio.open("predictions.tif", "w", driver="GTiff", width=512, height=512, count=1, dtype=rasterio.float32) as dst:
    dst.write(predictions[0,:,:,1].astype(rasterio.float32), 1)