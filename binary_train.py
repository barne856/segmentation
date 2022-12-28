import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import rasterio

# Load the Segmentation dataset
# Load the data
image_path = "dem.tif"
labels_path = "labels.tif"

with rasterio.open(image_path, "r") as dataset:
    image = dataset.read()

with rasterio.open(labels_path, "r") as dataset:
    labels = dataset.read()

# labels = tf.cast(labels, tf.float32)

image = tf.reshape(image, shape=[image.shape[1], image.shape[2], 1])
image = tf.concat([image, image, image], axis=2)
labels = tf.reshape(labels, shape=[labels.shape[1], labels.shape[2], 1])

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

dataset = {"image": image, "segmentation_mask": labels}
dataset = tf.data.Dataset.from_tensors((dataset))
dataset = dataset.prefetch(1)

# Use the image and label to train the model
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / tf.cast(tf.reduce_max(input_image), tf.float32)
    # input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (512, 512))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (512, 512),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


train_images = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
TRAIN_LENGTH = 1  # info.splits["train"].num_examples
BATCH_SIZE = 1
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


train_batches = (
    train_images.cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_images.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[512, 512, 3], include_top=False
)

# Use the activations of these layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

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

OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
#model = create_unet_model(input_shape=(512,512,3), num_classes=OUTPUT_CLASSES)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
tf.keras.utils.plot_model(model, show_shapes=True)


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
            new_mask = create_mask(pred_mask)
            new_mask = tf.image.resize(new_mask, (512, 512))
            new_mask = tf.transpose(new_mask, perm=[2, 0, 1])
            with rasterio.open("pred_mask.tif", "w", driver="GTiff", width=512, height=512, count=1, dtype=rasterio.int64) as dst:
                dst.write(new_mask)
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...])),
            ]
        )


# show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


EPOCHS = 1000
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits["test"].num_examples // BATCH_SIZE // VAL_SUBSPLITS

image = tf.reshape(image, shape=[1, image.shape[0], image.shape[1], image.shape[2]])
labels = tf.reshape(labels, shape=[1, labels.shape[0], labels.shape[1], labels.shape[2]])

model_history = model.fit(
    train_batches,
    #image,
    #labels,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=test_batches,
    callbacks=[DisplayCallback()],
)

loss = model_history.history["loss"]
val_loss = model_history.history["val_loss"]

# plt.figure()
# plt.plot(model_history.epoch, loss, "r", label="Training loss")
# plt.plot(model_history.epoch, val_loss, "bo", label="Validation loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss Value")
# plt.ylim([0, 1])
# plt.legend()
# plt.show()
# 
show_predictions(test_batches)
        
