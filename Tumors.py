from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import random

# ====== Load Data ======
# umors_training_directory = "/home/mmyer027/dataset/brainTumors/Training"
# umors_testing_directory = "/home/mmyer027/dataset/brainTumors/Testing"
tumors_training_directory = "/home/mmyer027/dataset/brainTumorBasic/Training"
tumors_testing_directory = "/home/mmyer027/dataset/brainTumorBasic/Testing"

tumors_trainFull_dataSet = image_dataset_from_directory(
    tumors_training_directory,
    image_size=(224, 224),
    batch_size=None,
    shuffle=True,
)

tumors_testFull_dataSet = image_dataset_from_directory(
    tumors_testing_directory,
    image_size=(224, 224),
    batch_size=32,
)

train_class_names = image_dataset_from_directory(
    tumors_training_directory,
    image_size=(224, 224),
    batch_size=32
).class_names

print("Train classes:", train_class_names)

# ==== Augment data, synthetically expand "no_tumors" ====#
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ====== Pre Process Data ======
tumors_trainFull_dataSet = tumors_trainFull_dataSet.map(
    lambda x, y: (data_augmentation(preprocess_input(x), training=True), y))
tumors_trainFull_dataSet = tumors_trainFull_dataSet.cache().prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

# ====== Convert Full Training Data to Numpy for Sampling ======
tumor_x_full, tumor_y_full = zip(*list(tumors_trainFull_dataSet))
tumor_x_full = np.stack(tumor_x_full)
tumor_y_full = np.array(tumor_y_full)

unique, counts = np.unique(tumor_y_full, return_counts=True)
print(dict(zip(unique, counts)))

# === subset sizes ===
subset_sizes = [10, 50, 200, 500, 2000, len(tumor_x_full)]
# subset_sizes = [len(tumor_x_full)]
num_classes = 2
# num_classes = 4
epochs = 15


# ====== Model Creation ======
def build_model(num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ====== Train Model With Different Data Set Sizes
results = {}

# ==== Compute class weights ====#
classes = np.unique(tumor_y_full)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=tumor_y_full)
class_weight = dict(zip(classes, weights))
print(class_weight)

for n in subset_sizes:
    print(f"\n Training with {n} samples")

    # Get random selection of subset of n samples
    idx = random.sample(range(len(tumor_x_full)), n)
    x_subset = preprocess_input(tumor_x_full[idx])
    y_subset = tumor_y_full[idx]

    # Create TensorFlow DataSet
    tumors_trainFull_dataSet = tf.data.Dataset.from_tensor_slices((x_subset, y_subset)).batch(32)
    tumors_trainFull_dataSet = tumors_trainFull_dataSet.cache().shuffle(1000).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # Build and Train Model
    model = build_model(num_classes)
    history = model.fit(tumors_trainFull_dataSet, epochs=epochs, verbose=1, class_weight=class_weight)
    # history = model.fit(tumors_trainFull_dataSet, epochs=epochs, verbose=1)

    # Evaluate on Test Set
    test_loss, test_acc = model.evaluate(tumors_testFull_dataSet, verbose=0)
    results[n] = test_acc
    print(f"Test accuracy with {n} samples: {test_acc:.4f}")

# ====== Plot Results ======
plt.figure(figsize=(8, 4))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xscale('log')
plt.xlabel('Training Dataset Size (samples)')
plt.ylabel('Test Accuracy')
plt.title('Effect of Training Dataset Size on Model Performance')
plt.grid(True)
plt.savefig("tumorsModelTraining.png")
plt.close()

# ====== Summary Table ======
print("\n Summary of Results:")
for n, acc in results.items():
    print(f"{n:>6} Samples -> Test Accuracy = {acc:.4f}")
