import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory

# ====== Load Data ======
dogCat_training_directory = "/home/myersm/dataset/catdog/training_set/training_set/"
dogCat_testing_directory = "/home/myersm/dataset/catdog/test_set/test_set/"

dogCat_trainFull_dataSet = image_dataset_from_directory(
    dogCat_training_directory,
    image_size=(224, 224),
    batch_size=None,
    shuffle=True,
)

dogCat_testFull_dataSet = image_dataset_from_directory(
    dogCat_testing_directory,
    image_size=(224, 224),
    batch_size=32
)

train_class_names = image_dataset_from_directory(
    dogCat_training_directory,
    image_size=(224, 224),
    batch_size=32
).class_names

print("Train classes:", train_class_names)

# ====== Pre Process Data ======
dogCat_trainFull_dataSet = dogCat_trainFull_dataSet.map(lambda x, y: (preprocess_input(x), y))
dogCat_trainFull_dataSet = dogCat_trainFull_dataSet.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ====== Convert Full Training Data to Numpy for Sampling ======
dogCat_x_full, dogCat_y_full = zip(*list(dogCat_trainFull_dataSet))
dogCat_x_full = np.stack(dogCat_x_full)
dogCat_y_full = np.array(dogCat_y_full)

unique, counts = np.unique(dogCat_y_full, return_counts=True)
print(dict(zip(unique, counts)))

# ====== Subset Sizes ======
subset_sizes = [10, 50, 200, 500, 2000, len(dogCat_x_full)]
# subset_sizes = [len(dogCat_x_full)]
num_classes = 2
epochs = 5


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

for n in subset_sizes:
    print(f"\n Training with {n} samples")

    # Get random selection of subset of n samples
    idx = random.sample(range(len(dogCat_x_full)), n)
    x_subset = preprocess_input(dogCat_x_full[idx])
    y_subset = dogCat_y_full[idx]

    # Create TensorFlow DataSet
    dogCat_trainFull_dataSet = tf.data.Dataset.from_tensor_slices((x_subset, y_subset)).batch(32)
    dogCat_trainFull_dataSet = dogCat_trainFull_dataSet.cache().shuffle(1000).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # Build and Train Model
    model = build_model(num_classes)
    history = model.fit(dogCat_trainFull_dataSet, epochs=epochs, verbose=1)

    # Evaluate on Test Set
    test_loss, test_acc = model.evaluate(dogCat_testFull_dataSet, verbose=0)
    results[n] = test_acc
    print(f"Test accuracy with {n} samples: {test_acc:.4f}")

# ====== Plot Results ======
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xscale('log')
plt.xlabel('Training Dataset Size (samples)')
plt.ylabel('Test Accuracy')
plt.title('Effect of Training Dataset Size on Model Performance')
plt.grid(True)
plt.savefig("dogsAndCatsModelTraining.png")
plt.close()

# ====== Summary Table ======
print("\n Summary of Results:")
for n, acc in results.items():
    print(f"{n:>6} Samples -> Test Accuracy = {acc:.4f}")
