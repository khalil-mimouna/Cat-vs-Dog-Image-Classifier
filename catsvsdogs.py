import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset():
    (train_dataset, test_dataset), dataset_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], shuffle_files=True, with_info=True, as_supervised=True)
    
    # Resize and normalize the images
    def preprocess_image(image, label):
        image = tf.image.resize(image, (150, 150))
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label
    
    train_dataset = train_dataset.map(preprocess_image)
    test_dataset = test_dataset.map(preprocess_image)
    
    return train_dataset, test_dataset, dataset_info


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_dataset, test_dataset, dataset_info = load_dataset()

num_train_examples = dataset_info.splits['train[:80%]'].num_examples
num_test_examples = dataset_info.splits['train[80%:]'].num_examples

BATCH_SIZE = 32

train_dataset = train_dataset.shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = build_model()

model.fit(train_dataset, epochs=10, validation_data=test_dataset)