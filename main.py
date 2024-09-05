import tensorflow as tf
from MLP import Dense, MLP
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28*28) / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 28*28) / 255.0    # Flatten and normalize

y_train = to_categorical(y_train, 10)  # One-hot encode the labels
y_test = to_categorical(y_test, 10)    # One-hot encode the labels

# Initialize the MLP model
model = MLP()

# Add layers
model.addlayer(Dense(input_size=784, output_size=128, activation='relu'))
model.addlayer(Dense(input_size=128, output_size=64, activation='relu'))
model.addlayer(Dense(input_size=64, output_size=10, activation='sigmoid'))  # Output layer for 10 classes

# Set hyperparameters
learning_rate = 0.01
epochs = 10
batch_size = 32

# Đảm bảo TensorFlow sử dụng GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Thiết lập để chỉ sử dụng GPU cụ thể (nếu có nhiều GPU)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU: ", gpus[0])
    except RuntimeError as e:
        print(e)

# Train the model
model.fit(x_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)

# Predict on test data
predictions = model.predict(x_test)

# Calculate the accuracy
correct_predictions = np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
accuracy = np.mean(correct_predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


def visualize_predictions(x_test, y_test, predictions, num_images=10):
    # Chọn ngẫu nhiên một số hình ảnh để trực quan hóa
    indices = np.random.choice(len(x_test), num_images, replace=False)

    plt.figure(figsize=(10, 2 * num_images))

    for i, idx in enumerate(indices):
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True Label: {np.argmax(y_test[idx])}")
        plt.axis('off')

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.bar(range(10), predictions[idx])
        plt.title(f"Predicted Label: {np.argmax(predictions[idx])}")
        plt.xticks(range(10))

    plt.tight_layout()
    plt.show()


# Sử dụng hàm để trực quan hóa dự đoán
visualize_predictions(x_test, y_test, predictions, num_images=5)
