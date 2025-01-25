# CODTECH_DS_02
COMPANY   : CODTECH IT SOLUTIONS
NAME      : DEEPSIKA A
INTERN ID : CT08FOT
DOMAIN    : DATA SCIENCE 
DURATION  : 4 WEEKS
MENTOR    : NEELA SANTOSH
This task involves creating a model using Deep learning for image recognition and it gives you a solid foundation for building a CNN-based image recognition model with data augmentation, transfer learning, and evaluation.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
import matplotlib.pyplot as plt
# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to categorical (one-hot encoding)
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Check the shape of the dataset
print(f'Training images shape: {train_images.shape}')
print(f'Test images shape: {test_images.shape}')
# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)

# Fit the generator on the training data
train_datagen.fit(train_images)
# Define the CNN model architecture
model = models.Sequential()

# Add convolutional layers with ReLU activation and MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# Add fully connected layers
model.add(layers.Dense(64, activation='relu'))

# Output layer with softmax activation (for multi-class classification)
model.add(layers.Dense(10, activation='softmax'))

# Summary of the model
model.summary()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# Train the model with data augmentation
history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=64),
                    epochs=10,
                    validation_data=(test_images, test_labels))
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
# Load the pretrained ResNet50 model (excluding the top layers)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of ResNet50
resnet_model.trainable = False

# Create a new model with ResNet50 as the base
model_transfer = models.Sequential([
    resnet_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])

# Compile the transfer learning model
model_transfer.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Train the model with the augmented data
history_transfer = model_transfer.fit(train_datagen.flow(train_images, train_labels, batch_size=64),
                                      epochs=10,
                                      validation_data=(test_images, test_labels))
# Evaluate the transfer learning model on the test set
test_loss_transfer, test_acc_transfer = model_transfer.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy (Transfer Learning): {test_acc_transfer}')
# Plot training and validation accuracy
plt.plot(history_transfer.history['accuracy'], label='Training Accuracy')
plt.plot(history_transfer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history_transfer.history['loss'], label='Training Loss')
plt.plot(history_transfer.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


