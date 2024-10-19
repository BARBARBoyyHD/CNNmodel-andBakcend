import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define the image data generator for training
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # normalize images and split for validation

train_generator = datagen.flow_from_directory(
    'train/',             # Path to the train folder
    target_size=(224, 224),  # Resize images to 224x224 for MobileNet
    batch_size=32,          # Number of images per batch
    class_mode='binary',    # If you're doing binary classification (cats vs. not-cats)
    subset='training'       # Use the training subset
)

validation_generator = datagen.flow_from_directory(
    'train/',             # Same path for validation
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'    # Use the validation subset
)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model so its weights won't change during training
base_model.trainable = False

# Add new classification layers on top of the pre-trained model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # For binary classification, use 1 output neuron with sigmoid
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,   # You can adjust the number of epochs
    validation_data=validation_generator
)

# Unfreeze the base model
base_model.trainable = True

# Re-compile the model after unfreezing
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    epochs=10,   # Fine-tune for more epochs
    validation_data=validation_generator
)
# Save the trained model
model.save('cat_classifier.h5')

# For inference (to classify new images), load the model
model = tf.keras.models.load_model('cat_classifier.h5')
