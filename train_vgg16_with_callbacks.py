from src.data_preprocessing import get_data_generators
from src.transfer_learning import build_vgg16_model, get_vgg16_callbacks

# Dataset paths
TRAIN_DIR = 'datasets/train'
VAL_DIR = 'datasets/validation'
TEST_DIR = 'datasets/test'

# Load data generators
train_gen, val_gen, test_gen = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

# Build the VGG16 model
model = build_vgg16_model()

# Get callbacks for training
callbacks = get_vgg16_callbacks()

# Phase 1: Train the model with frozen base layers
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  # Number of epochs for initial training
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size,
    callbacks=callbacks,  # Apply early stopping and learning rate reduction
    verbose=1
)

# Unfreeze the last few layers of the base model for fine-tuning
for layer in model.layers[0].layers[-8:]:  # Unfreeze the last 8 layers of VGG16
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Phase 2: Fine-tune the model
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  # Additional epochs for fine-tuning
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size,
    callbacks=callbacks,
    verbose=1
)

# Save the fine-tuned model
model.save('kitchenware_vgg16_finetuned_model.h5')
