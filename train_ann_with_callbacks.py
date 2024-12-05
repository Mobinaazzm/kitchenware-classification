from src.data_preprocessing import get_data_generators
from src.ann_model import build_ann_model, get_ann_callbacks

# Dataset paths
TRAIN_DIR = 'datasets/train'
VAL_DIR = 'datasets/validation'
TEST_DIR = 'datasets/test'

# Load data generators
train_gen, val_gen, test_gen = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

# Build the ANN model
model = build_ann_model()

# Get callbacks for training
callbacks = get_ann_callbacks()

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,  # Define the number of epochs
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size,
    callbacks=callbacks,  # Use callbacks for early stopping and learning rate reduction
    verbose=1
)

# Save the trained model
model.save('kitchenware_ann_model_with_callbacks.h5')
