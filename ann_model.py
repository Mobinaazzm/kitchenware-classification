import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the ANN model
def build_ann_model():
    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),

        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),

        Dense(4, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Define callbacks
def get_ann_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    return [early_stopping, reduce_lr]

# Example usage
if __name__ == "__main__":
    # Assume train_generator and validation_generator are already defined
    from data_preprocessing import get_data_generators

    # Dataset paths
    TRAIN_DIR = 'datasets/train'
    VAL_DIR = 'datasets/validation'

    # Load data generators
    train_generator, validation_generator, _ = get_data_generators(TRAIN_DIR, VAL_DIR, None)

    # Build the model
    model = build_ann_model()

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(
        train_generator,
        epochs=30,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        verbose=1,
        callbacks=get_ann_callbacks()
    )

    # Save the trained model
    model.save('kitchenware_ann_model_with_callbacks.h5')

