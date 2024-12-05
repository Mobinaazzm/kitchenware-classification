from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_cnn_model():
    """
    Builds and compiles a Convolutional Neural Network (CNN) for kitchenware classification.

    Returns:
        Sequential: A compiled CNN model.
    """
    # Define the CNN architecture
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Output layer (4 classes)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_callbacks():
    """
    Defines and returns the callbacks for training the CNN model.

    Returns:
        list: A list of callback functions.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',     
        patience=5,             # Stop training if no improvement for 5 epochs
        restore_best_weights=True  # Restore the best weights after stopping
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',     
        factor=0.5,             
        patience=3,            
        min_lr=1e-6             
    )

    return [early_stopping, reduce_lr]


