from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_vgg16_model():
    """
    Builds a transfer learning model using VGG16 as the base model.

    Returns:
        Sequential: A compiled VGG16-based model.
    """
    # Load pre-trained VGG16 as the base model, excluding its fully connected layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    
    # Freeze all layers of the base model to retain pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of VGG16
    model = Sequential([
        base_model,                           
        GlobalAveragePooling2D(),            # Pool the feature maps into a single vector
        Dense(1024, activation='relu'),      
        Dropout(0.5),                        
        Dense(256, activation='relu'),       
        Dropout(0.5),
        Dense(4, activation='softmax')       # Output layer for 4 classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_vgg16_callbacks():
    """
    Defines and returns the callbacks for training the VGG16 model.

    Returns:
        list: A list of callback functions.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',         
        patience=5,                 # Stop training if no improvement for 5 epochs
        restore_best_weights=True   # Restore the best weights after stopping
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',         
        factor=0.5,                 
        patience=3,                 
        min_lr=1e-6                 
    )

    return [early_stopping, reduce_lr]


