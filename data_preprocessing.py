from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(training_dir, validation_dir, test_dir):

    """
    Creates data generators for training, validation, and testing datasets.

    Args:
        training_dir (str): Path to the training dataset.
        validation_dir (str): Path to the validation dataset.
        test_dir (str): Path to the testing dataset.

    Returns:
        tuple: A tuple containing training, validation, and testing generators.
    """
    # Data augmentation for training
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling for validation and test datasets
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = training_datagen.flow_from_directory(
        training_dir, target_size=(150, 150), batch_size=64, class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=64, class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=64, class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator

