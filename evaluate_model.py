from tensorflow.keras.models import load_model
from src.data_preprocessing import get_data_generators

TEST_DIR = 'datasets/test'
_, _, test_gen = get_data_generators(None, None, TEST_DIR)

model = load_model('kitchenware_model.h5')
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc * 100:.2f}%")
