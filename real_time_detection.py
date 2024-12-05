import cv2
import numpy as np
from tensorflow.keras.models import load_model

def run_real_time_detection(model_path, labels):
    """
    Runs real-time detection using a trained model and a webcam.

    Args:
        model_path (str): Path to the trained model.
        labels (dict): Dictionary of class labels.
    """
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, (150, 150))
        img = np.expand_dims(img.astype('float32') / 255.0, axis=0)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = f"{labels[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Real-Time Kitchenware Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

