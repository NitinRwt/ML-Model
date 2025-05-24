import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Define class map: label name -> integer index
class_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'C': 10, 'dot': 11
}

class YoloCNN_OCR:
    def __init__(
        self,
        yolo_model_path: str,
        cnn_model_path: str,
        image_size: tuple = (128, 128),
        conf_threshold: float = 0.25
    ):
        # Load YOLO detector
        self.detector = YOLO(yolo_model_path)
        # Load CNN classifier
        self.cnn = load_model(cnn_model_path)
        # Invert class_map: index -> label name
        self.inv_map = {v: k for k, v in class_map.items()}
        # Parameters
        self.image_size = image_size
        self.conf_threshold = conf_threshold

    def ocr_image(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")

        # 1) Detect with YOLO
        res = self.detector.predict(source=img, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        # Filter by confidence
        mask = confs >= self.conf_threshold
        boxes = boxes[mask]

        if len(boxes) == 0:
            return ""

        # 2) Sort detections left-to-right
        boxes = boxes[np.argsort(boxes[:, 0])]

        digits = []
        for x1, y1, x2, y2 in boxes:
            # Crop detection
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            # Preprocess for CNN
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.image_size)
            inp = resized.astype("float32") / 255.0
            inp = inp.reshape(1, self.image_size[0], self.image_size[1], 1)
            # Predict
            probs = self.cnn.predict(inp, verbose=False)
            label_int = np.argmax(probs, axis=1)[0]
            label = self.inv_map[label_int]
            digits.append('.' if label == 'dot' else label)

        return "".join(digits)

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    yolo_model_path = "Models/res_detect.pt"
    cnn_model_path = "Models/digit_cnn_model_1.2.h5"
    test_image_path = "new_data/32.jpg"

    ocr = YoloCNN_OCR(
        yolo_model_path=yolo_model_path,
        cnn_model_path=cnn_model_path,
        image_size=(128, 128),
        conf_threshold=0.5
    )
    raw = ocr.ocr_image(test_image_path)
    print(f"Processed OCR: {raw}")
