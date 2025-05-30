import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import tempfile

# ----------------- Helper Functions -----------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def crop_regions(image: np.ndarray, res, conf_threshold: float=0.6) -> list:
    """
    Crops and deskews regions based on OBB detection, returning (crop, x_min).
    Uses perspective transform to rotate the region upright.
    """
    regions = []
    if hasattr(res, 'obb') and res.obb is not None:
        polys = res.obb.xyxyxyxy.cpu().numpy()
        confs = res.obb.conf.cpu().numpy()
        for poly, conf in zip(polys, confs):
            if conf < conf_threshold:
                continue
            pts = poly.reshape(4, 2).astype(np.float32)
            # Order points and compute destination rectangle
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxW - 1, 0],
                [maxW - 1, maxH - 1],
                [0, maxH - 1]
            ], dtype="float32")

            # Perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(image, M, (maxW, maxH))
            x_min = int(rect[:, 0].min())
            regions.append((warp, x_min))
    return regions

# ----------------- Main OCR Class -----------------
class TwoStageOCR:
    def __init__(
        self,
        box_model_path: str,
        yolo_model_path: str,
        cnn_model_path: str,
        image_size=(28, 28),
        conf_threshold=0.25
    ):
        # Stage 1: panel/region detector
        self.box_detector = YOLO(box_model_path)
        # Stage 2: digit detector for refined localization
        self.digit_detector = YOLO(yolo_model_path)
        # CNN for final classification
        self.cnn = tf.keras.models.load_model(cnn_model_path, compile=False)

        # Embedded class names for LeNet
        class_names = ['0','1','2','3','4','5','6','7','8','9','C','dot']
        self.inv_map = {i: label for i, label in enumerate(class_names)}

        self.image_size = image_size
        self.conf_threshold = conf_threshold

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size)
        normed = resized.astype(np.float32) / 255.0
        return normed.reshape(1, *self.image_size, 1)

    def ocr_panel(self, panel: np.ndarray) -> str:
        """
        Detect digits in a cropped panel and classify using CNN.
        """
        res = self.digit_detector.predict(source=panel, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        # Filter by confidence
        mask = confs >= self.conf_threshold
        boxes = boxes[mask]
        if boxes.size == 0:
            return ""
        # Sort left-to-right
        boxes = boxes[np.argsort(boxes[:, 0])]

        digits = []
        for x1, y1, x2, y2 in boxes:
            c = panel[int(y1):int(y2), int(x1):int(x2)]
            inp = self.preprocess_crop(c)
            probs = self.cnn.predict(inp, verbose=False)
            idx = int(np.argmax(probs, axis=1)[0])
            label = self.inv_map[idx]
            digits.append(label)
        return ''.join(digits)

    def ocr_image(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")

        # Stage 1: detect and crop panels
        res_panels = self.box_detector.predict(source=img, verbose=False)[0]
        panels = crop_regions(img, res_panels)
        if not panels:
            return ""
        # Sort panels by x-coordinate
        panels = sorted(panels, key=lambda x: x[1])

        # Stage 2: OCR each panel
        results = []
        for panel_crop, _ in panels:
            text = self.ocr_panel(panel_crop)
            if text:
                results.append(text)
        return ' '.join(results)


# -------------------
# Example pipeline
# -------------------
# ...existing code...

if __name__ == '__main__':
    box_model = 'Models/res_temp_box.pt'
    temp_yolo_model = 'Models/temp_detection.pt'
    temp_cnn_model  = 'Models/lenet7seg.h5'
    res_yolo_model = 'Models/res_detect.pt'
    res_cnn_model  = 'Models/lenet_res.h5'

    # Initialize both OCRs
    temp_ocr = TwoStageOCR(
        box_model_path=box_model,
        yolo_model_path=temp_yolo_model,
        cnn_model_path=temp_cnn_model,
        image_size=(28,28),
        conf_threshold=0.3
    )
    from Lenet_res import YoloLeNetOCR 
    res_ocr = YoloLeNetOCR(
        yolo_model_path=res_yolo_model,
        lenet_model_path=res_cnn_model, 
        image_size=(28,28),
        conf_threshold=0.5
    )

    input_dir = 'test_images/cd_cr/resistance'
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        full = os.path.join(input_dir, fname)
        img = cv2.imread(full)
        if img is None:
            print(f"Cannot read {full}")
            continue

        # Detect panels (res or temp)
        box_detector = temp_ocr.box_detector
        res_panels = box_detector.predict(source=img, verbose=False)[0]
        if not hasattr(res_panels, 'obb') or res_panels.obb is None:
            print(f"{fname} -> No panels detected")
            continue

        polys = res_panels.obb.xyxyxyxy.cpu().numpy()   
        confs = res_panels.obb.conf.cpu().numpy()
        class_ids = res_panels.obb.cls.cpu().numpy().astype(int)
        class_names = box_detector.model.names  # Should be ['res', 'temp']

        for poly, conf, cls_id in zip(polys, confs, class_ids):
            if conf < 0.3:
                continue
            pts = poly.reshape(4, 2).astype(np.float32)
            rect = order_points(pts)  # <-- FIXED HERE
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([
                [0, 0],
                [maxW - 1, 0],
                [maxW - 1, maxH - 1],
                [0, maxH - 1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            crop = cv2.warpPerspective(img, M, (maxW, maxH))

            class_name = class_names[cls_id]
            if class_name == 'temp':
                raw = temp_ocr.ocr_panel(crop)
                # Remove 'C' if present
                temp_digits = raw.replace('C', '')
                # Only format if at least 2 digits
                if len(temp_digits) > 1:
                    formatted = temp_digits[:-1] + '.' + temp_digits[-1] + '°C'
                else:
                    formatted = temp_digits + '°C'
                print(f"{fname} [TEMP] -> {formatted}")
            elif class_name == 'res':
                # Save crop to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, crop)
                    tmp_path = tmp.name
                raw = res_ocr.ocr_image(tmp_path)
                os.remove(tmp_path)  # Clean up temp file
                raw = raw.replace("dot", ".")  
                print(f"{fname} [RES] -> {raw}")

