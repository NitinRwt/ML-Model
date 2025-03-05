from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from typing import Dict, Any, Set

app = FastAPI()

# Initialize models
model = YOLO("Models/model_1_0_2.pt")
reader = easyocr.Reader(["en"])

def calculate_combined_confidence(yolo_confs: list, ocr_confs: list) -> float:
    """Calculate weighted confidence score from YOLO and OCR results"""
    if not yolo_confs and not ocr_confs:
        return 0.0
        
    yolo_avg = sum(yolo_confs) / len(yolo_confs) if yolo_confs else 0.0
    ocr_avg = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0
    
    # Weighted average (adjust weights as needed)
    return round((yolo_avg * 0.5 + ocr_avg * 0.5) * 100, 1)

def detect_and_recognize(image_path: str) -> Dict[str, Dict[str, Any]]:
    results = model.predict(source=image_path, conf=0.15, save=False)
    output = {}

    for result in results:
        for box in result.boxes:
            # Extract detection info
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = result.names[class_id]
            yolo_conf = box.conf.item()

            # Process image region
            image = cv2.imread(image_path)
            cropped = image[y_min:y_max, x_min:x_max]
            
            # OCR processing
            ocr_results = reader.readtext(cropped)
            ocr_texts = []
            ocr_confs = []

            for detection in ocr_results:
                _, text, conf = detection
                ocr_texts.append(text)
                ocr_confs.append(conf)

            # Update output structure
            if class_name not in output:
                output[class_name] = {
                    "yolo_confs": [],
                    "ocr_confs": [],
                    "ocr_texts": []
                }

            output[class_name]["yolo_confs"].append(yolo_conf)
            output[class_name]["ocr_confs"].extend(ocr_confs)
            output[class_name]["ocr_texts"].extend(ocr_texts)

    # Format final response
    final_result = {}
    for class_name, data in output.items():
        final_result[class_name] = {
            "text": " ".join(data["ocr_texts"]) if data["ocr_texts"] else "",
            "conf": calculate_combined_confidence(
                data["yolo_confs"],
                data["ocr_confs"]
            )
        }
    
    return final_result

# Predefined test values with expected YOLO classes
TEST_VALUES = {
    "test1": {"kv", "TimeLeft"},
    "test2": {"res", "temp"},
    "test3": {"res"},
    "test4": {"Volt"},
    "test5": {"qcValue"},
    "test6": {"qValue"},
    "test7": {"temp"},
}

@app.post("/test/")
async def test(test_name: str = Form(...), file: UploadFile = File(...)):
    try:
        if test_name not in TEST_VALUES:
            return JSONResponse(content={
                "error": "Invalid test name provided",
                "valid_tests": list(TEST_VALUES.keys())
            }, status_code=400)

        # Save uploaded image temporarily
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Check if the file is a valid image
        if not os.path.exists(image_path) or not cv2.haveImageReader(image_path):
            os.remove(image_path)
            return JSONResponse(content={
                "error": "No valid image selected. Supported formats are: images: {'jpeg', 'tiff', 'pfm', 'mpo', 'bmp', 'tif', 'heic', 'jpg', 'dng', 'png', 'webp'}"
            }, status_code=400)

        # Process image
        detected_data = detect_and_recognize(image_path)
        extracted_classes: Set[str] = set(detected_data.keys())

        # Cleanup temp file
        os.remove(image_path)

        expected_classes = TEST_VALUES[test_name]

        combined_conf = sum(data["conf"] for data in detected_data.values()) / len(detected_data) if detected_data else 0

        if extracted_classes == expected_classes:
            return JSONResponse(content={
                "results": detected_data,
                "combined_conf": combined_conf
            })
        else:
            return JSONResponse(content={
                "message": "Extracted classes do not match expected classes",
                "expected_classes": list(expected_classes),
                "extracted_classes": list(extracted_classes),
            }, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "11"}
