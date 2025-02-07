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

def detect_and_recognize(image_path: str) -> Dict[str, Dict[str, Any]]:
    results = model.predict(source=image_path, conf=0.15, save=False)
    output = {}

    for result in results:
        for box in result.boxes:
            # Extract detection info
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = model.names[class_id]
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
                    "text": "",
                    "conf": 0
                }

            if ocr_texts:
                output[class_name]["text"] = " ".join(ocr_texts)
                output[class_name]["conf"] = sum(ocr_confs) / len(ocr_confs)

    return output

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
