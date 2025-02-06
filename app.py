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
            class_name = results.names[class_id]  
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

    return output

@app.post("/test/")
async def test(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Process image
        detected_data = detect_and_recognize(image_path)
        extracted_classes: Set[str] = set(detected_data.keys())

        # Cleanup temp file
        os.remove(image_path)

        # Prepare detailed results
        detailed_results = {}
        for class_name, data in detected_data.items():
            combined_conf = sum(data["ocr_confs"]) / len(data["ocr_confs"]) if data["ocr_confs"] else 0
            detailed_results[class_name] = {
                "ocr_texts": data["ocr_texts"],
                "combined_conf": combined_conf
            }

        return JSONResponse(content={
            "extracted_classes": list(extracted_classes),
            "detailed_results": detailed_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "1.0.3"}
