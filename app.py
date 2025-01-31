from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from typing import Dict, Any

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

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Save temporary file
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Process image
        result = detect_and_recognize(image_path)
        
        # Cleanup
        os.remove(image_path)
        
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "1.0.2"}