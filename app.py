from fastapi import FastAPI, HTTPException
import os
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the YOLO model once during startup
model_path = "best.pt"
model = YOLO(model_path)
reader = easyocr.Reader(["en"])

def detect_and_recognize(image_path: str):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    # Load the image
    image = cv2.imread(image_path)

    # Perform inference
    results = model.predict(source=image, save=False, conf=0.15)

    # Dictionary to store final results
    output = {"TimeLeft": None, "kV": None}

    # Process results
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_region = image[y_min:y_max, x_min:x_max]

            # OCR on the cropped region
            ocr_results = reader.readtext(cropped_region)

            # Check each detected text
            for ocr_result in ocr_results:
                text = ocr_result[1]  # Extract the detected text

                # Match and assign the text to keys
                if "TimeLeft" in text or ":" in text:
                    output["TimeLeft"] = text
                elif "kV" in text or text.replace(".", "", 1).isdigit():
                    output["kV"] = text

    return output

@app.get("/detect/")
async def detect_image(image_path: str):
    try:
        # Process the image and get results
        result = detect_and_recognize(image_path)

        # Return the results as JSON response
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

