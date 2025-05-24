from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import tensorflow as tf
import easyocr
from new_apparatus import draw_obb
from res_temp_N2N import TwoStageOCR, order_points
from Cnn_res import YoloCNN_OCR  
from Remaining_test import draw_obb  
from analog import crop_region, calculate_meter_reading, get_center_point

app = FastAPI()

try:
    analog_box = YOLO("Models/analog_box_v2.pt")
    analog_reading = YOLO("Models/analog_reading_v2.pt")
    remaining_test_model = YOLO("Models/Remaining_tests_model.pt")
    new_apparatus_model = YOLO("Models/New_Apparatus_model.pt")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

reader = easyocr.Reader(['en'])

def process_res_temp(file_bytes):
    try:
        # Load models (do this once globally in production)
        box_model = 'Models/res_temp_box.pt'
        temp_yolo_model = 'Models/temp_detection.pt'
        temp_cnn_model  = 'Models/lenet7seg.h5'
        res_yolo_model = 'Models/res_detect.pt'
        res_cnn_model  = 'Models/digit_cnn_model_1.2.h5'

        from res_temp_N2N import TwoStageOCR, order_points
        from Cnn_res import YoloCNN_OCR

        temp_ocr = TwoStageOCR(
            box_model_path=box_model,
            yolo_model_path=temp_yolo_model,
            cnn_model_path=temp_cnn_model,
            image_size=(28,28),
            conf_threshold=0.3
        )
        res_ocr = YoloCNN_OCR(
            yolo_model_path=res_yolo_model,
            cnn_model_path=res_cnn_model,
            image_size=(128,128),
            conf_threshold=0.5
        )

        # Decode image
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for processing")

        # Detect panels (res or temp)
        box_detector = temp_ocr.box_detector
        res_panels = box_detector.predict(source=image_cv, verbose=False)[0]
        if not hasattr(res_panels, 'obb') or res_panels.obb is None:
            raise HTTPException(status_code=400, detail="No panels detected")

        polys = res_panels.obb.xyxyxyxy.cpu().numpy()
        confs = res_panels.obb.conf.cpu().numpy()
        class_ids = res_panels.obb.cls.cpu().numpy().astype(int)
        class_names = box_detector.model.names  # Should be ['res', 'temp']

        results = []
        confidence_scores = []
        for poly, conf, cls_id in zip(polys, confs, class_ids):
            if conf < 0.3:
                continue
            pts = poly.reshape(4, 2).astype(np.float32)
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
            M = cv2.getPerspectiveTransform(rect, dst)
            crop = cv2.warpPerspective(image_cv, M, (maxW, maxH))

            class_name = class_names[cls_id]
            if class_name == 'temp':
                raw = temp_ocr.ocr_panel(crop)
                formatted = raw
                if len(raw) > 2:
                    formatted = raw[:2] + '.' + raw[2:]
                if formatted.endswith('C'):
                    formatted = f"{formatted[:-1]}°C"
                results.append({
                    "keyName": "Temperature",
                    "keyValue": formatted,
                    "actualValue": formatted,
                    "confidenceScore": round(float(conf), 2)
                })
                confidence_scores.append(float(conf))
            elif class_name == 'res':
                # Save crop to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, crop)
                    tmp_path = tmp.name
                raw = res_ocr.ocr_image(tmp_path)
                os.remove(tmp_path)  # Clean up temp file
                results.append({
                    "keyName": "Resistance",
                    "keyValue": raw,
                    "actualValue": raw,
                    "confidenceScore": round(float(conf), 2)
                })
                confidence_scores.append(float(conf))

        if not results:
            raise HTTPException(status_code=400, detail="No valid panels detected")

        overall_confidence = round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.75

        return {"ocs": overall_confidence, "extractions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")  

def process_remaining_test(file_bytes, expected_classes):
    try:
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for processing")
        
        # Run inference using the remaining tests model
        results = remaining_test_model(image_cv)
        
        extracted_data = {}
        confidence_scores = {}
        
        # Process results and extract text from detected regions
        for r in results:
            if r.obb is not None:
                # Get confidence scores from the detections
                confidences = r.obb.conf.cpu().numpy() if hasattr(r.obb, 'conf') else None
                
                # Use the draw_obb function from Remaining_test.py to extract text
                _, extracted_texts = draw_obb(image_cv.copy(), r.obb)
                
                # Match the extracted texts with their class names
                for i, class_id in enumerate(r.obb.cls.cpu().numpy()):
                    class_name = r.names[int(class_id)]
                    
                    # Only process classes that we expect for this test type
                    if class_name in expected_classes and i < len(extracted_texts) and extracted_texts[i]:
                        # Store the detected text with its class name
                        extracted_data[class_name] = extracted_texts[i]
                        
                        # Store confidence score if available
                        if confidences is not None and i < len(confidences):
                            confidence_scores[class_name] = float(confidences[i])
                        else:
                            confidence_scores[class_name] = 0.75  # Default fallback
        
        # Calculate overall confidence score (average of individual scores)
        overall_confidence = 0.0
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            overall_confidence = 0.75  # Default if no scores available
            
        # Round overall confidence to 2 decimal places
        overall_confidence = round(overall_confidence, 2)
        
        # Format response with individual rounded confidence scores
        kv_list = []
        for k, v in extracted_data.items():
            conf = round(confidence_scores.get(k, 0.75), 2)
            kv_list.append({
                "keyName": k, 
                "keyValue": v, 
                "actualValue": v, 
                "confidenceScore": conf
            })
        
        # Determine test type based on expected classes
        test_type = "extractions"
        
        # If no data was extracted
        if not kv_list:
            raise HTTPException(status_code=400, detail=f"No data extracted for the expected classes: {expected_classes}")
            
        return {"ocs": overall_confidence, test_type: kv_list}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing test data: {str(e)}")

def process_dc_test(file_bytes):
    """
    Implements the DC_TEST pipeline using functions from analog.py.
    It decodes the image, ensures consistent color format, detects and crops the meter
    region using the analog_box model, and then uses the analog_reading model along with
    calculate_meter_reading and get_center_point to compute the meter reading.
    """
    try:
        # Decode file bytes into a CV image (BGR)
        image_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data for DC_TEST")
        
        results = analog_box(image_cv)
        cropped_meter = None
        for r in results:
            if hasattr(r, "obb") and r.obb is not None:
                cropped_meter = crop_region(image_cv, r.obb)
                if cropped_meter is not None:
                    break
        
        if cropped_meter is None:
            raise HTTPException(status_code=400, detail="No analog meter detected in image")
        
        meter_results = analog_reading(cropped_meter)
        needle_corners = None
        needle_corners = None
        number_positions = []
        needle_confidence = 0
        number_confidences = []
        
        for r in meter_results:
            if hasattr(r, "obb") and r.obb is not None:
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                confidences = r.obb.conf.cpu().numpy()  # Get confidence scores
                
                for box, class_id, conf in zip(boxes, classes, confidences):
                    class_name = r.names[int(class_id)]
                    center = get_center_point(box)
                    
                    if class_name.lower() == "needle":
                        needle_corners = box.reshape(4, 2)
                        needle_confidence = float(conf)
                    elif (class_name.isdigit() or 
                          class_name in ["0", "5", "10", "15", "20", "25", "30"] or 
                          class_name.lower() == "numbers"):
                        number_positions.append((0, center))
                        number_confidences.append(float(conf))
        
        if needle_corners is not None and number_positions:
            reading, method = calculate_meter_reading(needle_corners, number_positions)
            
            overall_confidence = (2 * needle_confidence + sum(number_confidences)) / (2 + len(number_confidences))
            overall_confidence = round(overall_confidence, 2)
            
            reading = round(float(reading), 2)
            
            list = [{
                "keyName": "MeterReading",
                "keyValue": str(reading),
                "actualValue": str(reading),
                "confidenceScore": overall_confidence,
            }]
            
            return {
                "ocs": overall_confidence,
                "extractions": list
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DC_TEST: {str(e)}")


@app.post("/detect/")
async def detect(file: UploadFile = File(...), test_type: str = Form(...)):
    file_bytes = await file.read()
    if test_type == "CONDUCTOR_RESISTANCE_TEST":
        return process_res_temp(file_bytes)
    elif test_type == "DC_TEST":
        return process_dc_test(file_bytes)
    elif test_type == "PARTIAL_DISCHARGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["UVolt", "qCValue"])
    elif test_type == "HIGH_VOLTAGE_TEST":
        return process_remaining_test(file_bytes, expected_classes=["kV", "TimeLeft", "q(IEC) value"])
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Choose 'CONDUCTOR_RESISTANCE_TEST', 'DC_TEST', 'PARTIAL_DISCHARGE_TEST', or 'HIGH_VOLTAGE_TEST'")

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "v2.1"}