import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_crop(image, model_obb):
    # Run detection on the image array
    results = model_obb(image, conf=0.2)
    obb_data = results[0].obb
    if obb_data is None:
        return None, "No objects detected"
    
    # Crop the first detected meter region
    for i, class_id in enumerate(obb_data.cls.cpu().numpy()):
        corners_flat = obb_data.xyxyxyxy.cpu().numpy()[i]
        corners = np.array(corners_flat).reshape(4, 2)
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        return cropped_image, None
    return None, "No meter detected"

def get_meter_reading(cropped_image, model):
    results = model(cropped_image, conf=0.2)
    obb_data = results[0].obb
    class_names = results[0].names

    if obb_data is None:
        return "No objects detected"

    number_positions = []
    needle_corners = None
    number_values = [0, 5, 10, 15, 20, 25, 30]
    
    for i, class_id in enumerate(obb_data.cls.cpu().numpy()):
        class_name = class_names[int(class_id)]
        if hasattr(obb_data, "xyxyxyxy") and obb_data.xyxyxyxy is not None:
            corners_flat = obb_data.xyxyxyxy.cpu().numpy()[i]
            corners = np.array(corners_flat).reshape(4, 2)
            
            if class_name.lower() == "needle":
                needle_corners = corners
            elif class_name.lower() == "numbers":
                center_x = np.mean(corners[:, 0])
                number_positions.append((corners, center_x))
    
    number_positions.sort(key=lambda x: x[1])
    
    # More comprehensive data for labeled numbers
    labeled_numbers = []
    for i, (corners, center_x) in enumerate(number_positions):
        if i < len(number_values):
            value = number_values[i]
            min_x = np.min(corners[:, 0])
            max_x = np.max(corners[:, 0])
            labeled_numbers.append((corners, value, center_x, min_x, max_x))
    
    interpolated_value = None
    if needle_corners is not None and labeled_numbers:
        # Calculate average x position of needle tips (corners 3 and 4)
        needle_tip1_x = needle_corners[2][0]  # Corner 3 x-coordinate
        needle_tip2_x = needle_corners[3][0]  # Corner 4 x-coordinate
        needle_tip_avg_x = (needle_tip1_x + needle_tip2_x) / 2
        
        # Find the two adjacent number values
        left_value = None
        right_value = None
        left_center_x = None
        right_center_x = None
        
        # Sort labeled_numbers by center_x for finding left and right neighbors
        sorted_numbers = sorted(labeled_numbers, key=lambda x: x[2])
        
        # Find the two values that the needle is between
        for corners, value, center_x, min_x, max_x in sorted_numbers:
            if center_x <= needle_tip_avg_x:
                left_value = value
                left_center_x = center_x
            else:
                right_value = value
                right_center_x = center_x
                break
        
        # If needle is before the first value
        if left_value is None and right_value is not None:
            interpolated_value = right_value
            
        # If needle is after the last value
        elif right_value is None and left_value is not None:
            interpolated_value = left_value
            
        # If needle is between two values, interpolate
        elif left_value is not None and right_value is not None:
            # Calculate the ratio based on x positions
            total_distance = right_center_x - left_center_x
            if total_distance > 0:  # Avoid division by zero
                needle_distance = needle_tip_avg_x - left_center_x
                ratio = needle_distance / total_distance
                
                # Interpolate between the two values
                value_range = right_value - left_value
                interpolated_value = left_value + (ratio * value_range)
                # Round to 1 decimal place
                interpolated_value = round(interpolated_value, 1)
    
    return interpolated_value
