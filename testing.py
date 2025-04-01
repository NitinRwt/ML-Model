import sys
import cv2
import numpy as np
from ultralytics import YOLO

def draw_obb(image, obb):
    boxes = obb.xyxyxyxy.cpu().numpy()
    
    for box in boxes:
        pts = box.reshape(4, 2).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return image

def get_center_point(box):
    # Calculate center point of a bounding box
    pts = box.reshape(4, 2)
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    return (center_x, center_y)

def calculate_meter_reading(needle_corners, number_positions):
    # Define standard number values for labeling from left to right
    number_values = [0, 5, 10, 15, 20, 25, 30]
    
    # Sort number positions by x-coordinate (left to right)
    sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
    
    # Replace the detected values with our standardized values
    labeled_positions = []
    for i, (_, position) in enumerate(sorted_positions):
        if i < len(number_values):
            labeled_positions.append((number_values[i], position))
    
    # Calculate midpoint between corner 3 and corner 4 as the needle tip
    needle_tip_x = (needle_corners[2][0] + needle_corners[3][0]) / 2
    needle_tip_y = (needle_corners[2][1] + needle_corners[3][1]) / 2
    needle_tip = np.array([needle_tip_x, needle_tip_y])
    
    # First check if needle is exactly at a number position
    for value, position in labeled_positions:
        distance = np.sqrt((needle_tip[0] - position[0])**2 + (needle_tip[1] - position[1])**2)
        if distance < 15:  # Threshold for "exact match"
            return value, "exact_midpoint"
    
    # If not exact, find the two numbers the needle is between
    left_value = None
    right_value = None
    left_position = None
    right_position = None
    
    for i in range(len(labeled_positions) - 1):
        curr_value, curr_pos = labeled_positions[i]
        next_value, next_pos = labeled_positions[i + 1]
        
        # Check if needle tip is between these two numbers (x-coordinate)
        if curr_pos[0] <= needle_tip[0] <= next_pos[0]:
            left_value = curr_value
            right_value = next_value
            left_position = curr_pos
            right_position = next_pos
            break
    
    # If needle is not between any two numbers, return the closest one
    if left_value is None or right_value is None:
        # Find the closest number
        min_distance = float('inf')
        closest_value = None
        
        for value, position in labeled_positions:
            distance = np.sqrt((needle_tip[0] - position[0])**2 + (needle_tip[1] - position[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_value = value
        
        return closest_value, "closest_midpoint"
    
    # Calculate interpolation based on x-coordinate position
    total_x_distance = right_position[0] - left_position[0]
    needle_x_distance = needle_tip[0] - left_position[0]
    
    # Calculate the ratio (0 to 1) of where the needle is between the two numbers
    ratio = needle_x_distance / total_x_distance if total_x_distance > 0 else 0
    
    # Calculate the interpolated value with one decimal place precision
    value_range = right_value - left_value
    interpolated_value = left_value + (ratio * value_range)
    
    # Round to one decimal place
    interpolated_value = round(interpolated_value, 1)
    
    return interpolated_value, "interpolated_midpoint"

def main(model_path_3, image_path):
    # Load the YOLO OBB model for detection
    model_3 = YOLO(model_path_3)
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image at", image_path)
        sys.exit(1)
    
    # Run inference using model_3 for detection
    results = model_3(image)
    
    # Variables to store needle and number positions
    needle_corners = None
    number_positions = []
    
    # Iterate over the results and process detections
    for r in results:
        if r.obb is not None:
            image = draw_obb(image, r.obb)
            
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            classes = r.obb.cls.cpu().numpy()
            
            for i, (box, class_id) in enumerate(zip(boxes, classes)):
                class_name = r.names[int(class_id)]
                center = get_center_point(box)
                
                # Draw the center point for all detections
                cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
                
                if class_name.lower() == "needle":
                    # Store all corners of the needle
                    needle_corners = box.reshape(4, 2)
                    # Draw the needle corners 3 and 4
                    cv2.circle(image, (int(needle_corners[2][0]), int(needle_corners[2][1])), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(needle_corners[3][0]), int(needle_corners[3][1])), 5, (0, 255, 255), -1)
                    cv2.putText(image, class_name, (int(center[0]), int(center[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif class_name.isdigit() or class_name in ["0", "5", "10", "15", "20", "25", "30"] or class_name.lower() == "numbers":
                    number_positions.append((0, center))  # Store with placeholder value, we'll label them later
                
                print(f"Detected: {class_name} at position {center}")
    
    # Label the numbers from left to right for visualization
    if number_positions:
        number_values = [0, 5, 10, 15, 20, 25, 30]
        sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
        
        # Draw the labels on the image
        for i, (_, position) in enumerate(sorted_positions):
            if i < len(number_values):
                label = str(number_values[i])
                cv2.putText(image, label, 
                           (int(position[0]), int(position[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Calculate the meter reading if we have needle and numbers
    if needle_corners is not None and number_positions:
        # Calculate midpoint between corner 3 and corner 4 for visualization
        needle_tip_x = (needle_corners[2][0] + needle_corners[3][0]) / 2
        needle_tip_y = (needle_corners[2][1] + needle_corners[3][1]) / 2
        needle_tip = np.array([needle_tip_x, needle_tip_y])
        
        # Draw the midpoint
        cv2.circle(image, (int(needle_tip[0]), int(needle_tip[1])), 6, (0, 255, 0), -1)
        
        reading, method = calculate_meter_reading(needle_corners, number_positions)
        if reading is not None:
            result_text = f"Meter reading: {reading}"
            print(result_text)
            
            # Visualize the connection between the used needle midpoint and closest number
            sorted_positions = sorted(number_positions, key=lambda x: x[1][0])
            labeled_positions = []
            for i, (_, position) in enumerate(sorted_positions):
                if i < len(number_values):
                    labeled_positions.append((number_values[i], position))
            
            # Find the two values the needle is between (for interpolation visualization)
            left_pos = None
            right_pos = None
            
            for i in range(len(labeled_positions) - 1):
                curr_value, curr_pos = labeled_positions[i]
                next_value, next_pos = labeled_positions[i + 1]
                
                # Check if needle tip is between these two numbers (x-coordinate)
                if curr_pos[0] <= needle_tip[0] <= next_pos[0]:
                    left_pos = curr_pos
                    right_pos = next_pos
                    break
            
            # Draw visualization lines
            if "interpolated" in method and left_pos is not None and right_pos is not None:
                # Draw lines to both adjacent numbers
                cv2.line(image, 
                         (int(needle_tip[0]), int(needle_tip[1])), 
                         (int(left_pos[0]), int(left_pos[1])), 
                         (255, 0, 255), 1, cv2.LINE_AA)
                cv2.line(image, 
                         (int(needle_tip[0]), int(needle_tip[1])), 
                         (int(right_pos[0]), int(right_pos[1])), 
                         (255, 0, 255), 1, cv2.LINE_AA)
                
                # Draw the interpolation region
                pts = np.array([
                    [int(left_pos[0]), int(left_pos[1])],
                    [int(right_pos[0]), int(right_pos[1])],
                    [int(needle_tip[0]), int(needle_tip[1])]
                ], np.int32)
                cv2.fillPoly(image, [pts], (255, 0, 255, 80))
            else:
                # Find the closest position for non-interpolated readings
                closest_position = None
                min_distance = float('inf')
                
                for _, position in labeled_positions:
                    distance = np.sqrt((needle_tip[0] - position[0])**2 + 
                                     (needle_tip[1] - position[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_position = position
                
                if closest_position is not None:
                    # Draw a line connecting the needle midpoint and the closest number
                    cv2.line(image, 
                            (int(needle_tip[0]), int(needle_tip[1])), 
                            (int(closest_position[0]), int(closest_position[1])), 
                            (255, 0, 255), 2)
            
            # Add interpolation info text
            method_text = f"Method: {method}"
            cv2.putText(image, method_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(image, result_text, (20, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("Needle position is out of range")
    else:
        if needle_corners is None:
            print("Needle not detected")
        if not number_positions:
            print("No numbers detected")
    
    # Display the resulting image with bounding boxes
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path_3 = "Models/analog_reading_v2.pt"
    image_path = "cropped_images/200.png"
    main(model_path_3, image_path)