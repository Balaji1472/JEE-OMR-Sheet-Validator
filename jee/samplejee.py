import cv2
import numpy as np
import json
from datetime import datetime
import csv

def load_template(json_path):
    """Load template configuration"""
    with open(json_path, 'r') as f:
        return json.load(f)

def enhance_image(image):
    """
    Enhanced image preprocessing specifically tuned for OMR sheets
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply adaptive thresholding with optimized parameters
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,  # Increased block size for better local adaptation
        C=15  # Increased constant for better bubble detection
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def detect_marked_bubble(binary_image, x, y, bubble_dims, threshold=0.45):
    """
    Improved bubble detection with support for multiple marked answers
    Returns confidence score without early threshold filtering
    """
    bubble_width, bubble_height = bubble_dims
    half_width, half_height = bubble_width // 2, bubble_height // 2
    
    # Extract ROI with padding
    padding = 2
    roi = binary_image[
        max(0, y - half_height - padding):min(binary_image.shape[0], y + half_height + padding),
        max(0, x - half_width - padding):min(binary_image.shape[1], x + half_width + padding)
    ]
    
    if roi.size == 0:
        return 0.0
    
    # Calculate fill ratio
    total_pixels = roi.size
    filled_pixels = cv2.countNonZero(roi)
    fill_ratio = filled_pixels / float(total_pixels)
    
    # Analyze local contrast
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)
    
    # Calculate confidence score based on multiple factors
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        expected_area = np.pi * (min(bubble_width, bubble_height)/2)**2
        area_ratio = contour_area / expected_area
        
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
        
        confidence = (
            fill_ratio * 0.4 +
            (1.0 - abs(1.0 - area_ratio)) * 0.3 +
            circularity * 0.3
        )
        
        return confidence
    
    return 0.0

def process_field_block(binary_image, block_config, output_image=None):
    """
    Process a field block with support for multiple marked answers
    """
    origin_x, origin_y = block_config['origin']
    h_gap = block_config['bubblesGap']
    v_gap = block_config['labelsGap']
    bubble_dims = block_config.get('bubbleDimensions', [12, 12])
    threshold = block_config.get('threshold', 0.45)
    
    is_vertical = block_config.get('direction', 'horizontal') == 'vertical'
    num_options = len(block_config['bubbleValues'])
    
    # Handle range notation in field labels
    if '..' in block_config['fieldLabels'][0]:
        label_parts = block_config['fieldLabels'][0].split('..')
        base_label = ''.join(filter(str.isalpha, label_parts[0]))
        start_num = int(''.join(filter(str.isdigit, label_parts[0])))
        end_num = int(label_parts[1])
        num_questions = end_num - start_num + 1
    else:
        num_questions = len(block_config['fieldLabels'])
    
    results = []
    confidence_scores = []
    
    for q in range(num_questions):
        question_scores = []
        marked_options = []
        
        for opt in range(num_options):
            if is_vertical:
                x = int(origin_x + (q * h_gap))
                y = int(origin_y + (opt * v_gap))
            else:
                x = int(origin_x + (opt * h_gap))
                y = int(origin_y + (q * v_gap))
            
            confidence = detect_marked_bubble(binary_image, x, y, bubble_dims)
            question_scores.append(confidence)
            
            if confidence > threshold:
                marked_options.append(opt)
            
            # Visualization
            if output_image is not None:
                if confidence > threshold:
                    color = (0, int(255 * confidence), 0)
                    thickness = 2
                else:
                    color = (0, 0, 255)
                    thickness = 1
                cv2.circle(output_image, (x, y), bubble_dims[0]//2, color, thickness)
        
        # Determine result based on marked options
        if len(marked_options) > 1:
            results.append("Multiple")
        elif len(marked_options) == 0:
            results.append(-1)  # No answer
        else:
            results.append(marked_options[0])
        
        confidence_scores.append(question_scores)
    
    return results, confidence_scores

def save_results_to_csv(results_dict, template_config, filename="omr_results.csv"):
    """
    Save results in horizontal format with options (A, B, C, D) and handle 'No Answer' and 'Multiple'.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row
        header = ["roll_number"] + [f"p{i}" for i in range(1, 31)] \
                 + [f"c{i}" for i in range(31, 61)] + [f"m{i}" for i in range(61, 91)]
        writer.writerow(header)
        
        # Write results row
        row = [results_dict['roll_number']]
        for section in ['physics', 'chemistry', 'mathematics']:
            answers = results_dict.get(section, [])
            for answer in answers:
                if answer == "Multiple":
                    row.append("Multiple")
                elif answer == -1:
                    row.append("No Answer")
                else:
                    # Convert numerical index to letter option
                    row.append(chr(65 + answer))  # A=0, B=1, C=2, D=3
        writer.writerow(row)



def process_omr_sheet(image_path, template_path):
    """
    Main processing function with results for roll number, physics, chemistry, and mathematics.
    """
    # Load template and image
    template_config = load_template(template_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create output image and enhance input image
    output_image = image.copy()
    binary_image = enhance_image(image)
    
    # Process fields
    results = {}
    
    # Process roll number
    roll_block = template_config['fieldBlocks']['RollNumber']
    roll_digits, _ = process_field_block(binary_image, roll_block, output_image)
    results['roll_number'] = ''.join(str(d) if d != -1 else 'X' for d in roll_digits)
    
    # Process subject sections
    for section_name, section_config in template_config['fieldBlocks'].items():
        if section_name != 'RollNumber':
            section_results, _ = process_field_block(binary_image, section_config, output_image)
            results[section_name.lower()] = section_results
    
    # Save results
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #csv_filename = f"omr_results.csv"
    #save_results_to_csv(results, template_config, csv_filename)
    
    # Save visualization
    #output_filename = f"processed_omr_{timestamp}.jpg"
    #cv2.imwrite(output_filename, output_image)
    
    return results, output_image


def main():
    """
    Main entry point
    """
    image_path = "corrected_omr.jpg"
    template_path = "template.json"
    
    try:
        process_omr_sheet(image_path, template_path)
    except Exception as e:
        print(f"Error processing OMR sheet: {str(e)}")

if __name__ == "__main__":
    main()
