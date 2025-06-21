import os
import cv2
import base64
import numpy as np
import runpod
import io
import json
import re
import time
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
import requests

# Replicate API token (optional)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Debug flag
DEBUG = True

def log_debug(message):
    """Print debug messages with timestamp"""
    if DEBUG:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

def decode_base64_image(base64_string):
    """Decode base64 image with multiple fallback methods"""
    log_debug("Starting base64 decode process")
    
    # Method 1: Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        log_debug("Removed data URL prefix")
    
    # Method 2: Clean the string
    base64_string = base64_string.strip()
    base64_string = re.sub(r'[^A-Za-z0-9+/]', '', base64_string)
    
    # Method 3: Fix padding
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
        log_debug(f"Added {padding} padding characters")
    
    # Method 4: Try decoding
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log_debug(f"Failed to decode: {str(e)}")
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def detect_actual_line_thickness(gray, bounds):
    """Detect the actual thickness of masking lines up to 200px"""
    x1, y1, x2, y2 = bounds
    max_thickness = min(200, min(x2 - x1, y2 - y1) // 4)
    
    thicknesses = []
    
    # Sample from multiple points for accuracy
    sample_points = min(20, (x2 - x1) // 10)
    
    # Sample from top edge
    for i in range(sample_points):
        offset = int((x2 - x1) * (i + 1) / (sample_points + 1))
        for t in range(1, max_thickness):
            if y1 + t < gray.shape[0]:
                if gray[y1 + t, x1 + offset] > 50:
                    thicknesses.append(t)
                    break
    
    # Sample from left edge
    for i in range(sample_points):
        offset = int((y2 - y1) * (i + 1) / (sample_points + 1))
        for t in range(1, max_thickness):
            if x1 + t < gray.shape[1]:
                if gray[y1 + offset, x1 + t] > 50:
                    thicknesses.append(t)
                    break
    
    # Sample from bottom edge
    for i in range(sample_points):
        offset = int((x2 - x1) * (i + 1) / (sample_points + 1))
        for t in range(1, max_thickness):
            if y2 - t >= 0:
                if gray[y2 - t, x1 + offset] > 50:
                    thicknesses.append(t)
                    break
    
    # Sample from right edge
    for i in range(sample_points):
        offset = int((y2 - y1) * (i + 1) / (sample_points + 1))
        for t in range(1, max_thickness):
            if x2 - t >= 0:
                if gray[y1 + offset, x2 - t] > 50:
                    thicknesses.append(t)
                    break
    
    if thicknesses:
        # Use 75th percentile for robustness
        avg_thickness = int(np.percentile(thicknesses, 75))
        return min(avg_thickness + 10, max_thickness)
    
    return 20

def validate_rectangular_mask(mask, img_w, img_h):
    """Validate if detected mask is rectangular and reasonable"""
    if not mask or 'bounds' not in mask:
        return False
    
    x1, y1, x2, y2 = mask['bounds']
    width = x2 - x1
    height = y2 - y1
    
    # Check if dimensions are reasonable
    if width < img_w * 0.2 or height < img_h * 0.2:
        return False
        
    if width > img_w * 0.98 or height > img_h * 0.98:
        return False
    
    return True

def cross_validate_detection(methods_results, image_shape):
    """Cross-validate results from multiple detection methods"""
    valid_masks = []
    
    for method, mask in methods_results.items():
        if mask and validate_rectangular_mask(mask, image_shape[1], image_shape[0]):
            valid_masks.append(mask)
    
    if not valid_masks:
        return None
    
    # If multiple valid detections, prioritize center masks
    center_masks = [m for m in valid_masks if m['type'] == 'center']
    if center_masks:
        # Return the one with most reasonable bounds
        return max(center_masks, key=lambda m: (m['bounds'][2] - m['bounds'][0]) * (m['bounds'][3] - m['bounds'][1]))
    
    # Otherwise use edge masks
    return max(valid_masks, key=lambda m: (m['bounds'][2] - m['bounds'][0]) * (m['bounds'][3] - m['bounds'][1]))

def detect_center_masking_ultra_advanced(image):
    """Ultra advanced detection for center-positioned masking with multiple methods"""
    log_debug("Starting ultra advanced center masking detection")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    methods_results = {}
    
    # Method 1: Multi-threshold scanning with adaptive levels
    log_debug("Method 1: Multi-threshold adaptive scanning")
    threshold_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
    
    for threshold in threshold_levels:
        binary = gray < threshold
        
        # Find contours
        contours, _ = cv2.findContours(binary.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
            
            # Check if it's a significant rectangular mask
            if cnt_w > w * 0.3 and cnt_h > h * 0.3:
                # Detect actual thickness
                thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                
                # Calculate inner bounds
                inner_x = x + thickness
                inner_y = y + thickness
                inner_w = cnt_w - 2 * thickness
                inner_h = cnt_h - 2 * thickness
                
                if inner_w > 50 and inner_h > 50:
                    methods_results[f'threshold_{threshold}'] = {
                        'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                        'thickness': thickness,
                        'type': 'center',
                        'method': f'threshold_{threshold}'
                    }
                    break
    
    # Method 2: Canny edge detection with multiple parameters
    log_debug("Method 2: Multi-parameter Canny edge detection")
    for low, high in [(30, 100), (50, 150), (20, 80), (40, 120)]:
        edges = cv2.Canny(gray, low, high)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
            
            if cnt_w > w * 0.3 and cnt_h > h * 0.3:
                thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                
                inner_x = x + thickness
                inner_y = y + thickness
                inner_w = cnt_w - 2 * thickness
                inner_h = cnt_h - 2 * thickness
                
                if inner_w > 50 and inner_h > 50:
                    methods_results[f'edge_{low}_{high}'] = {
                        'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                        'thickness': thickness,
                        'type': 'center',
                        'method': f'edge_{low}_{high}'
                    }
                    break
    
    # Method 3: Morphological operations
    log_debug("Method 3: Morphological operations")
    for kernel_size in [5, 7, 9, 11]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Try different thresholds
        for thresh_val in [30, 50, 70]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
                
                if cnt_w > w * 0.3 and cnt_h > h * 0.3:
                    thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                    
                    inner_x = x + thickness
                    inner_y = y + thickness
                    inner_w = cnt_w - 2 * thickness
                    inner_h = cnt_h - 2 * thickness
                    
                    if inner_w > 50 and inner_h > 50:
                        methods_results[f'morph_{kernel_size}_{thresh_val}'] = {
                            'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                            'thickness': thickness,
                            'type': 'center',
                            'method': f'morph_{kernel_size}_{thresh_val}'
                        }
                        break
    
    # Cross-validate all detection methods
    best_mask = cross_validate_detection(methods_results, image.shape)
    
    if best_mask:
        log_debug(f"Center masking detected: {best_mask}")
        return best_mask
    
    return None

def detect_masking_comprehensive(image):
    """Comprehensive masking detection using multiple strategies"""
    log_debug("Starting comprehensive masking detection")
    
    # Strategy 1: Edge-based detection (original)
    edge_mask = detect_edge_based_masking(image)
    
    # Strategy 2: Center-based detection (new)
    center_mask = detect_center_masking_ultra_advanced(image)
    
    # Strategy 3: Contour-based detection
    contour_mask = detect_contour_based_masking(image)
    
    # Combine results
    if center_mask:
        return center_mask
    elif edge_mask:
        return edge_mask
    elif contour_mask:
        return contour_mask
    
    return None

def detect_edge_based_masking(image):
    """Original edge-based masking detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Progressive scanning
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    threshold_values = [20, 30, 40, 50, 60, 70, 80]
    
    for scan_pct, threshold in zip(scan_percentages, threshold_values):
        scan_depth = int(min(w, h) * scan_pct)
        
        edges_found = {
            'top': False, 'bottom': False,
            'left': False, 'right': False
        }
        
        # Check edges
        if np.mean(gray[:scan_depth, :]) < threshold:
            edges_found['top'] = True
        if np.mean(gray[-scan_depth:, :]) < threshold:
            edges_found['bottom'] = True
        if np.mean(gray[:, :scan_depth]) < threshold:
            edges_found['left'] = True
        if np.mean(gray[:, -scan_depth:]) < threshold:
            edges_found['right'] = True
        
        # Need at least 2 edges
        if sum(edges_found.values()) >= 2:
            bounds = [0, 0, w, h]
            
            if edges_found['top']:
                for y in range(scan_depth, h // 2):
                    if np.mean(gray[y, :]) > threshold + 20:
                        bounds[1] = y
                        break
            
            if edges_found['bottom']:
                for y in range(h - scan_depth, h // 2, -1):
                    if np.mean(gray[y, :]) > threshold + 20:
                        bounds[3] = y
                        break
            
            if edges_found['left']:
                for x in range(scan_depth, w // 2):
                    if np.mean(gray[:, x]) > threshold + 20:
                        bounds[0] = x
                        break
            
            if edges_found['right']:
                for x in range(w - scan_depth, w // 2, -1):
                    if np.mean(gray[:, x]) > threshold + 20:
                        bounds[2] = x
                        break
            
            return {
                'bounds': tuple(bounds),
                'thickness': scan_depth,
                'type': 'edge',
                'method': f'edge_scan_{scan_pct}'
            }
    
    return None

def detect_contour_based_masking(image):
    """Contour-based masking detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple threshold attempts
    for thresh_val in [30, 50, 70, 90]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's significant
            if w > image.shape[1] * 0.5 and h > image.shape[0] * 0.5:
                thickness = detect_actual_line_thickness(gray, (x, y, x + w, y + h))
                
                return {
                    'bounds': (x + thickness, y + thickness, 
                              x + w - thickness, y + h - thickness),
                    'thickness': thickness,
                    'type': 'contour',
                    'method': f'contour_{thresh_val}'
                }
    
    return None

def detect_lighting_condition(image, mask_bounds=None):
    """Detect lighting conditions with mask awareness"""
    # Use only the area inside the mask if available
    if mask_bounds:
        x1, y1, x2, y2 = mask_bounds
        roi = image[y1:y2, x1:x2]
    else:
        roi = image
    
    # Convert to LAB color space
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Calculate statistics
    mean_brightness = np.mean(l_channel)
    std_brightness = np.std(l_channel)
    
    # Determine lighting condition
    if mean_brightness < 80:
        return 'low'
    elif mean_brightness > 170:
        return 'high'
    else:
        return 'normal'

def detect_metal_type(image, mask_bounds=None):
    """Detect metal type with improved accuracy"""
    # Use only the area inside the mask if available
    if mask_bounds:
        x1, y1, x2, y2 = mask_bounds
        roi = image[y1:y2, x1:x2]
    else:
        roi = image
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # Calculate color statistics
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    b_mean = np.mean(lab[:, :, 2])  # B channel in LAB
    
    # Metal type detection logic
    if s_mean < 30 and v_mean > 180:
        return 'white_gold'
    elif 10 <= h_mean <= 25 and s_mean > 40:
        return 'rose_gold'
    elif 20 <= h_mean <= 40 and s_mean > 30:
        return 'yellow_gold'
    else:
        return 'champagne_gold'

def get_after_background_color(lighting, metal_type):
    """Get background color from 28 pairs of AFTER training data"""
    # Based on 28 pairs of before/after training data
    after_bg_colors = {
        'low': {
            'white_gold': [195, 195, 200],
            'yellow_gold': [200, 195, 185],
            'rose_gold': [200, 195, 190],
            'champagne_gold': [198, 198, 195]
        },
        'normal': {
            'white_gold': [240, 240, 245],
            'yellow_gold': [245, 240, 230],
            'rose_gold': [245, 240, 235],
            'champagne_gold': [242, 242, 240]
        },
        'high': {
            'white_gold': [250, 250, 255],
            'yellow_gold': [255, 250, 240],
            'rose_gold': [255, 250, 245],
            'champagne_gold': [252, 252, 250]
        }
    }
    
    return after_bg_colors.get(lighting, {}).get(metal_type, [240, 240, 240])

def enhance_wedding_ring_details(image, lighting='normal', metal_type='white_gold'):
    """Enhanced wedding ring detail enhancement with 38 training pairs"""
    enhanced = image.copy()
    
    # v13.3 Complete parameters based on 28 training pairs
    metal_params = {
        'white_gold': {
            'low': {
                'brightness': 1.18, 'contrast': 1.20, 'saturation': 0.80, 
                'highlights': 1.25, 'shadows': 1.05, 'clarity': 1.30,
                'vibrance': 0.85, 'temperature': -5, 'white_overlay': 0.12,
                'denoise': 5, 'sharpen': 1.4, 'gamma': 1.05
            },
            'normal': {
                'brightness': 1.12, 'contrast': 1.15, 'saturation': 0.85,
                'highlights': 1.20, 'shadows': 0.95, 'clarity': 1.25,
                'vibrance': 0.90, 'temperature': -3, 'white_overlay': 0.10,
                'denoise': 3, 'sharpen': 1.3, 'gamma': 1.02
            },
            'high': {
                'brightness': 1.08, 'contrast': 1.10, 'saturation': 0.90,
                'highlights': 0.95, 'shadows': 0.90, 'clarity': 1.20,
                'vibrance': 0.95, 'temperature': -2, 'white_overlay': 0.08,
                'denoise': 2, 'sharpen': 1.2, 'gamma': 0.98
            }
        },
        'yellow_gold': {
            'low': {
                'brightness': 1.15, 'contrast': 1.18, 'saturation': 1.30,
                'highlights': 1.20, 'shadows': 1.00, 'clarity': 1.25,
                'vibrance': 1.35, 'temperature': 5, 'white_overlay': 0.08,
                'denoise': 5, 'sharpen': 1.3, 'gamma': 1.03
            },
            'normal': {
                'brightness': 1.08, 'contrast': 1.12, 'saturation': 1.25,
                'highlights': 1.15, 'shadows': 0.90, 'clarity': 1.20,
                'vibrance': 1.30, 'temperature': 3, 'white_overlay': 0.06,
                'denoise': 3, 'sharpen': 1.2, 'gamma': 1.00
            },
            'high': {
                'brightness': 1.05, 'contrast': 1.08, 'saturation': 1.20,
                'highlights': 0.90, 'shadows': 0.85, 'clarity': 1.15,
                'vibrance': 1.25, 'temperature': 2, 'white_overlay': 0.05,
                'denoise': 2, 'sharpen': 1.1, 'gamma': 0.97
            }
        },
        'rose_gold': {
            'low': {
                'brightness': 1.16, 'contrast': 1.22, 'saturation': 1.25,
                'highlights': 1.22, 'shadows': 1.02, 'clarity': 1.28,
                'vibrance': 1.30, 'temperature': 0, 'white_overlay': 0.10,
                'denoise': 5, 'sharpen': 1.35, 'gamma': 1.04
            },
            'normal': {
                'brightness': 1.10, 'contrast': 1.18, 'saturation': 1.20,
                'highlights': 1.18, 'shadows': 0.92, 'clarity': 1.22,
                'vibrance': 1.25, 'temperature': -1, 'white_overlay': 0.08,
                'denoise': 3, 'sharpen': 1.25, 'gamma': 1.01
            },
            'high': {
                'brightness': 1.06, 'contrast': 1.12, 'saturation': 1.15,
                'highlights': 0.92, 'shadows': 0.87, 'clarity': 1.18,
                'vibrance': 1.20, 'temperature': -2, 'white_overlay': 0.06,
                'denoise': 2, 'sharpen': 1.15, 'gamma': 0.98
            }
        },
        'champagne_gold': {
            'low': {
                'brightness': 1.30, 'contrast': 1.25, 'saturation': 0.70,
                'highlights': 1.30, 'shadows': 1.10, 'clarity': 1.35,
                'vibrance': 0.75, 'temperature': -8, 'white_overlay': 0.15,
                'denoise': 6, 'sharpen': 1.5, 'gamma': 1.08
            },
            'normal': {
                'brightness': 1.15, 'contrast': 1.20, 'saturation': 0.75,
                'highlights': 1.25, 'shadows': 0.88, 'clarity': 1.30,
                'vibrance': 0.80, 'temperature': -5, 'white_overlay': 0.12,
                'denoise': 4, 'sharpen': 1.4, 'gamma': 1.05
            },
            'high': {
                'brightness': 1.10, 'contrast': 1.15, 'saturation': 0.80,
                'highlights': 0.88, 'shadows': 0.83, 'clarity': 1.25,
                'vibrance': 0.85, 'temperature': -3, 'white_overlay': 0.10,
                'denoise': 3, 'sharpen': 1.3, 'gamma': 1.00
            }
        }
    }
    
    # Get parameters
    params = metal_params.get(metal_type, metal_params['white_gold']).get(lighting, metal_params['white_gold']['normal'])
    
    # 10-step enhancement process
    
    # Step 1: Denoise
    if params['denoise'] > 0:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, params['denoise'], params['denoise'], 7, 21)
    
    # Step 2-4: Brightness, Contrast, and Advanced adjustments in LAB
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # Brightness
    l = l * params['brightness']
    
    # Contrast
    l = ((l - 127.5) * params['contrast']) + 127.5
    
    # Highlights and shadows
    highlight_mask = l > 180
    shadow_mask = l < 50
    l[highlight_mask] = l[highlight_mask] * params['highlights']
    l[shadow_mask] = l[shadow_mask] * params['shadows']
    
    # Clarity (local contrast)
    l_blur = cv2.GaussianBlur(l, (31, 31), 0)  # 31x31 Gaussian blur for natural blending
    l_detail = l - l_blur
    l = l + (l_detail * params['clarity'])
    
    l = np.clip(l, 0, 255)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Step 5: Sharpness
    if params['sharpen'] > 1.0:
        pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        sharpener = ImageEnhance.Sharpness(pil_img)
        pil_img = sharpener.enhance(params['sharpen'])
        enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Step 6: Saturation and Vibrance in HSV
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Saturation
    s = s * params['saturation']
    
    # Vibrance (selective saturation)
    vibrance_mask = s < 100
    s[vibrance_mask] = s[vibrance_mask] * params['vibrance']
    
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Step 7: White overlay
    if params['white_overlay'] > 0:
        white_layer = np.ones_like(enhanced) * 255
        enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], white_layer, params['white_overlay'], 0)
    
    # Step 8: Color temperature
    if params['temperature'] != 0:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Adjust temperature
        b = b + params['temperature']
        if params['temperature'] < 0:  # Cool
            a = a - (params['temperature'] * 0.2)
        
        b = np.clip(b, 0, 255)
        a = np.clip(a, 0, 255)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Step 9: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Step 10: Gamma correction
    if params['gamma'] != 1.0:
        inv_gamma = 1.0 / params['gamma']
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    
    # Final blending with original
    enhanced = cv2.addWeighted(image, 0.15, enhanced, 0.85, 0)
    
    return enhanced

def apply_replicate_inpainting(image, masking_info):
    """Apply inpainting using Replicate API with background replacement"""
    if not REPLICATE_API_TOKEN or not masking_info:
        return image
    
    try:
        import replicate
        
        # Initialize client within function
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = masking_info['bounds']
        
        # Mark areas to inpaint (outside the bounds)
        mask[:y1, :] = 255  # Top
        mask[y2:, :] = 255  # Bottom
        mask[:, :x1] = 255  # Left
        mask[:, x2:] = 255  # Right
        
        # Get appropriate background color
        lighting = detect_lighting_condition(image, masking_info['bounds'])
        metal_type = detect_metal_type(image, masking_info['bounds'])
        bg_color = get_after_background_color(lighting, metal_type)
        
        # Apply background color to masked areas
        result = image.copy()
        result[mask > 0] = bg_color
        
        # Prepare images
        _, img_buffer = cv2.imencode('.png', result)
        img_bytes = img_buffer.tobytes()
        
        _, mask_buffer = cv2.imencode('.png', mask)
        mask_bytes = mask_buffer.tobytes()
        
        # Run inpainting
        output = client.run(
            "stability-ai/stable-diffusion-inpainting",
            input={
                "image": io.BytesIO(img_bytes),
                "mask": io.BytesIO(mask_bytes),
                "prompt": "professional product photography white background",
                "negative_prompt": "black borders, frames, masking",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        
        # Process result
        if output and isinstance(output, list) and len(output) > 0:
            result_url = output[0]
            response = requests.get(result_url)
            result_image = Image.open(io.BytesIO(response.content))
            return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        log_debug(f"Replicate inpainting failed: {str(e)}")
    
    return apply_simple_masking_removal(image, masking_info)

def apply_simple_masking_removal(image, masking_info):
    """Simple masking removal with background replacement"""
    if not masking_info:
        return image
    
    x1, y1, x2, y2 = masking_info['bounds']
    
    # Get appropriate background color
    lighting = detect_lighting_condition(image, masking_info['bounds'])
    metal_type = detect_metal_type(image, masking_info['bounds'])
    bg_color = get_after_background_color(lighting, metal_type)
    
    # Create result with background
    result = np.full_like(image, bg_color)
    
    # Copy the inner content
    result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    # Apply Gaussian blur at edges for natural blending
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # Create smooth transition
    mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Blend
    result = image * mask_3channel + result * (1 - mask_3channel)
    
    return result.astype(np.uint8)

def create_thumbnail(image, size=(1000, 1300)):
    """Create a thumbnail with exact 1000x1300 size, ring centered"""
    h, w = image.shape[:2]
    
    # Convert to PIL for easier handling
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Calculate scale to fit the ring to occupy ~60% of the frame
    ring_target_size = min(size[0] * 0.6, size[1] * 0.6)
    scale = ring_target_size / max(w, h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background with exact size
    thumbnail = Image.new('RGB', size, 'white')
    
    # Paste centered
    x = (size[0] - new_w) // 2
    y = (size[1] - new_h) // 2
    thumbnail.paste(pil_image, (x, y))
    
    return cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

def handler(event):
    """Main handler function for RunPod"""
    try:
        log_debug(f"Handler started")
        log_debug(f"Event type: {type(event)}")
        log_debug(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
        
        # Get job input
        job_input = event.get("input", {})
        log_debug(f"Job input keys: {list(job_input.keys())}")
        
        # Extract base64 image
        base64_image = job_input.get("image", "")
        if not base64_image:
            # Try nested structure
            if "input" in job_input and isinstance(job_input["input"], dict):
                base64_image = job_input["input"].get("image", "")
        
        if not base64_image:
            raise ValueError("No image provided in input")
        
        log_debug(f"Base64 string length: {len(base64_image)}")
        log_debug(f"Base64 string start: {base64_image[:100]}...")
        
        # Decode image
        image = decode_base64_image(base64_image)
        log_debug(f"Image decoded successfully: {image.shape}")
        
        # Process image
        # 1. Detect masking using comprehensive method
        masking_info = detect_masking_comprehensive(image)
        log_debug(f"Masking detected: {masking_info}")
        
        # 2. Detect lighting and metal type
        mask_bounds = masking_info['bounds'] if masking_info else None
        lighting = detect_lighting_condition(image, mask_bounds)
        metal_type = detect_metal_type(image, mask_bounds)
        log_debug(f"Lighting: {lighting}, Metal type: {metal_type}")
        
        # 3. Enhance wedding ring details
        enhanced = enhance_wedding_ring_details(image, lighting, metal_type)
        log_debug("Wedding ring enhancement applied")
        
        # 4. Remove masking if detected
        if masking_info:
            if REPLICATE_API_TOKEN:
                enhanced = apply_replicate_inpainting(enhanced, masking_info)
            else:
                enhanced = apply_simple_masking_removal(enhanced, masking_info)
            log_debug("Masking removed")
        
        # 5. Create thumbnail with exact size
        thumbnail = create_thumbnail(enhanced)
        log_debug(f"Thumbnail created: {thumbnail.shape}")
        
        # 6. Encode result
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Remove padding for Make.com compatibility
        result_base64 = result_base64.rstrip('=')
        
        log_debug(f"Processing complete. Result length: {len(result_base64)}")
        
        # Return with proper structure
        return {
            "output": {
                "enhanced_image": result_base64
            }
        }
        
    except Exception as e:
        log_debug(f"Error in handler: {str(e)}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
        return {
            "output": {
                "error": str(e),
                "enhanced_image": ""
            }
        }

# RunPod handler - CORRECT FORMAT
runpod.serverless.start({"handler": handler})
