#!/usr/bin/env python3
import runpod
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import io
import base64
import sys
import traceback
import replicate
import os
from collections import Counter

# Remove global Replicate initialization that causes crashes
# Will initialize when needed in the functions

def log_debug(message):
    """Enhanced debug logging"""
    print(f"[DEBUG v121 FULL] {message}", file=sys.stderr)

# v13.3 Complete Parameters - Based on 28 pairs of training data
COMPLETE_PARAMS_V13_3 = {
    'yellow_gold': {
        'natural': {
            'brightness': 1.24,
            'contrast': 1.08,
            'white_overlay': 0.02,
            'sharpness': 1.22,
            'color_temp_a': 5,
            'color_temp_b': 2,
            'original_blend': 0.15,
            'saturation': 1.06,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.28,
            'contrast': 1.12,
            'white_overlay': 0.03,
            'sharpness': 1.25,
            'color_temp_a': 7,
            'color_temp_b': 4,
            'original_blend': 0.12,
            'saturation': 1.10,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.05,
            'white_overlay': 0.04,
            'sharpness': 1.18,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.18,
            'saturation': 1.03,
            'gamma': 1.03
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.22,
            'contrast': 1.10,
            'white_overlay': 0.05,
            'sharpness': 1.20,
            'color_temp_a': -2,
            'color_temp_b': 3,
            'original_blend': 0.16,
            'saturation': 1.08,
            'gamma': 0.99
        },
        'warm': {
            'brightness': 1.26,
            'contrast': 1.14,
            'white_overlay': 0.06,
            'sharpness': 1.23,
            'color_temp_a': -1,
            'color_temp_b': 5,
            'original_blend': 0.14,
            'saturation': 1.12,
            'gamma': 0.96
        },
        'cool': {
            'brightness': 1.18,
            'contrast': 1.07,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -4,
            'color_temp_b': 1,
            'original_blend': 0.19,
            'saturation': 1.04,
            'gamma': 1.02
        }
    },
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.06,
            'white_overlay': 0.08,
            'sharpness': 1.24,
            'color_temp_a': -3,
            'color_temp_b': -2,
            'original_blend': 0.22,
            'saturation': 0.95,
            'gamma': 1.04
        },
        'warm': {
            'brightness': 1.22,
            'contrast': 1.10,
            'white_overlay': 0.09,
            'sharpness': 1.26,
            'color_temp_a': -1,
            'color_temp_b': 0,
            'original_blend': 0.20,
            'saturation': 0.98,
            'gamma': 1.01
        },
        'cool': {
            'brightness': 1.15,
            'contrast': 1.03,
            'white_overlay': 0.10,
            'sharpness': 1.20,
            'color_temp_a': -5,
            'color_temp_b': -3,
            'original_blend': 0.25,
            'saturation': 0.92,
            'gamma': 1.06
        }
    },
    'plain_white': {
        'natural': {
            'brightness': 1.30,
            'contrast': 1.15,
            'white_overlay': 0.15,
            'sharpness': 1.28,
            'color_temp_a': -6,
            'color_temp_b': -4,
            'original_blend': 0.10,
            'saturation': 0.85,
            'gamma': 1.08
        },
        'warm': {
            'brightness': 1.32,
            'contrast': 1.18,
            'white_overlay': 0.16,
            'sharpness': 1.30,
            'color_temp_a': -4,
            'color_temp_b': -2,
            'original_blend': 0.08,
            'saturation': 0.88,
            'gamma': 1.05
        },
        'cool': {
            'brightness': 1.28,
            'contrast': 1.12,
            'white_overlay': 0.17,
            'sharpness': 1.25,
            'color_temp_a': -8,
            'color_temp_b': -5,
            'original_blend': 0.12,
            'saturation': 0.82,
            'gamma': 1.10
        }
    }
}

# 28 pairs AFTER background colors
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [235, 232, 228],
        'medium': [228, 225, 221],
        'default': [235, 232, 228]
    },
    'warm': {
        'light': [238, 233, 225],
        'medium': [232, 227, 219],
        'default': [238, 233, 225]
    },
    'cool': {
        'light': [232, 235, 238],
        'medium': [225, 228, 232],
        'default': [232, 235, 238]
    }
}

def calculate_bounds_from_edges(gray, edges_with_black, scan_depth, threshold):
    """Calculate exact bounds of black borders"""
    h, w = gray.shape
    
    # Initialize with full image bounds
    left, top, right, bottom = 0, 0, w, h
    
    # Find exact boundaries for each edge with adaptive thickness
    if 'top' in edges_with_black:
        for y in range(min(scan_depth * 2, h)):
            if np.mean(gray[y, :] < threshold) < 0.5:
                top = y
                break
    
    if 'bottom' in edges_with_black:
        for y in range(h-1, max(h-scan_depth*2, -1), -1):
            if np.mean(gray[y, :] < threshold) < 0.5:
                bottom = y + 1
                break
    
    if 'left' in edges_with_black:
        for x in range(min(scan_depth * 2, w)):
            if np.mean(gray[:, x] < threshold) < 0.5:
                left = x
                break
    
    if 'right' in edges_with_black:
        for x in range(w-1, max(w-scan_depth*2, -1), -1):
            if np.mean(gray[:, x] < threshold) < 0.5:
                right = x + 1
                break
    
    # Validate bounds
    if left < right and top < bottom:
        return (left, top, right, bottom)
    return None

def detect_actual_line_thickness(gray, bounds):
    """Detect actual thickness of black lines - can handle up to 100px"""
    left, top, right, bottom = bounds
    h, w = gray.shape
    thicknesses = []
    
    # Sample from multiple points for accurate measurement
    # Top edge
    if top > 0:
        for x in range(left + 10, right - 10, 50):
            for y in range(top, min(top + 200, h)):  # Check up to 200px
                if gray[y, x] > 50:
                    thicknesses.append(y - top)
                    break
    
    # Left edge
    if left > 0:
        for y in range(top + 10, bottom - 10, 50):
            for x in range(left, min(left + 200, w)):  # Check up to 200px
                if gray[y, x] > 50:
                    thicknesses.append(x - left)
                    break
    
    # Bottom edge
    if bottom < h:
        for x in range(left + 10, right - 10, 50):
            for y in range(bottom - 1, max(bottom - 200, 0), -1):
                if gray[y, x] > 50:
                    thicknesses.append(bottom - y)
                    break
    
    # Right edge
    if right < w:
        for y in range(top + 10, bottom - 10, 50):
            for x in range(right - 1, max(right - 200, 0), -1):
                if gray[y, x] > 50:
                    thicknesses.append(right - x)
                    break
    
    if thicknesses:
        # Return median thickness with 50% safety margin
        return int(np.median(thicknesses) * 1.5 + 20)
    return 50  # Default safe thickness

def validate_rectangular_mask(mask, img_w, img_h):
    """Validate if the mask represents a proper rectangular masking"""
    bounds = mask['bounds']
    x1, y1, x2, y2 = bounds
    
    # Check if bounds are reasonable
    width = x2 - x1
    height = y2 - y1
    
    if width < img_w * 0.3 or height < img_h * 0.3:
        return False
        
    if width > img_w * 0.95 or height > img_h * 0.95:
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
    
    # If multiple valid detections, use the most conservative (smallest removal)
    best_mask = max(valid_masks, key=lambda m: (m['bounds'][2] - m['bounds'][0]) * (m['bounds'][3] - m['bounds'][1]))
    
    return best_mask

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
                
                methods_results[f'threshold_{threshold}'] = {
                    'bounds': (x, y, x + cnt_w, y + cnt_h),
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
                
                methods_results[f'edge_{low}_{high}'] = {
                    'bounds': (x, y, x + cnt_w, y + cnt_h),
                    'thickness': thickness,
                    'type': 'center',
                    'method': f'edge_{low}_{high}'
                }
                break
    
    # Method 3: Gradient analysis with Sobel
    log_debug("Method 3: Sobel gradient analysis")
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    for percentile in [85, 90, 95]:
        gradient_binary = gradient > np.percentile(gradient, percentile)
        
        contours, _ = cv2.findContours(gradient_binary.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
            
            if cnt_w > w * 0.3 and cnt_h > h * 0.3:
                thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                
                methods_results[f'gradient_{percentile}'] = {
                    'bounds': (x, y, x + cnt_w, y + cnt_h),
                    'thickness': thickness,
                    'type': 'center',
                    'method': f'gradient_{percentile}'
                }
                break
    
    # Method 4: Color variance detection
    log_debug("Method 4: Color variance detection")
    for y in range(0, h - 50, 10):
        row_variance = np.var(image[y:y+50, :])
        if row_variance < 100:  # Low variance indicates solid color (black line)
            for x in range(0, w - 50, 10):
                region_variance = np.var(image[y:y+50, x:x+50])
                if region_variance < 50:
                    # Found potential masking area
                    # Expand to find full bounds
                    top, left = y, x
                    bottom, right = y + 50, x + 50
                    
                    # Expand bounds
                    while top > 0 and np.mean(gray[top-1, :] < 50) > 0.5:
                        top -= 1
                    while bottom < h and np.mean(gray[bottom, :] < 50) > 0.5:
                        bottom += 1
                    while left > 0 and np.mean(gray[:, left-1] < 50) > 0.5:
                        left -= 1
                    while right < w and np.mean(gray[:, right] < 50) > 0.5:
                        right += 1
                    
                    if (right - left) > w * 0.3 and (bottom - top) > h * 0.3:
                        thickness = detect_actual_line_thickness(gray, (left, top, right, bottom))
                        
                        methods_results['variance'] = {
                            'bounds': (left, top, right, bottom),
                            'thickness': thickness,
                            'type': 'center',
                            'method': 'variance'
                        }
                        break
            if 'variance' in methods_results:
                break
    
    # Cross-validate results
    log_debug(f"Found {len(methods_results)} potential maskings")
    final_mask = cross_validate_detection(methods_results, (h, w))
    
    if final_mask:
        log_debug(f"Confirmed masking: bounds={final_mask['bounds']}, thickness={final_mask['thickness']}")
        return final_mask
    
    log_debug("No masking detected")
    return None

def detect_edge_masking_ultra(image):
    """Ultra precise edge masking detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check each edge with multiple scan depths
    edges_with_black = []
    
    # Try multiple scan depths
    for scan_depth in [50, 100, 150, 200]:
        if scan_depth > min(h, w) // 4:
            continue
            
        # Adaptive threshold based on image
        mean_brightness = np.mean(gray)
        threshold = min(40, mean_brightness * 0.2)
        
        # Top edge
        if np.mean(gray[:scan_depth, :] < threshold) > 0.7:
            if 'top' not in edges_with_black:
                edges_with_black.append('top')
        
        # Bottom edge
        if np.mean(gray[-scan_depth:, :] < threshold) > 0.7:
            if 'bottom' not in edges_with_black:
                edges_with_black.append('bottom')
        
        # Left edge
        if np.mean(gray[:, :scan_depth] < threshold) > 0.7:
            if 'left' not in edges_with_black:
                edges_with_black.append('left')
        
        # Right edge
        if np.mean(gray[:, -scan_depth:] < threshold) > 0.7:
            if 'right' not in edges_with_black:
                edges_with_black.append('right')
    
    if not edges_with_black:
        return None
    
    # Calculate bounds with the largest scan depth that found edges
    bounds = calculate_bounds_from_edges(gray, edges_with_black, scan_depth, threshold)
    
    if bounds:
        thickness = detect_actual_line_thickness(gray, bounds)
        return {
            'bounds': bounds,
            'thickness': thickness,
            'type': 'edge',
            'edges': edges_with_black
        }
    
    return None

def detect_metal_type_advanced(image):
    """Advanced metal type detection with 28 pairs training data knowledge"""
    # Convert to RGB for analysis
    if len(image.shape) == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return "plain_white"
    
    # Get center region for analysis
    h, w = rgb.shape[:2]
    center_y, center_x = h // 2, w // 2
    region_size = min(h, w) // 3  # Larger region for better analysis
    
    center_region = rgb[
        max(0, center_y - region_size):min(h, center_y + region_size),
        max(0, center_x - region_size):min(w, center_x + region_size)
    ]
    
    # Calculate color statistics
    avg_color = np.mean(center_region.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    
    # Convert to HSV for better color analysis
    hsv_region = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    h_channel = hsv_region[:, :, 0]
    s_channel = hsv_region[:, :, 1]
    v_channel = hsv_region[:, :, 2]
    
    avg_hue = np.mean(h_channel)
    avg_saturation = np.mean(s_channel)
    avg_brightness = np.mean(v_channel)
    
    # Calculate color differences
    rg_diff = abs(r - g)
    gb_diff = abs(g - b)
    rb_diff = abs(r - b)
    
    # Metal type detection logic based on 28 pairs analysis
    # Priority: plain_white > rose_gold > white_gold > yellow_gold
    
    # Check for plain white / champagne gold first
    if avg_saturation < 15 and avg_brightness > 180:
        return "plain_white"
    
    # Check for very low saturation (likely white metals)
    if avg_saturation < 20:
        if avg_brightness > 150:
            return "plain_white" if avg_brightness > 180 else "white_gold"
    
    # Rose gold detection (pinkish hue)
    if r > g * 1.1 and r > b * 1.15:
        if 10 <= avg_hue <= 25 or avg_hue > 170:  # Pink/red hues
            return "rose_gold"
    
    # White gold detection
    if avg_saturation < 25 and avg_brightness > 140:
        if rg_diff < 10 and gb_diff < 10:  # Very neutral colors
            return "white_gold"
    
    # Yellow gold detection (only if clearly yellow)
    if avg_saturation > 25 and g > b * 1.1:
        if 20 <= avg_hue <= 40:  # Yellow hues
            # Additional check to avoid misidentifying plain white
            if avg_brightness < 180:
                return "yellow_gold"
    
    # Default to plain white when uncertain
    return "plain_white"

def detect_lighting_condition(image):
    """Detect lighting condition for optimal parameter selection"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Calculate distribution metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Analyze histogram peaks
    peaks = []
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i]))
    
    # Sort peaks by prominence
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Determine lighting condition
    if mean_brightness > 180 and std_brightness < 40:
        return 'cool'  # Bright, even lighting
    elif mean_brightness < 100:
        return 'warm'  # Dark, likely warm lighting
    elif len(peaks) >= 2 and abs(peaks[0][0] - peaks[1][0]) > 100:
        return 'natural'  # High contrast, natural lighting
    else:
        # Default based on color temperature
        b, g, r = cv2.split(image)
        if np.mean(r) > np.mean(b) * 1.1:
            return 'warm'
        elif np.mean(b) > np.mean(r) * 1.1:
            return 'cool'
        else:
            return 'natural'

def apply_v13_complete_enhancement(image, metal_type, lighting):
    """Apply complete v13.3 enhancement with 10-step process"""
    log_debug(f"Applying v13.3 enhancement: {metal_type} / {lighting}")
    
    # Get parameters for this combination
    params = COMPLETE_PARAMS_V13_3[metal_type][lighting]
    
    # Step 1: Noise reduction
    enhanced = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    
    # Convert to PIL for enhancement steps
    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Step 2: Brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(params['brightness'])
    
    # Step 3: Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(params['contrast'])
    
    # Step 4: Sharpness enhancement
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(params['sharpness'])
    
    # Step 5: Saturation adjustment
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(params['saturation'])
    
    # Convert back to numpy for advanced processing
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Step 6: White overlay (especially for plain white)
    if params['white_overlay'] > 0:
        white_layer = np.full_like(enhanced, 255)
        enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                                  white_layer, params['white_overlay'], 0)
    
    # Step 7: Color temperature adjustment
    if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Step 8: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Step 9: Gamma correction
    if params['gamma'] != 1.0:
        inv_gamma = 1.0 / params['gamma']
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    
    # Step 10: Blend with original
    if params['original_blend'] > 0:
        enhanced = cv2.addWeighted(enhanced, 1 - params['original_blend'],
                                  image, params['original_blend'], 0)
    
    return enhanced

def remove_masking_with_after_background(image, masking_info, lighting='natural'):
    """Remove masking and fill with AFTER background color"""
    if not masking_info:
        return image
    
    log_debug(f"Removing {masking_info['type']} masking with AFTER background")
    
    # Get AFTER background color
    bg_color = AFTER_BACKGROUND_COLORS[lighting]['default']
    bg_color_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]])  # RGB to BGR
    
    h, w = image.shape[:2]
    result = image.copy()
    
    if masking_info['type'] == 'center':
        # For center masking, remove the frame and fill with background
        x1, y1, x2, y2 = masking_info['bounds']
        thickness = masking_info['thickness']
        
        # Create mask for the frame
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mark frame areas
        mask[y1:y1+thickness, x1:x2] = 255  # Top
        mask[y2-thickness:y2, x1:x2] = 255  # Bottom
        mask[y1:y2, x1:x1+thickness] = 255  # Left
        mask[y1:y2, x2-thickness:x2] = 255  # Right
        
        # Apply Gaussian blur for natural blending
        mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend background color
        background = np.full_like(result, bg_color_bgr)
        result = (result * (1 - mask_3channel) + background * mask_3channel).astype(np.uint8)
        
    elif masking_info['type'] == 'edge':
        # For edge masking, fill edges with background
        thickness = masking_info['thickness']
        
        for edge in masking_info.get('edges', []):
            if edge == 'top':
                result[:thickness, :] = bg_color_bgr
                # Gradient blend
                for i in range(20):
                    alpha = i / 20.0
                    y = thickness + i
                    if y < h:
                        result[y, :] = (result[y, :] * alpha + bg_color_bgr * (1 - alpha)).astype(np.uint8)
                        
            elif edge == 'bottom':
                result[-thickness:, :] = bg_color_bgr
                # Gradient blend
                for i in range(20):
                    alpha = i / 20.0
                    y = h - thickness - i - 1
                    if y >= 0:
                        result[y, :] = (result[y, :] * alpha + bg_color_bgr * (1 - alpha)).astype(np.uint8)
                        
            elif edge == 'left':
                result[:, :thickness] = bg_color_bgr
                # Gradient blend
                for i in range(20):
                    alpha = i / 20.0
                    x = thickness + i
                    if x < w:
                        result[:, x] = (result[:, x] * alpha + bg_color_bgr * (1 - alpha)).astype(np.uint8)
                        
            elif edge == 'right':
                result[:, -thickness:] = bg_color_bgr
                # Gradient blend
                for i in range(20):
                    alpha = i / 20.0
                    x = w - thickness - i - 1
                    if x >= 0:
                        result[:, x] = (result[:, x] * alpha + bg_color_bgr * (1 - alpha)).astype(np.uint8)
    
    return result

def create_perfect_thumbnail(image, target_size=(1000, 1300)):
    """Create perfectly sized thumbnail with minimal padding"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling to maximize ring size
    scale = min(target_w / w, target_h / h) * 0.98  # 98% to leave minimal padding
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize with high quality
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    thumbnail = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Apply subtle vignette effect
    center_x, center_y = target_w // 2, target_h // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(target_h):
        for x in range(target_w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist > max_dist * 0.7:
                factor = 1.0 - (dist - max_dist * 0.7) / (max_dist * 0.3) * 0.05
                thumbnail[y, x] = (thumbnail[y, x] * factor).astype(np.uint8)
    
    return thumbnail

def call_replicate_api_with_retry(image, mask, prompt="", max_retries=3):
    """Call Replicate API with retry logic"""
    for attempt in range(max_retries):
        try:
            # Get API token
            api_token = os.environ.get("REPLICATE_API_TOKEN")
            if not api_token:
                log_debug("No Replicate API token found")
                return None
            
            # Initialize client (create new instance each time)
            client = replicate.Client(api_token=api_token)
            
            # Convert images to base64
            img_buffer = io.BytesIO()
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            mask_buffer = io.BytesIO()
            Image.fromarray(mask).save(mask_buffer, format="PNG")
            mask_buffer.seek(0)
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
            
            # Determine which model to use based on mask size
            mask_area = np.sum(mask > 0)
            total_area = mask.shape[0] * mask.shape[1]
            mask_percentage = (mask_area / total_area) * 100
            
            log_debug(f"Mask percentage: {mask_percentage:.1f}%")
            
            if mask_percentage < 5:
                # Small mask - use faster model
                model = "ideogram-ai/ideogram-v2-turbo"
                inference_steps = 15
            else:
                # Larger mask - use higher quality model
                model = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
                inference_steps = 25
            
            log_debug(f"Using model: {model}")
            
            # Run inpainting
            output = client.run(
                model,
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": prompt or "clean white background, product photography, professional lighting",
                    "num_inference_steps": inference_steps
                }
            )
            
            # Process result
            if output:
                result_url = output[0] if isinstance(output, list) else output
                # Download and convert result
                import requests
                response = requests.get(result_url)
                result_image = Image.open(io.BytesIO(response.content))
                return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            log_debug(f"Replicate API attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                continue
    
    return None

def handler(job):
    """Main handler function for RunPod"""
    try:
        log_debug("Handler started - v121 FULL")
        input_data = job["input"]
        
        # Handle multiple possible input formats
        image_base64 = (
            input_data.get("image") or 
            input_data.get("image_base64") or 
            input_data.get("base64") or
            input_data.get("imageBase64")
        )
        
        if not image_base64:
            raise ValueError("No image data provided in any expected format")
        
        log_debug("Decoding image")
        # Handle data URL format
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        # Add padding if needed
        padding = 4 - len(image_base64) % 4
        if padding != 4:
            image_base64 += "=" * padding
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        log_debug(f"Image decoded: {image.shape}")
        original_shape = image.shape
        
        # Detect masking with ultra advanced methods
        log_debug("Detecting masking")
        center_mask = detect_center_masking_ultra_advanced(image)
        edge_mask = detect_edge_masking_ultra(image)
        
        masking_info = center_mask or edge_mask
        
        # Detect lighting condition first (before any modifications)
        lighting = detect_lighting_condition(image)
        log_debug(f"Detected lighting condition: {lighting}")
        
        # Remove masking if detected
        if masking_info:
            log_debug(f"Masking detected: {masking_info['type']}, thickness: {masking_info['thickness']}px")
            
            # Try Replicate API first for natural inpainting
            if os.environ.get("REPLICATE_API_TOKEN"):
                # Create mask for inpainting
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                if masking_info['type'] == 'center':
                    # For center masking, inpaint the black frame
                    x1, y1, x2, y2 = masking_info['bounds']
                    thickness = masking_info['thickness']
                    
                    # Create mask for the frame only (with extra margin)
                    mask[y1:y1+thickness+10, x1:x2] = 255  # Top
                    mask[y2-thickness-10:y2, x1:x2] = 255  # Bottom
                    mask[y1:y2, x1:x1+thickness+10] = 255  # Left
                    mask[y1:y2, x2-thickness-10:x2] = 255  # Right
                    
                elif masking_info['type'] == 'edge':
                    # For edge masking, inpaint the edges
                    thickness = masking_info['thickness']
                    for edge in masking_info.get('edges', []):
                        if edge == 'top':
                            mask[:thickness+10, :] = 255
                        elif edge == 'bottom':
                            mask[-thickness-10:, :] = 255
                        elif edge == 'left':
                            mask[:, :thickness+10] = 255
                        elif edge == 'right':
                            mask[:, -thickness-10:] = 255
                
                # Dilate mask slightly for better coverage
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Call Replicate for inpainting
                inpainted = call_replicate_api_with_retry(image, mask)
                if inpainted is not None:
                    image = inpainted
                    log_debug("Successfully inpainted with Replicate")
                else:
                    # Fallback to AFTER background method
                    image = remove_masking_with_after_background(image, masking_info, lighting)
                    log_debug("Fallback to AFTER background method")
            else:
                # No API token, use AFTER background method
                image = remove_masking_with_after_background(image, masking_info, lighting)
                log_debug("Used AFTER background method (no API token)")
        
        # Detect metal type with advanced method
        log_debug("Detecting metal type")
        metal_type = detect_metal_type_advanced(image)
        log_debug(f"Detected metal type: {metal_type}")
        
        # Apply complete v13.3 enhancement
        log_debug("Applying v13.3 complete enhancement")
        enhanced = apply_v13_complete_enhancement(image, metal_type, lighting)
        
        # Additional enhancement for plain white (champagne gold)
        if metal_type == "plain_white":
            log_debug("Applying additional plain white enhancement")
            # Extra brightness and white overlay
            pil_enhanced = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            brightness = ImageEnhance.Brightness(pil_enhanced)
            pil_enhanced = brightness.enhance(1.05)
            enhanced = cv2.cvtColor(np.array(pil_enhanced), cv2.COLOR_RGB2BGR)
        
        # Create perfect thumbnail
        log_debug("Creating perfect thumbnail")
        thumbnail = create_perfect_thumbnail(enhanced)
        
        # Encode results to base64 (remove padding for Make.com)
        log_debug("Encoding results")
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        enhanced_buffer = io.BytesIO()
        enhanced_pil.save(enhanced_buffer, format='PNG', quality=95)
        enhanced_buffer.seek(0)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode().rstrip('=')
        
        thumbnail_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        thumbnail_buffer = io.BytesIO()
        thumbnail_pil.save(thumbnail_buffer, format='PNG', quality=95)
        thumbnail_buffer.seek(0)
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode().rstrip('=')
        
        # Return with proper nesting for Make.com
        # Make.com expects: {{4.data.output.output.enhanced_image}}
        return {
            "output": {
                "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
                "thumbnail": f"data:image/png;base64,{thumbnail_base64}",
                "metal_type": metal_type,
                "lighting_condition": lighting,
                "masking_detected": masking_info is not None,
                "masking_info": {
                    "type": masking_info['type'],
                    "thickness": masking_info['thickness'],
                    "bounds": masking_info['bounds']
                } if masking_info else None,
                "parameters_used": COMPLETE_PARAMS_V13_3[metal_type][lighting],
                "original_size": f"{original_shape[1]}x{original_shape[0]}",
                "version": "v121-FULL",
                "status": "success"
            }
        }
        
    except Exception as e:
        log_debug(f"Error in processing: {str(e)}")
        traceback.print_exc()
        
        # Even on error, try to return something useful
        try:
            # If we have an image, at least brighten it
            if 'image' in locals() and image is not None:
                brightened = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
                
                # Simple thumbnail
                h, w = brightened.shape[:2]
                scale = min(1000/w, 1300/h) * 0.9
                new_size = (int(w*scale), int(h*scale))
                thumbnail = cv2.resize(brightened, new_size)
                
                # Encode
                _, buffer = cv2.imencode('.png', brightened)
                enhanced_base64 = base64.b64encode(buffer).decode().rstrip('=')
                
                _, buffer = cv2.imencode('.png', thumbnail)
                thumbnail_base64 = base64.b64encode(buffer).decode().rstrip('=')
                
                return {
                    "output": {
                        "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
                        "thumbnail": f"data:image/png;base64,{thumbnail_base64}",
                        "error": str(e),
                        "status": "error_with_fallback"
                    }
                }
        except:
            pass
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler - CORRECT FORMAT with dictionary
runpod.serverless.start({"handler": handler})
