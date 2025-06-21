import runpod
import cv2
import numpy as np
import base64
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
import traceback
import replicate
from typing import Dict, Tuple, Optional, List
import time

# v134 - Complete version with all features
VERSION = "v134-COMPLETE"

# Get Replicate API token
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# v13.3 parameters from 28 pairs + 10 additional pairs training data
RING_PARAMS = {
    'yellow_gold': {
        'low': {
            'brightness': 1.20, 'contrast': 1.08, 'saturation': 1.05,
            'sharpness': 1.18, 'highlights': 1.22, 'shadows': 0.85,
            'temperature': 1.15, 'exposure': 1.10, 'whites': 0.08,
            'blacks': -0.03, 'vibrance': 1.12, 'clarity': 1.08,
            'dehaze': 0.15, 'grain': 0.02, 'vignette': -0.05,
            'highlight_priority': 0.7, 'shadow_priority': 0.3
        },
        'normal': {
            'brightness': 1.15, 'contrast': 1.10, 'saturation': 1.08,
            'sharpness': 1.25, 'highlights': 1.12, 'shadows': 0.92,
            'temperature': 1.08, 'exposure': 1.05, 'whites': 0.05,
            'blacks': -0.02, 'vibrance': 1.08, 'clarity': 1.12,
            'dehaze': 0.10, 'grain': 0.01, 'vignette': -0.03,
            'highlight_priority': 0.6, 'shadow_priority': 0.4
        },
        'high': {
            'brightness': 1.08, 'contrast': 1.12, 'saturation': 1.03,
            'sharpness': 1.35, 'highlights': 1.05, 'shadows': 0.88,
            'temperature': 1.02, 'exposure': 0.98, 'whites': 0.03,
            'blacks': -0.05, 'vibrance': 1.05, 'clarity': 1.15,
            'dehaze': 0.08, 'grain': 0.01, 'vignette': -0.02,
            'highlight_priority': 0.5, 'shadow_priority': 0.5
        }
    },
    'rose_gold': {
        'low': {
            'brightness': 1.22, 'contrast': 1.05, 'saturation': 1.10,
            'sharpness': 1.20, 'highlights': 1.18, 'shadows': 0.82,
            'temperature': 1.12, 'exposure': 1.12, 'whites': 0.06,
            'blacks': -0.02, 'vibrance': 1.15, 'clarity': 1.10,
            'dehaze': 0.12, 'grain': 0.02, 'vignette': -0.04,
            'red_adjust': 1.08, 'pink_tone': 0.85,
            'highlight_priority': 0.65, 'shadow_priority': 0.35
        },
        'normal': {
            'brightness': 1.18, 'contrast': 1.08, 'saturation': 1.12,
            'sharpness': 1.28, 'highlights': 1.10, 'shadows': 0.90,
            'temperature': 1.05, 'exposure': 1.08, 'whites': 0.04,
            'blacks': -0.03, 'vibrance': 1.12, 'clarity': 1.13,
            'dehaze': 0.08, 'grain': 0.01, 'vignette': -0.02,
            'red_adjust': 1.05, 'pink_tone': 0.90,
            'highlight_priority': 0.55, 'shadow_priority': 0.45
        },
        'high': {
            'brightness': 1.10, 'contrast': 1.10, 'saturation': 1.05,
            'sharpness': 1.38, 'highlights': 1.02, 'shadows': 0.85,
            'temperature': 1.00, 'exposure': 1.00, 'whites': 0.02,
            'blacks': -0.04, 'vibrance': 1.08, 'clarity': 1.18,
            'dehaze': 0.05, 'grain': 0.01, 'vignette': -0.01,
            'red_adjust': 1.02, 'pink_tone': 0.95,
            'highlight_priority': 0.45, 'shadow_priority': 0.55
        }
    },
    'white_gold': {
        'low': {
            'brightness': 1.25, 'contrast': 1.12, 'saturation': 0.95,
            'sharpness': 1.22, 'highlights': 1.25, 'shadows': 0.78,
            'temperature': 0.92, 'exposure': 1.15, 'whites': 0.10,
            'blacks': -0.04, 'vibrance': 0.98, 'clarity': 1.14,
            'dehaze': 0.18, 'grain': 0.02, 'vignette': -0.06,
            'blue_adjust': 1.05, 'cool_tone': 1.10,
            'highlight_priority': 0.75, 'shadow_priority': 0.25
        },
        'normal': {
            'brightness': 1.20, 'contrast': 1.15, 'saturation': 0.98,
            'sharpness': 1.30, 'highlights': 1.15, 'shadows': 0.88,
            'temperature': 0.95, 'exposure': 1.10, 'whites': 0.08,
            'blacks': -0.03, 'vibrance': 0.95, 'clarity': 1.16,
            'dehaze': 0.12, 'grain': 0.01, 'vignette': -0.04,
            'blue_adjust': 1.03, 'cool_tone': 1.08,
            'highlight_priority': 0.65, 'shadow_priority': 0.35
        },
        'high': {
            'brightness': 1.12, 'contrast': 1.18, 'saturation': 0.92,
            'sharpness': 1.40, 'highlights': 1.08, 'shadows': 0.82,
            'temperature': 0.98, 'exposure': 1.02, 'whites': 0.05,
            'blacks': -0.05, 'vibrance': 0.92, 'clarity': 1.20,
            'dehaze': 0.08, 'grain': 0.01, 'vignette': -0.02,
            'blue_adjust': 1.02, 'cool_tone': 1.05,
            'highlight_priority': 0.55, 'shadow_priority': 0.45
        }
    },
    'champagne': {
        'low': {
            'brightness': 1.18, 'contrast': 1.06, 'saturation': 1.02,
            'sharpness': 1.25, 'highlights': 1.20, 'shadows': 0.80,
            'temperature': 1.08, 'exposure': 1.08, 'whites': 0.07,
            'blacks': -0.02, 'vibrance': 1.10, 'clarity': 1.12,
            'dehaze': 0.14, 'grain': 0.02, 'vignette': -0.05,
            'champagne_enhance': 1.12, 'warm_tone': 0.92,
            'pearl_effect': 1.15, 'highlight_priority': 0.68
        },
        'normal': {
            'brightness': 1.12, 'contrast': 1.08, 'saturation': 1.05,
            'sharpness': 1.32, 'highlights': 1.08, 'shadows': 0.85,
            'temperature': 1.03, 'exposure': 1.02, 'whites': 0.05,
            'blacks': -0.03, 'vibrance': 1.06, 'clarity': 1.14,
            'dehaze': 0.10, 'grain': 0.01, 'vignette': -0.03,
            'champagne_enhance': 1.08, 'warm_tone': 0.95,
            'pearl_effect': 1.10, 'highlight_priority': 0.58
        },
        'high': {
            'brightness': 1.05, 'contrast': 1.10, 'saturation': 1.00,
            'sharpness': 1.42, 'highlights': 1.00, 'shadows': 0.80,
            'temperature': 1.00, 'exposure': 0.95, 'whites': 0.03,
            'blacks': -0.04, 'vibrance': 1.02, 'clarity': 1.18,
            'dehaze': 0.06, 'grain': 0.01, 'vignette': -0.01,
            'champagne_enhance': 1.05, 'warm_tone': 0.98,
            'pearl_effect': 1.05, 'highlight_priority': 0.48
        }
    }
}

# 28 pairs AFTER background colors based on training data
AFTER_BG_COLORS = {
    'pair_1': (241, 238, 234), 'pair_2': (243, 239, 236), 'pair_3': (237, 233, 229),
    'pair_4': (245, 241, 238), 'pair_5': (240, 236, 232), 'pair_6': (244, 240, 237),
    'pair_7': (239, 235, 231), 'pair_8': (242, 238, 235), 'pair_9': (238, 234, 230),
    'pair_10': (246, 242, 239), 'pair_11': (236, 232, 228), 'pair_12': (243, 239, 236),
    'pair_13': (241, 237, 234), 'pair_14': (245, 241, 238), 'pair_15': (239, 235, 232),
    'pair_16': (242, 238, 235), 'pair_17': (240, 236, 233), 'pair_18': (244, 240, 237),
    'pair_19': (238, 234, 231), 'pair_20': (243, 239, 236), 'pair_21': (241, 237, 234),
    'pair_22': (246, 242, 239), 'pair_23': (237, 233, 230), 'pair_24': (242, 238, 235),
    'pair_25': (240, 236, 233), 'pair_26': (245, 241, 238), 'pair_27': (239, 235, 232),
    'pair_28': (244, 240, 237)
}

def is_valid_base64(s):
    """Check if string is valid base64"""
    try:
        if isinstance(s, str):
            # Remove data URL prefix if present
            if 'base64,' in s:
                s = s.split('base64,')[1]
            # Check if it's valid base64
            base64.b64decode(s)
            return True
    except:
        pass
    return False

def decode_base64_image(base64_string):
    """Decode base64 image with multiple format support"""
    try:
        # Handle data URL format
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Add padding if necessary
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        return np.array(img)
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise

def encode_image_to_base64(image):
    """Encode image to base64 without padding for Make.com"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
    else:
        img_pil = image
    
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG", quality=95, optimize=True)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64.rstrip('=')

def detect_actual_line_thickness(gray_img, initial_bounds):
    """Detect actual thickness of black lines"""
    h, w = gray_img.shape
    left, top, right, bottom = initial_bounds
    
    thicknesses = []
    
    # Check top edge
    if top > 0:
        for y in range(min(200, top)):
            if np.mean(gray_img[y, :]) < 40:
                continue
            else:
                thicknesses.append(y)
                break
    
    # Check bottom edge
    if bottom < h:
        for y in range(min(200, h - bottom)):
            if np.mean(gray_img[h-1-y, :]) < 40:
                continue
            else:
                thicknesses.append(y)
                break
    
    # Check left edge
    if left > 0:
        for x in range(min(200, left)):
            if np.mean(gray_img[:, x]) < 40:
                continue
            else:
                thicknesses.append(x)
                break
    
    # Check right edge
    if right < w:
        for x in range(min(200, w - right)):
            if np.mean(gray_img[:, w-1-x]) < 40:
                continue
            else:
                thicknesses.append(x)
                break
    
    if thicknesses:
        return max(thicknesses)
    return 50  # default

def validate_rectangular_mask(bounds, image_shape, min_area_ratio=0.3):
    """Validate if detected bounds form a valid rectangular mask"""
    h, w = image_shape[:2]
    left, top, right, bottom = bounds
    
    # Check if bounds are valid
    if left >= right or top >= bottom:
        return False
    
    # Check if the masked area is reasonable
    mask_width = right - left
    mask_height = bottom - top
    mask_area = mask_width * mask_height
    image_area = w * h
    
    if mask_area < image_area * min_area_ratio:
        return False
    
    # Check if it's roughly rectangular (not too skewed)
    if mask_width < w * 0.2 or mask_height < h * 0.2:
        return False
    
    return True

def detect_center_masking_ultra_advanced(image):
    """Ultra-advanced masking detection with multiple methods and adaptive thickness"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize detection results
    detection_results = []
    
    # Method 1: Progressive threshold scanning with multiple levels
    threshold_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for threshold in threshold_levels:
        for scan_percent in scan_percentages:
            scan_depth = int(min(w, h) * scan_percent)
            if scan_depth < 10:
                continue
            
            # Analyze each edge independently
            edges = {
                'top': np.mean(gray[:scan_depth, :] < threshold),
                'bottom': np.mean(gray[-scan_depth:, :] < threshold),
                'left': np.mean(gray[:, :scan_depth] < threshold),
                'right': np.mean(gray[:, -scan_depth:] < threshold)
            }
            
            # Count edges with significant black content
            black_edges = sum(1 for edge_val in edges.values() if edge_val > 0.7)
            
            if black_edges >= 2:
                # Calculate precise bounds
                top = scan_depth if edges['top'] > 0.7 else 0
                bottom = h - scan_depth if edges['bottom'] > 0.7 else h
                left = scan_depth if edges['left'] > 0.7 else 0
                right = w - scan_depth if edges['right'] > 0.7 else w
                
                bounds = (left, top, right, bottom)
                
                if validate_rectangular_mask(bounds, image.shape):
                    # Calculate confidence score
                    edge_strength = sum(edges.values()) / 4
                    consistency = 1.0 - (max(edges.values()) - min(edges.values()))
                    threshold_score = 1.0 - (threshold / 100)
                    
                    confidence = (edge_strength * 0.4 + 
                                consistency * 0.3 + 
                                threshold_score * 0.2 +
                                (black_edges / 4) * 0.1)
                    
                    # Determine masking type
                    if black_edges == 4:
                        mask_type = 'full_frame'
                    elif edges['top'] > 0.7 and edges['bottom'] > 0.7:
                        mask_type = 'horizontal_bars'
                    elif edges['left'] > 0.7 and edges['right'] > 0.7:
                        mask_type = 'vertical_bars'
                    else:
                        mask_type = 'partial'
                    
                    # Detect actual thickness
                    thickness = detect_actual_line_thickness(gray, bounds)
                    
                    detection_results.append({
                        'bounds': bounds,
                        'type': mask_type,
                        'thickness': thickness,
                        'confidence': confidence,
                        'method': 'threshold'
                    })
    
    # Method 2: Gradient-based edge detection
    edges = cv2.Canny(gray, 30, 100)
    edge_map = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Analyze edge density in border regions
    border_size = int(min(w, h) * 0.3)
    edge_densities = {
        'top': np.mean(edge_map[:border_size, :] > 0),
        'bottom': np.mean(edge_map[-border_size:, :] > 0),
        'left': np.mean(edge_map[:, :border_size] > 0),
        'right': np.mean(edge_map[:, -border_size:] > 0)
    }
    
    # Method 3: Morphological analysis
    kernel_sizes = [(5, 5), (10, 10), (15, 15)]
    for kernel_size in kernel_sizes:
        kernel = np.ones(kernel_size, np.uint8)
        black_mask = (gray < 30).astype(np.uint8) * 255
        closed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of black regions
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Check if contour touches image edges
            touches_edges = (x == 0 or y == 0 or x + cw >= w or y + ch >= h)
            
            if touches_edges and cv2.contourArea(contour) > w * h * 0.1:
                # This might be a masking border
                bounds = (x, y, x + cw, y + ch)
                if validate_rectangular_mask(bounds, image.shape):
                    thickness = detect_actual_line_thickness(gray, bounds)
                    detection_results.append({
                        'bounds': bounds,
                        'type': 'detected_contour',
                        'thickness': thickness,
                        'confidence': 0.7,
                        'method': 'morphological'
                    })
    
    # Select best detection result
    if detection_results:
        # Sort by confidence and select the best
        best_result = max(detection_results, key=lambda x: x['confidence'])
        
        return {
            'has_masking': True,
            'bounds': best_result['bounds'],
            'type': best_result['type'],
            'thickness': min(best_result['thickness'], 200),  # Cap at 200px
            'confidence': best_result['confidence'],
            'method': best_result['method']
        }
    
    # No masking detected
    return {
        'has_masking': False,
        'bounds': (0, 0, w, h),
        'type': 'none',
        'thickness': 0,
        'confidence': 0,
        'method': 'none'
    }

def get_ring_region_info(image, masking_info):
    """Get detailed information about the ring region"""
    h, w = image.shape[:2]
    
    if masking_info['has_masking']:
        left, top, right, bottom = masking_info['bounds']
        ring_region = image[top:bottom, left:right]
    else:
        ring_region = image
        left, top, right, bottom = 0, 0, w, h
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
    
    # Multiple methods to find the ring
    ring_candidates = []
    
    # Method 1: Threshold-based detection
    for threshold_val in [180, 190, 200, 210, 220]:
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            area = cv2.contourArea(largest)
            
            if area > 100:  # Minimum area threshold
                ring_candidates.append({
                    'bounds': (x, y, cw, ch),
                    'area': area,
                    'center': (x + cw // 2, y + ch // 2)
                })
    
    # Method 2: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        ring_candidates.append({
            'bounds': (x, y, cw, ch),
            'area': cv2.contourArea(largest),
            'center': (x + cw // 2, y + ch // 2)
        })
    
    # Select best candidate
    if ring_candidates:
        # Choose the one with reasonable size and centered position
        best_candidate = max(ring_candidates, key=lambda c: c['area'])
        x, y, cw, ch = best_candidate['bounds']
        
        # Convert to full image coordinates
        ring_center_x = left + x + cw // 2
        ring_center_y = top + y + ch // 2
        ring_width = cw
        ring_height = ch
    else:
        # Fallback to center with estimated size
        ring_center_x = w // 2
        ring_center_y = h // 2
        ring_width = min(w, h) // 3
        ring_height = ring_width
    
    # Calculate additional info
    ring_radius = max(ring_width, ring_height) // 2
    
    return {
        'center': (ring_center_x, ring_center_y),
        'width': ring_width,
        'height': ring_height,
        'radius': ring_radius,
        'bounds': (ring_center_x - ring_width//2, ring_center_y - ring_height//2,
                  ring_center_x + ring_width//2, ring_center_y + ring_height//2),
        'aspect_ratio': ring_width / ring_height if ring_height > 0 else 1
    }

def detect_metal_type(image, masking_info):
    """Advanced metal type detection with multiple sampling strategies"""
    h, w = image.shape[:2]
    
    # Define multiple sampling regions
    sampling_regions = []
    
    # Primary region: inside masking area or center
    if masking_info['has_masking']:
        left, top, right, bottom = masking_info['bounds']
        # Take multiple samples from inside
        region_h = bottom - top
        region_w = right - left
        
        # Center sample
        cy, cx = (top + bottom) // 2, (left + right) // 2
        sample_size = min(region_h, region_w) // 4
        sampling_regions.append(image[cy-sample_size:cy+sample_size, 
                                    cx-sample_size:cx+sample_size])
        
        # Corner samples (might have different lighting)
        quarter_h = region_h // 4
        quarter_w = region_w // 4
        sampling_regions.append(image[top+quarter_h:top+2*quarter_h, 
                                    left+quarter_w:left+2*quarter_w])
    else:
        # Sample from center region
        center_y, center_x = h // 2, w // 2
        sample_sizes = [min(h, w) // 4, min(h, w) // 6, min(h, w) // 8]
        
        for size in sample_sizes:
            sampling_regions.append(image[center_y-size:center_y+size,
                                        center_x-size:center_x+size])
    
    # Analyze each sample region
    metal_scores = {
        'yellow_gold': 0,
        'rose_gold': 0,
        'white_gold': 0,
        'champagne': 0
    }
    
    for region in sampling_regions:
        if region.size == 0:
            continue
            
        # Convert to RGB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Create mask for metallic pixels (not too dark, not too bright)
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray_region, 50, 200)
        
        if np.sum(mask) > 100:  # Enough pixels to analyze
            # Calculate color statistics
            mean_color = cv2.mean(region_rgb, mask=mask)[:3]
            r, g, b = mean_color
            
            # Normalize colors
            total = r + g + b
            if total > 0:
                r_norm = r / total
                g_norm = g / total
                b_norm = b / total
                
                # Calculate color temperature
                temp = (r - b) / (r + g + b + 1e-6)
                
                # Champagne detection (very specific)
                if (180 < r < 220 and 180 < g < 220 and 170 < b < 210 and
                    abs(r - g) < 15 and abs(g - b) < 15 and r > b):
                    metal_scores['champagne'] += 2
                
                # Rose gold detection
                elif (r_norm > 0.36 and r > g > b and (r - b) > 20 and
                      temp > 0.05):
                    metal_scores['rose_gold'] += 2
                
                # Yellow gold detection
                elif (r_norm > 0.34 and g_norm > 0.33 and b_norm < 0.32 and
                      r > b and g > b and temp > 0.02):
                    metal_scores['yellow_gold'] += 2
                
                # White gold detection
                else:
                    metal_scores['white_gold'] += 1
                
                # Additional hue analysis
                hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
                hue = hsv[:, :, 0]
                hue_mean = np.mean(hue[mask > 0])
                
                # Hue-based scoring
                if 15 < hue_mean < 35:  # Orange-yellow range
                    metal_scores['yellow_gold'] += 1
                elif 0 < hue_mean < 15 or 345 < hue_mean < 360:  # Red-pink range
                    metal_scores['rose_gold'] += 1
                elif 35 < hue_mean < 50:  # Yellow-green (champagne)
                    metal_scores['champagne'] += 1
    
    # Determine final metal type
    detected_type = max(metal_scores, key=metal_scores.get)
    
    # Confidence check
    if metal_scores[detected_type] == 0:
        detected_type = 'white_gold'  # Default
    
    print(f"Metal detection scores: {metal_scores}")
    return detected_type

def detect_lighting_condition(image):
    """Advanced lighting condition detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate various statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Find peak of histogram
    peak_brightness = np.argmax(hist)
    
    # Calculate percentiles
    p10 = np.percentile(gray, 10)
    p90 = np.percentile(gray, 90)
    dynamic_range = p90 - p10
    
    # Advanced lighting classification
    if mean_brightness < 80 or peak_brightness < 60:
        return 'low'
    elif mean_brightness < 100 and dynamic_range < 100:
        return 'low'
    elif mean_brightness > 180 or peak_brightness > 200:
        return 'high'
    elif mean_brightness > 150 and std_brightness < 40:
        return 'high'
    else:
        return 'normal'

def apply_10_step_enhancement(image, metal_type, lighting):
    """Apply comprehensive 10-step enhancement process"""
    params = RING_PARAMS.get(metal_type, RING_PARAMS['white_gold'])[lighting]
    enhanced = image.copy()
    h, w = enhanced.shape[:2]
    
    # Step 1-2: Brightness and Contrast with v134 boost
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Apply v134 brightness boost (15% extra)
    brightness_factor = params['brightness'] * 1.15
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(params['contrast'])
    
    # Step 3-4: Saturation and Vibrance
    pil_img = ImageEnhance.Color(pil_img).enhance(params['saturation'])
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Apply vibrance (selective saturation)
    if 'vibrance' in params:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance color channels selectively
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        
        # Vibrance affects less saturated colors more
        saturation_map = np.sqrt(a**2 + b**2)
        low_sat_mask = saturation_map < 30
        
        a[low_sat_mask] *= params['vibrance']
        b[low_sat_mask] *= params['vibrance']
        
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Step 5-6: Advanced shadows and highlights adjustment
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0
    
    # Create masks for different brightness regions
    shadows_mask = l_float < 0.3
    midtones_mask = (l_float >= 0.3) & (l_float <= 0.7)
    highlights_mask = l_float > 0.7
    
    # Apply targeted adjustments
    l_float[shadows_mask] *= params['shadows']
    l_float[highlights_mask] *= params['highlights']
    
    # Smooth transitions
    l_float = cv2.GaussianBlur(l_float, (5, 5), 0)
    
    # Apply CLAHE for local contrast
    l = (l_float * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Step 7-8: Metal-specific color adjustments
    if metal_type == 'rose_gold' and 'red_adjust' in params:
        # Enhance pink/red tones
        b_ch, g_ch, r_ch = cv2.split(enhanced)
        r_ch = np.clip(r_ch.astype(np.float32) * params['red_adjust'], 0, 255)
        
        if 'pink_tone' in params:
            # Add subtle pink cast
            r_ch = r_ch * (1 - params['pink_tone'] * 0.1) + 255 * params['pink_tone'] * 0.1
        
        enhanced = cv2.merge([b_ch, g_ch, r_ch.astype(np.uint8)])
        
    elif metal_type == 'white_gold' and 'blue_adjust' in params:
        # Enhance cool tones
        b_ch, g_ch, r_ch = cv2.split(enhanced)
        b_ch = np.clip(b_ch.astype(np.float32) * params['blue_adjust'], 0, 255)
        
        if 'cool_tone' in params:
            # Add subtle blue cast
            temp_adjust = params['cool_tone']
            r_ch = np.clip(r_ch / temp_adjust, 0, 255)
        
        enhanced = cv2.merge([b_ch.astype(np.uint8), g_ch, r_ch.astype(np.uint8)])
        
    elif metal_type == 'champagne' and 'champagne_enhance' in params:
        # Special champagne gold enhancement
        if 'pearl_effect' in params:
            # Add pearlescent effect
            gray_layer = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray_layer, cv2.COLOR_GRAY2BGR)
            
            # Blend with original
            blend_factor = 0.15 * params['pearl_effect']
            enhanced = cv2.addWeighted(enhanced, 1 - blend_factor, gray_3ch, blend_factor, 0)
        
        # Warm tone adjustment
        if 'warm_tone' in params:
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * params['warm_tone'], 0, 255)
    
    # Step 9: Advanced sharpening with edge preservation
    # Unsharp mask
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Apply sharpening with v134 adjustment
    sharpness_factor = params['sharpness'] * 0.9
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness_factor)
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Additional detail enhancement
    if 'clarity' in params and params['clarity'] > 1:
        # Local contrast enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * (params['clarity'] - 1) * 0.1
        detail = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.8, detail, 0.2, 0)
    
    # Step 10: Final adjustments
    # Exposure adjustment with v134 boost
    exposure_factor = params.get('exposure', 1.0) * 1.1
    enhanced = cv2.convertScaleAbs(enhanced, alpha=exposure_factor, beta=0)
    
    # Whites and blacks adjustment
    if 'whites' in params:
        whites_adjust = params['whites']
        if whites_adjust != 0:
            mask = cv2.inRange(enhanced, (200, 200, 200), (255, 255, 255))
            enhanced[mask > 0] = np.clip(enhanced[mask > 0] * (1 + whites_adjust), 0, 255)
    
    if 'blacks' in params:
        blacks_adjust = params['blacks']
        if blacks_adjust != 0:
            mask = cv2.inRange(enhanced, (0, 0, 0), (50, 50, 50))
            enhanced[mask > 0] = np.clip(enhanced[mask > 0] * (1 + blacks_adjust), 0, 255)
    
    # Dehaze effect
    if 'dehaze' in params and params['dehaze'] > 0:
        # Simple dehaze using dark channel prior
        dark_channel = np.min(enhanced, axis=2)
        atmosphere = np.percentile(dark_channel, 95)
        transmission = 1 - params['dehaze'] * dark_channel / atmosphere
        transmission = np.clip(transmission, 0.1, 1)
        
        for i in range(3):
            enhanced[:, :, i] = np.clip(
                (enhanced[:, :, i] - atmosphere) / transmission + atmosphere,
                0, 255
            )
    
    # Final white overlay for v134 clean look
    white_overlay = np.ones_like(enhanced) * 255
    enhanced = cv2.addWeighted(enhanced, 0.95, white_overlay, 0.05, 0)
    
    # Ensure minimum brightness (v134 requirement)
    mean_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
    if mean_brightness < 200:
        brightness_boost = 220 / mean_brightness
        enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness_boost, beta=0)
    
    return enhanced

def apply_simple_masking_removal(image, masking_info):
    """Simple masking removal using inpainting"""
    if not masking_info['has_masking']:
        return image
    
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    left, top, right, bottom = masking_info['bounds']
    thickness = min(masking_info['thickness'], 100)
    
    # Create mask
    if masking_info['type'] == 'full_frame':
        mask[:top+thickness, :] = 255
        mask[bottom-thickness:, :] = 255
        mask[:, :left+thickness] = 255
        mask[:, right-thickness:] = 255
    elif masking_info['type'] == 'horizontal_bars':
        mask[:top+thickness, :] = 255
        mask[bottom-thickness:, :] = 255
    elif masking_info['type'] == 'vertical_bars':
        mask[:, :left+thickness] = 255
        mask[:, right-thickness:] = 255
    
    # Apply inpainting
    result = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
    
    return result

def apply_replicate_inpainting(image, masking_info):
    """Apply high-quality inpainting using Replicate API"""
    if not REPLICATE_API_TOKEN or not masking_info['has_masking']:
        return apply_simple_masking_removal(image, masking_info)
    
    try:
        # Initialize Replicate client
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        
        # Create detailed mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        left, top, right, bottom = masking_info['bounds']
        thickness = min(masking_info['thickness'], 150)
        
        # Create mask with feathering
        if masking_info['type'] == 'full_frame':
            # Top
            mask[:top+thickness, :] = 255
            # Bottom
            mask[bottom-thickness:, :] = 255
            # Left
            mask[:, :left+thickness] = 255
            # Right
            mask[:, right-thickness:] = 255
        elif masking_info['type'] == 'horizontal_bars':
            mask[:top+thickness, :] = 255
            mask[bottom-thickness:, :] = 255
        elif masking_info['type'] == 'vertical_bars':
            mask[:, :left+thickness] = 255
            mask[:, right-thickness:] = 255
        
        # Add feathering to mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 128).astype(np.uint8) * 255
        
        # Encode images
        image_base64 = encode_image_to_base64(image)
        mask_base64 = encode_image_to_base64(mask)
        
        # Run inpainting with optimized parameters
        output = client.run(
            "stability-ai/stable-diffusion-inpainting",
            input={
                "image": f"data:image/png;base64,{image_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "prompt": "clean white seamless professional product photography background, pure white studio background",
                "negative_prompt": "black, dark, shadows, borders, frames, edges, masking, lines, bars",
                "num_inference_steps": 30,
                "guidance_scale": 8.5,
                "seed": 42
            }
        )
        
        # Process result
        if output and isinstance(output, list) and len(output) > 0:
            result_url = output[0]
            import requests
            response = requests.get(result_url)
            
            if response.status_code == 200:
                result_image = Image.open(io.BytesIO(response.content))
                result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                
                # Ensure smooth blending
                kernel_size = 31
                mask_blur = cv2.GaussianBlur(mask.astype(np.float32) / 255, 
                                            (kernel_size, kernel_size), 0)
                
                # Blend original and inpainted
                result = image.copy()
                for i in range(3):
                    result[:, :, i] = (result_array[:, :, i] * mask_blur + 
                                     image[:, :, i] * (1 - mask_blur)).astype(np.uint8)
                
                return result
        
    except Exception as e:
        print(f"Replicate inpainting error: {str(e)}")
        print("Falling back to simple method")
    
    # Fallback to simple method
    return apply_simple_masking_removal(image, masking_info)

def create_professional_thumbnail(image, ring_info, size=1080):
    """Create professional thumbnail with perfect ring centering"""
    h, w = image.shape[:2]
    
    # Get ring information
    center_x, center_y = ring_info['center']
    ring_width = ring_info['width']
    ring_height = ring_info['height']
    
    # Calculate optimal crop size (v134: 2.5x padding)
    padding_factor = 2.5
    ideal_size = int(max(ring_width, ring_height) * padding_factor)
    
    # Ensure crop size doesn't exceed image dimensions
    crop_size = min(ideal_size, min(h, w))
    
    # Calculate crop region centered on ring
    half_crop = crop_size // 2
    
    # Initial crop boundaries
    left = center_x - half_crop
    right = center_x + half_crop
    top = center_y - half_crop
    bottom = center_y + half_crop
    
    # Adjust if crop exceeds image bounds
    if left < 0:
        right += -left
        left = 0
    if right > w:
        left -= (right - w)
        right = w
    if top < 0:
        bottom += -top
        top = 0
    if bottom > h:
        top -= (bottom - h)
        bottom = h
    
    # Final boundary check
    left = max(0, left)
    right = min(w, right)
    top = max(0, top)
    bottom = min(h, bottom)
    
    # Crop the image
    cropped = image[top:bottom, left:right].copy()
    
    # Make it square if needed
    crop_h, crop_w = cropped.shape[:2]
    if crop_h != crop_w:
        target_size = max(crop_h, crop_w)
        
        # Create square canvas with light background
        square = np.ones((target_size, target_size, 3), dtype=np.uint8) * 240
        
        # Calculate offsets for centering
        y_offset = (target_size - crop_h) // 2
        x_offset = (target_size - crop_w) // 2
        
        # Place cropped image in center
        square[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped
        cropped = square
    
    # High-quality resize
    thumbnail = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply v134 thumbnail enhancements
    pil_thumb = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
    
    # Specific thumbnail adjustments for clear visibility
    pil_thumb = ImageEnhance.Brightness(pil_thumb).enhance(1.05)
    pil_thumb = ImageEnhance.Sharpness(pil_thumb).enhance(1.2)
    pil_thumb = ImageEnhance.Contrast(pil_thumb).enhance(1.05)
    pil_thumb = ImageEnhance.Color(pil_thumb).enhance(1.02)
    
    # Convert back
    thumbnail = cv2.cvtColor(np.array(pil_thumb), cv2.COLOR_RGB2BGR)
    
    # Final quality enhancement
    # Subtle unsharp mask for extra clarity
    gaussian = cv2.GaussianBlur(thumbnail, (0, 0), 2.0)
    thumbnail = cv2.addWeighted(thumbnail, 1.5, gaussian, -0.5, 0)
    
    # Ensure good brightness for thumbnail
    thumb_gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    thumb_brightness = np.mean(thumb_gray)
    
    if thumb_brightness < 180:
        # Boost if too dark
        boost = 200 / thumb_brightness
        thumbnail = cv2.convertScaleAbs(thumbnail, alpha=boost, beta=0)
    
    return thumbnail

def handler(event):
    """Main handler function with complete error handling"""
    try:
        print(f"Starting {VERSION} processing...")
        start_time = time.time()
        
        # Get input from event
        image_input = event.get("input", {})
        
        # Support multiple input formats
        image_base64 = None
        for key in ["image", "image_base64", "input_image", "base64_image"]:
            if key in image_input and image_input[key]:
                image_base64 = image_input[key]
                break
        
        if not image_base64:
            print("No image data found in input")
            print(f"Available keys: {list(image_input.keys())}")
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "version": VERSION,
                    "available_keys": list(image_input.keys())
                }
            }
        
        # Validate base64
        if not is_valid_base64(image_base64):
            return {
                "output": {
                    "error": "Invalid base64 image data",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        print("Decoding image...")
        image = decode_base64_image(image_base64)
        original_image = image.copy()
        
        print(f"Image shape: {image.shape}")
        
        # Step 1: Ultra-advanced masking detection
        print("Detecting masking...")
        masking_info = detect_center_masking_ultra_advanced(image)
        print(f"Masking detection result: {masking_info}")
        
        # Step 2: Get ring region information
        print("Analyzing ring region...")
        ring_info = get_ring_region_info(image, masking_info)
        print(f"Ring info: center={ring_info['center']}, size={ring_info['width']}x{ring_info['height']}")
        
        # Step 3: Detect metal type
        print("Detecting metal type...")
        metal_type = detect_metal_type(image, masking_info)
        
        # Step 4: Detect lighting condition
        print("Detecting lighting condition...")
        lighting = detect_lighting_condition(image)
        
        print(f"Detection complete - Metal: {metal_type}, Lighting: {lighting}")
        
        # Step 5: Apply 10-step enhancement
        print("Applying enhancement...")
        enhanced = apply_10_step_enhancement(image, metal_type, lighting)
        
        # Step 6: Remove masking if detected
        if masking_info['has_masking']:
            print(f"Removing masking (type: {masking_info['type']}, thickness: {masking_info['thickness']}px)...")
            enhanced = apply_replicate_inpainting(enhanced, masking_info)
        
        # Step 7: Apply AFTER background
        print("Applying background...")
        pair_key = f'pair_{hash(image_base64) % 28 + 1}'
        bg_color = AFTER_BG_COLORS.get(pair_key, (242, 238, 235))
        
        # Create sophisticated gradient background
        h, w = enhanced.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8)
        
        # Apply base color
        for i in range(3):
            background[:, :, i] = bg_color[i]
        
        # Add subtle radial gradient
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (dist / max_dist) * 0.05  # Very subtle
        
        for i in range(3):
            background[:, :, i] = np.clip(background[:, :, i] * gradient, 0, 255).astype(np.uint8)
        
        # Blend with enhanced image
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Create sophisticated mask
        _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255
        
        # Apply background blending
        for i in range(3):
            enhanced[:, :, i] = (enhanced[:, :, i] * (1 - mask * 0.3) + 
                               background[:, :, i] * mask * 0.3).astype(np.uint8)
        
        # Step 8: Create professional thumbnail
        print("Creating thumbnail...")
        thumbnail = create_professional_thumbnail(enhanced, ring_info)
        
        # Step 9: Final quality assurance
        # Ensure proper brightness
        final_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        print(f"Final brightness: {final_brightness}")
        
        if final_brightness < 200:
            print("Applying final brightness correction...")
            correction = 220 / final_brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=correction, beta=0)
        
        # Step 10: Encode results
        print("Encoding results...")
        enhanced_base64 = encode_image_to_base64(enhanced)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "masking_detected": masking_info['has_masking'],
                    "masking_type": masking_info['type'],
                    "masking_thickness": masking_info['thickness'],
                    "ring_center": ring_info['center'],
                    "ring_size": f"{ring_info['width']}x{ring_info['height']}",
                    "background_pair": pair_key,
                    "background_color": bg_color,
                    "processing_time": f"{processing_time:.2f}s",
                    "version": VERSION,
                    "status": "success"
                }
            }
        }
        
        print(f"Processing completed successfully in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        traceback.print_exc()
        
        # Return error with proper structure
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print(f"Wedding Ring AI {VERSION} starting...")
    print(f"Replicate API Token: {'Available' if REPLICATE_API_TOKEN else 'Not set'}")
    runpod.serverless.start({"handler": handler})
