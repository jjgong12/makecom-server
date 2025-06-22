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
from typing import Dict, Tuple, Optional, List, Any
import time

# v135 - Ultra Precision Center Box Masking Detection & Removal
VERSION = "v135-ULTRA-PRECISION"

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

# 28 pairs AFTER background colors
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
            if 'base64,' in s:
                s = s.split('base64,')[1]
            base64.b64decode(s)
            return True
    except:
        pass
    return False

def decode_base64_image(base64_string):
    """Decode base64 image with multiple format support"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
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

def detect_center_box_masking_ultra(image):
    """Ultra-precision center box masking detection for 6720x4480 images"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"[V135] Starting ultra-precision masking detection on {w}x{h} image")
    
    # Expected box characteristics from analysis
    expected_box_width = int(w * 0.31)  # ~2080 pixels
    expected_box_height = int(h * 0.34)  # ~1520 pixels
    center_x, center_y = w // 2, h // 2
    
    # Multiple detection passes
    detection_results = []
    
    # Pass 1: Scan from center outward to find box edges
    # This is most reliable for center box masking
    for thickness in range(10, 30):  # 15-20 pixel line expected
        # Scan horizontally from center
        left_edge = None
        right_edge = None
        
        # Find left edge
        for x in range(center_x, 0, -10):
            column = gray[:, max(0, x-thickness):x+thickness]
            black_ratio = np.mean(column < 20)
            if black_ratio > 0.8:
                left_edge = x
                break
        
        # Find right edge
        for x in range(center_x, w, 10):
            column = gray[:, max(0, x-thickness):x+thickness]
            black_ratio = np.mean(column < 20)
            if black_ratio > 0.8:
                right_edge = x
                break
        
        # Scan vertically from center
        top_edge = None
        bottom_edge = None
        
        # Find top edge
        for y in range(center_y, 0, -10):
            row = gray[max(0, y-thickness):y+thickness, :]
            black_ratio = np.mean(row < 20)
            if black_ratio > 0.8:
                top_edge = y
                break
        
        # Find bottom edge
        for y in range(center_y, h, 10):
            row = gray[max(0, y-thickness):y+thickness, :]
            black_ratio = np.mean(row < 20)
            if black_ratio > 0.8:
                bottom_edge = y
                break
        
        if all([left_edge, right_edge, top_edge, bottom_edge]):
            box_width = right_edge - left_edge
            box_height = bottom_edge - top_edge
            
            # Verify it's close to expected dimensions
            width_ratio = box_width / expected_box_width
            height_ratio = box_height / expected_box_height
            
            if 0.8 < width_ratio < 1.2 and 0.8 < height_ratio < 1.2:
                detection_results.append({
                    'bounds': (left_edge, top_edge, right_edge, bottom_edge),
                    'thickness': thickness,
                    'confidence': 1.0 - abs(1.0 - width_ratio) - abs(1.0 - height_ratio),
                    'method': 'center_scan'
                })
    
    # Pass 2: Edge detection with precise line finding
    edges = cv2.Canny(gray, 30, 100)
    
    # Find horizontal lines
    horizontal_kernel = np.ones((1, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find vertical lines
    vertical_kernel = np.ones((50, 1), np.uint8)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine and find rectangle
    combined = cv2.bitwise_or(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Check if it's centered and right size
        contour_center_x = x + cw // 2
        contour_center_y = y + ch // 2
        
        center_dist = np.sqrt((contour_center_x - center_x)**2 + (contour_center_y - center_y)**2)
        
        if center_dist < min(w, h) * 0.1:  # Within 10% of center
            if 0.8 < cw/expected_box_width < 1.2 and 0.8 < ch/expected_box_height < 1.2:
                # Measure line thickness
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Sample thickness at multiple points
                thickness_samples = []
                for i in range(10):
                    sample_y = y + int(ch * i / 10)
                    line = mask[sample_y, :]
                    transitions = np.diff(line > 0).astype(int)
                    edges_found = np.where(transitions != 0)[0]
                    
                    if len(edges_found) >= 2:
                        thickness_samples.append(edges_found[1] - edges_found[0])
                
                if thickness_samples:
                    avg_thickness = int(np.mean(thickness_samples))
                    if 10 <= avg_thickness <= 30:
                        detection_results.append({
                            'bounds': (x, y, x + cw, y + ch),
                            'thickness': avg_thickness,
                            'confidence': 0.9,
                            'method': 'edge_detection'
                        })
    
    # Pass 3: Template matching for black rectangular frame
    # Create template of expected box
    template_size = (100, 100)
    template = np.ones(template_size, dtype=np.uint8) * 255
    cv2.rectangle(template, (10, 10), (90, 90), 0, 15)  # 15 pixel thick line
    
    # Resize and match
    scale_factor = min(w, h) / 1000
    scaled_template = cv2.resize(template, None, fx=scale_factor, fy=scale_factor)
    
    result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(result >= threshold)
    
    # Process matches
    for pt in zip(*loc[::-1]):
        # Extrapolate full box from corner match
        detected_thickness = int(15 * scale_factor)
        estimated_left = pt[0] + detected_thickness
        estimated_top = pt[1] + detected_thickness
        estimated_right = estimated_left + expected_box_width
        estimated_bottom = estimated_top + expected_box_height
        
        # Verify it's centered
        detected_center_x = (estimated_left + estimated_right) // 2
        detected_center_y = (estimated_top + estimated_bottom) // 2
        
        if abs(detected_center_x - center_x) < w * 0.1 and abs(detected_center_y - center_y) < h * 0.1:
            detection_results.append({
                'bounds': (estimated_left, estimated_top, estimated_right, estimated_bottom),
                'thickness': detected_thickness,
                'confidence': 0.8,
                'method': 'template_matching'
            })
    
    # Select best result
    if detection_results:
        # Sort by confidence
        best_result = max(detection_results, key=lambda x: x['confidence'])
        
        # Refine thickness detection
        left, top, right, bottom = best_result['bounds']
        
        # Sample actual thickness at multiple points
        thickness_samples = []
        
        # Top edge
        for x in range(left + 50, right - 50, 100):
            for t in range(5, 40):
                region = gray[max(0, top-t):top+t, x-5:x+5]
                if region.size > 0 and np.mean(region) < 20:
                    thickness_samples.append(t)
                    break
        
        # Bottom edge
        for x in range(left + 50, right - 50, 100):
            for t in range(5, 40):
                region = gray[bottom-t:min(h, bottom+t), x-5:x+5]
                if region.size > 0 and np.mean(region) < 20:
                    thickness_samples.append(t)
                    break
        
        # Left edge
        for y in range(top + 50, bottom - 50, 100):
            for t in range(5, 40):
                region = gray[y-5:y+5, max(0, left-t):left+t]
                if region.size > 0 and np.mean(region) < 20:
                    thickness_samples.append(t)
                    break
        
        # Right edge
        for y in range(top + 50, bottom - 50, 100):
            for t in range(5, 40):
                region = gray[y-5:y+5, right-t:min(w, right+t)]
                if region.size > 0 and np.mean(region) < 20:
                    thickness_samples.append(t)
                    break
        
        if thickness_samples:
            refined_thickness = int(np.median(thickness_samples))
        else:
            refined_thickness = best_result['thickness']
        
        print(f"[V135] Masking detected: bounds={best_result['bounds']}, thickness={refined_thickness}, method={best_result['method']}")
        
        return {
            'has_masking': True,
            'bounds': best_result['bounds'],
            'type': 'center_box',
            'thickness': refined_thickness,
            'confidence': best_result['confidence'],
            'method': best_result['method']
        }
    
    print("[V135] No center box masking detected")
    return {
        'has_masking': False,
        'bounds': (0, 0, w, h),
        'type': 'none',
        'thickness': 0,
        'confidence': 0,
        'method': 'none'
    }

def verify_masking_removal(image, original_masking_info):
    """Verify if masking has been successfully removed"""
    if not original_masking_info['has_masking']:
        return True
    
    # Re-detect masking
    current_masking = detect_center_box_masking_ultra(image)
    
    # If no masking detected, removal was successful
    if not current_masking['has_masking']:
        return True
    
    # If masking still detected, check if it's significantly reduced
    original_thickness = original_masking_info['thickness']
    current_thickness = current_masking['thickness']
    
    return current_thickness < original_thickness * 0.3

def get_ring_region_with_masking(image, masking_info):
    """Get ring region considering masking boundaries"""
    h, w = image.shape[:2]
    
    if masking_info['has_masking']:
        left, top, right, bottom = masking_info['bounds']
        
        # The ring is inside the masking box
        # Add padding to avoid the black lines
        padding = masking_info['thickness'] + 20
        
        ring_left = left + padding
        ring_top = top + padding
        ring_right = right - padding
        ring_bottom = bottom - padding
        
        # Ensure valid bounds
        ring_left = max(0, ring_left)
        ring_top = max(0, ring_top)
        ring_right = min(w, ring_right)
        ring_bottom = min(h, ring_bottom)
        
        ring_region = image[ring_top:ring_bottom, ring_left:ring_right]
    else:
        ring_region = image
        ring_left, ring_top = 0, 0
        ring_right, ring_bottom = w, h
    
    # Find ring within the region
    gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
    
    # Multiple threshold attempts
    ring_found = False
    for threshold_val in [180, 190, 200, 210, 220]:
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour that's likely a ring
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Ring should be somewhat circular
                if area > 1000 and 0.5 < cw/ch < 2.0:
                    valid_contours.append(contour)
            
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest)
                
                # Convert to full image coordinates
                ring_center_x = ring_left + x + cw // 2
                ring_center_y = ring_top + y + ch // 2
                
                ring_found = True
                break
    
    if not ring_found:
        # Fallback to center of the masking box
        ring_center_x = (ring_left + ring_right) // 2
        ring_center_y = (ring_top + ring_bottom) // 2
        cw = ch = min(ring_right - ring_left, ring_bottom - ring_top) // 3
    
    return {
        'center': (ring_center_x, ring_center_y),
        'width': cw,
        'height': ch,
        'bounds': (ring_center_x - cw//2, ring_center_y - ch//2,
                  ring_center_x + cw//2, ring_center_y + ch//2),
        'region_bounds': (ring_left, ring_top, ring_right, ring_bottom)
    }

def detect_metal_type(image, masking_info):
    """Detect metal type from the ring inside masking area"""
    h, w = image.shape[:2]
    
    # Get ring region
    ring_info = get_ring_region_with_masking(image, masking_info)
    
    # Sample from ring area
    rx1, ry1, rx2, ry2 = ring_info['bounds']
    rx1 = max(0, rx1 - 50)
    ry1 = max(0, ry1 - 50)
    rx2 = min(w, rx2 + 50)
    ry2 = min(h, ry2 + 50)
    
    ring_area = image[ry1:ry2, rx1:rx2]
    
    if ring_area.size == 0:
        return 'white_gold'
    
    # Convert to RGB
    ring_rgb = cv2.cvtColor(ring_area, cv2.COLOR_BGR2RGB)
    
    # Create mask for ring pixels
    gray = cv2.cvtColor(ring_area, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 50, 200)
    
    if np.sum(mask) > 100:
        mean_color = cv2.mean(ring_rgb, mask=mask)[:3]
        r, g, b = mean_color
        
        total = r + g + b
        if total > 0:
            r_norm = r / total
            g_norm = g / total
            b_norm = b / total
            
            # Champagne detection
            if (180 < r < 220 and 180 < g < 220 and 170 < b < 210 and
                abs(r - g) < 15 and abs(g - b) < 15 and r > b):
                return 'champagne'
            # Rose gold
            elif r_norm > 0.36 and r > g > b and (r - b) > 20:
                return 'rose_gold'
            # Yellow gold
            elif r_norm > 0.34 and g_norm > 0.33 and b_norm < 0.32:
                return 'yellow_gold'
            # White gold
            else:
                return 'white_gold'
    
    return 'white_gold'

def detect_lighting_condition(image):
    """Detect lighting condition"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 100:
        return 'low'
    elif mean_brightness < 150:
        return 'normal'
    else:
        return 'high'

def apply_light_enhancement_v135(image, metal_type, lighting):
    """Apply light enhancement for ring detail preservation"""
    params = RING_PARAMS.get(metal_type, RING_PARAMS['white_gold'])[lighting]
    enhanced = image.copy()
    
    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Very light adjustments to preserve original quality
    # Brightness - much lighter than before
    brightness_factor = 1.0 + (params['brightness'] - 1.0) * 0.3  # Only 30% of original adjustment
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
    
    # Contrast - minimal
    contrast_factor = 1.0 + (params['contrast'] - 1.0) * 0.2  # Only 20% of original
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)
    
    # Sharpness - light
    sharpness_factor = 1.0 + (params['sharpness'] - 1.0) * 0.4  # 40% of original
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness_factor)
    
    # Convert back
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Ensure minimum brightness without over-processing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 180:  # Lower threshold than before
        boost = 180 / mean_brightness
        enhanced = cv2.convertScaleAbs(enhanced, alpha=boost, beta=0)
    
    return enhanced

def apply_replicate_inpainting_v135(image, masking_info, max_attempts=3):
    """Apply Replicate inpainting with multiple attempts if needed"""
    if not REPLICATE_API_TOKEN or not masking_info['has_masking']:
        return image
    
    current_image = image.copy()
    
    for attempt in range(max_attempts):
        print(f"[V135] Inpainting attempt {attempt + 1}/{max_attempts}")
        
        try:
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            
            # Create precise mask
            h, w = current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            left, top, right, bottom = masking_info['bounds']
            thickness = masking_info['thickness'] + 10  # Add extra pixels
            
            # Draw the box frame
            # Top
            mask[max(0, top-thickness):top+thickness, left-thickness:right+thickness] = 255
            # Bottom
            mask[bottom-thickness:min(h, bottom+thickness), left-thickness:right+thickness] = 255
            # Left
            mask[top-thickness:bottom+thickness, max(0, left-thickness):left+thickness] = 255
            # Right
            mask[top-thickness:bottom+thickness, right-thickness:min(w, right+thickness)] = 255
            
            # Dilate mask slightly
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Encode images
            image_base64 = encode_image_to_base64(current_image)
            mask_base64 = encode_image_to_base64(mask)
            
            # Run inpainting
            output = client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{image_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean pure white seamless background, professional product photography studio background",
                    "negative_prompt": "black lines, borders, frames, edges, dark areas, shadows, masking",
                    "num_inference_steps": 35,
                    "guidance_scale": 9.0,
                    "seed": 42 + attempt  # Different seed each attempt
                }
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                result_url = output[0]
                import requests
                response = requests.get(result_url)
                
                if response.status_code == 200:
                    result_image = Image.open(io.BytesIO(response.content))
                    current_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                    
                    # Verify removal
                    if verify_masking_removal(current_image, masking_info):
                        print(f"[V135] Masking successfully removed on attempt {attempt + 1}")
                        return current_image
                    else:
                        print(f"[V135] Masking still detected after attempt {attempt + 1}")
                        # Update masking info for next attempt
                        masking_info = detect_center_box_masking_ultra(current_image)
            
        except Exception as e:
            print(f"[V135] Inpainting attempt {attempt + 1} failed: {str(e)}")
    
    return current_image

def create_professional_thumbnail_v135(image, ring_info, target_size=(1000, 1300)):
    """Create thumbnail with ring filling 90% of frame"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Get ring bounds
    ring_cx, ring_cy = ring_info['center']
    ring_w = ring_info['width']
    ring_h = ring_info['height']
    
    # Calculate crop to make ring fill 90% of frame
    # Account for aspect ratio difference
    target_aspect = target_w / target_h
    ring_aspect = ring_w / ring_h
    
    if ring_aspect > target_aspect:
        # Ring is wider - fit to width
        crop_width = int(ring_w / 0.9)
        crop_height = int(crop_width / target_aspect)
    else:
        # Ring is taller - fit to height
        crop_height = int(ring_h / 0.9)
        crop_width = int(crop_height * target_aspect)
    
    # Calculate crop boundaries
    left = ring_cx - crop_width // 2
    right = ring_cx + crop_width // 2
    top = ring_cy - crop_height // 2
    bottom = ring_cy + crop_height // 2
    
    # Adjust if out of bounds
    if left < 0:
        right -= left
        left = 0
    if right > w:
        left -= (right - w)
        right = w
    if top < 0:
        bottom -= top
        top = 0
    if bottom > h:
        top -= (bottom - h)
        bottom = h
    
    # Final bounds check
    left = max(0, left)
    right = min(w, right)
    top = max(0, top)
    bottom = min(h, bottom)
    
    # Crop
    cropped = image[top:bottom, left:right].copy()
    
    # Resize to target size
    thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Light enhancement for thumbnail
    pil_thumb = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
    
    # Subtle adjustments
    pil_thumb = ImageEnhance.Brightness(pil_thumb).enhance(1.05)
    pil_thumb = ImageEnhance.Sharpness(pil_thumb).enhance(1.15)
    pil_thumb = ImageEnhance.Contrast(pil_thumb).enhance(1.03)
    
    thumbnail = cv2.cvtColor(np.array(pil_thumb), cv2.COLOR_RGB2BGR)
    
    return thumbnail

def handler(event):
    """Main handler function with v135 improvements"""
    try:
        print(f"[V135] Starting {VERSION} processing...")
        start_time = time.time()
        
        # Get input
        image_input = event.get("input", {})
        
        image_base64 = None
        for key in ["image", "image_base64", "input_image", "base64_image"]:
            if key in image_input and image_input[key]:
                image_base64 = image_input[key]
                break
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Validate and decode
        if not is_valid_base64(image_base64):
            return {
                "output": {
                    "error": "Invalid base64 image data",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        print("[V135] Decoding image...")
        image = decode_base64_image(image_base64)
        original_image = image.copy()
        h, w = image.shape[:2]
        print(f"[V135] Image size: {w}x{h}")
        
        # Step 1: Ultra-precision masking detection
        print("[V135] Detecting center box masking...")
        masking_info = detect_center_box_masking_ultra(image)
        
        # Step 2: Get ring info before any processing
        print("[V135] Analyzing ring region...")
        ring_info = get_ring_region_with_masking(image, masking_info)
        
        # Step 3: Detect metal type
        print("[V135] Detecting metal type...")
        metal_type = detect_metal_type(image, masking_info)
        
        # Step 4: Detect lighting
        lighting = detect_lighting_condition(image)
        
        print(f"[V135] Metal: {metal_type}, Lighting: {lighting}")
        
        # Step 5: Process ring area with light enhancement
        if masking_info['has_masking']:
            # Process only the ring area inside masking
            left, top, right, bottom = ring_info['region_bounds']
            ring_region = image[top:bottom, left:right].copy()
            
            # Apply light enhancement
            enhanced_region = apply_light_enhancement_v135(ring_region, metal_type, lighting)
            
            # Put back the enhanced region
            enhanced = image.copy()
            enhanced[top:bottom, left:right] = enhanced_region
        else:
            # Process entire image
            enhanced = apply_light_enhancement_v135(image, metal_type, lighting)
        
        # Step 6: Remove masking with multiple attempts
        if masking_info['has_masking']:
            print("[V135] Removing masking...")
            enhanced = apply_replicate_inpainting_v135(enhanced, masking_info, max_attempts=3)
        
        # Step 7: Apply subtle background
        pair_key = f'pair_{hash(image_base64) % 28 + 1}'
        bg_color = AFTER_BG_COLORS.get(pair_key, (242, 238, 235))
        
        # Very subtle background application
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255
        
        background = np.ones_like(enhanced)
        for i in range(3):
            background[:, :, i] = bg_color[i]
        
        for i in range(3):
            enhanced[:, :, i] = (enhanced[:, :, i] * (1 - mask * 0.2) + 
                               background[:, :, i] * mask * 0.2).astype(np.uint8)
        
        # Step 8: Create thumbnail with 90% ring coverage
        print("[V135] Creating thumbnail...")
        thumbnail = create_professional_thumbnail_v135(enhanced, ring_info)
        
        # Step 9: Final quality check
        final_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        if final_brightness < 200:
            correction = 210 / final_brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=correction, beta=0)
        
        # Encode results
        enhanced_base64 = encode_image_to_base64(enhanced)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
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
                    "masking_bounds": masking_info['bounds'] if masking_info['has_masking'] else None,
                    "ring_center": ring_info['center'],
                    "ring_size": f"{ring_info['width']}x{ring_info['height']}",
                    "thumbnail_size": "1000x1300",
                    "background_pair": pair_key,
                    "processing_time": f"{processing_time:.2f}s",
                    "version": VERSION,
                    "status": "success"
                }
            }
        }
        
        print(f"[V135] Processing completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"[V135] Error: {str(e)}")
        traceback.print_exc()
        
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
