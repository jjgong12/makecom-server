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

# CRITICAL MASKING PARAMETERS FOR 6720x4480 IMAGES
MASKING_PARAMS = {
    'color_threshold': 50,  # RGB values below this are considered black/dark gray
    'min_thickness': 80,    # Minimum expected thickness in pixels
    'max_thickness': 150,   # Maximum expected thickness in pixels
    'scan_depth': 200,      # How deep to scan from edges
    'detection_confidence': 0.8  # 80% of pixels must be dark to confirm masking
}

def log_debug(message):
    """Print debug messages with timestamp"""
    if DEBUG:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

def find_image_in_event(event):
    """Find image data in various possible locations in the event"""
    log_debug("Searching for image in event structure...")
    
    # Log entire event structure for debugging
    log_debug(f"Full event: {json.dumps(event, default=str)[:1000]}...")
    
    # List of possible paths to check
    paths_to_check = [
        lambda e: e.get("image"),
        lambda e: e.get("image_base64"),
        lambda e: e.get("input", {}).get("image"),
        lambda e: e.get("input", {}).get("image_base64"),
        lambda e: e.get("input", {}).get("data", {}).get("image"),
        lambda e: e.get("input", {}).get("data", {}).get("image_base64"),
        lambda e: e.get("data", {}).get("image"),
        lambda e: e.get("data", {}).get("image_base64"),
        lambda e: e.get("input", {}).get("input", {}).get("image"),
        lambda e: e.get("input", {}).get("input", {}).get("image_base64"),
    ]
    
    # Try each path
    for i, path_func in enumerate(paths_to_check):
        try:
            result = path_func(event)
            if result and isinstance(result, str) and len(result) > 100:
                log_debug(f"Found image at path {i}")
                return result
        except Exception as e:
            continue
    
    # If not found in standard paths, search recursively
    def search_dict(d, depth=0, max_depth=5):
        if depth > max_depth:
            return None
        if isinstance(d, dict):
            for key, value in d.items():
                log_debug(f"Checking key '{key}' at depth {depth}")
                if key in ['image', 'image_base64', 'enhanced_image', 'base64']:
                    if isinstance(value, str) and len(value) > 100:
                        log_debug(f"Found image in key '{key}'")
                        return value
                elif isinstance(value, dict):
                    result = search_dict(value, depth + 1)
                    if result:
                        return result
        return None
    
    # Try recursive search
    image_data = search_dict(event)
    if image_data:
        return image_data
    
    # Last resort: find any large string that might be base64
    def find_large_strings(d, min_length=1000):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, str) and len(value) > min_length:
                    # Check if it looks like base64
                    if re.match(r'^[A-Za-z0-9+/]*={0,2}$', value[:100]):
                        log_debug(f"Found potential base64 in key '{key}'")
                        return value
                elif isinstance(value, dict):
                    result = find_large_strings(value)
                    if result:
                        return result
        return None
    
    return find_large_strings(event)

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
    """Detect the actual thickness of masking lines (80-150px for 6720x4480)"""
    x1, y1, x2, y2 = bounds
    
    # For high-res images, expect thicker lines
    max_thickness = MASKING_PARAMS['max_thickness']
    
    thicknesses = []
    
    # Sample from multiple points for accuracy
    sample_points = 30  # More samples for better accuracy
    
    # Sample from all edges
    for edge in ['top', 'left', 'bottom', 'right']:
        for i in range(sample_points):
            if edge == 'top':
                offset = int((x2 - x1) * (i + 1) / (sample_points + 1))
                for t in range(1, max_thickness):
                    if y1 + t < gray.shape[0]:
                        if gray[y1 + t, x1 + offset] > MASKING_PARAMS['color_threshold']:
                            thicknesses.append(t)
                            break
            elif edge == 'left':
                offset = int((y2 - y1) * (i + 1) / (sample_points + 1))
                for t in range(1, max_thickness):
                    if x1 + t < gray.shape[1]:
                        if gray[y1 + offset, x1 + t] > MASKING_PARAMS['color_threshold']:
                            thicknesses.append(t)
                            break
            elif edge == 'bottom':
                offset = int((x2 - x1) * (i + 1) / (sample_points + 1))
                for t in range(1, max_thickness):
                    if y2 - t >= 0:
                        if gray[y2 - t, x1 + offset] > MASKING_PARAMS['color_threshold']:
                            thicknesses.append(t)
                            break
            elif edge == 'right':
                offset = int((y2 - y1) * (i + 1) / (sample_points + 1))
                for t in range(1, max_thickness):
                    if x2 - t >= 0:
                        if gray[y1 + offset, x2 - t] > MASKING_PARAMS['color_threshold']:
                            thicknesses.append(t)
                            break
    
    if thicknesses:
        # Use median for robustness
        median_thickness = int(np.median(thicknesses))
        log_debug(f"Detected masking thickness: {median_thickness}px (samples: {len(thicknesses)})")
        return max(median_thickness + 10, MASKING_PARAMS['min_thickness'])
    
    # Default to expected thickness for high-res images
    return MASKING_PARAMS['min_thickness']

def detect_dark_gray_masking(image):
    """Specialized detection for dark gray masking (RGB 25-45)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    log_debug(f"Detecting dark gray masking on {w}x{h} image")
    
    # Initial aggressive crop (2% from each edge)
    initial_crop = int(min(w, h) * 0.02)
    
    # Create mask for dark pixels (RGB < 50)
    dark_mask = gray < MASKING_PARAMS['color_threshold']
    
    # Find edges where dark pixels end
    bounds = {
        'top': initial_crop,
        'bottom': h - initial_crop,
        'left': initial_crop,
        'right': w - initial_crop
    }
    
    # Scan from edges inward up to scan_depth
    scan_depth = min(MASKING_PARAMS['scan_depth'], min(w, h) // 4)
    
    # Top edge
    for y in range(initial_crop, min(scan_depth, h // 2)):
        dark_ratio = np.mean(dark_mask[y, :])
        if dark_ratio < MASKING_PARAMS['detection_confidence']:
            bounds['top'] = y + 10  # Add small margin
            break
    
    # Bottom edge
    for y in range(h - initial_crop, max(h - scan_depth, h // 2), -1):
        dark_ratio = np.mean(dark_mask[y, :])
        if dark_ratio < MASKING_PARAMS['detection_confidence']:
            bounds['bottom'] = y - 10
            break
    
    # Left edge
    for x in range(initial_crop, min(scan_depth, w // 2)):
        dark_ratio = np.mean(dark_mask[:, x])
        if dark_ratio < MASKING_PARAMS['detection_confidence']:
            bounds['left'] = x + 10
            break
    
    # Right edge
    for x in range(w - initial_crop, max(w - scan_depth, w // 2), -1):
        dark_ratio = np.mean(dark_mask[:, x])
        if dark_ratio < MASKING_PARAMS['detection_confidence']:
            bounds['right'] = x - 10
            break
    
    # Verify we found significant masking
    if (bounds['top'] > initial_crop or bounds['bottom'] < h - initial_crop or
        bounds['left'] > initial_crop or bounds['right'] < w - initial_crop):
        
        masking_info = {
            'bounds': (bounds['left'], bounds['top'], bounds['right'], bounds['bottom']),
            'thickness': max(
                bounds['top'] - initial_crop,
                h - initial_crop - bounds['bottom'],
                bounds['left'] - initial_crop,
                w - initial_crop - bounds['right']
            ),
            'type': 'dark_gray_frame',
            'method': 'specialized_dark_detection'
        }
        
        log_debug(f"Dark gray masking detected: {masking_info}")
        return masking_info
    
    return None

def detect_masking_comprehensive(image):
    """Comprehensive masking detection using multiple strategies"""
    log_debug("Starting comprehensive masking detection")
    
    # First try specialized dark gray detection (for RGB 25-45)
    dark_mask = detect_dark_gray_masking(image)
    if dark_mask:
        return dark_mask
    
    # Then try other methods
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Progressive threshold scanning
    for threshold in [30, 40, 50, 60, 70, 80, 90, 100, 120]:
        # Check edges for dark pixels
        edge_samples = {
            'top': gray[:50, :].flatten(),
            'bottom': gray[-50:, :].flatten(),
            'left': gray[:, :50].flatten(),
            'right': gray[:, -50:].flatten()
        }
        
        dark_edges = sum(1 for edge, pixels in edge_samples.items() 
                        if np.mean(pixels < threshold) > 0.8)
        
        if dark_edges >= 2:  # At least 2 edges are dark
            # Find exact boundaries
            bounds = [0, 0, w, h]
            
            # Scan inward from each edge
            for y in range(min(200, h // 2)):
                if np.mean(gray[y, :] < threshold) < 0.8:
                    bounds[1] = y
                    break
            
            for y in range(h - 1, max(h - 200, h // 2), -1):
                if np.mean(gray[y, :] < threshold) < 0.8:
                    bounds[3] = y + 1
                    break
            
            for x in range(min(200, w // 2)):
                if np.mean(gray[:, x] < threshold) < 0.8:
                    bounds[0] = x
                    break
            
            for x in range(w - 1, max(w - 200, w // 2), -1):
                if np.mean(gray[:, x] < threshold) < 0.8:
                    bounds[2] = x + 1
                    break
            
            # Verify significant masking found
            if (bounds[0] > 20 or bounds[1] > 20 or 
                bounds[2] < w - 20 or bounds[3] < h - 20):
                
                thickness = detect_actual_line_thickness(gray, tuple(bounds))
                
                return {
                    'bounds': tuple(bounds),
                    'thickness': thickness,
                    'type': 'edge_frame',
                    'method': f'threshold_{threshold}'
                }
    
    # Method 2: Contour-based detection
    for thresh_val in [30, 50, 70, 90]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to connect nearby dark regions
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour that touches image edges
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
            
            # Check if contour touches at least 2 edges
            touches_edges = sum([
                x <= 10,  # Left edge
                y <= 10,  # Top edge
                x + cnt_w >= w - 10,  # Right edge
                y + cnt_h >= h - 10   # Bottom edge
            ])
            
            if touches_edges >= 2 and cnt_w > w * 0.8 and cnt_h > h * 0.8:
                thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                
                # Calculate inner bounds
                inner_bounds = (
                    x + thickness,
                    y + thickness,
                    x + cnt_w - thickness,
                    y + cnt_h - thickness
                )
                
                return {
                    'bounds': inner_bounds,
                    'thickness': thickness,
                    'type': 'contour_frame',
                    'method': f'contour_{thresh_val}'
                }
    
    log_debug("No masking detected")
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
    """Simple masking removal by cropping to inner bounds"""
    if not masking_info:
        return image
    
    x1, y1, x2, y2 = masking_info['bounds']
    
    # Ensure bounds are within image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    # Crop to inner content
    cropped = image[y1:y2, x1:x2]
    
    log_debug(f"Cropped from {image.shape} to {cropped.shape}")
    
    return cropped

def create_thumbnail(image, size=(1000, 1300)):
    """Create a thumbnail with exact 1000x1300 size, ring centered and filling frame"""
    h, w = image.shape[:2]
    
    # Convert to PIL for easier handling
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Calculate aspect ratios
    img_aspect = w / h
    target_aspect = size[0] / size[1]
    
    if img_aspect > target_aspect:
        # Image is wider than target - fit by height
        new_h = size[1]
        new_w = int(new_h * img_aspect)
    else:
        # Image is taller than target - fit by width
        new_w = size[0]
        new_h = int(new_w / img_aspect)
    
    # Resize to ensure it fills the frame
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background with exact size
    thumbnail = Image.new('RGB', size, 'white')
    
    # Center crop to fit exactly
    x = (new_w - size[0]) // 2
    y = (new_h - size[1]) // 2
    
    # If resized image is smaller than target, paste centered
    if new_w < size[0] or new_h < size[1]:
        x = (size[0] - new_w) // 2
        y = (size[1] - new_h) // 2
        thumbnail.paste(pil_image, (x, y))
    else:
        # Crop to fit
        cropped = pil_image.crop((x, y, x + size[0], y + size[1]))
        thumbnail.paste(cropped, (0, 0))
    
    return cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

def handler(event):
    """Main handler function for RunPod"""
    try:
        log_debug(f"Handler started - v133")
        log_debug(f"Event type: {type(event)}")
        
        # Find image in event structure
        base64_image = find_image_in_event(event)
        
        if not base64_image:
            log_debug("No image found after extensive search")
            return {
                "output": {
                    "error": "No image provided in input",
                    "enhanced_image": "",
                    "thumbnail": "",
                    "processing_info": {}
                }
            }
        
        log_debug(f"Image found! Length: {len(base64_image)}")
        log_debug(f"Base64 string start: {base64_image[:100]}...")
        
        # Decode image
        image = decode_base64_image(base64_image)
        log_debug(f"Image decoded successfully: {image.shape}")
        
        # Process image
        # 1. Detect masking using comprehensive method
        masking_info = detect_masking_comprehensive(image)
        log_debug(f"Masking detected: {masking_info}")
        
        # 2. Remove masking first if detected
        if masking_info:
            # Remove masking
            if REPLICATE_API_TOKEN:
                processed = apply_replicate_inpainting(image, masking_info)
            else:
                processed = apply_simple_masking_removal(image, masking_info)
            log_debug("Masking removed")
        else:
            processed = image
        
        # 3. Detect lighting and metal type from processed image
        lighting = detect_lighting_condition(processed)
        metal_type = detect_metal_type(processed)
        log_debug(f"Lighting: {lighting}, Metal type: {metal_type}")
        
        # 4. Enhance wedding ring details
        enhanced = enhance_wedding_ring_details(processed, lighting, metal_type)
        log_debug("Wedding ring enhancement applied")
        
        # 5. Create thumbnail from enhanced image
        thumbnail = create_thumbnail(enhanced)
        log_debug(f"Thumbnail created: {thumbnail.shape}")
        
        # 6. Encode results
        # Enhanced image
        _, enhanced_buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_base64 = base64.b64encode(enhanced_buffer).decode('utf-8')
        enhanced_base64 = enhanced_base64.rstrip('=')  # Remove padding for Make.com
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding for Make.com
        
        # Processing info
        processing_info = {
            "masking_detected": bool(masking_info),
            "masking_removed": bool(masking_info),
            "masking_type": masking_info['type'] if masking_info else None,
            "masking_method": masking_info['method'] if masking_info else None,
            "thickness": masking_info['thickness'] if masking_info else None,
            "lighting": lighting,
            "metal_type": metal_type,
            "replicate_used": bool(REPLICATE_API_TOKEN and masking_info),
            "original_size": f"{image.shape[1]}x{image.shape[0]}",
            "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
            "thumbnail_size": f"{thumbnail.shape[1]}x{thumbnail.shape[0]}"
        }
        
        log_debug(f"Processing complete. Enhanced: {len(enhanced_base64)}, Thumbnail: {len(thumb_base64)}")
        log_debug(f"Processing info: {processing_info}")
        
        # Return with proper structure
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "processing_info": processing_info
            }
        }
        
    except Exception as e:
        log_debug(f"Error in handler: {str(e)}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
        return {
            "output": {
                "error": str(e),
                "enhanced_image": "",
                "thumbnail": "",
                "processing_info": {}
            }
        }

# RunPod handler - CORRECT FORMAT
runpod.serverless.start({"handler": handler})
