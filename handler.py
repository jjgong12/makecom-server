import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import base64
import json
import os
import requests
import time
import replicate

# Replicate API configuration
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '')

def log_debug(message):
    """Debug logging"""
    print(f"[DEBUG v122] {message}")

def decode_base64_image(base64_string):
    """Decode base64 image with comprehensive error handling for Make.com"""
    try:
        if not base64_string:
            raise ValueError("Empty base64 string")
        
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Clean the string
        base64_string = base64_string.strip()
        
        # Try standard decode first (without adding padding)
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except:
            pass
        
        # If failed, try with padding
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string_padded = base64_string + '=' * (4 - missing_padding)
            try:
                image_data = base64.b64decode(base64_string_padded)
                image = Image.open(io.BytesIO(image_data))
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except:
                pass
        
        # Try URL-safe decode
        try:
            # Replace URL-safe characters
            base64_string_safe = base64_string.replace('-', '+').replace('_', '/')
            missing_padding = len(base64_string_safe) % 4
            if missing_padding:
                base64_string_safe += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string_safe)
            image = Image.open(io.BytesIO(image_data))
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except:
            pass
        
        raise Exception("All decoding methods failed")
        
    except Exception as e:
        log_debug(f"Error decoding base64: {str(e)}")
        raise Exception(f"Could not decode image: {str(e)}")

def encode_image_to_base64(image, format='JPEG'):
    """Encode image to base64 WITHOUT padding for Make.com"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=95)
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # Remove padding for Make.com
    return base64_string.rstrip('=')

def detect_actual_line_thickness(gray, bounds):
    """Detect actual thickness of black lines - can handle up to 100px"""
    left, top, right, bottom = bounds
    h, w = gray.shape
    thicknesses = []
    
    # Sample from multiple points for accurate measurement
    # Top edge
    if top > 0:
        for x in range(left + 10, min(right - 10, w), 50):
            for y in range(top, min(top + 200, h)):  # Check up to 200px
                if gray[y, x] > 50:
                    thicknesses.append(y - top)
                    break
    
    # Left edge
    if left > 0:
        for y in range(top + 10, min(bottom - 10, h), 50):
            for x in range(left, min(left + 200, w)):  # Check up to 200px
                if gray[y, x] > 50:
                    thicknesses.append(x - left)
                    break
    
    # Bottom edge
    if bottom < h:
        for x in range(left + 10, min(right - 10, w), 50):
            for y in range(bottom - 1, max(bottom - 200, 0), -1):
                if gray[y, x] > 50:
                    thicknesses.append(bottom - y)
                    break
    
    # Right edge
    if right < w:
        for y in range(top + 10, min(bottom - 10, h), 50):
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
    for kernel_size in [3, 5, 7]:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Try different morphological operations
        for threshold in [30, 40, 50]:
            binary = (gray < threshold).astype(np.uint8) * 255
            
            # Close to fill gaps
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, cnt_w, cnt_h = cv2.boundingRect(contour)
                
                if cnt_w > w * 0.3 and cnt_h > h * 0.3:
                    # Check if rectangular
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    
                    if len(approx) == 4:  # Rectangular
                        thickness = detect_actual_line_thickness(gray, (x, y, x + cnt_w, y + cnt_h))
                        
                        inner_x = x + thickness
                        inner_y = y + thickness
                        inner_w = cnt_w - 2 * thickness
                        inner_h = cnt_h - 2 * thickness
                        
                        if inner_w > 50 and inner_h > 50:
                            methods_results[f'morph_{kernel_size}_{threshold}'] = {
                                'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                                'thickness': thickness,
                                'type': 'center',
                                'method': f'morph_{kernel_size}_{threshold}'
                            }
                            break
            if f'morph_{kernel_size}_{threshold}' in methods_results:
                break
    
    # Cross-validate results
    log_debug(f"Found {len(methods_results)} potential center maskings")
    final_mask = cross_validate_detection(methods_results, (h, w))
    
    if final_mask:
        log_debug(f"Confirmed center masking: bounds={final_mask['bounds']}, thickness={final_mask['thickness']}")
        return final_mask
    
    log_debug("No center masking detected")
    return None

def detect_edge_masking_ultra(image):
    """Ultra precise edge masking detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check each edge with multiple scan depths
    edges_with_black = []
    best_scan_depth = 0
    
    # Try multiple scan depths
    for scan_depth in [50, 100, 150, 200]:
        if scan_depth > min(h, w) // 4:
            continue
            
        # Adaptive threshold based on image
        mean_brightness = np.mean(gray)
        threshold = min(40, mean_brightness * 0.2)
        
        current_edges = []
        
        # Top edge
        if np.mean(gray[:scan_depth, :] < threshold) > 0.7:
            current_edges.append('top')
        
        # Bottom edge
        if np.mean(gray[-scan_depth:, :] < threshold) > 0.7:
            current_edges.append('bottom')
        
        # Left edge
        if np.mean(gray[:, :scan_depth] < threshold) > 0.7:
            current_edges.append('left')
        
        # Right edge
        if np.mean(gray[:, -scan_depth:] < threshold) > 0.7:
            current_edges.append('right')
        
        if len(current_edges) >= 2:
            edges_with_black = current_edges
            best_scan_depth = scan_depth
            break
    
    if not edges_with_black:
        return None
    
    # Calculate bounds with detected edges
    left, top, right, bottom = 0, 0, w, h
    
    if 'top' in edges_with_black:
        for y in range(min(best_scan_depth * 2, h)):
            if np.mean(gray[y, :] < 40) < 0.5:
                top = y
                break
    
    if 'bottom' in edges_with_black:
        for y in range(h-1, max(h-best_scan_depth*2, -1), -1):
            if np.mean(gray[y, :] < 40) < 0.5:
                bottom = y + 1
                break
    
    if 'left' in edges_with_black:
        for x in range(min(best_scan_depth * 2, w)):
            if np.mean(gray[:, x] < 40) < 0.5:
                left = x
                break
    
    if 'right' in edges_with_black:
        for x in range(w-1, max(w-best_scan_depth*2, -1), -1):
            if np.mean(gray[:, x] < 40) < 0.5:
                right = x + 1
                break
    
    if left < right and top < bottom:
        thickness = detect_actual_line_thickness(gray, (left, top, right, bottom))
        return {
            'bounds': (left, top, right, bottom),
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
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Calculate color statistics
    avg_hue = np.mean(hsv[:, :, 0])
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_brightness = np.mean(hsv[:, :, 2])
    
    # RGB analysis
    avg_r = np.mean(center_region[:, :, 0])
    avg_g = np.mean(center_region[:, :, 1])
    avg_b = np.mean(center_region[:, :, 2])
    
    log_debug(f"Color analysis - H:{avg_hue:.1f}, S:{avg_saturation:.1f}, V:{avg_brightness:.1f}, R:{avg_r:.1f}, G:{avg_g:.1f}, B:{avg_b:.1f}")
    
    # Metal type detection based on 28 pairs knowledge
    # Priority order: rose_gold -> yellow_gold -> white_gold -> plain_white
    
    # Rose gold detection
    if (avg_r > avg_g * 1.15 and avg_r > avg_b * 1.2 and 
        avg_saturation > 20 and avg_hue < 20):
        return "rose_gold"
    
    # Yellow gold detection
    elif (avg_r > avg_b * 1.2 and avg_g > avg_b * 1.15 and 
          avg_saturation > 25 and 20 < avg_hue < 40):
        return "yellow_gold"
    
    # White gold detection (includes bright silver)
    elif avg_saturation < 20 and avg_brightness > 150:
        return "white_gold"
    
    # Plain white / Champagne gold (무도금화이트)
    else:
        # If very low saturation and high brightness
        if avg_saturation < 15 and avg_brightness > 180:
            return "plain_white"  # This is champagne gold in Korean terms
        else:
            return "white_gold"  # Default to white gold

def detect_lighting_condition(image):
    """Detect lighting condition from image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Analyze distribution
    dark_pixels = np.sum(hist[:85])  # 0-85
    mid_pixels = np.sum(hist[85:170])  # 85-170
    bright_pixels = np.sum(hist[170:])  # 170-255
    
    log_debug(f"Lighting - Mean: {mean_brightness:.1f}, Std: {std_brightness:.1f}, Dark: {dark_pixels:.2f}, Mid: {mid_pixels:.2f}, Bright: {bright_pixels:.2f}")
    
    # Determine lighting condition
    if mean_brightness < 100:
        return "low"
    elif mean_brightness > 160:
        return "high"
    else:
        return "normal"

def enhance_wedding_ring_details(image, metal_type, lighting, is_ring_region=True):
    """Enhanced wedding ring details based on 38 pairs of training data"""
    # Parameters based on metal type and lighting (from 28+10 pairs)
    params = {
        'yellow_gold': {
            'low': {'brightness': 1.25, 'contrast': 1.20, 'saturation': 1.15, 'sharpness': 1.3, 'gamma': 0.85},
            'normal': {'brightness': 1.18, 'contrast': 1.15, 'saturation': 1.10, 'sharpness': 1.2, 'gamma': 0.90},
            'high': {'brightness': 1.10, 'contrast': 1.10, 'saturation': 1.05, 'sharpness': 1.1, 'gamma': 0.95}
        },
        'rose_gold': {
            'low': {'brightness': 1.22, 'contrast': 1.18, 'saturation': 1.12, 'sharpness': 1.3, 'gamma': 0.87},
            'normal': {'brightness': 1.15, 'contrast': 1.12, 'saturation': 1.08, 'sharpness': 1.2, 'gamma': 0.92},
            'high': {'brightness': 1.08, 'contrast': 1.08, 'saturation': 1.03, 'sharpness': 1.1, 'gamma': 0.96}
        },
        'white_gold': {
            'low': {'brightness': 1.28, 'contrast': 1.22, 'saturation': 1.02, 'sharpness': 1.4, 'gamma': 0.83},
            'normal': {'brightness': 1.20, 'contrast': 1.15, 'saturation': 1.00, 'sharpness': 1.3, 'gamma': 0.88},
            'high': {'brightness': 1.12, 'contrast': 1.10, 'saturation': 0.98, 'sharpness': 1.2, 'gamma': 0.93}
        },
        'plain_white': {  # Champagne gold (무도금화이트)
            'low': {'brightness': 1.30, 'contrast': 1.25, 'saturation': 0.95, 'sharpness': 1.4, 'gamma': 0.82},
            'normal': {'brightness': 1.22, 'contrast': 1.18, 'saturation': 0.92, 'sharpness': 1.3, 'gamma': 0.87},
            'high': {'brightness': 1.15, 'contrast': 1.12, 'saturation': 0.90, 'sharpness': 1.2, 'gamma': 0.92}
        }
    }
    
    # Get parameters
    if metal_type not in params:
        metal_type = 'white_gold'  # Default
    if lighting not in params[metal_type]:
        lighting = 'normal'  # Default
    
    p = params[metal_type][lighting]
    
    # If not ring region, use lighter enhancement
    if not is_ring_region:
        p = {
            'brightness': 1.05 + (p['brightness'] - 1.0) * 0.3,
            'contrast': 1.03 + (p['contrast'] - 1.0) * 0.3,
            'saturation': 1.0 + (p['saturation'] - 1.0) * 0.3,
            'sharpness': 1.05 + (p['sharpness'] - 1.0) * 0.3,
            'gamma': 0.95 + (p['gamma'] - 0.95) * 0.3
        }
    
    # Convert to PIL for enhancements
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply enhancements
    # 1. Brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(p['brightness'])
    
    # 2. Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(p['contrast'])
    
    # 3. Color (Saturation)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(p['saturation'])
    
    # 4. Sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(p['sharpness'])
    
    # Convert back to numpy
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 5. Gamma correction
    gamma_corrected = np.power(result / 255.0, p['gamma']) * 255.0
    result = gamma_corrected.astype(np.uint8)
    
    # 6. Additional detail enhancement for ring regions
    if is_ring_region:
        # Unsharp mask for extra detail
        gaussian = cv2.GaussianBlur(result, (5, 5), 1.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
        
        # CLAHE for local contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result

def apply_replicate_inpainting(image, masking_info):
    """Apply Replicate API inpainting for natural masking removal"""
    try:
        if not REPLICATE_API_TOKEN:
            log_debug("No Replicate API token")
            return None
        
        log_debug("Applying Replicate inpainting")
        
        # Initialize Replicate client
        replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        
        # Create mask for inpainting
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        thickness = masking_info['thickness']
        bounds = masking_info['bounds']
        
        if masking_info['type'] == 'center':
            # For center masking, create mask for the black frame
            x1, y1, x2, y2 = bounds
            
            # Expand bounds to include the black frame
            frame_x1 = max(0, x1 - thickness - 10)
            frame_y1 = max(0, y1 - thickness - 10)
            frame_x2 = min(w, x2 + thickness + 10)
            frame_y2 = min(h, y2 + thickness + 10)
            
            # Create mask for the frame
            mask[frame_y1:y1+10, frame_x1:frame_x2] = 255  # Top
            mask[y2-10:frame_y2, frame_x1:frame_x2] = 255  # Bottom
            mask[frame_y1:frame_y2, frame_x1:x1+10] = 255  # Left
            mask[frame_y1:frame_y2, x2-10:frame_x2] = 255  # Right
            
        elif masking_info['type'] == 'edge':
            # For edge masking, mask the edges
            for edge in masking_info.get('edges', []):
                if edge == 'top':
                    mask[:thickness+20, :] = 255
                elif edge == 'bottom':
                    mask[-thickness-20:, :] = 255
                elif edge == 'left':
                    mask[:, :thickness+20] = 255
                elif edge == 'right':
                    mask[:, -thickness-20:] = 255
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        
        # Save to bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        mask_bytes = io.BytesIO()
        pil_mask.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
        
        # Run inpainting
        output = replicate_client.run(
            "stability-ai/stable-diffusion-inpainting",
            input={
                "image": image_bytes,
                "mask": mask_bytes,
                "prompt": "clean white seamless product photography background, smooth gradient",
                "negative_prompt": "black, dark, shadows, borders, frames, lines, marks",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        
        if output and len(output) > 0:
            # Download result
            response = requests.get(output[0])
            result_image = Image.open(io.BytesIO(response.content))
            result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            
            # Preserve the ring area
            if masking_info['type'] == 'center':
                x1, y1, x2, y2 = bounds
                result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            
            log_debug("Replicate inpainting successful")
            return result
            
    except Exception as e:
        log_debug(f"Replicate inpainting failed: {str(e)}")
    
    return None

def apply_simple_masking_removal(image, masking_info):
    """Simple masking removal using averaging and smooth blending"""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Get background color from edges
    edge_pixels = []
    margin = 100
    
    # Sample from all edges (avoiding masked areas)
    if masking_info['type'] == 'edge':
        edges = masking_info.get('edges', [])
        if 'top' not in edges and margin < h//2:
            edge_pixels.extend(image[0:margin, :].reshape(-1, 3))
        if 'bottom' not in edges and margin < h//2:
            edge_pixels.extend(image[h-margin:h, :].reshape(-1, 3))
        if 'left' not in edges and margin < w//2:
            edge_pixels.extend(image[:, 0:margin].reshape(-1, 3))
        if 'right' not in edges and margin < w//2:
            edge_pixels.extend(image[:, w-margin:w].reshape(-1, 3))
    else:
        # For center masking, sample from corners
        corner_size = 50
        if corner_size < min(h, w) // 4:
            edge_pixels.extend(image[0:corner_size, 0:corner_size].reshape(-1, 3))
            edge_pixels.extend(image[0:corner_size, w-corner_size:w].reshape(-1, 3))
            edge_pixels.extend(image[h-corner_size:h, 0:corner_size].reshape(-1, 3))
            edge_pixels.extend(image[h-corner_size:h, w-corner_size:w].reshape(-1, 3))
    
    if edge_pixels:
        bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
    else:
        # Default light background
        bg_color = np.array([240, 240, 240], dtype=np.uint8)
    
    # Create gradient background
    gradient = np.ones((h, w, 3), dtype=np.uint8) * bg_color
    
    # Add subtle gradient
    for y in range(h):
        factor = 1.0 - (abs(y - h//2) / (h//2)) * 0.05
        gradient[y] = bg_color * factor
    
    # Apply masking removal
    thickness = masking_info['thickness']
    
    if masking_info['type'] == 'center':
        x1, y1, x2, y2 = masking_info['bounds']
        
        # Fill the black frame area
        # Top
        if y1 > thickness:
            result[max(0, y1-thickness-20):y1, x1-thickness-20:x2+thickness+20] = \
                gradient[max(0, y1-thickness-20):y1, x1-thickness-20:x2+thickness+20]
        
        # Bottom
        if y2 < h - thickness:
            result[y2:min(h, y2+thickness+20), x1-thickness-20:x2+thickness+20] = \
                gradient[y2:min(h, y2+thickness+20), x1-thickness-20:x2+thickness+20]
        
        # Left
        if x1 > thickness:
            result[y1-thickness-20:y2+thickness+20, max(0, x1-thickness-20):x1] = \
                gradient[y1-thickness-20:y2+thickness+20, max(0, x1-thickness-20):x1]
        
        # Right
        if x2 < w - thickness:
            result[y1-thickness-20:y2+thickness+20, x2:min(w, x2+thickness+20)] = \
                gradient[y1-thickness-20:y2+thickness+20, x2:min(w, x2+thickness+20)]
    
    elif masking_info['type'] == 'edge':
        edges = masking_info.get('edges', [])
        
        if 'top' in edges:
            result[:thickness+20, :] = gradient[:thickness+20, :]
        if 'bottom' in edges:
            result[-thickness-20:, :] = gradient[-thickness-20:, :]
        if 'left' in edges:
            result[:, :thickness+20] = gradient[:, :thickness+20]
        if 'right' in edges:
            result[:, -thickness-20:] = gradient[:, -thickness-20:]
    
    # Smooth blending
    for i in range(3):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

def create_thumbnail(image, target_size=(800, 800)):
    """Create thumbnail with ring centered and filling most of the frame"""
    h, w = image.shape[:2]
    
    # Convert to grayscale for ring detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the ring using threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (should be the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_ring, h_ring = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = int(max(w_ring, h_ring) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_ring = min(w - x, w_ring + 2 * padding)
        h_ring = min(h - y, h_ring + 2 * padding)
        
        # Make it square
        if w_ring > h_ring:
            diff = w_ring - h_ring
            y = max(0, y - diff // 2)
            h_ring = min(h - y, w_ring)
        else:
            diff = h_ring - w_ring
            x = max(0, x - diff // 2)
            w_ring = min(w - x, h_ring)
        
        # Crop to ring area
        cropped = image[y:y+h_ring, x:x+w_ring]
    else:
        # Fallback: use center crop
        crop_size = int(min(w, h) * 0.8)
        x = (w - crop_size) // 2
        y = (h - crop_size) // 2
        cropped = image[y:y+crop_size, x:x+crop_size]
    
    # Resize to target size
    thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """Main handler function for RunPod"""
    try:
        log_debug("Handler started")
        
        # Parse input
        job_input = event.get("input", {})
        
        # Try different possible input formats
        image_base64 = None
        if "image" in job_input:
            image_base64 = job_input["image"]
        elif "image_base64" in job_input:
            image_base64 = job_input["image_base64"]
        elif "enhanced_image" in job_input:
            image_base64 = job_input["enhanced_image"]
        
        if not image_base64:
            log_debug(f"Available keys: {list(job_input.keys())}")
            return {
                "output": {
                    "error": "No image provided",
                    "available_keys": list(job_input.keys())
                }
            }
        
        # Decode image
        image = decode_base64_image(image_base64)
        log_debug(f"Image decoded successfully: {image.shape}")
        
        # Step 1: Detect masking (both center and edge)
        center_mask = detect_center_masking_ultra_advanced(image)
        edge_mask = detect_edge_masking_ultra(image)
        
        # Prioritize center masking if found
        masking_info = center_mask or edge_mask
        
        # Step 2: Detect lighting condition
        lighting = detect_lighting_condition(image)
        log_debug(f"Lighting condition: {lighting}")
        
        # Step 3: Process based on masking detection
        if masking_info:
            log_debug(f"Masking detected: {masking_info['type']}, thickness: {masking_info['thickness']}px")
            
            # Detect metal type within masked area
            x1, y1, x2, y2 = masking_info['bounds']
            ring_region = image[y1:y2, x1:x2]
            metal_type = detect_metal_type_advanced(ring_region)
            log_debug(f"Metal type: {metal_type}")
            
            # Enhance ring details within mask
            enhanced_ring = enhance_wedding_ring_details(ring_region, metal_type, lighting, is_ring_region=True)
            
            # Remove masking
            if REPLICATE_API_TOKEN:
                result = apply_replicate_inpainting(image, masking_info)
                if result is None:
                    result = apply_simple_masking_removal(image, masking_info)
            else:
                result = apply_simple_masking_removal(image, masking_info)
            
            # Put enhanced ring back
            result[y1:y2, x1:x2] = enhanced_ring
            
            # Light enhancement for full image
            result = enhance_wedding_ring_details(result, metal_type, lighting, is_ring_region=False)
            
        else:
            log_debug("No masking detected, applying direct enhancement")
            # No masking, just enhance
            metal_type = detect_metal_type_advanced(image)
            log_debug(f"Metal type: {metal_type}")
            result = enhance_wedding_ring_details(image, metal_type, lighting, is_ring_region=True)
        
        # Create thumbnail
        thumbnail = create_thumbnail(result)
        
        # Encode results (without padding for Make.com)
        enhanced_base64 = encode_image_to_base64(result)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        # Return with correct structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "masking_detected": masking_info is not None,
                    "masking_type": masking_info['type'] if masking_info else None,
                    "version": "v122"
                }
            }
        }
        
    except Exception as e:
        log_debug(f"Error in handler: {str(e)}")
        import traceback
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
