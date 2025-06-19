import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import time

# v13.3 Complete Parameters from 28 pairs of learning data
COMPLETE_PARAMETERS = {
    'white_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.10, 'exposure': 0.05,
            'highlights': -0.10, 'shadows': 0.15, 'vibrance': 1.20,
            'saturation': 1.08, 'clarity': 15, 'color_temp': -5,
            'white_overlay': 0.06
        },
        'bright': {
            'brightness': 1.10, 'contrast': 1.08, 'exposure': 0.02,
            'highlights': -0.15, 'shadows': 0.10, 'vibrance': 1.15,
            'saturation': 1.06, 'clarity': 12, 'color_temp': -4,
            'white_overlay': 0.05
        },
        'shadow': {
            'brightness': 1.25, 'contrast': 1.12, 'exposure': 0.08,
            'highlights': -0.05, 'shadows': 0.25, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 18, 'color_temp': -6,
            'white_overlay': 0.08
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.12, 'contrast': 1.15, 'exposure': 0.03,
            'highlights': -0.08, 'shadows': 0.12, 'vibrance': 1.30,
            'saturation': 1.12, 'clarity': 12, 'color_temp': 8,
            'white_overlay': 0.03
        },
        'bright': {
            'brightness': 1.08, 'contrast': 1.12, 'exposure': 0.00,
            'highlights': -0.12, 'shadows': 0.08, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 10, 'color_temp': 6,
            'white_overlay': 0.02
        },
        'shadow': {
            'brightness': 1.20, 'contrast': 1.18, 'exposure': 0.06,
            'highlights': -0.03, 'shadows': 0.20, 'vibrance': 1.35,
            'saturation': 1.15, 'clarity': 15, 'color_temp': 10,
            'white_overlay': 0.04
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.14, 'contrast': 1.12, 'exposure': 0.04,
            'highlights': -0.07, 'shadows': 0.14, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 13, 'color_temp': 3,
            'white_overlay': 0.05
        },
        'bright': {
            'brightness': 1.10, 'contrast': 1.10, 'exposure': 0.01,
            'highlights': -0.10, 'shadows': 0.10, 'vibrance': 1.20,
            'saturation': 1.08, 'clarity': 11, 'color_temp': 2,
            'white_overlay': 0.04
        },
        'shadow': {
            'brightness': 1.22, 'contrast': 1.15, 'exposure': 0.07,
            'highlights': -0.04, 'shadows': 0.22, 'vibrance': 1.30,
            'saturation': 1.12, 'clarity': 16, 'color_temp': 4,
            'white_overlay': 0.06
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.30, 'contrast': 1.08, 'exposure': 0.06,
            'highlights': -0.12, 'shadows': 0.18, 'vibrance': 1.15,
            'saturation': 0.90, 'clarity': 14, 'color_temp': -6,
            'white_overlay': 0.15
        },
        'bright': {
            'brightness': 1.28, 'contrast': 1.06, 'exposure': 0.03,
            'highlights': -0.18, 'shadows': 0.12, 'vibrance': 1.10,
            'saturation': 0.88, 'clarity': 11, 'color_temp': -7,
            'white_overlay': 0.18
        },
        'shadow': {
            'brightness': 1.35, 'contrast': 1.10, 'exposure': 0.10,
            'highlights': -0.06, 'shadows': 0.28, 'vibrance': 1.20,
            'saturation': 0.85, 'clarity': 17, 'color_temp': -8,
            'white_overlay': 0.20
        }
    }
}

def detect_metal_type(image):
    """Detect metal type from image"""
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get center region (more reliable)
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        roi_size = min(h, w) // 4
        roi = hsv[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        # Calculate average hue and saturation
        avg_hue = np.mean(roi[:, :, 0])
        avg_sat = np.mean(roi[:, :, 1])
        avg_val = np.mean(roi[:, :, 2])
        
        # Metal detection logic
        if avg_sat < 30:  # Low saturation = white gold
            return 'white_gold'
        elif 15 <= avg_hue <= 25 and avg_sat > 50:  # Orange hue = yellow gold
            return 'yellow_gold'
        elif 5 <= avg_hue <= 15 and avg_sat > 30:  # Red-orange = rose gold
            return 'rose_gold'
        elif avg_hue < 20 and avg_val > 180:  # Bright low hue = champagne
            return 'champagne_gold'
        else:
            return 'white_gold'
    except:
        return 'white_gold'

def detect_lighting(image):
    """Detect lighting condition from image brightness"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 180:
            return 'bright'
        elif avg_brightness < 100:
            return 'shadow'
        else:
            return 'natural'
    except:
        return 'natural'

def remove_black_borders(image):
    """Remove black borders - EXTREME version with direct replacement"""
    h, w = image.shape[:2]
    max_scan = int(min(h, w) * 0.4)  # 40% scan
    
    # Find borders with lower threshold for better detection
    threshold = 15  # Lower threshold for black detection
    
    # Top border - check center 50% for better accuracy
    top = 0
    center_start = w // 4
    center_end = 3 * w // 4
    for i in range(max_scan):
        row_center = image[i, center_start:center_end]
        if np.mean(row_center) > threshold:
            top = i
            break
    
    # Bottom border
    bottom = h
    for i in range(h-1, h-max_scan-1, -1):
        row_center = image[i, center_start:center_end]
        if np.mean(row_center) > threshold:
            bottom = i + 1
            break
    
    # Left border - check center 50%
    left = 0
    center_start = h // 4
    center_end = 3 * h // 4
    for i in range(max_scan):
        col_center = image[center_start:center_end, i]
        if np.mean(col_center) > threshold:
            left = i
            break
    
    # Right border
    right = w
    for i in range(w-1, w-max_scan-1, -1):
        col_center = image[center_start:center_end, i]
        if np.mean(col_center) > threshold:
            right = i + 1
            break
    
    # Get background color from edges (after borders)
    bg_samples = []
    if top > 0:
        bg_samples.append(image[top+5:top+15, center_start:center_end])
    if bottom < h:
        bg_samples.append(image[bottom-15:bottom-5, center_start:center_end])
    if left > 0:
        bg_samples.append(image[center_start:center_end, left+5:left+15])
    if right < w:
        bg_samples.append(image[center_start:center_end, right-15:right-5])
    
    if bg_samples:
        bg_color = np.median([np.median(s, axis=(0,1)) for s in bg_samples], axis=0).astype(np.uint8)
    else:
        bg_color = np.array([248, 248, 248], dtype=np.uint8)  # Default white
    
    # Create clean image with background color
    result = np.full_like(image, bg_color)
    
    # Copy the non-border region
    if top < bottom and left < right:
        result[top:bottom, left:right] = image[top:bottom, left:right]
        removed = True
    else:
        result = image
        removed = False
    
    return result, removed

def enhance_image(image, params):
    """Apply v13.3 enhancement with safe PIL operations"""
    # Convert to PIL
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(image)
    
    # Step 1-2: Brightness and Contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(params['brightness'])
    pil_img = ImageEnhance.Contrast(pil_img).enhance(params['contrast'])
    
    # Step 6-7: Saturation (safe method)
    pil_img = ImageEnhance.Color(pil_img).enhance(params['saturation'])
    
    # Convert to numpy for advanced processing
    img_array = np.array(pil_img).astype(np.float32)
    
    # Step 3: Exposure
    img_array = np.clip(img_array * (1 + params['exposure']), 0, 255)
    
    # Step 8: Simple temperature adjustment
    if params['color_temp'] > 0:  # Warmer
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)  # More red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.95, 0, 255)  # Less blue
    elif params['color_temp'] < 0:  # Cooler (for champagne gold)
        cool_factor = 1 + (abs(params['color_temp']) * 0.02)  # Stronger cooling
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] / cool_factor, 0, 255)  # Much less red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * cool_factor, 0, 255)  # Much more blue
    
    # Step 9: White overlay - STRONGER for champagne gold
    if params['white_overlay'] > 0:
        white = np.ones_like(img_array) * 255
        # Apply stronger overlay for champagne gold
        overlay_strength = params['white_overlay'] * 1.5 if 'champagne' in str(params) else params['white_overlay']
        img_array = (1 - overlay_strength) * img_array + overlay_strength * white
    
    # Convert back to uint8
    result = np.clip(img_array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def create_thumbnail(image):
    """Create 1000x1300 thumbnail with ring centered and enlarged"""
    h, w = image.shape[:2]
    target_w, target_h = 1000, 1300
    
    # Find the ring area (bright region in center)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_y, center_x = h // 2, w // 2
    
    # Use binary threshold to find ring
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, ring_w, ring_h = cv2.boundingRect(largest_contour)
        
        # Add padding around the ring
        pad = int(max(ring_w, ring_h) * 0.3)
        x = max(0, x - pad)
        y = max(0, y - pad)
        ring_w = min(w - x, ring_w + 2 * pad)
        ring_h = min(h - y, ring_h + 2 * pad)
        
        # Crop to ring area
        cropped = image[y:y+ring_h, x:x+ring_w]
    else:
        # Fallback: use center crop
        crop_size = min(h, w) // 2
        start_y = max(0, center_y - crop_size)
        start_x = max(0, center_x - crop_size)
        cropped = image[start_y:start_y+crop_size*2, start_x:start_x+crop_size*2]
    
    # Scale to fit thumbnail with ring taking 95% of space
    crop_h, crop_w = cropped.shape[:2]
    scale = min(target_w * 0.95 / crop_w, target_h * 0.95 / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    thumb = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
    
    # Center the ring
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    thumb[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumb

def handler(job):
    """Wedding Ring AI v58 Handler - Extreme White Champagne Gold + Better Borders"""
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get("image") or job_input.get("image_base64")
        
        if not image_data:
            return {"output": {"error": "No image provided", "status": "error", "version": "v58"}}
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            return {"output": {"error": f"Image decode error: {str(e)}", "status": "error", "version": "v58"}}
        
        # Step 1: Remove black borders
        image, border_removed = remove_black_borders(image)
        
        # Step 2: Detect metal and lighting
        metal_type = job_input.get("metal_type", "auto")
        lighting = job_input.get("lighting", "auto")
        
        if metal_type == "auto":
            metal_type = detect_metal_type(image)
        
        if lighting == "auto":
            lighting = detect_lighting(image)
        
        # Step 3: Get parameters and enhance
        params = COMPLETE_PARAMETERS.get(metal_type, {}).get(lighting, COMPLETE_PARAMETERS['white_gold']['natural'])
        enhanced = enhance_image(image, params)
        
        # Step 4: Create thumbnail
        thumbnail = create_thumbnail(enhanced)
        
        # Step 5: Convert to base64
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        # CRITICAL: Return with proper output structure
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_removed": border_removed,
                    "processing_time": time.time() - start_time,
                    "version": "v58",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "error",
                "version": "v58"
            }
        }

# RunPod serverless start
runpod.serverless.start({"handler": handler})
