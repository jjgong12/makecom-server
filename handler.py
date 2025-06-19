import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io
import base64
import time

# v13.3 complete parameters from 28 pairs of training data
METAL_PARAMS = {
    "rosegold": {
        "soft": {
            "bilateral_d": 15, "bilateral_sigma": 80,
            "brightness": 1.25, "contrast": 1.15, "saturation": 1.05,
            "highlights": 1.20, "shadows": 0.85, "whites": 1.10, "blacks": 0.05,
            "white_overlay": 0.10, "color_temp": 0
        },
        "bright": {
            "bilateral_d": 18, "bilateral_sigma": 90,
            "brightness": 1.28, "contrast": 1.20, "saturation": 1.08,
            "highlights": 1.25, "shadows": 0.82, "whites": 1.15, "blacks": 0.03,
            "white_overlay": 0.12, "color_temp": 0
        },
        "mixed": {
            "bilateral_d": 16, "bilateral_sigma": 85,
            "brightness": 1.26, "contrast": 1.18, "saturation": 1.06,
            "highlights": 1.22, "shadows": 0.84, "whites": 1.12, "blacks": 0.04,
            "white_overlay": 0.11, "color_temp": 0
        }
    },
    "champagnegold": {
        "soft": {
            "bilateral_d": 20, "bilateral_sigma": 100,
            "brightness": 1.30, "contrast": 1.20, "saturation": 1.10,
            "highlights": 1.25, "shadows": 0.80, "whites": 1.20, "blacks": 0.02,
            "white_overlay": 0.15, "color_temp": -6
        },
        "bright": {
            "bilateral_d": 20, "bilateral_sigma": 100,
            "brightness": 1.30, "contrast": 1.25, "saturation": 1.12,
            "highlights": 1.30, "shadows": 0.78, "whites": 1.25, "blacks": 0.00,
            "white_overlay": 0.18, "color_temp": -6
        },
        "mixed": {
            "bilateral_d": 20, "bilateral_sigma": 100,
            "brightness": 1.30, "contrast": 1.22, "saturation": 1.11,
            "highlights": 1.28, "shadows": 0.79, "whites": 1.22, "blacks": 0.01,
            "white_overlay": 0.16, "color_temp": -6
        }
    },
    "yellowgold": {
        "soft": {
            "bilateral_d": 12, "bilateral_sigma": 70,
            "brightness": 1.20, "contrast": 1.10, "saturation": 1.00,
            "highlights": 1.15, "shadows": 0.90, "whites": 1.05, "blacks": 0.08,
            "white_overlay": 0.08, "color_temp": 2
        },
        "bright": {
            "bilateral_d": 15, "bilateral_sigma": 80,
            "brightness": 1.22, "contrast": 1.15, "saturation": 1.02,
            "highlights": 1.20, "shadows": 0.88, "whites": 1.10, "blacks": 0.06,
            "white_overlay": 0.10, "color_temp": 2
        },
        "mixed": {
            "bilateral_d": 13, "bilateral_sigma": 75,
            "brightness": 1.21, "contrast": 1.12, "saturation": 1.01,
            "highlights": 1.18, "shadows": 0.89, "whites": 1.08, "blacks": 0.07,
            "white_overlay": 0.09, "color_temp": 2
        }
    },
    "whitegold": {
        "soft": {
            "bilateral_d": 25, "bilateral_sigma": 120,
            "brightness": 1.35, "contrast": 1.25, "saturation": 0.95,
            "highlights": 1.30, "shadows": 0.75, "whites": 1.25, "blacks": 0.00,
            "white_overlay": 0.20, "color_temp": -10
        },
        "bright": {
            "bilateral_d": 28, "bilateral_sigma": 130,
            "brightness": 1.38, "contrast": 1.30, "saturation": 0.92,
            "highlights": 1.35, "shadows": 0.72, "whites": 1.30, "blacks": -0.02,
            "white_overlay": 0.22, "color_temp": -10
        },
        "mixed": {
            "bilateral_d": 26, "bilateral_sigma": 125,
            "brightness": 1.36, "contrast": 1.28, "saturation": 0.94,
            "highlights": 1.32, "shadows": 0.74, "whites": 1.28, "blacks": -0.01,
            "white_overlay": 0.21, "color_temp": -10
        }
    }
}

def find_border_thickness(img_gray, direction='top', max_check=200):
    """Find black border thickness from each direction"""
    h, w = img_gray.shape
    
    if direction == 'top':
        for i in range(min(max_check, h)):
            if np.mean(img_gray[i, :]) > 50:
                return i
    elif direction == 'bottom':
        for i in range(min(max_check, h)):
            if np.mean(img_gray[h-1-i, :]) > 50:
                return i
    elif direction == 'left':
        for i in range(min(max_check, w)):
            if np.mean(img_gray[:, i]) > 50:
                return i
    elif direction == 'right':
        for i in range(min(max_check, w)):
            if np.mean(img_gray[:, w-1-i]) > 50:
                return i
    
    return 0

def remove_black_border_smart(img_array):
    """Smart black border removal with ring position tracking"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Find border thickness from each direction
    top = find_border_thickness(gray, 'top')
    bottom = find_border_thickness(gray, 'bottom')
    left = find_border_thickness(gray, 'left')
    right = find_border_thickness(gray, 'right')
    
    # Add safety margin
    margin = 10
    top = max(0, top - margin)
    bottom = max(0, bottom - margin)
    left = max(0, left - margin)
    right = max(0, right - margin)
    
    h, w = img_array.shape[:2]
    
    # Calculate crop area
    crop_h = h - top - bottom
    crop_w = w - left - right
    
    if crop_h > 100 and crop_w > 100:  # Minimum size check
        # Crop image
        cropped = img_array[top:h-bottom, left:w-right]
        
        # Ring position relative to original image
        ring_bbox = {
            'x': left,
            'y': top,
            'width': crop_w,
            'height': crop_h
        }
        
        return cropped, ring_bbox, True
    
    return img_array, None, False

def analyze_image_for_params(img_array):
    """Analyze image to determine metal type and lighting"""
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # Remove background using center crop
    h, w = img_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    crop_size = min(h, w) // 3
    center_region = img_lab[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]
    
    # Analyze color channels
    avg_a = np.mean(center_region[:, :, 1])
    avg_b = np.mean(center_region[:, :, 2])
    avg_l = np.mean(center_region[:, :, 0])
    
    # Determine metal type
    if avg_a > 130 and avg_b > 140:
        metal_type = "rosegold"
    elif avg_b > 135 and avg_a < 130:
        metal_type = "yellowgold"
    elif avg_l > 180 and avg_a < 128 and avg_b < 130:
        metal_type = "whitegold"
    else:
        metal_type = "champagnegold"
    
    # Determine lighting
    bright_pixels = np.sum(l_channel > 200)
    total_pixels = l_channel.size
    bright_ratio = bright_pixels / total_pixels
    
    if bright_ratio > 0.15:
        lighting = "bright"
    elif bright_ratio < 0.05:
        lighting = "soft"
    else:
        lighting = "mixed"
    
    return metal_type, lighting

def adjust_color_temperature(img, temp_adjust):
    """Adjust color temperature"""
    if temp_adjust == 0:
        return img
    
    img_array = np.array(img).astype(np.float32)
    
    if temp_adjust > 0:  # Warmer
        img_array[:, :, 0] *= 1 + (temp_adjust * 0.01)  # Red
        img_array[:, :, 2] *= 1 - (temp_adjust * 0.01)  # Blue
    else:  # Cooler
        img_array[:, :, 0] *= 1 + (temp_adjust * 0.01)  # Red
        img_array[:, :, 2] *= 1 - (temp_adjust * 0.01)  # Blue
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_highlights_shadows(img, highlights, shadows, whites, blacks):
    """Apply highlights, shadows, whites, and blacks adjustments"""
    img_array = np.array(img).astype(np.float32)
    
    # Convert to Lab
    img_lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = img_lab[:, :, 0]
    
    # Apply adjustments
    l_normalized = l_channel / 255.0
    
    # Highlights (affects bright areas)
    highlight_mask = np.power(l_normalized, 0.5)
    l_channel = l_channel * (1 + (highlights - 1) * highlight_mask)
    
    # Shadows (affects dark areas)
    shadow_mask = 1 - np.power(l_normalized, 2)
    l_channel = l_channel * (1 + (shadows - 1) * shadow_mask)
    
    # Whites (affects very bright areas)
    white_mask = np.power(l_normalized, 0.25)
    l_channel = l_channel * (1 + (whites - 1) * white_mask * 0.5)
    
    # Blacks (affects very dark areas)
    black_mask = 1 - np.power(l_normalized, 4)
    l_channel = l_channel + (blacks * 255 * black_mask)
    
    # Apply back
    l_channel = np.clip(l_channel, 0, 255)
    img_lab[:, :, 0] = l_channel
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_rgb)

def apply_white_overlay(img, opacity):
    """Apply white overlay for 'slightly white-coated' look"""
    if opacity <= 0:
        return img
    
    white_layer = Image.new('RGB', img.size, (255, 255, 255))
    return Image.blend(img, white_layer, opacity)

def create_thumbnail_simple(img, ring_bbox=None, target_size=(1000, 1300)):
    """Create thumbnail with ring centered - simple version without rembg"""
    if ring_bbox:
        # Use saved ring position
        x, y, w, h = ring_bbox['x'], ring_bbox['y'], ring_bbox['width'], ring_bbox['height']
        
        # Add 30% margin
        margin_w = int(w * 0.3)
        margin_h = int(h * 0.3)
        
        # Calculate crop area
        crop_x1 = max(0, x - margin_w)
        crop_y1 = max(0, y - margin_h)
        crop_x2 = min(img.width, x + w + margin_w)
        crop_y2 = min(img.height, y + h + margin_h)
        
        # Crop to ring area
        img_cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    else:
        # Fallback: center crop
        w, h = img.size
        if w > h:
            left = (w - h) // 2
            img_cropped = img.crop((left, 0, left + h, h))
        else:
            top = (h - w) // 2
            img_cropped = img.crop((0, top, w, top + w))
    
    # Calculate scaling to fit target size
    crop_w, crop_h = img_cropped.size
    scale = min(target_size[0] / crop_w, target_size[1] / crop_h)
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize
    img_resized = img_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background
    result = Image.new('RGB', target_size, (255, 255, 255))
    
    # Paste centered
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    result.paste(img_resized, (x_offset, y_offset))
    
    return result

def process_image(img_cv):
    """Main processing function with smart border removal"""
    # Remove black border and get ring position
    img_cv, ring_bbox, border_removed = remove_black_border_smart(img_cv)
    
    # Analyze image
    metal_type, lighting = analyze_image_for_params(img_cv)
    params = METAL_PARAMS[metal_type][lighting]
    
    # Step 1: Noise reduction
    img_cv = cv2.bilateralFilter(img_cv, params['bilateral_d'], params['bilateral_sigma'], params['bilateral_sigma'])
    
    # Convert to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Step 2-4: Basic adjustments
    img_pil = ImageEnhance.Brightness(img_pil).enhance(params['brightness'])
    img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])
    img_pil = ImageEnhance.Color(img_pil).enhance(params['saturation'])
    
    # Step 5-8: Advanced adjustments
    img_pil = apply_highlights_shadows(img_pil, params['highlights'], params['shadows'], 
                                      params['whites'], params['blacks'])
    
    # Step 9: White overlay
    img_pil = apply_white_overlay(img_pil, params['white_overlay'])
    
    # Step 10: Color temperature
    img_pil = adjust_color_temperature(img_pil, params['color_temp'])
    
    # Generate thumbnail using saved ring position
    thumbnail = create_thumbnail_simple(img_pil, ring_bbox)
    
    return img_pil, thumbnail, metal_type, lighting, border_removed, ring_bbox

def handler(event):
    """RunPod handler function"""
    try:
        start_time = time.time()
        
        # Get input
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        
        # Handle test requests
        if not image_data:
            return {
                "output": {
                    "status": "ready",
                    "message": "Wedding Ring AI v25 - Ready for processing",
                    "version": "v25"
                }
            }
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "error"
                }
            }
        
        if img_cv is None:
            return {
                "output": {
                    "error": "Failed to decode image - invalid format",
                    "status": "error"
                }
            }
        
        # Process image
        processed_img, thumbnail_img, metal_type, lighting, border_removed, ring_bbox = process_image(img_cv)
        
        # Convert to base64
        # Main image
        main_buffer = io.BytesIO()
        processed_img.save(main_buffer, format='JPEG', quality=95)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = io.BytesIO()
        thumbnail_img.save(thumb_buffer, format='JPEG', quality=90)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Return with correct structure for Make.com
        return {
            "output": {  # This is critical - RunPod wraps this in data.output
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_removed": border_removed,
                    "ring_position": ring_bbox,
                    "processing_time": time.time() - start_time,
                    "status": "success",
                    "version": "v25"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v25"
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
