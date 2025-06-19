import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io
import base64
import time
from rembg import remove

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

def detect_black_border(img_array, threshold=30):
    """Detect black border width (supports up to 200px)"""
    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    max_border = min(200, h//4, w//4)
    
    for border_width in range(1, max_border + 1):
        edges = np.concatenate([
            gray[:border_width, :].flatten(),
            gray[-border_width:, :].flatten(),
            gray[:, :border_width].flatten(),
            gray[:, -border_width:].flatten()
        ])
        
        if np.mean(edges) > threshold:
            return max(0, border_width - 1)
    
    return max_border

def remove_black_border_complete(img_array):
    """Remove black border completely with safety margin"""
    border_width = detect_black_border(img_array)
    
    if border_width > 0:
        h, w = img_array.shape[:2]
        safe_border = min(int(border_width * 1.5 + 20), h//4, w//4)
        
        # Crop with safety margin
        img_cropped = img_array[safe_border:h-safe_border, safe_border:w-safe_border]
        
        # Fill borders with background color (gray)
        result = np.full_like(img_array, [200, 200, 200] if len(img_array.shape) == 3 else 200)
        y_offset = (h - img_cropped.shape[0]) // 2
        x_offset = (w - img_cropped.shape[1]) // 2
        result[y_offset:y_offset+img_cropped.shape[0], x_offset:x_offset+img_cropped.shape[1]] = img_cropped
        
        return result, True
    
    return img_array, False

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

def create_thumbnail(img, target_size=(1000, 1300)):
    """Create thumbnail with ring taking up 80% of frame"""
    # Remove background first
    img_no_bg = remove(img)
    
    # Find ring bounds
    img_array = np.array(img_no_bg)
    alpha = img_array[:, :, 3] if img_array.shape[2] == 4 else None
    
    if alpha is not None:
        coords = np.column_stack(np.where(alpha > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop to ring with padding
            padding = 20
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(img_array.shape[0], y_max + padding)
            x_max = min(img_array.shape[1], x_max + padding)
            
            ring_cropped = img_array[y_min:y_max, x_min:x_max]
            
            # Calculate scaling to make ring 80% of target
            ring_h, ring_w = ring_cropped.shape[:2]
            scale = min(target_size[0] * 0.8 / ring_w, target_size[1] * 0.8 / ring_h)
            
            new_w = int(ring_w * scale)
            new_h = int(ring_h * scale)
            
            # Resize ring
            ring_resized = cv2.resize(ring_cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create white background
            result = np.full((target_size[1], target_size[0], 4), [255, 255, 255, 255], dtype=np.uint8)
            
            # Center the ring
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            
            # Composite
            for c in range(3):
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                    ring_resized[:, :, c] * (ring_resized[:, :, 3] / 255.0) + \
                    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] * (1 - ring_resized[:, :, 3] / 255.0)
            
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3] = \
                np.maximum(result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3], ring_resized[:, :, 3])
            
            return Image.fromarray(result[:, :, :3])
    
    # Fallback
    return img.resize(target_size, Image.Resampling.LANCZOS)

def process_image(img_cv):
    """Main processing function with 10-step enhancement"""
    # Remove black border first
    img_cv, border_removed = remove_black_border_complete(img_cv)
    
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
    
    # Generate thumbnail
    thumbnail = create_thumbnail(img_pil)
    
    return img_pil, thumbnail, metal_type, lighting, border_removed

def handler(event):
    """RunPod handler function"""
    try:
        start_time = time.time()
        
        # Get input
        image_data = event['input'].get('image')
        if not image_data:
            return {
                "error": "No image provided",
                "status": "error"
            }
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return {
                "error": "Failed to decode image",
                "status": "error"
            }
        
        # Process image
        processed_img, thumbnail_img, metal_type, lighting, border_removed = process_image(img_cv)
        
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
                    "processing_time": time.time() - start_time,
                    "status": "success",
                    "version": "v24"
                }
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "version": "v24"
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
