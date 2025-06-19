import cv2
import numpy as np
import runpod
import base64
import io
from PIL import Image, ImageEnhance, ImageOps
import time

# Complete v13.3 parameters (28 pairs)
COMPLETE_PARAMETERS = {
    'yellow_gold': {
        'bright': {
            'h_shift': 8, 's_mult': 1.25, 'v_mult': 1.15,
            'contrast': 1.25, 'brightness': 15, 'shadows': 25,
            'highlights': 15, 'vibrance': 1.2, 'clarity': 45,
            'exposure': 0.3, 'gamma': 1.15, 'temp_adjust': -0.05,
            'denoise': 15, 'sharpen': 65, 'halo_removal': True,
            'edge_preserve': 0.8, 'highlight_protection': 0.9,
            'texture_enhance': 1.2, 'micro_contrast': 35,
            'tint_correction': {'r': 1.02, 'g': 1.0, 'b': 0.97},
            'local_contrast': 25, 'color_boost': {'yellow': 1.15, 'orange': 1.1},
            'surface_smooth': 20, 'specular_enhance': 1.3,
            'ambient_lift': 15, 'metal_shine': 1.25,
            'detail_boost': 40, 'color_temp': 5800
        },
        'mixed': {
            'h_shift': 5, 's_mult': 1.15, 'v_mult': 1.1,
            'contrast': 1.2, 'brightness': 10, 'shadows': 20,
            'highlights': 10, 'vibrance': 1.15, 'clarity': 40,
            'exposure': 0.2, 'gamma': 1.1, 'temp_adjust': -0.03,
            'denoise': 20, 'sharpen': 60, 'halo_removal': True,
            'edge_preserve': 0.85, 'highlight_protection': 0.85,
            'texture_enhance': 1.15, 'micro_contrast': 30,
            'tint_correction': {'r': 1.01, 'g': 1.0, 'b': 0.98},
            'local_contrast': 20, 'color_boost': {'yellow': 1.1, 'orange': 1.05},
            'surface_smooth': 25, 'specular_enhance': 1.2,
            'ambient_lift': 10, 'metal_shine': 1.2,
            'detail_boost': 35, 'color_temp': 5600
        },
        'ambient': {
            'h_shift': 3, 's_mult': 1.08, 'v_mult': 1.05,
            'contrast': 1.15, 'brightness': 5, 'shadows': 15,
            'highlights': 5, 'vibrance': 1.1, 'clarity': 35,
            'exposure': 0.1, 'gamma': 1.05, 'temp_adjust': 0,
            'denoise': 25, 'sharpen': 55, 'halo_removal': True,
            'edge_preserve': 0.9, 'highlight_protection': 0.8,
            'texture_enhance': 1.1, 'micro_contrast': 25,
            'tint_correction': {'r': 1.0, 'g': 1.0, 'b': 0.99},
            'local_contrast': 15, 'color_boost': {'yellow': 1.05, 'orange': 1.0},
            'surface_smooth': 30, 'specular_enhance': 1.15,
            'ambient_lift': 5, 'metal_shine': 1.15,
            'detail_boost': 30, 'color_temp': 5400
        }
    },
    'rose_gold': {
        'bright': {
            'h_shift': -3, 's_mult': 1.2, 'v_mult': 1.15,
            'contrast': 1.25, 'brightness': 15, 'shadows': 25,
            'highlights': 15, 'vibrance': 1.25, 'clarity': 45,
            'exposure': 0.3, 'gamma': 1.15, 'temp_adjust': 0.05,
            'denoise': 15, 'sharpen': 65, 'halo_removal': True,
            'edge_preserve': 0.8, 'highlight_protection': 0.9,
            'texture_enhance': 1.2, 'micro_contrast': 35,
            'tint_correction': {'r': 1.03, 'g': 0.99, 'b': 0.97},
            'local_contrast': 25, 'color_boost': {'red': 1.1, 'pink': 1.15},
            'surface_smooth': 20, 'specular_enhance': 1.3,
            'ambient_lift': 15, 'metal_shine': 1.25,
            'detail_boost': 40, 'color_temp': 5900
        },
        'mixed': {
            'h_shift': -2, 's_mult': 1.12, 'v_mult': 1.1,
            'contrast': 1.2, 'brightness': 10, 'shadows': 20,
            'highlights': 10, 'vibrance': 1.18, 'clarity': 40,
            'exposure': 0.2, 'gamma': 1.1, 'temp_adjust': 0.03,
            'denoise': 20, 'sharpen': 60, 'halo_removal': True,
            'edge_preserve': 0.85, 'highlight_protection': 0.85,
            'texture_enhance': 1.15, 'micro_contrast': 30,
            'tint_correction': {'r': 1.02, 'g': 0.995, 'b': 0.98},
            'local_contrast': 20, 'color_boost': {'red': 1.05, 'pink': 1.1},
            'surface_smooth': 25, 'specular_enhance': 1.2,
            'ambient_lift': 10, 'metal_shine': 1.2,
            'detail_boost': 35, 'color_temp': 5700
        },
        'ambient': {
            'h_shift': -1, 's_mult': 1.05, 'v_mult': 1.05,
            'contrast': 1.15, 'brightness': 5, 'shadows': 15,
            'highlights': 5, 'vibrance': 1.12, 'clarity': 35,
            'exposure': 0.1, 'gamma': 1.05, 'temp_adjust': 0.02,
            'denoise': 25, 'sharpen': 55, 'halo_removal': True,
            'edge_preserve': 0.9, 'highlight_protection': 0.8,
            'texture_enhance': 1.1, 'micro_contrast': 25,
            'tint_correction': {'r': 1.01, 'g': 1.0, 'b': 0.99},
            'local_contrast': 15, 'color_boost': {'red': 1.0, 'pink': 1.05},
            'surface_smooth': 30, 'specular_enhance': 1.15,
            'ambient_lift': 5, 'metal_shine': 1.15,
            'detail_boost': 30, 'color_temp': 5500
        }
    },
    'white_gold': {
        'bright': {
            'h_shift': 0, 's_mult': 0.95, 'v_mult': 1.2,
            'contrast': 1.3, 'brightness': 20, 'shadows': 30,
            'highlights': 20, 'vibrance': 0.9, 'clarity': 50,
            'exposure': 0.4, 'gamma': 1.2, 'temp_adjust': -0.08,
            'denoise': 10, 'sharpen': 70, 'halo_removal': True,
            'edge_preserve': 0.75, 'highlight_protection': 0.95,
            'texture_enhance': 1.25, 'micro_contrast': 40,
            'tint_correction': {'r': 0.98, 'g': 1.0, 'b': 1.02},
            'local_contrast': 30, 'color_boost': {'blue': 1.05, 'silver': 1.1},
            'surface_smooth': 15, 'specular_enhance': 1.4,
            'ambient_lift': 20, 'metal_shine': 1.3,
            'detail_boost': 45, 'color_temp': 6500
        },
        'mixed': {
            'h_shift': 0, 's_mult': 0.92, 'v_mult': 1.15,
            'contrast': 1.25, 'brightness': 15, 'shadows': 25,
            'highlights': 15, 'vibrance': 0.85, 'clarity': 45,
            'exposure': 0.3, 'gamma': 1.15, 'temp_adjust': -0.05,
            'denoise': 15, 'sharpen': 65, 'halo_removal': True,
            'edge_preserve': 0.8, 'highlight_protection': 0.9,
            'texture_enhance': 1.2, 'micro_contrast': 35,
            'tint_correction': {'r': 0.99, 'g': 1.0, 'b': 1.01},
            'local_contrast': 25, 'color_boost': {'blue': 1.0, 'silver': 1.05},
            'surface_smooth': 20, 'specular_enhance': 1.3,
            'ambient_lift': 15, 'metal_shine': 1.25,
            'detail_boost': 40, 'color_temp': 6200
        },
        'ambient': {
            'h_shift': 0, 's_mult': 0.9, 'v_mult': 1.1,
            'contrast': 1.2, 'brightness': 10, 'shadows': 20,
            'highlights': 10, 'vibrance': 0.8, 'clarity': 40,
            'exposure': 0.2, 'gamma': 1.1, 'temp_adjust': -0.02,
            'denoise': 20, 'sharpen': 60, 'halo_removal': True,
            'edge_preserve': 0.85, 'highlight_protection': 0.85,
            'texture_enhance': 1.15, 'micro_contrast': 30,
            'tint_correction': {'r': 1.0, 'g': 1.0, 'b': 1.0},
            'local_contrast': 20, 'color_boost': {'blue': 0.95, 'silver': 1.0},
            'surface_smooth': 25, 'specular_enhance': 1.2,
            'ambient_lift': 10, 'metal_shine': 1.2,
            'detail_boost': 35, 'color_temp': 6000
        }
    },
    'platinum': {
        'bright': {
            'h_shift': 0, 's_mult': 0.85, 'v_mult': 1.25,
            'contrast': 1.35, 'brightness': 25, 'shadows': 35,
            'highlights': 25, 'vibrance': 0.8, 'clarity': 55,
            'exposure': 0.5, 'gamma': 1.25, 'temp_adjust': -0.1,
            'denoise': 8, 'sharpen': 75, 'halo_removal': True,
            'edge_preserve': 0.7, 'highlight_protection': 1.0,
            'texture_enhance': 1.3, 'micro_contrast': 45,
            'tint_correction': {'r': 0.97, 'g': 1.0, 'b': 1.03},
            'local_contrast': 35, 'color_boost': {'blue': 1.1, 'silver': 1.15},
            'surface_smooth': 10, 'specular_enhance': 1.5,
            'ambient_lift': 25, 'metal_shine': 1.35,
            'detail_boost': 50, 'color_temp': 6800
        },
        'mixed': {
            'h_shift': 0, 's_mult': 0.88, 'v_mult': 1.2,
            'contrast': 1.3, 'brightness': 20, 'shadows': 30,
            'highlights': 20, 'vibrance': 0.82, 'clarity': 50,
            'exposure': 0.4, 'gamma': 1.2, 'temp_adjust': -0.07,
            'denoise': 12, 'sharpen': 70, 'halo_removal': True,
            'edge_preserve': 0.75, 'highlight_protection': 0.95,
            'texture_enhance': 1.25, 'micro_contrast': 40,
            'tint_correction': {'r': 0.98, 'g': 1.0, 'b': 1.02},
            'local_contrast': 30, 'color_boost': {'blue': 1.05, 'silver': 1.1},
            'surface_smooth': 15, 'specular_enhance': 1.4,
            'ambient_lift': 20, 'metal_shine': 1.3,
            'detail_boost': 45, 'color_temp': 6500
        },
        'ambient': {
            'h_shift': 0, 's_mult': 0.87, 'v_mult': 1.15,
            'contrast': 1.25, 'brightness': 15, 'shadows': 25,
            'highlights': 15, 'vibrance': 0.78, 'clarity': 45,
            'exposure': 0.3, 'gamma': 1.15, 'temp_adjust': -0.03,
            'denoise': 18, 'sharpen': 65, 'halo_removal': True,
            'edge_preserve': 0.8, 'highlight_protection': 0.9,
            'texture_enhance': 1.2, 'micro_contrast': 35,
            'tint_correction': {'r': 0.99, 'g': 1.0, 'b': 1.01},
            'local_contrast': 25, 'color_boost': {'blue': 1.0, 'silver': 1.05},
            'surface_smooth': 20, 'specular_enhance': 1.3,
            'ambient_lift': 15, 'metal_shine': 1.25,
            'detail_boost': 40, 'color_temp': 6300
        }
    }
}

def detect_metal_type(image):
    """Detect metal type based on color analysis"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get center region
    h, w = image.shape[:2]
    center_y, center_x = h//2, w//2
    sample_size = min(h, w) // 4
    center_region = hsv[center_y-sample_size:center_y+sample_size,
                       center_x-sample_size:center_x+sample_size]
    
    # Calculate color statistics
    avg_hue = np.mean(center_region[:,:,0])
    avg_sat = np.mean(center_region[:,:,1])
    avg_val = np.mean(center_region[:,:,2])
    
    # Metal type detection logic
    if avg_sat < 30 and avg_val > 180:
        return 'white_gold'
    elif avg_sat < 25 and avg_val > 200:
        return 'platinum'
    elif 5 <= avg_hue <= 25 and avg_sat > 40:
        return 'rose_gold'
    elif 15 <= avg_hue <= 35 and avg_sat > 30:
        return 'yellow_gold'
    else:
        # Default based on value
        if avg_val > 200:
            return 'platinum'
        elif avg_val > 180:
            return 'white_gold'
        else:
            return 'yellow_gold'

def detect_lighting(image):
    """Detect lighting condition"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Analyze distribution
    lower_quarter = np.sum(hist[:64])
    upper_quarter = np.sum(hist[192:])
    
    # Lighting detection logic
    if avg_brightness > 180 and upper_quarter > 0.3:
        return 'bright'
    elif avg_brightness < 100 or lower_quarter > 0.3:
        return 'ambient'
    else:
        return 'mixed'

def remove_black_borders(image):
    """Remove black borders if present"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find non-black pixels
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        for cnt in contours[1:]:
            x2, y2, w2, h2 = cv2.boundingRect(cnt)
            x = min(x, x2)
            y = min(y, y2)
            w = max(x + w, x2 + w2) - x
            h = max(y + h, y2 + h2) - y
        
        # Check if cropping is needed
        if x > 5 or y > 5 or w < image.shape[1] - 10 or h < image.shape[0] - 10:
            return image[y:y+h, x:x+w], True
    
    return image, False

def apply_shadow_highlight(img, shadows=0, highlights=0):
    """Apply shadow/highlight adjustments"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply adjustments
    if shadows != 0:
        shadow_mask = (l < 100).astype(np.float32)
        l = l + (shadows * shadow_mask * (100 - l) / 100).astype(np.uint8)
    
    if highlights != 0:
        highlight_mask = (l > 150).astype(np.float32)
        l = l + (highlights * highlight_mask * (255 - l) / 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def adjust_vibrance(img, vibrance):
    """Adjust vibrance (saturation of less saturated colors)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Calculate saturation mask
    sat_mask = 1.0 - (hsv[:,:,1] / 255.0)
    
    # Apply vibrance
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + (vibrance - 1) * sat_mask), 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_clarity(img, clarity):
    """Apply clarity adjustment (mid-tone contrast)"""
    if clarity == 0:
        return img
    
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    
    # High pass filter
    high_pass = cv2.subtract(img, blurred)
    
    # Apply only to mid-tones
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid_mask = np.exp(-((gray.astype(np.float32) - 128) ** 2) / (2 * 50 ** 2))
    mid_mask = cv2.cvtColor(mid_mask, cv2.COLOR_GRAY2BGR)
    
    # Blend
    result = img + (high_pass * mid_mask * clarity / 100).astype(np.uint8)
    return np.clip(result, 0, 255).astype(np.uint8)

def adjust_color_temperature(img, temp_adjust):
    """Adjust color temperature"""
    if temp_adjust == 0:
        return img
    
    result = img.copy().astype(np.float32)
    
    if temp_adjust > 0:  # Warmer
        result[:,:,2] *= (1 - temp_adjust * 0.1)  # Reduce blue
        result[:,:,0] *= (1 + temp_adjust * 0.05)  # Increase red
    else:  # Cooler
        result[:,:,0] *= (1 + temp_adjust * 0.1)  # Reduce red
        result[:,:,2] *= (1 - temp_adjust * 0.05)  # Increase blue
    
    return np.clip(result, 0, 255).astype(np.uint8)

def process_image(img_cv, metal_type=None, lighting=None):
    """Process image with v13.3 parameters"""
    # Step 1: Remove black borders
    img_cv, border_removed = remove_black_borders(img_cv)
    
    # Step 2: Auto-detect metal and lighting if not provided
    if not metal_type:
        metal_type = detect_metal_type(img_cv)
    if not lighting:
        lighting = detect_lighting(img_cv)
    
    # Step 3: Get parameters
    params = COMPLETE_PARAMETERS[metal_type][lighting]
    
    # Step 4: Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Step 5: Apply basic adjustments
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(params['contrast'])
    
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(1 + params['brightness'] / 100)
    
    # Step 6: Convert back to CV2 for advanced processing
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Step 7: Apply shadow/highlight
    img_cv = apply_shadow_highlight(img_cv, params['shadows'], params['highlights'])
    
    # Step 8: Apply vibrance and clarity
    img_cv = adjust_vibrance(img_cv, params['vibrance'])
    img_cv = apply_clarity(img_cv, params['clarity'])
    
    # Step 9: Back to PIL for final adjustments
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Step 10: Color temperature
    img_pil = adjust_color_temperature(img_pil, params['color_temp'])
    
    # Generate thumbnail
    thumbnail = create_thumbnail(img_pil)
    
    return img_pil, thumbnail, metal_type, lighting, border_removed

def create_thumbnail(img_pil):
    """Create 1000x1300 thumbnail with white background"""
    # Target size
    target_w, target_h = 1000, 1300
    
    # Calculate scaling to fit 90% of frame
    img_w, img_h = img_pil.size
    scale = min(target_w * 0.9 / img_w, target_h * 0.9 / img_h)
    
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # Resize image
    resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background
    thumbnail = Image.new('RGB', (target_w, target_h), 'white')
    
    # Paste centered
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    thumbnail.paste(resized, (x, y))
    
    return thumbnail

def handler(event):
    """RunPod handler function - v26 with field name fix"""
    try:
        start_time = time.time()
        
        # Get input - check both 'image' and 'image_base64'
        input_data = event.get('input', {})
        image_data = input_data.get('image') or input_data.get('image_base64')
        
        if not image_data:
            return {
                "output": {
                    "error": "No image provided. Expected 'image' or 'image_base64' field",
                    "status": "error"
                }
            }
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return {
                "output": {
                    "error": "Failed to decode image",
                    "status": "error"
                }
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
                    "version": "v26"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v26"
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
