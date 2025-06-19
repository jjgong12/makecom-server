import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_black_borders_extreme(img_array):
    """Extreme black border detection - based on project knowledge"""
    h, w = img_array.shape[:2]
    logger.info(f"Extreme border detection - Image size: {w}x{h}")
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # Project knowledge: threshold 100-120, scan 50% of image
    threshold = 120  # More aggressive than 190
    max_scan = int(min(h, w) * 0.5)  # 50% of image
    
    # Top border - check ENTIRE line, not just center
    for i in range(min(max_scan, h)):
        if np.mean(gray[i, :]) < threshold:  # Full line check
            borders['top'] = i + 1
        else:
            break
    
    # Bottom border
    for i in range(min(max_scan, h)):
        if np.mean(gray[h-1-i, :]) < threshold:
            borders['bottom'] = i + 1
        else:
            break
    
    # Left border
    for i in range(min(max_scan, w)):
        if np.mean(gray[:, i]) < threshold:
            borders['left'] = i + 1
        else:
            break
    
    # Right border
    for i in range(min(max_scan, w)):
        if np.mean(gray[:, w-1-i]) < threshold:
            borders['right'] = i + 1
        else:
            break
    
    # Extra safety margin
    safety_margin = 50
    borders = {k: v + safety_margin if v > 0 else 0 for k, v in borders.items()}
    
    logger.info(f"Detected borders with margin: {borders}")
    return borders

def create_thumbnail_ultra_zoom(img_array, size=800):
    """Create thumbnail with ultra zoom - absolutely no padding"""
    h, w = img_array.shape[:2]
    
    # Convert to grayscale for ring detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Find non-white pixels (ring area)
    # Lower threshold to catch more of the ring
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w_c)
            y_max = max(y_max, y + h_c)
        
        # Add minimal padding (2%)
        padding = 0.02
        w_pad = int((x_max - x_min) * padding)
        h_pad = int((y_max - y_min) * padding)
        
        x_min = max(0, x_min - w_pad)
        y_min = max(0, y_min - h_pad)
        x_max = min(w, x_max + w_pad)
        y_max = min(h, y_max + h_pad)
        
        # Crop to ring area
        cropped = img_array[y_min:y_max, x_min:x_max]
    else:
        # Fallback to center crop
        crop_size = int(min(w, h) * 0.98)
        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        cropped = img_array[y1:y1+crop_size, x1:x1+crop_size]
    
    # Resize to target size
    thumbnail = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def enhance_wedding_ring_premium(img_array, metal_type='white_gold'):
    """Premium wedding ring enhancement with extreme border removal"""
    # First remove borders with extreme settings
    borders = find_black_borders_extreme(img_array)
    
    if any(borders.values()):
        h, w = img_array.shape[:2]
        img_array = img_array[
            borders['top']:h-borders['bottom'],
            borders['left']:w-borders['right']
        ]
        logger.info(f"Borders removed - New size: {img_array.shape[1]}x{img_array.shape[0]}")
    
    # Convert to PIL for enhancement
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    # Base adjustments
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.15)
    
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.12)
    
    # Metal-specific enhancements
    if metal_type == 'yellow_gold':
        color = ImageEnhance.Color(img)
        img = color.enhance(1.25)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0, 
                                   np.full_like(img_array, [0, 10, 25]), 0.08, 0)
    
    elif metal_type == 'rose_gold':
        color = ImageEnhance.Color(img)
        img = color.enhance(1.18)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0,
                                   np.full_like(img_array, [0, 5, 20]), 0.12, 0)
    
    elif metal_type == 'champagne_gold':
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.30)
        color = ImageEnhance.Color(img)
        img = color.enhance(0.9)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0,
                                   np.full_like(img_array, [245, 240, 235]), 0.15, 0)
        img_array[:,:,0] = np.clip(img_array[:,:,0] + 6, 0, 255)
    
    else:  # white_gold
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.20)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0,
                                   np.full_like(img_array, [240, 245, 250]), 0.10, 0)
    
    # Professional sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]]) / 1.5
    img_array = cv2.filter2D(img_array, -1, kernel)
    
    # Final adjustments
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.add(lab[:,:,0], 3)
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img_array

def detect_metal_type(img_array):
    """Detect metal type from image colors"""
    h, w = img_array.shape[:2]
    center = img_array[h//3:2*h//3, w//3:2*w//3]
    
    if len(center.shape) == 3 and center.shape[2] == 3:
        center_rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
    else:
        center_rgb = center
    
    avg_color = np.mean(center_rgb, axis=(0,1))
    r, g, b = avg_color
    
    warmth = (r - b) / 255.0
    saturation = np.std(avg_color) / np.mean(avg_color)
    brightness = np.mean(avg_color) / 255.0
    
    logger.info(f"Color analysis - R:{r:.1f} G:{g:.1f} B:{b:.1f}, Warmth:{warmth:.2f}")
    
    if brightness > 0.80 and 0.02 < warmth < 0.06:
        return 'champagne_gold'
    elif warmth > 0.08 and r > g > b and saturation > 0.05:
        return 'rose_gold'
    elif warmth > 0.05 and saturation > 0.03:
        return 'yellow_gold'
    else:
        return 'white_gold'

def handler(event):
    """RunPod handler function"""
    try:
        # Get base64 image from request
        image_base64 = event.get('input', {}).get('image', '')
        if not image_base64:
            raise ValueError("No image data provided")
        
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_array is None:
            raise ValueError("Failed to decode image")
        
        logger.info(f"Processing image: {img_array.shape[1]}x{img_array.shape[0]}")
        
        # Detect metal type
        metal_type = detect_metal_type(img_array)
        logger.info(f"Detected metal type: {metal_type}")
        
        # Get lighting preference
        lighting = event.get('input', {}).get('lighting', 'professional')
        
        # Apply wedding ring enhancement with extreme border removal
        enhanced = enhance_wedding_ring_premium(img_array, metal_type)
        
        # Create thumbnail with ultra zoom
        thumbnail = create_thumbnail_ultra_zoom(enhanced)
        
        # Convert to base64 - using high quality for main image
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        enhanced_buffer = BytesIO()
        enhanced_pil.save(enhanced_buffer, format='JPEG', quality=98, optimize=True)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail with slightly lower quality to reduce size
        thumbnail_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        thumbnail_buffer = BytesIO()
        thumbnail_pil.save(thumbnail_buffer, format='JPEG', quality=95, optimize=True)
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')
        
        # Log sizes for debugging
        logger.info(f"Enhanced size: {len(enhanced_base64)} chars")
        logger.info(f"Thumbnail size: {len(thumbnail_base64)} chars")
        
        # Return with correct structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
                    "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                    "status": "success",
                    "version": "v65.0"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v65.0"
            }
        }

# RunPod endpoint
runpod.serverless.start({"handler": handler})
