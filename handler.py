import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adaptive_threshold_scan(gray_img):
    """Multi-stage adaptive threshold scanning for black borders"""
    h, w = gray_img.shape
    borders = {'top': 0, 'bottom': h, 'left': 0, 'right': w}
    
    # Stage 1: Aggressive scan with multiple thresholds (10 to 80)
    thresholds = [10, 20, 30, 40, 50, 60, 70, 80]
    scan_depth = int(min(h, w) * 0.5)  # 50% scan depth
    
    for threshold in thresholds:
        # Top border
        for i in range(min(scan_depth, h//3)):
            if np.mean(gray_img[i, :]) > threshold:  # Check entire row
                borders['top'] = max(borders['top'], i)
                break
        
        # Bottom border
        for i in range(h-1, max(h-scan_depth, 2*h//3), -1):
            if np.mean(gray_img[i, :]) > threshold:  # Check entire row
                borders['bottom'] = min(borders['bottom'], i+1)
                break
        
        # Left border
        for i in range(min(scan_depth, w//3)):
            if np.mean(gray_img[:, i]) > threshold:  # Check entire column
                borders['left'] = max(borders['left'], i)
                break
        
        # Right border
        for i in range(w-1, max(w-scan_depth, 2*w//3), -1):
            if np.mean(gray_img[:, i]) > threshold:  # Check entire column
                borders['right'] = min(borders['right'], i+1)
                break
    
    return borders

def canny_edge_border_detection(gray_img):
    """Use Canny edge detection to find content boundaries"""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        x_min, y_min = gray_img.shape[1], gray_img.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        return {'top': y_min, 'bottom': y_max, 'left': x_min, 'right': x_max}
    
    return None

def ultra_aggressive_border_removal(image):
    """Ultimate black border removal combining multiple techniques"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive threshold scanning
    borders1 = adaptive_threshold_scan(gray)
    
    # Method 2: Canny edge detection
    borders2 = canny_edge_border_detection(gray)
    
    # Method 3: Ultra scan with threshold 120 (from v23.1 ULTRA)
    borders3 = {'top': 0, 'bottom': h, 'left': 0, 'right': w}
    max_scan = int(min(h, w) * 0.5)
    
    # Top
    for i in range(max_scan):
        if np.mean(gray[i, :]) > 120:
            borders3['top'] = i
            break
    
    # Bottom
    for i in range(h-1, h-max_scan, -1):
        if np.mean(gray[i, :]) > 120:
            borders3['bottom'] = i + 1
            break
    
    # Left
    for i in range(max_scan):
        if np.mean(gray[:, i]) > 120:
            borders3['left'] = i
            break
    
    # Right
    for i in range(w-1, w-max_scan, -1):
        if np.mean(gray[:, i]) > 120:
            borders3['right'] = i + 1
            break
    
    # Combine all methods - take the most aggressive crop
    if borders2:
        final_top = max(borders1['top'], borders2['top'], borders3['top'])
        final_bottom = min(borders1['bottom'], borders2['bottom'], borders3['bottom'])
        final_left = max(borders1['left'], borders2['left'], borders3['left'])
        final_right = min(borders1['right'], borders2['right'], borders3['right'])
    else:
        final_top = max(borders1['top'], borders3['top'])
        final_bottom = min(borders1['bottom'], borders3['bottom'])
        final_left = max(borders1['left'], borders3['left'])
        final_right = min(borders1['right'], borders3['right'])
    
    # Safety margin
    safety_margin = 10
    final_top += safety_margin
    final_bottom -= safety_margin
    final_left += safety_margin
    final_right -= safety_margin
    
    # Ensure valid crop
    final_top = max(0, final_top)
    final_bottom = min(h, final_bottom)
    final_left = max(0, final_left)
    final_right = min(w, final_right)
    
    # Additional edge cleaning (25 pixels if black detected)
    cropped = image[final_top:final_bottom, final_left:final_right]
    
    if cropped.size > 0:
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edge_cut = 25
        
        # Check edges and remove if dark
        if np.mean(gray_cropped[:20, :]) < 100:
            cropped = cropped[edge_cut:, :]
        if np.mean(gray_cropped[-20:, :]) < 100:
            cropped = cropped[:-edge_cut, :]
        if np.mean(gray_cropped[:, :20]) < 100:
            cropped = cropped[:, edge_cut:]
        if np.mean(gray_cropped[:, -20:]) < 100:
            cropped = cropped[:, :-edge_cut]
    
    # If crop would be too small, return original
    if cropped.shape[0] < h * 0.4 or cropped.shape[1] < w * 0.4:
        logger.warning("Crop would remove too much content, applying minimal crop")
        # Try minimal crop
        minimal_crop = 50
        return image[minimal_crop:-minimal_crop, minimal_crop:-minimal_crop]
    
    logger.info(f"Border removal: Original {w}x{h} -> Cropped {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped

def detect_metal_type_accurate(image):
    """Accurately detect metal type with stricter yellow gold criteria"""
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    sample_size = min(h, w) // 4
    
    # Extract center region
    center_region = image[
        max(0, center_y - sample_size):min(h, center_y + sample_size),
        max(0, center_x - sample_size):min(w, center_x + sample_size)
    ]
    
    # Calculate average colors in BGR
    avg_colors = np.mean(center_region.reshape(-1, 3), axis=0)
    b, g, r = avg_colors
    
    # Calculate color characteristics
    brightness = np.mean(avg_colors)
    rg_diff = r - g
    rb_diff = r - b
    gb_diff = g - b
    
    # Stricter detection logic
    if rg_diff > 15 and rb_diff > 20:  # Clear pink/rose tint
        return "rose_gold"
    elif brightness > 200 and abs(rg_diff) < 5 and abs(gb_diff) < 5:  # Very bright and neutral
        return "white_gold"
    elif rg_diff > 8 and gb_diff > 8 and brightness < 180:  # Strict yellow criteria
        return "yellow_gold"
    else:  # Default to unplated white
        return "unplated_white"

def enhance_wedding_ring_v94(image, metal_type):
    """Enhanced wedding ring processing without color distortion"""
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Metal-specific enhancement
    if metal_type == "rose_gold":
        brightness = ImageEnhance.Brightness(pil_image)
        pil_image = brightness.enhance(1.15)
        contrast = ImageEnhance.Contrast(pil_image)
        pil_image = contrast.enhance(1.25)
    elif metal_type == "yellow_gold":
        brightness = ImageEnhance.Brightness(pil_image)
        pil_image = brightness.enhance(1.12)
        contrast = ImageEnhance.Contrast(pil_image)
        pil_image = contrast.enhance(1.2)
    elif metal_type == "white_gold":
        brightness = ImageEnhance.Brightness(pil_image)
        pil_image = brightness.enhance(1.25)
        contrast = ImageEnhance.Contrast(pil_image)
        pil_image = contrast.enhance(1.35)
    else:  # unplated_white
        brightness = ImageEnhance.Brightness(pil_image)
        pil_image = brightness.enhance(1.2)
        contrast = ImageEnhance.Contrast(pil_image)
        pil_image = contrast.enhance(1.3)
    
    # Sharp enhancement for all
    sharpness = ImageEnhance.Sharpness(pil_image)
    pil_image = sharpness.enhance(1.8)
    
    # Convert back to numpy
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return enhanced

def create_perfect_thumbnail(image, target_size=(1000, 1300)):
    """Create thumbnail with ring filling 98% of frame"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fill 98% of frame
    scale = max(target_w / w, target_h / h) * 0.98
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize with high quality
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    thumbnail = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    
    # Calculate centered position
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image
    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumbnail

def handler(event):
    """Main handler function v94"""
    try:
        # Get input
        image_input = event.get("input", {})
        image_base64 = image_input.get("image") or image_input.get("image_base64")
        
        if not image_base64:
            logger.error("No image data provided")
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "failed",
                    "version": "v94-ultimate"
                }
            }
        
        # Handle base64 padding
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_base64 = image_base64.strip()
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += '=' * (4 - missing_padding)
        
        # Decode base64
        try:
            img_data = base64.b64decode(image_base64)
            img_buffer = io.BytesIO(img_data)
            img_buffer.seek(0)  # Critical for BytesIO
            img = Image.open(img_buffer)
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "failed",
                    "version": "v94-ultimate"
                }
            }
        
        logger.info(f"Processing image: {img_array.shape}")
        
        # 1. Ultra aggressive black border removal
        no_border = ultra_aggressive_border_removal(img_array)
        logger.info(f"After border removal: {no_border.shape}")
        
        # 2. Accurate metal type detection
        metal_type = detect_metal_type_accurate(no_border)
        logger.info(f"Detected metal type: {metal_type}")
        
        # 3. Enhanced wedding ring processing
        enhanced = enhance_wedding_ring_v94(no_border, metal_type)
        
        # 4. Perfect thumbnail creation
        thumbnail = create_perfect_thumbnail(enhanced)
        
        # Convert to base64 without padding
        _, buffer = cv2.imencode('.png', enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8').rstrip('=')
        
        _, thumb_buffer = cv2.imencode('.png', thumbnail, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8').rstrip('=')
        
        logger.info("Processing completed successfully")
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "original_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
                    "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                    "thumbnail_size": f"{thumbnail.shape[1]}x{thumbnail.shape[0]}",
                    "status": "success",
                    "version": "v94-ultimate"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v94-ultimate"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
