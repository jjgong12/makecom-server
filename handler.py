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

def detect_ring_area(img_array):
    """Detect wedding ring location using multiple methods"""
    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect circles (rings are circular)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=min(w, h) // 2
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        # Expand protection area
        protection_radius = int(r * 1.5)
        return (x - protection_radius, y - protection_radius, 
                x + protection_radius, y + protection_radius)
    
    # Method 2: Find brightest region (rings reflect light)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    
    # Create protection zone around bright area
    protection_size = min(w, h) // 3
    x, y = maxLoc
    return (max(0, x - protection_size), max(0, y - protection_size),
            min(w, x + protection_size), min(h, y + protection_size))

def remove_black_borders_with_protection(image):
    """Remove black borders while protecting the ring area"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    
    # Detect ring area to protect
    ring_area = detect_ring_area(image)
    rx1, ry1, rx2, ry2 = ring_area
    
    # Create mask for black pixels (but not in ring area)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find borders to crop
    top, bottom, left, right = 0, h, 0, w
    
    # Top border - stop before ring area
    for i in range(min(ry1, h // 3)):
        if np.mean(gray[i, :]) > 20:
            top = i
            break
    
    # Bottom border - stop before ring area
    for i in range(h - 1, max(ry2, 2 * h // 3), -1):
        if np.mean(gray[i, :]) > 20:
            bottom = i + 1
            break
    
    # Left border - stop before ring area
    for i in range(min(rx1, w // 3)):
        if np.mean(gray[:, i]) > 20:
            left = i
            break
    
    # Right border - stop before ring area
    for i in range(w - 1, max(rx2, 2 * w // 3), -1):
        if np.mean(gray[:, i]) > 20:
            right = i + 1
            break
    
    # Crop image
    cropped = image[top:bottom, left:right]
    
    # If cropped too much, return original
    if cropped.shape[0] < h * 0.5 or cropped.shape[1] < w * 0.5:
        logger.warning("Cropping would remove too much, returning original")
        return image
    
    return cropped

def enhance_wedding_ring(image, metal_type):
    """Enhance wedding ring based on metal type without color distortion"""
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Basic enhancements only - no color space conversion
    enhancer = ImageEnhance.Brightness(pil_image)
    if metal_type in ['yellow_gold', 'rose_gold']:
        pil_image = enhancer.enhance(1.15)
    else:
        pil_image = enhancer.enhance(1.2)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.3)
    
    # Sharpness for detail
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    # Convert back to numpy
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return enhanced

def create_centered_thumbnail(image, target_size=(1000, 1300)):
    """Create perfectly centered thumbnail"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling to fill frame
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    thumbnail = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    
    # Calculate position to center
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Handle overflow
    if x_offset < 0:
        resized = resized[:, -x_offset:target_w-x_offset]
        x_offset = 0
    if y_offset < 0:
        resized = resized[-y_offset:target_h-y_offset, :]
        y_offset = 0
    
    # Place resized image
    end_x = min(x_offset + resized.shape[1], target_w)
    end_y = min(y_offset + resized.shape[0], target_h)
    
    thumbnail[y_offset:end_y, x_offset:end_x] = resized[:end_y-y_offset, :end_x-x_offset]
    
    return thumbnail

def detect_metal_type(image):
    """Detect metal type from the ring"""
    # Get center region where ring likely is
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
    
    # Determine metal type based on color ratios
    if r > g > b and (r - b) > 20:
        return "rose_gold"
    elif r > b and g > b and abs(r - g) < 20:
        return "yellow_gold"
    elif abs(r - g) < 10 and abs(g - b) < 10 and np.mean(avg_colors) > 180:
        return "white_gold"
    else:
        return "unplated_white"

def handler(event):
    """Main handler function"""
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
                    "version": "v92-fixed-bytesio"
                }
            }
        
        # Handle base64 padding
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_base64 = image_base64.strip()
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += '=' * (4 - missing_padding)
        
        # Decode base64 - FIX: Added seek(0) after BytesIO creation
        try:
            img_data = base64.b64decode(image_base64)
            img_buffer = io.BytesIO(img_data)
            img_buffer.seek(0)  # CRITICAL FIX: Reset buffer position to start
            img = Image.open(img_buffer)
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "failed",
                    "version": "v92-fixed-bytesio"
                }
            }
        
        logger.info(f"Processing image: {img_array.shape}")
        
        # Detect metal type
        metal_type = detect_metal_type(img_array)
        logger.info(f"Detected metal type: {metal_type}")
        
        # 1. Remove black borders with ring protection
        no_border = remove_black_borders_with_protection(img_array)
        
        # 2. Enhance wedding ring without color distortion
        enhanced = enhance_wedding_ring(no_border, metal_type)
        
        # 3. Create centered thumbnail
        thumbnail = create_centered_thumbnail(enhanced)
        
        # Convert to base64 with padding removed
        _, buffer = cv2.imencode('.png', enhanced)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8').rstrip('=')
        
        _, thumb_buffer = cv2.imencode('.png', thumbnail)
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
                    "version": "v92-fixed-bytesio"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v92-fixed-bytesio"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
