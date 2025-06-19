import runpod
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

def log(message):
    """Simple logging function"""
    print(f"[Wedding Ring AI v71 Supreme] {message}")

def detect_and_remove_black_border_supreme(image):
    """
    Supreme black border removal - Based on v23.1 ULTRA success
    Ultra aggressive 3-pass system with OpenCV
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    original_h, original_w = h, w
    
    # PASS 1: Ultra aggressive scan (50% of image, like v23.1 ULTRA)
    log("PASS 1: Ultra aggressive border scan - 50% area")
    max_border = min(int(h * 0.5), int(w * 0.5), 500)
    
    # Top border - check entire row
    top_crop = 0
    for y in range(max_border):
        row_mean = np.mean(gray[y, :])  # Entire row like v23.1 ULTRA
        if row_mean < 100:  # More aggressive than v70's 120
            top_crop = y + 1
        else:
            # Check if 90% of pixels are dark
            dark_pixels = np.sum(gray[y, :] < 100)
            if dark_pixels > w * 0.9:
                top_crop = y + 1
            else:
                break
    
    # Bottom border
    bottom_crop = h
    for y in range(h - 1, h - max_border, -1):
        row_mean = np.mean(gray[y, :])
        if row_mean < 100:
            bottom_crop = y
        else:
            dark_pixels = np.sum(gray[y, :] < 100)
            if dark_pixels > w * 0.9:
                bottom_crop = y
            else:
                break
    
    # Left border
    left_crop = 0
    for x in range(max_border):
        col_mean = np.mean(gray[:, x])
        if col_mean < 100:
            left_crop = x + 1
        else:
            dark_pixels = np.sum(gray[:, x] < 100)
            if dark_pixels > h * 0.9:
                left_crop = x + 1
            else:
                break
    
    # Right border
    right_crop = w
    for x in range(w - 1, w - max_border, -1):
        col_mean = np.mean(gray[:, x])
        if col_mean < 100:
            right_crop = x
        else:
            dark_pixels = np.sum(gray[:, x] < 100)
            if dark_pixels > h * 0.9:
                right_crop = x
            else:
                break
    
    # Apply PASS 1
    if top_crop > 0 or bottom_crop < h or left_crop > 0 or right_crop < w:
        log(f"PASS 1 removed: top={top_crop}, bottom={h-bottom_crop}, left={left_crop}, right={w-right_crop}")
        image = image[top_crop:bottom_crop, left_crop:right_crop]
        gray = gray[top_crop:bottom_crop, left_crop:right_crop]
        h, w = gray.shape
    
    # PASS 2: Secondary precision crop (like v23.1)
    log("PASS 2: Precision border removal")
    
    # Check smaller area for remaining borders
    check_size = min(30, int(h * 0.1), int(w * 0.1))
    
    # Top edge check
    if np.mean(gray[:check_size, :]) < 80:
        for y in range(check_size):
            if np.mean(gray[y, :]) < 80:
                top_extra = y + 1
            else:
                break
        if top_extra > 0:
            image = image[top_extra:, :]
            gray = gray[top_extra:, :]
            log(f"PASS 2 removed additional {top_extra}px from top")
    
    # Bottom edge check
    h, w = gray.shape
    if np.mean(gray[-check_size:, :]) < 80:
        for y in range(h - 1, h - check_size, -1):
            if np.mean(gray[y, :]) < 80:
                bottom_extra = h - y
            else:
                break
        if bottom_extra > 0:
            image = image[:-bottom_extra, :]
            gray = gray[:-bottom_extra, :]
            log(f"PASS 2 removed additional {bottom_extra}px from bottom")
    
    # Left edge check
    h, w = gray.shape
    if np.mean(gray[:, :check_size]) < 80:
        for x in range(check_size):
            if np.mean(gray[:, x]) < 80:
                left_extra = x + 1
            else:
                break
        if left_extra > 0:
            image = image[:, left_extra:]
            gray = gray[:, left_extra:]
            log(f"PASS 2 removed additional {left_extra}px from left")
    
    # Right edge check
    h, w = gray.shape
    if np.mean(gray[:, -check_size:]) < 80:
        for x in range(w - 1, w - check_size, -1):
            if np.mean(gray[:, x]) < 80:
                right_extra = w - x
            else:
                break
        if right_extra > 0:
            image = image[:, :-right_extra]
            gray = gray[:, :-right_extra]
            log(f"PASS 2 removed additional {right_extra}px from right")
    
    # PASS 3: Final fine-tuning (very aggressive)
    log("PASS 3: Final edge cleaning")
    h, w = gray.shape
    
    # Ultra fine edge removal - even more aggressive
    edge_check = 15  # Fixed 15 pixels like v23.1
    
    if np.mean(gray[:edge_check, :]) < 100:
        image = image[edge_check:, :]
        gray = gray[edge_check:, :]
        log("PASS 3 removed 15px from top")
    
    if gray.shape[0] > edge_check and np.mean(gray[-edge_check:, :]) < 100:
        image = image[:-edge_check, :]
        gray = gray[:-edge_check, :]
        log("PASS 3 removed 15px from bottom")
    
    if gray.shape[1] > edge_check and np.mean(gray[:, :edge_check]) < 100:
        image = image[:, edge_check:]
        gray = gray[:, edge_check:]
        log("PASS 3 removed 15px from left")
    
    if gray.shape[1] > edge_check and np.mean(gray[:, -edge_check:]) < 100:
        image = image[:, :-edge_check]
        log("PASS 3 removed 15px from right")
    
    # Final safety check
    final_h, final_w = image.shape[:2]
    if final_h < original_h * 0.3 or final_w < original_w * 0.3:
        log("Warning: Removed too much, but proceeding anyway")
    
    log(f"Supreme removal complete: {original_h}x{original_w} -> {final_h}x{final_w}")
    return image

def enhance_wedding_ring(image):
    """
    Enhanced wedding ring processing with strong corrections
    """
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Strong brightness boost for wedding rings
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.4)
    
    # Strong contrast for metal definition
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.3)
    
    # Maximum sharpness for details
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(2.0)
    
    # Slight color enhancement
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Convert back to OpenCV
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Additional OpenCV processing
    # Strong denoise
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Unsharp mask for extra pop
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    enhanced = cv2.addWeighted(enhanced, 1.7, gaussian, -0.7, 0)
    
    # Ensure pure white background (248, 248, 248)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    enhanced[mask == 255] = [248, 248, 248]
    
    return enhanced

def create_thumbnail_supreme(image, target_size=(1000, 1300)):
    """
    Create thumbnail with supreme border removal - matching main image logic
    """
    log("Creating supreme thumbnail with same border removal")
    
    # Apply same supreme border removal as main image
    image_clean = detect_and_remove_black_border_supreme(image.copy())
    
    h, w = image_clean.shape[:2]
    
    # Convert to grayscale for ring detection
    gray = cv2.cvtColor(image_clean, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold for ring detection
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours (entire ring set)
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w_c)
            y_max = max(y_max, y + h_c)
        
        # Minimal padding (1%)
        padding = int(max(x_max - x_min, y_max - y_min) * 0.01)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop to ring area
        ring_crop = image_clean[y_min:y_max, x_min:x_max]
    else:
        log("No contours found, using cleaned image directly")
        ring_crop = image_clean
    
    # Create white background
    background = np.full((target_size[1], target_size[0], 3), 248, dtype=np.uint8)
    
    # Calculate scale to fill 99% of canvas
    scale_x = (target_size[0] * 0.99) / ring_crop.shape[1]
    scale_y = (target_size[1] * 0.99) / ring_crop.shape[0]
    scale = min(scale_x, scale_y)
    
    new_width = int(ring_crop.shape[1] * scale)
    new_height = int(ring_crop.shape[0] * scale)
    
    # Resize with high quality
    ring_resized = cv2.resize(ring_crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Enhance the thumbnail
    ring_resized = enhance_wedding_ring(ring_resized)
    
    # Center on background
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = ring_resized
    
    log(f"Thumbnail created: {new_width}x{new_height} ring in {target_size[0]}x{target_size[1]} canvas")
    return background

def handler(job):
    """RunPod handler function"""
    log("Starting v71 Supreme processing")
    
    try:
        job_input = job['input']
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get('image') or job_input.get('image_base64')
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Handle base64 padding
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        log(f"Image loaded: {image.shape}")
        
        # Apply supreme black border removal
        image_clean = detect_and_remove_black_border_supreme(image)
        log(f"After border removal: {image_clean.shape}")
        
        # Enhance the cleaned image
        enhanced = enhance_wedding_ring(image_clean)
        
        # Convert main image to base64
        _, buffer = cv2.imencode('.png', enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        enhanced_base64 = enhanced_base64.rstrip('=')  # Remove padding
        
        # Create thumbnail with same supreme border removal
        thumbnail = create_thumbnail_supreme(image, target_size=(1000, 1300))
        
        # Convert thumbnail to base64
        _, thumb_buffer = cv2.imencode('.png', thumbnail, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumbnail_base64 = thumbnail_base64.rstrip('=')  # Remove padding
        
        log("Processing complete - v71 Supreme")
        
        # Return with correct structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": "v71_supreme",
                    "original_size": list(image.shape[:2]),
                    "cleaned_size": list(image_clean.shape[:2]),
                    "border_removed": True,
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        log(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
