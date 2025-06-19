import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def handler(event):
    """Wedding Ring AI v79 - Ultimate Black Border Destroyer with Multiple Protection Layers"""
    try:
        # Get image from event
        image_input = event.get("input", {})
        image_base64 = image_input.get("image") or image_input.get("image_base64")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error"
                }
            }
        
        # Handle base64 prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "output": {
                    "error": "Failed to decode image",
                    "status": "error"
                }
            }
        
        # STEP 1: First aggressive black removal
        image_no_border = remove_black_border_phase1(image)
        
        # STEP 2: Second pass - edge cleaning
        image_no_border = remove_black_border_phase2(image_no_border)
        
        # STEP 3: Third pass - pixel-level cleaning
        image_no_border = remove_black_pixels_phase3(image_no_border)
        
        # STEP 4: Fourth pass - final safety check
        image_no_border = final_black_removal_phase4(image_no_border)
        
        # Detect metal type
        metal_type = detect_metal_type(image_no_border)
        
        # Apply powerful enhancement
        enhanced_image = enhance_wedding_ring_ultimate(image_no_border, metal_type)
        
        # Create perfect thumbnail
        thumbnail = create_thumbnail_full_frame(enhanced_image)
        
        # Convert to base64 (WITHOUT padding for Make.com)
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        main_base64 = main_base64.rstrip('=')
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "version": "v79_ultimate",
                    "black_removal_passes": 4,
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }

def remove_black_border_phase1(image):
    """Phase 1: Aggressive initial black border removal"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Very low threshold for black detection
    black_threshold = 25
    
    # Find borders by scanning from edges
    # Top
    top = 0
    for y in range(h // 2):
        row = gray[y, :]
        non_black_pixels = np.sum(row > black_threshold)
        if non_black_pixels > w * 0.5:  # If more than 50% of pixels are non-black
            top = max(0, y - 2)
            break
    
    # Bottom
    bottom = h
    for y in range(h - 1, h // 2, -1):
        row = gray[y, :]
        non_black_pixels = np.sum(row > black_threshold)
        if non_black_pixels > w * 0.5:
            bottom = min(h, y + 2)
            break
    
    # Left
    left = 0
    for x in range(w // 2):
        col = gray[:, x]
        non_black_pixels = np.sum(col > black_threshold)
        if non_black_pixels > h * 0.5:
            left = max(0, x - 2)
            break
    
    # Right
    right = w
    for x in range(w - 1, w // 2, -1):
        col = gray[:, x]
        non_black_pixels = np.sum(col > black_threshold)
        if non_black_pixels > h * 0.5:
            right = min(w, x + 2)
            break
    
    return image[top:bottom, left:right]

def remove_black_border_phase2(image):
    """Phase 2: Remove remaining black edges"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check edges and remove up to 100 pixels if mostly black
    edge_size = 100
    threshold = 30
    
    # Top edge
    for i in range(min(edge_size, h // 4)):
        if np.mean(gray[i, :]) < threshold:
            image = image[1:, :]
            gray = gray[1:, :]
        else:
            break
    
    # Update dimensions
    h, w = image.shape[:2]
    
    # Bottom edge
    for i in range(min(edge_size, h // 4)):
        if np.mean(gray[-(i+1), :]) < threshold:
            image = image[:-1, :]
            gray = gray[:-1, :]
        else:
            break
    
    # Update dimensions
    h, w = image.shape[:2]
    
    # Left edge
    for i in range(min(edge_size, w // 4)):
        if np.mean(gray[:, i]) < threshold:
            image = image[:, 1:]
            gray = gray[:, 1:]
        else:
            break
    
    # Update dimensions
    h, w = image.shape[:2]
    
    # Right edge
    for i in range(min(edge_size, w // 4)):
        if np.mean(gray[:, -(i+1)]) < threshold:
            image = image[:, :-1]
            gray = gray[:, :-1]
        else:
            break
    
    return image

def remove_black_pixels_phase3(image):
    """Phase 3: Pixel-level black removal from edges"""
    h, w = image.shape[:2]
    
    # Convert to RGB for easier processing
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask for non-black pixels
    # Black is < 40 in all channels
    mask = np.logical_or(
        rgb[:, :, 0] > 40,
        np.logical_or(rgb[:, :, 1] > 40, rgb[:, :, 2] > 40)
    )
    
    # Find bounds of non-black content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Add small safety margin
        ymin = max(0, ymin - 1)
        ymax = min(h, ymax + 2)
        xmin = max(0, xmin - 1)
        xmax = min(w, xmax + 2)
        
        return image[ymin:ymax, xmin:xmax]
    
    return image

def final_black_removal_phase4(image):
    """Phase 4: Final aggressive black removal"""
    h, w = image.shape[:2]
    
    # Check corners and edges one more time
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define regions to check (corners and edges)
    corner_size = min(50, h // 10, w // 10)
    
    # Check all corners
    corners_black = [
        np.mean(gray[:corner_size, :corner_size]) < 35,  # Top-left
        np.mean(gray[:corner_size, -corner_size:]) < 35,  # Top-right
        np.mean(gray[-corner_size:, :corner_size]) < 35,  # Bottom-left
        np.mean(gray[-corner_size:, -corner_size:]) < 35  # Bottom-right
    ]
    
    # If any corner is black, crop more aggressively
    if any(corners_black):
        crop_amount = corner_size // 2
        image = image[crop_amount:-crop_amount, crop_amount:-crop_amount]
    
    # Final edge check
    h, w = image.shape[:2]
    if h > 20 and w > 20:
        edge_width = 10
        if np.mean(gray[:edge_width, :]) < 40:
            image = image[edge_width:, :]
        if np.mean(gray[-edge_width:, :]) < 40:
            image = image[:-edge_width, :]
        if np.mean(gray[:, :edge_width]) < 40:
            image = image[:, edge_width:]
        if np.mean(gray[:, -edge_width:]) < 40:
            image = image[:, :-edge_width]
    
    return image

def detect_metal_type(image):
    """Detect wedding ring metal type"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get center region
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_size = min(h, w) // 3
    
    roi = hsv[center_y-roi_size:center_y+roi_size, 
              center_x-roi_size:center_x+roi_size]
    
    # Analyze colors
    avg_hue = np.mean(roi[:, :, 0])
    avg_sat = np.mean(roi[:, :, 1])
    avg_val = np.mean(roi[:, :, 2])
    
    # Metal detection
    if avg_sat < 20 and avg_val > 200:
        return "white_gold"
    elif avg_sat < 35:
        return "champagne_gold"
    elif 10 <= avg_hue <= 20 and avg_sat > 30:
        return "rose_gold"
    else:
        return "yellow_gold"

def enhance_wedding_ring_ultimate(image, metal_type):
    """Ultimate wedding ring enhancement"""
    
    # 1. Strong denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    
    # 2. LAB enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Massive brightness increase
    l = cv2.add(l, 50)
    
    # CLAHE with high contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Metal-specific adjustments
    if metal_type == "champagne_gold" or metal_type == "white_gold":
        # Make very white
        a = cv2.subtract(a, 8)
        b = cv2.subtract(b, 10)
    elif metal_type == "rose_gold":
        a = cv2.add(a, 2)
        b = cv2.subtract(b, 3)
    else:
        b = cv2.subtract(b, 5)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Super sharpening
    # Multiple sharpening passes
    for _ in range(2):
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # 4. Edge enhancement
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(enhanced, 1.0, edges_colored, 0.2, 0)
    
    # 5. PIL adjustments
    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Maximum brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.35)
    
    # Maximum contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.35)
    
    # Maximum sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    # Reduce saturation
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.8)
    
    final = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 6. Pure white background
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    
    # Create ring mask
    _, ring_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    
    # Clean mask
    kernel = np.ones((7,7), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=3)
    
    # Pure white background
    white_bg = np.full_like(final, (250, 248, 245), dtype=np.uint8)
    
    # Apply
    result = np.where(ring_mask[..., None] > 0, final, white_bg)
    
    # Final smoothing
    result = cv2.bilateralFilter(result, 11, 80, 80)
    
    return result

def create_thumbnail_full_frame(image):
    """Create 1000x1300 thumbnail with ring filling entire frame"""
    target_w, target_h = 1000, 1300
    
    # Find ring bounds using multiple methods
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Threshold
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding box of all significant contours
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    found_ring = False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Significant contour
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            found_ring = True
    
    if found_ring and x_max > x_min and y_max > y_min:
        # NO PADDING AT ALL
        ring_crop = image[y_min:y_max, x_min:x_max]
    else:
        # Fallback: use center 90%
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.05)
        ring_crop = image[margin:-margin, margin:-margin]
    
    # Calculate scale to FILL entire thumbnail
    crop_h, crop_w = ring_crop.shape[:2]
    scale_w = target_w / crop_w
    scale_h = target_h / crop_h
    scale = max(scale_w, scale_h) * 1.02  # 2% extra to ensure full coverage
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize
    resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Center crop to exact size
    y_start = (new_h - target_h) // 2
    x_start = (new_w - target_w) // 2
    
    # Ensure we have exact dimensions
    if new_h >= target_h and new_w >= target_w:
        thumbnail = resized[y_start:y_start+target_h, x_start:x_start+target_w]
    else:
        # Should not happen, but safety fallback
        thumbnail = np.full((target_h, target_w, 3), (250, 248, 245), dtype=np.uint8)
        y_off = max(0, (target_h - new_h) // 2)
        x_off = max(0, (target_w - new_w) // 2)
        y_end = min(y_off + new_h, target_h)
        x_end = min(x_off + new_w, target_w)
        thumbnail[y_off:y_end, x_off:x_end] = resized[:y_end-y_off, :x_end-x_off]
    
    return thumbnail

# RunPod handler
runpod.serverless.start({"handler": handler})
