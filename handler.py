import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def handler(event):
    """Wedding Ring AI v78 - Ultra Border Removal with Enhanced Processing"""
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
        
        # Step 1: Ultra aggressive black border removal
        image_no_border = remove_black_border_ultra(image)
        
        # Step 2: Detect metal type
        metal_type = detect_metal_type(image_no_border)
        
        # Step 3: Apply enhanced wedding ring processing (based on before/after data)
        enhanced_image = enhance_wedding_ring_v78(image_no_border, metal_type)
        
        # Step 4: Create perfect thumbnail with NO margins
        thumbnail = create_thumbnail_no_margins(enhanced_image)
        
        # Convert to base64 (WITHOUT padding for Make.com)
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "version": "v78_ultra",
                    "border_removed": True,
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

def remove_black_border_ultra(image):
    """Ultra aggressive black border removal"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # More aggressive thresholds for black detection
    black_threshold = 30  # Lower threshold to catch more black
    
    # Scan from edges with larger scan area
    max_scan = int(min(h, w) * 0.4)  # Scan up to 40% from each edge
    
    # Find actual borders by scanning inward
    # Top border
    top = 0
    for y in range(max_scan):
        row = gray[y, :]
        if np.mean(row) > black_threshold and np.max(row) > 50:
            # Check if we have consistent non-black content
            if y > 5 and np.mean(gray[y:y+10, :]) > black_threshold:
                top = max(0, y - 5)
                break
    
    # Bottom border
    bottom = h
    for y in range(h - 1, h - max_scan, -1):
        row = gray[y, :]
        if np.mean(row) > black_threshold and np.max(row) > 50:
            if y < h - 5 and np.mean(gray[y-10:y, :]) > black_threshold:
                bottom = min(h, y + 5)
                break
    
    # Left border
    left = 0
    for x in range(max_scan):
        col = gray[:, x]
        if np.mean(col) > black_threshold and np.max(col) > 50:
            if x > 5 and np.mean(gray[:, x:x+10]) > black_threshold:
                left = max(0, x - 5)
                break
    
    # Right border
    right = w
    for x in range(w - 1, w - max_scan, -1):
        col = gray[:, x]
        if np.mean(col) > black_threshold and np.max(col) > 50:
            if x < w - 5 and np.mean(gray[:, x-10:x]) > black_threshold:
                right = min(w, x + 5)
                break
    
    # Crop the image
    cropped = image[top:bottom, left:right]
    
    # Second pass - check if edges are still black
    h2, w2 = cropped.shape[:2]
    if h2 > 100 and w2 > 100:
        gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Additional edge removal if needed
        edge_check = 20
        if np.mean(gray2[:edge_check, :]) < 40:
            cropped = cropped[edge_check:, :]
        if np.mean(gray2[-edge_check:, :]) < 40:
            cropped = cropped[:-edge_check, :]
        if np.mean(gray2[:, :edge_check]) < 40:
            cropped = cropped[:, edge_check:]
        if np.mean(gray2[:, -edge_check:]) < 40:
            cropped = cropped[:, :-edge_check]
    
    return cropped

def detect_metal_type(image):
    """Detect wedding ring metal type"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get center region where ring is
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_size = min(h, w) // 3
    
    roi = hsv[center_y-roi_size:center_y+roi_size, 
              center_x-roi_size:center_x+roi_size]
    
    # Analyze colors
    avg_hue = np.mean(roi[:, :, 0])
    avg_sat = np.mean(roi[:, :, 1])
    avg_val = np.mean(roi[:, :, 2])
    
    # Determine metal type based on color characteristics
    if avg_sat < 25 and avg_val > 200:  # Very bright, low saturation
        return "white_gold"
    elif avg_sat < 40 and avg_val > 180:  # Bright, low-medium saturation
        return "champagne_gold"
    elif 10 <= avg_hue <= 20 and avg_sat > 30:  # Rose tone
        return "rose_gold"
    elif 15 <= avg_hue <= 30 and avg_sat > 40:  # Yellow tone
        return "yellow_gold"
    else:
        return "champagne_gold"  # Default

def enhance_wedding_ring_v78(image, metal_type):
    """Enhanced wedding ring processing based on before/after analysis"""
    
    # 1. Strong denoising while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    
    # 2. Powerful brightness and contrast enhancement
    # Convert to LAB for better control
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Significant brightness increase (based on before/after analysis)
    l = cv2.add(l, 45)  # Strong brightness boost
    
    # Apply CLAHE for better local contrast
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Metal-specific color adjustments
    if metal_type == "champagne_gold":
        # Make it much whiter (like white gold)
        a = cv2.subtract(a, 5)  # Reduce red
        b = cv2.subtract(b, 8)  # Reduce yellow significantly
    elif metal_type == "rose_gold":
        # Brighten while keeping some warmth
        a = cv2.add(a, 3)
        b = cv2.subtract(b, 2)
    elif metal_type == "yellow_gold":
        # Reduce yellow saturation
        b = cv2.subtract(b, 5)
    elif metal_type == "white_gold":
        # Cool tones
        a = cv2.subtract(a, 2)
        b = cv2.subtract(b, 3)
    
    # Merge back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Strong sharpening for details
    # Unsharp mask with high strength
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
    
    # 4. Edge enhancement for ring details
    edges = cv2.Canny(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY), 30, 100)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    detail_enhanced = cv2.addWeighted(sharpened, 1.0, edges_colored, 0.15, 0)
    
    # 5. Final adjustments with PIL
    pil_image = Image.fromarray(cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2RGB))
    
    # Strong brightness increase
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.25)
    
    # High contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.25)
    
    # Extra sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.4)
    
    # Reduce saturation for cleaner look
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.85)
    
    # Convert back
    final = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 6. Create bright, clean background
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    
    # Create mask for ring
    _, ring_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Clean mask
    kernel = np.ones((5,5), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate to include ring edges
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=2)
    
    # Create bright background (matching after samples)
    bright_bg = np.full_like(final, (245, 243, 240), dtype=np.uint8)
    
    # Apply mask
    result = np.where(ring_mask[..., None] > 0, final, bright_bg)
    
    # Smooth transitions
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

def create_thumbnail_no_margins(image):
    """Create 1000x1300 thumbnail with ring filling entire frame"""
    target_w, target_h = 1000, 1300
    
    # Find ring bounds
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get combined bounding box
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        if x_max > x_min and y_max > y_min:
            # NO PADDING - crop exactly to ring bounds
            ring_crop = image[y_min:y_max, x_min:x_max]
        else:
            ring_crop = image
    else:
        # Use aggressive center crop
        h, w = image.shape[:2]
        size = int(min(h, w) * 0.9)
        center_y, center_x = h // 2, w // 2
        ring_crop = image[center_y-size//2:center_y+size//2,
                          center_x-size//2:center_x+size//2]
    
    # Resize to fill entire thumbnail
    crop_h, crop_w = ring_crop.shape[:2]
    
    # Calculate scale to fill 1000x1300 completely
    scale_w = target_w / crop_w
    scale_h = target_h / crop_h
    scale = max(scale_w, scale_h)  # Use max to ensure full coverage
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize
    resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # If oversized, crop from center
    if new_w > target_w or new_h > target_h:
        y_start = (new_h - target_h) // 2
        x_start = (new_w - target_w) // 2
        thumbnail = resized[y_start:y_start+target_h, x_start:x_start+target_w]
    else:
        # This shouldn't happen with max scale, but just in case
        thumbnail = np.full((target_h, target_w, 3), (245, 243, 240), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumbnail

# RunPod handler
runpod.serverless.start({"handler": handler})
