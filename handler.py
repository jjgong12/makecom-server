import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def detect_wedding_ring_area(image):
    """Detect wedding ring area more accurately"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Find bright metallic areas (rings are usually brighter than background)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Method 2: Edge detection for ring contours
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Method 3: Center-weighted detection (rings are usually in center)
    center_mask = np.zeros_like(gray)
    cv2.circle(center_mask, (w//2, h//2), min(w,h)//3, 255, -1)
    
    # Combine methods
    combined_mask = cv2.bitwise_or(bright_mask, edges_dilated)
    combined_mask = cv2.bitwise_and(combined_mask, center_mask)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create ring mask
    ring_mask = np.zeros_like(gray)
    if contours:
        # Sort by area and take largest ones (likely rings)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i, contour in enumerate(contours[:2]):  # Max 2 rings
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(ring_mask, [contour], -1, 255, -1)
    
    # Dilate to include ring edges
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=3)
    
    return ring_mask

def enhance_ring_only(image, ring_mask):
    """Enhance only the ring area"""
    # Create a copy for enhancement
    enhanced = image.copy()
    
    # Extract ring area
    ring_area = cv2.bitwise_and(image, image, mask=ring_mask)
    
    # 1. Denoise
    ring_denoised = cv2.fastNlMeansDenoisingColored(ring_area, None, 5, 5, 7, 21)
    
    # 2. Sharpen strongly
    gaussian = cv2.GaussianBlur(ring_denoised, (0, 0), 3.0)
    ring_sharpened = cv2.addWeighted(ring_denoised, 2.5, gaussian, -1.5, 0)
    
    # 3. Enhance contrast
    pil_ring = Image.fromarray(cv2.cvtColor(ring_sharpened, cv2.COLOR_BGR2RGB))
    
    enhancer = ImageEnhance.Contrast(pil_ring)
    pil_ring = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Brightness(pil_ring)
    pil_ring = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(pil_ring)
    pil_ring = enhancer.enhance(1.8)
    
    ring_enhanced = cv2.cvtColor(np.array(pil_ring), cv2.COLOR_RGB2BGR)
    
    # Apply enhanced ring back to image
    enhanced[ring_mask > 0] = ring_enhanced[ring_mask > 0]
    
    return enhanced

def create_professional_background(image, ring_mask):
    """Create background like the AFTER files - natural bright beige/gray"""
    h, w = image.shape[:2]
    
    # Create background colors from AFTER files
    # Top: slightly cooler bright beige
    # Bottom: slightly warmer bright beige
    top_color = np.array([235, 233, 230])     # Bright beige-gray
    bottom_color = np.array([240, 238, 235])  # Warmer bright beige
    
    # Create smooth gradient
    background = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        weight = y / h
        background[y] = (1 - weight) * top_color + weight * bottom_color
    
    background = background.astype(np.uint8)
    
    # Apply background only where there's no ring
    result = image.copy()
    result[ring_mask == 0] = background[ring_mask == 0]
    
    # Smooth the transition between ring and background
    # Create transition zone
    kernel = np.ones((15,15), np.uint8)
    transition_mask = cv2.dilate(ring_mask, kernel, iterations=1)
    transition_mask = cv2.GaussianBlur(transition_mask.astype(np.float32), (21, 21), 10)
    transition_mask = transition_mask / 255.0
    
    # Blend
    for c in range(3):
        result[:,:,c] = (image[:,:,c] * transition_mask + 
                         background[:,:,c] * (1 - transition_mask)).astype(np.uint8)
    
    return result

def detect_metal_and_apply_correction(image, ring_mask):
    """Detect metal type and apply specific color correction"""
    # Extract ring area for analysis
    ring_pixels = image[ring_mask > 0]
    
    if len(ring_pixels) == 0:
        return image
    
    # Analyze color
    avg_color = np.mean(ring_pixels, axis=0)
    b, g, r = avg_color
    
    # Detect metal type
    if r > g and r > b and (r - b) > 20:  # Rose gold
        metal_type = 'rose_gold'
    elif g > b and abs(r - g) < 20:  # Yellow/Champagne gold
        metal_type = 'champagne_gold'
    else:  # White gold
        metal_type = 'white_gold'
    
    # Apply correction
    corrected = image.copy()
    
    if metal_type == 'champagne_gold':
        # Make it much whiter
        corrected = corrected.astype(np.float32)
        corrected[:,:,0] = np.clip(corrected[:,:,0] * 1.3, 0, 255)  # Blue up
        corrected[:,:,1] = np.clip(corrected[:,:,1] * 1.15, 0, 255)  # Green up
        corrected[:,:,2] = np.clip(corrected[:,:,2] * 1.0, 0, 255)  # Red same
        
        # Add white overlay
        white = np.full_like(corrected, 255)
        corrected = cv2.addWeighted(corrected, 0.7, white, 0.3, 0)
        corrected = corrected.astype(np.uint8)
    
    return corrected, metal_type

def create_perfect_thumbnail(image):
    """Create thumbnail with ring filling the frame"""
    # Detect ring area
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find bright areas (rings)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all rings
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        if x_max > x_min and y_max > y_min:
            # Add 2% padding
            pad_x = int((x_max - x_min) * 0.02)
            pad_y = int((y_max - y_min) * 0.02)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(image.shape[1], x_max + pad_x)
            y_max = min(image.shape[0], y_max + pad_y)
            
            # Crop
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            # Use center crop
            h, w = image.shape[:2]
            size = min(h, w) // 2
            cropped = image[h//2-size:h//2+size, w//2-size:w//2+size]
    else:
        # Fallback to center crop
        h, w = image.shape[:2]
        size = min(h, w) // 2
        cropped = image[h//2-size:h//2+size, w//2-size:w//2+size]
    
    # Resize to 1000x1300
    target_w, target_h = 1000, 1300
    h, w = cropped.shape[:2]
    
    # Calculate scale to fill 95% of frame
    scale = min(target_w * 0.95 / w, target_h * 0.95 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create canvas with same background as main image
    canvas = np.full((target_h, target_w, 3), (237, 235, 232), dtype=np.uint8)
    
    # Center paste
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def handler(event):
    """RunPod handler function"""
    try:
        # Get image from event
        image_input = event.get("input", {})
        base64_image = image_input.get("image") or image_input.get("image_base64")
        
        if not base64_image:
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error"
                }
            }
        
        # Handle base64 prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "output": {
                    "error": "Failed to decode image",
                    "status": "error"
                }
            }
        
        # Step 1: Detect wedding ring area
        ring_mask = detect_wedding_ring_area(image)
        
        # Step 2: Enhance only the ring
        enhanced = enhance_ring_only(image, ring_mask)
        
        # Step 3: Detect metal type and apply color correction
        enhanced, metal_type = detect_metal_and_apply_correction(enhanced, ring_mask)
        
        # Step 4: Create professional background
        final_image = create_professional_background(enhanced, ring_mask)
        
        # Step 5: Create thumbnail
        thumbnail = create_perfect_thumbnail(final_image)
        
        # Convert to base64
        # Main image
        _, main_buffer = cv2.imencode('.jpg', final_image, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 98])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "version": "v75_precise_detection",
                    "enhancements": [
                        "precise_ring_detection",
                        "ring_only_enhancement",
                        "professional_background",
                        "perfect_thumbnail_crop"
                    ]
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

# RunPod handler
runpod.serverless.start({"handler": handler})
