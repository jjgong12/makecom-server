import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def detect_ring_metal_type(image):
    """Detect wedding ring metal type based on color analysis"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate average hue and saturation in center region
    h, w = image.shape[:2]
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]
    avg_hue = np.mean(center_region[:, :, 0])
    avg_sat = np.mean(center_region[:, :, 1])
    avg_val = np.mean(center_region[:, :, 2])
    
    # Detect metal type based on HSV values
    if avg_sat < 30 and avg_val > 200:  # Low saturation, high brightness
        return 'white_gold'
    elif 10 <= avg_hue <= 20 and avg_sat > 30:  # Orange-pink hue
        return 'rose_gold'
    elif 15 <= avg_hue <= 25 and avg_sat > 50:  # Yellow hue
        return 'yellow_gold'
    elif avg_sat < 40 and 180 < avg_val < 220:  # Champagne - low sat, medium-high brightness
        return 'champagne_gold'
    else:
        return 'white_gold'  # Default

def enhance_ring_details_ultra(image):
    """Ultra-strong enhancement for ring details"""
    # 1. Strong denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    
    # 2. Ultra unsharp mask (2.2x strength)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 2.2, gaussian, -1.2, 0)
    
    # 3. CLAHE with higher clip limit
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. Strong edge enhancement
    edges = cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), 30, 100)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend edges back strongly
    detail_enhanced = cv2.addWeighted(enhanced, 1.0, edges_colored, 0.15, 0)
    
    return detail_enhanced

def apply_metal_specific_enhancement_ultra(image, metal_type):
    """Ultra-strong metal-specific adjustments"""
    enhanced = image.copy()
    
    # Convert to float for precise adjustments
    enhanced = enhanced.astype(np.float32)
    
    if metal_type == 'champagne_gold':
        # Make champagne gold VERY white (like image 5)
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.25, 0, 255)  # Blue channel up 25%
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.15, 0, 255)  # Green channel 15%
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.0, 0, 255)   # Red channel unchanged
        
        # 35% white overlay for strong whitening
        white_overlay = np.full_like(enhanced, 255)
        enhanced = cv2.addWeighted(enhanced, 0.65, white_overlay, 0.35, 0)
        
    elif metal_type == 'rose_gold':
        # Enhance pink tones while keeping brightness
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.15, 0, 255)  # Red channel
        # Slight white overlay
        white_overlay = np.full_like(enhanced, 255)
        enhanced = cv2.addWeighted(enhanced, 0.85, white_overlay, 0.15, 0)
        
    elif metal_type == 'yellow_gold':
        # Enhance yellow tones
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.1, 0, 255)   # Green
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.15, 0, 255)  # Red
        
    else:  # white_gold
        # Cool tones, high brightness
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.1, 0, 255)   # Blue
        # Slight white overlay
        white_overlay = np.full_like(enhanced, 255)
        enhanced = cv2.addWeighted(enhanced, 0.9, white_overlay, 0.1, 0)
    
    # Overall brightness boost for all metals
    enhanced = np.clip(enhanced * 1.25, 0, 255)
    
    return enhanced.astype(np.uint8)

def create_natural_bright_background(image):
    """Create natural bright background like the AFTER files"""
    # Create mask for the ring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to separate ring from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find ring contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create ring mask
    ring_mask = np.zeros_like(gray)
    if contours:
        # Draw all significant contours (for multiple rings)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(ring_mask, [contour], -1, 255, -1)
    
    # Dilate to include ring edges
    kernel = np.ones((7,7), np.uint8)
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=2)
    
    # Create natural gradient background (like AFTER files)
    h, w = image.shape[:2]
    
    # Sample colors from edges of the image
    top_color = np.mean(image[:50, :], axis=(0,1))
    bottom_color = np.mean(image[-50:, :], axis=(0,1))
    
    # Target bright beige/gray (from AFTER files)
    target_top = np.array([242, 240, 238])     # Slight blue-gray
    target_bottom = np.array([245, 243, 240])  # Warm beige
    
    # Create gradient
    gradient_bg = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        weight = y / h
        gradient_bg[y] = (1 - weight) * target_top + weight * target_bottom
    
    gradient_bg = gradient_bg.astype(np.uint8)
    
    # Apply mask - keep ring, use gradient for background
    result = np.where(ring_mask[..., None] > 0, image, gradient_bg)
    
    # Smooth transition
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

def create_thumbnail_ultra_tight(image, target_size=(1000, 1300)):
    """Create thumbnail with ring filling almost entire frame"""
    # Find ring contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all significant contours
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter noise
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        if x_max > x_min and y_max > y_min:
            # MINIMAL padding (1% only)
            pad = int(max(x_max - x_min, y_max - y_min) * 0.01)
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(image.shape[1], x_max + pad)
            y_max = min(image.shape[0], y_max + pad)
            
            # Crop to ring area
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image
    else:
        # If no contours, use center 80%
        h, w = image.shape[:2]
        margin = 0.1
        x_min = int(w * margin)
        y_min = int(h * margin)
        x_max = int(w * (1 - margin))
        y_max = int(h * (1 - margin))
        cropped = image[y_min:y_max, x_min:x_max]
    
    # Scale to fill 98% of target (almost full)
    h, w = cropped.shape[:2]
    scale_x = (target_size[0] * 0.98) / w
    scale_y = (target_size[1] * 0.98) / h
    scale = min(scale_x, scale_y)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize with highest quality
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create natural background (like AFTER files)
    thumbnail = np.full((target_size[1], target_size[0], 3), (243, 241, 238), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumbnail

def enhance_wedding_ring_v74(image):
    """Main enhancement pipeline v74 - Ultra strong"""
    # 1. Detect metal type
    metal_type = detect_ring_metal_type(image)
    
    # 2. Ultra detail enhancement
    detailed = enhance_ring_details_ultra(image)
    
    # 3. Ultra metal-specific adjustments
    metal_enhanced = apply_metal_specific_enhancement_ultra(detailed, metal_type)
    
    # 4. Create natural bright background (like AFTER files)
    natural_bg = create_natural_bright_background(metal_enhanced)
    
    # 5. Final ultra brightness and contrast using PIL
    pil_image = Image.fromarray(cv2.cvtColor(natural_bg, cv2.COLOR_BGR2RGB))
    
    # Strong enhancements
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.35)  # 35% brighter
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.25)  # 25% more contrast
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)   # 50% sharper
    
    # If champagne gold, one more white boost
    if metal_type == 'champagne_gold':
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)  # Extra 10% for champagne
    
    # Convert back
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return final_image, metal_type

def handler(event):
    """RunPod handler function"""
    try:
        # Get base64 image from event
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
        
        # Process the image
        enhanced_image, metal_type = enhance_wedding_ring_v74(image)
        
        # Create ultra tight thumbnail
        thumbnail = create_thumbnail_ultra_tight(enhanced_image)
        
        # Convert to base64
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced_image, 
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
                    "version": "v74_ultra_strong",
                    "enhancements": [
                        "ultra_detail_enhancement",
                        "metal_specific_ultra",
                        "natural_bright_background",
                        "ultra_tight_thumbnail"
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
