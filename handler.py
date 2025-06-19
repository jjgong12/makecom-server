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

def enhance_ring_details(image):
    """Enhance wedding ring details with multiple techniques"""
    # 1. Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    
    # 2. Unsharp mask for detail enhancement
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.8, gaussian, -0.8, 0)
    
    # 3. Local contrast enhancement using CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for better local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. Edge enhancement
    edges = cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend edges back
    detail_enhanced = cv2.addWeighted(enhanced, 1.0, edges_colored, 0.1, 0)
    
    return detail_enhanced

def apply_metal_specific_enhancement(image, metal_type):
    """Apply metal-specific color and brightness adjustments"""
    enhanced = image.copy()
    
    # Convert to float for precise adjustments
    enhanced = enhanced.astype(np.float32)
    
    if metal_type == 'champagne_gold':
        # Make champagne gold much more white (like image 5)
        # Strongly reduce yellow/gold tones
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.20, 0, 255)  # Blue channel up more
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.12, 0, 255)  # Green channel slightly up
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.02, 0, 255)  # Red channel minimal change
        
        # Add stronger white overlay to make it much closer to white gold
        white_overlay = np.full_like(enhanced, 255)
        enhanced = cv2.addWeighted(enhanced, 0.75, white_overlay, 0.25, 0)
        
    elif metal_type == 'rose_gold':
        # Enhance pink tones while keeping brightness
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.1, 0, 255)  # Red channel
        
    elif metal_type == 'yellow_gold':
        # Enhance yellow tones
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.08, 0, 255)  # Green
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.12, 0, 255)  # Red
        
    else:  # white_gold
        # Cool tones, high brightness
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.08, 0, 255)  # Blue
    
    # Overall brightness boost for all metals
    enhanced = np.clip(enhanced * 1.15, 0, 255)
    
    return enhanced.astype(np.uint8)

def create_clean_background(image):
    """Create clean, bright background while preserving the ring"""
    # Create mask for the ring using multiple methods
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Threshold for metallic objects
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Combine masks
    ring_mask = cv2.bitwise_or(thresh1, thresh2)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate to include ring edges
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=2)
    
    # Create bright background like after files (235-245 range)
    bright_bg = np.full_like(image, (245, 242, 238))  # Slight warm tone like after files
    
    # Apply mask - keep ring, replace background
    result = np.where(ring_mask[..., None] > 0, image, bright_bg)
    
    # Smooth edges
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create thumbnail with ring filling most of the frame"""
    # Find ring contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours (for multiple rings)
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        if x_max > x_min and y_max > y_min:
            # Add small padding (3% instead of larger padding)
            pad = int(max(x_max - x_min, y_max - y_min) * 0.03)
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(image.shape[1], x_max + pad)
            y_max = min(image.shape[0], y_max + pad)
            
            # Crop to ring area
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image
    else:
        # If no contours found, use center crop
        h, w = image.shape[:2]
        size = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        cropped = image[center_y-size:center_y+size, center_x-size:center_x+size]
    
    # Calculate scaling to fit target size
    h, w = cropped.shape[:2]
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    scale = min(scale_x, scale_y) * 0.95  # 95% to leave tiny margin
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create background matching after files
    thumbnail = np.full((target_size[1], target_size[0], 3), (245, 242, 238), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumbnail

def enhance_wedding_ring(image):
    """Main enhancement pipeline"""
    # 1. Detect metal type
    metal_type = detect_ring_metal_type(image)
    
    # 2. Initial detail enhancement
    detailed = enhance_ring_details(image)
    
    # 3. Apply metal-specific adjustments
    metal_enhanced = apply_metal_specific_enhancement(detailed, metal_type)
    
    # 4. Create clean, bright background
    clean_bg = create_clean_background(metal_enhanced)
    
    # 5. Final brightness and contrast adjustment using PIL
    pil_image = Image.fromarray(cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB))
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.15)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.3)
    
    # Convert back to OpenCV format
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return final_image, metal_type

def handler(event):
    """RunPod handler function"""
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
        
        # Process the image
        enhanced_image, metal_type = enhance_wedding_ring(image)
        
        # Create thumbnail
        thumbnail = create_thumbnail(enhanced_image)
        
        # Convert results to base64
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced_image, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 98])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding for Make.com
        
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
                    "version": "v73",
                    "enhancements": [
                        "detail_enhancement",
                        "metal_specific_color",
                        "bright_clean_background",
                        "optimal_thumbnail_crop"
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
