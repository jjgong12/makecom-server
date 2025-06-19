import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_black_borders_ultra(img_array, threshold=190, scan_range=400):
    """Ultra aggressive black border detection for 6720x4480 images"""
    h, w = img_array.shape[:2]
    logger.info(f"Ultra border detection - Image size: {w}x{h}")
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # Ultra aggressive threshold - only pure white passes
    # Top border - scan up to 400px for thick borders
    for i in range(min(scan_range, h//2)):
        row = gray[i, w//4:3*w//4]
        if np.mean(row) < threshold:
            borders['top'] = i + 1
        else:
            break
    
    # Bottom border
    for i in range(min(scan_range, h//2)):
        row = gray[h-1-i, w//4:3*w//4]
        if np.mean(row) < threshold:
            borders['bottom'] = i + 1
        else:
            break
    
    # Left border
    for i in range(min(scan_range, w//2)):
        col = gray[h//4:3*h//4, i]
        if np.mean(col) < threshold:
            borders['left'] = i + 1
        else:
            break
    
    # Right border
    for i in range(min(scan_range, w//2)):
        col = gray[h//4:3*h//4, w-1-i]
        if np.mean(col) < threshold:
            borders['right'] = i + 1
        else:
            break
    
    # Add extra safety margin for any remaining black pixels
    safety_margin = 60
    borders = {k: v + safety_margin if v > 0 else 0 for k, v in borders.items()}
    
    logger.info(f"Detected borders with margin: {borders}")
    return borders

def remove_black_borders_completely(img_array):
    """Remove black borders with ultra settings for 6720x4480"""
    borders = find_black_borders_ultra(img_array, threshold=190, scan_range=400)
    
    if any(borders.values()):
        h, w = img_array.shape[:2]
        new_img = img_array[
            borders['top']:h-borders['bottom'],
            borders['left']:w-borders['right']
        ]
        logger.info(f"Borders removed - New size: {new_img.shape[1]}x{new_img.shape[0]}")
        return new_img
    
    return img_array

def create_thumbnail_max_zoom(img_array, size=800):
    """Create thumbnail with maximum zoom - no padding"""
    h, w = img_array.shape[:2]
    
    # Find ring area (simple center crop)
    center_x, center_y = w // 2, h // 2
    
    # Crop to square around center
    crop_size = min(w, h)
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Ensure bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Crop and resize
    cropped = img_array[y1:y2, x1:x2]
    thumbnail = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def enhance_wedding_ring_premium(img_array, metal_type='white_gold'):
    """Premium wedding ring enhancement"""
    # First remove borders
    img_array = remove_black_borders_completely(img_array)
    
    # Convert to PIL
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    # Base adjustments
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.15)
    
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.12)
    
    # Metal-specific enhancements
    if metal_type == 'yellow_gold':
        # Warm, rich gold
        color = ImageEnhance.Color(img)
        img = color.enhance(1.25)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0, 
                                   np.full_like(img_array, [0, 10, 25]), 0.08, 0)
    
    elif metal_type == 'rose_gold':
        # Soft pink tones
        color = ImageEnhance.Color(img)
        img = color.enhance(1.18)
        img_array = np.array(img)
        img_array = cv2.addWeighted(img_array, 1.0,
                                   np.full_like(img_array, [0, 5, 20]), 0.12, 0)
    
    elif metal_type == 'champagne_gold':
        # Bright champagne - almost white gold
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.30)
        color = ImageEnhance.Color(img)
        img = color.enhance(0.9)
        img_array = np.array(img)
        # Cool white overlay
        img_array = cv2.addWeighted(img_array, 1.0,
                                   np.full_like(img_array, [245, 240, 235]), 0.15, 0)
        # Slight blue tint for white gold effect
        img_array[:,:,0] = np.clip(img_array[:,:,0] + 6, 0, 255)
    
    else:  # white_gold
        # Cool, bright white
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
    
    # Ensure no clipping
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Final subtle adjustments
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.add(lab[:,:,0], 3)
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img_array

def detect_metal_type(img_array):
    """Detect metal type from image colors"""
    # Sample center region
    h, w = img_array.shape[:2]
    center = img_array[h//3:2*h//3, w//3:2*w//3]
    
    # Convert to RGB for analysis
    if len(center.shape) == 3 and center.shape[2] == 3:
        center_rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
    else:
        center_rgb = center
    
    # Calculate color statistics
    avg_color = np.mean(center_rgb, axis=(0,1))
    r, g, b = avg_color
    
    # Calculate warmth and saturation
    warmth = (r - b) / 255.0
    saturation = np.std(avg_color) / np.mean(avg_color)
    brightness = np.mean(avg_color) / 255.0
    
    logger.info(f"Color analysis - R:{r:.1f} G:{g:.1f} B:{b:.1f}, Warmth:{warmth:.2f}, Sat:{saturation:.2f}")
    
    # Detect champagne gold (very bright, slight warmth)
    if brightness > 0.80 and 0.02 < warmth < 0.06:
        return 'champagne_gold'
    # Rose gold (pink tones)
    elif warmth > 0.08 and r > g > b and saturation > 0.05:
        return 'rose_gold'
    # Yellow gold (warm tones)
    elif warmth > 0.05 and saturation > 0.03:
        return 'yellow_gold'
    # White gold (cool/neutral)
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
        
        # Apply wedding ring enhancement
        enhanced = enhance_wedding_ring_premium(img_array, metal_type)
        
        # Create thumbnail with max zoom - no padding
        thumbnail = create_thumbnail_max_zoom(enhanced)
        
        # Convert to base64
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        enhanced_buffer = BytesIO()
        enhanced_pil.save(enhanced_buffer, format='JPEG', quality=98, optimize=True)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        thumbnail_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        thumbnail_buffer = BytesIO()
        thumbnail_pil.save(thumbnail_buffer, format='JPEG', quality=95, optimize=True)
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')
        
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
                    "version": "v64.0"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v64.0"
            }
        }

# RunPod endpoint
runpod.serverless.start({"handler": handler})
