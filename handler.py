import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import os
import traceback
from typing import Dict, Any, List, Optional, Tuple

def handler(event):
    """Wedding Ring AI v81 - Multi-Stage Black Border Removal with Fixed Handler"""
    try:
        # Get image from event - CORRECT STRUCTURE
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
        
        # Multi-stage black border removal
        cleaned_image = multi_stage_black_removal(image)
        
        # Detect metal type
        metal_type = detect_metal_type(cleaned_image)
        
        # Apply powerful enhancement
        enhanced_image = enhance_wedding_ring(cleaned_image, metal_type)
        
        # Create perfect thumbnail
        thumbnail = create_thumbnail(enhanced_image)
        
        # Convert to base64 (WITHOUT padding for Make.com)
        _, main_buffer = cv2.imencode('.jpg', enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding
        
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding
        
        # Return with CORRECT output structure
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "original_size": image.shape[:2],
                    "enhanced_size": enhanced_image.shape[:2],
                    "thumbnail_size": thumbnail.shape[:2],
                    "status": "success",
                    "version": "v81_fixed"
                }
            }
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {
            "output": {
                "error": error_msg,
                "status": "error"
            }
        }

def multi_stage_black_removal(img: np.ndarray) -> np.ndarray:
    """Multi-stage black border detection and removal"""
    result = img.copy()
    h, w = result.shape[:2]
    
    # Stage 1: Scan in 5-pixel increments
    print("Stage 1: 5-pixel increment scan...")
    for edge in ['top', 'bottom', 'left', 'right']:
        for depth in range(5, min(150, h//4 if edge in ['top','bottom'] else w//4), 5):
            if check_and_remove_black_strip(result, edge, depth, threshold=30):
                result = crop_edge(result, edge, depth)
                print(f"  Removed {depth}px from {edge}")
                break
    
    # Stage 2: Scan in 10-pixel increments with different threshold
    print("Stage 2: 10-pixel increment scan...")
    h, w = result.shape[:2]
    for edge in ['top', 'bottom', 'left', 'right']:
        for depth in range(10, min(100, h//4 if edge in ['top','bottom'] else w//4), 10):
            if check_and_remove_black_strip(result, edge, depth, threshold=40):
                result = crop_edge(result, edge, depth)
                print(f"  Removed {depth}px from {edge}")
                break
    
    # Stage 3: Pixel-by-pixel scan for remaining black edges
    print("Stage 3: Pixel-by-pixel scan...")
    for _ in range(3):  # Multiple passes
        old_shape = result.shape
        result = remove_black_pixels_aggressive(result)
        if result.shape == old_shape:
            break
        print(f"  Trimmed to {result.shape}")
    
    # Stage 4: Corner analysis
    print("Stage 4: Corner analysis...")
    result = remove_black_corners(result)
    
    # Stage 5: Final cleanup with multiple thresholds
    print("Stage 5: Final cleanup...")
    for threshold in [20, 30, 40, 50]:
        result = final_black_cleanup(result, threshold)
    
    # Ensure minimum size
    if result.shape[0] < 100 or result.shape[1] < 100:
        print("Warning: Image too small after border removal, using original")
        return img
    
    return result

def check_and_remove_black_strip(img: np.ndarray, edge: str, depth: int, threshold: int) -> bool:
    """Check if edge has black strip"""
    h, w = img.shape[:2]
    
    if edge == 'top':
        strip = img[:depth, :]
    elif edge == 'bottom':
        strip = img[-depth:, :]
    elif edge == 'left':
        strip = img[:, :depth]
    else:  # right
        strip = img[:, -depth:]
    
    # Convert to grayscale for checking
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_strip) < threshold

def crop_edge(img: np.ndarray, edge: str, pixels: int) -> np.ndarray:
    """Crop specified edge by given pixels"""
    h, w = img.shape[:2]
    
    if edge == 'top':
        return img[pixels:, :]
    elif edge == 'bottom':
        return img[:-pixels, :]
    elif edge == 'left':
        return img[:, pixels:]
    else:  # right
        return img[:, :-pixels]

def remove_black_pixels_aggressive(img: np.ndarray) -> np.ndarray:
    """Aggressively remove black pixels from all edges"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Find bounds
    top, bottom, left, right = 0, h, 0, w
    
    # Top
    for i in range(h//3):
        if np.mean(gray[i, :]) > 50:
            top = i
            break
    
    # Bottom
    for i in range(h-1, 2*h//3, -1):
        if np.mean(gray[i, :]) > 50:
            bottom = i + 1
            break
    
    # Left
    for i in range(w//3):
        if np.mean(gray[:, i]) > 50:
            left = i
            break
    
    # Right
    for i in range(w-1, 2*w//3, -1):
        if np.mean(gray[:, i]) > 50:
            right = i + 1
            break
    
    return img[top:bottom, left:right]

def remove_black_corners(img: np.ndarray) -> np.ndarray:
    """Remove black corners"""
    h, w = img.shape[:2]
    corner_size = min(100, h//4, w//4)
    
    # Check each corner
    corners = [
        (0, 0, corner_size, corner_size),  # top-left
        (0, w-corner_size, corner_size, w),  # top-right
        (h-corner_size, 0, h, corner_size),  # bottom-left
        (h-corner_size, w-corner_size, h, w)  # bottom-right
    ]
    
    crop_top = crop_bottom = crop_left = crop_right = 0
    
    for y1, x1, y2, x2 in corners:
        corner = img[y1:y2, x1:x2]
        gray_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_corner) < 30:
            if y1 == 0 and x1 == 0:  # top-left
                crop_top = max(crop_top, corner_size//2)
                crop_left = max(crop_left, corner_size//2)
            elif y1 == 0 and x2 == w:  # top-right
                crop_top = max(crop_top, corner_size//2)
                crop_right = max(crop_right, corner_size//2)
            elif y2 == h and x1 == 0:  # bottom-left
                crop_bottom = max(crop_bottom, corner_size//2)
                crop_left = max(crop_left, corner_size//2)
            else:  # bottom-right
                crop_bottom = max(crop_bottom, corner_size//2)
                crop_right = max(crop_right, corner_size//2)
    
    if crop_top + crop_bottom + crop_left + crop_right > 0:
        return img[crop_top:h-crop_bottom, crop_left:w-crop_right]
    return img

def final_black_cleanup(img: np.ndarray, threshold: int) -> np.ndarray:
    """Final cleanup pass with specific threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask of non-black pixels
    mask = gray > threshold
    
    # Find bounding box of non-black pixels
    coords = np.column_stack(np.where(mask))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return img[y_min:y_max+1, x_min:x_max+1]
    
    return img

def detect_metal_type(image: np.ndarray) -> str:
    """Detect metal type from ring image"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Analyze color channels
    mean_b = np.mean(b)
    
    if mean_b > 135:  # Warm/yellow tones
        return "yellow_gold"
    elif mean_b > 128:  # Slightly warm
        return "rose_gold"
    else:  # Cool tones
        return "white_gold"

def enhance_wedding_ring(image: np.ndarray, metal_type: str) -> np.ndarray:
    """Enhance wedding ring with powerful adjustments"""
    
    # Denoise first
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to LAB for better color manipulation
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Powerful brightness enhancement
    l = cv2.add(l, 50)  # Strong brightness boost
    
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Metal-specific color adjustments
    if metal_type == "yellow_gold":
        b = cv2.add(b, 10)  # Enhance yellow
    elif metal_type == "rose_gold":
        a = cv2.add(a, 5)   # Enhance pink
        b = cv2.add(b, 5)
    else:  # white_gold
        # Make it whiter
        a = cv2.subtract(a, 8)
        b = cv2.subtract(b, 10)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Convert to PIL for final adjustments
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Strong enhancements
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.35)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.35)
    
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)
    
    # Apply sharpening filter twice
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
    # Convert back to OpenCV format
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return enhanced

def create_thumbnail(image: np.ndarray) -> np.ndarray:
    """Create 1000x1300 thumbnail with ring filling 80% of frame"""
    target_size = (1000, 1300)
    
    # Find ring contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop ring area
        ring_crop = image[y:y+h, x:x+w]
    else:
        # Fallback: use center crop
        h, w = image.shape[:2]
        crop_size = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        ring_crop = image[center_y-crop_size:center_y+crop_size, 
                         center_x-crop_size:center_x+crop_size]
    
    # Calculate scale to fill 80% of target
    scale_w = (target_size[0] * 0.8) / ring_crop.shape[1]
    scale_h = (target_size[1] * 0.8) / ring_crop.shape[0]
    scale = max(scale_w, scale_h) * 1.02  # Slightly larger to ensure fill
    
    new_w = int(ring_crop.shape[1] * scale)
    new_h = int(ring_crop.shape[0] * scale)
    
    # Resize ring
    ring_resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # If oversized, crop to fit
    if new_w > target_size[0] or new_h > target_size[1]:
        # Center crop to target size
        start_x = (new_w - target_size[0]) // 2 if new_w > target_size[0] else 0
        start_y = (new_h - target_size[1]) // 2 if new_h > target_size[1] else 0
        
        ring_resized = ring_resized[start_y:start_y+target_size[1], 
                                    start_x:start_x+target_size[0]]
        
        # Ensure exact size
        if ring_resized.shape[:2] != (target_size[1], target_size[0]):
            ring_resized = cv2.resize(ring_resized, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return ring_resized
    else:
        # Create white background and center the ring
        background = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
        
        # Center position
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        # Place ring on background
        background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = ring_resized
        
        return background

# RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
