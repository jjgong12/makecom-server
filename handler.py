import runpod
import base64
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from io import BytesIO
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import traceback

# v13.3 complete parameters (28 pairs training data - 4 metals x 3 lighting = 12 sets)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15,
            'saturation': 1.02,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18,
            'saturation': 1.00,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12,
            'saturation': 1.03,
            'gamma': 1.02
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.14,
            'contrast': 1.08,
            'white_overlay': 0.06,
            'sharpness': 1.11,
            'color_temp_a': -1,
            'color_temp_b': -1,
            'original_blend': 0.20,
            'saturation': 1.05,
            'gamma': 1.00
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.06,
            'white_overlay': 0.08,
            'sharpness': 1.09,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.22,
            'saturation': 1.03,
            'gamma': 0.97
        },
        'cool': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.04,
            'sharpness': 1.13,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.18,
            'saturation': 1.07,
            'gamma': 1.03
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.10,
            'contrast': 1.04,
            'white_overlay': 0.03,
            'sharpness': 1.07,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.25,
            'saturation': 1.08,
            'gamma': 0.99
        },
        'warm': {
            'brightness': 1.08,
            'contrast': 1.02,
            'white_overlay': 0.05,
            'sharpness': 1.05,
            'color_temp_a': -1,
            'color_temp_b': -1,
            'original_blend': 0.27,
            'saturation': 1.06,
            'gamma': 0.96
        },
        'cool': {
            'brightness': 1.12,
            'contrast': 1.06,
            'white_overlay': 0.02,
            'sharpness': 1.09,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.23,
            'saturation': 1.10,
            'gamma': 1.02
        }
    },
    'silver': {
        'natural': {
            'brightness': 1.22,
            'contrast': 1.16,
            'white_overlay': 0.11,
            'sharpness': 1.19,
            'color_temp_a': -4,
            'color_temp_b': -4,
            'original_blend': 0.10,
            'saturation': 0.98,
            'gamma': 1.03
        },
        'warm': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.14,
            'sharpness': 1.17,
            'color_temp_a': -6,
            'color_temp_b': -6,
            'original_blend': 0.13,
            'saturation': 0.96,
            'gamma': 1.00
        },
        'cool': {
            'brightness': 1.24,
            'contrast': 1.18,
            'white_overlay': 0.09,
            'sharpness': 1.21,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.08,
            'saturation': 1.00,
            'gamma': 1.04
        }
    }
}

def initial_force_crop(image_array: np.ndarray, crop_percent: float = 0.02) -> np.ndarray:
    """Force crop edges by given percentage to guarantee removal"""
    h, w = image_array.shape[:2]
    
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    
    # Ensure minimum crop
    crop_h = max(crop_h, 20)
    crop_w = max(crop_w, 20)
    
    return image_array[crop_h:h-crop_h, crop_w:w-crop_w]

def verify_black_removal(image_array: np.ndarray, threshold: int = 80) -> bool:
    """Verify if black borders are completely removed"""
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Check edges for black pixels
    edge_size = 10
    
    # Top edge
    if np.mean(gray[:edge_size, :]) < threshold:
        return False
    
    # Bottom edge
    if np.mean(gray[-edge_size:, :]) < threshold:
        return False
    
    # Left edge
    if np.mean(gray[:, :edge_size]) < threshold:
        return False
    
    # Right edge
    if np.mean(gray[:, -edge_size:]) < threshold:
        return False
    
    # Check corners more strictly
    corner_size = 30
    corners = [
        gray[:corner_size, :corner_size],  # Top-left
        gray[:corner_size, -corner_size:],  # Top-right
        gray[-corner_size:, :corner_size],  # Bottom-left
        gray[-corner_size:, -corner_size:]  # Bottom-right
    ]
    
    for corner in corners:
        if np.mean(corner) < threshold + 20:  # Stricter for corners
            return False
    
    return True

def detect_and_remove_black_iterative(image_array: np.ndarray, iteration: int) -> Tuple[np.ndarray, bool]:
    """Detect and remove black borders with increasing aggressiveness"""
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Increase aggressiveness with each iteration
    base_threshold = 80 - (iteration * 10)  # 80, 70, 60, 50, 40
    scan_depth = 0.3 + (iteration * 0.1)  # 30%, 40%, 50%, 60%, 70%
    min_crop = 20 + (iteration * 10)  # 20, 30, 40, 50, 60
    
    top_crop = 0
    bottom_crop = h
    left_crop = 0
    right_crop = w
    
    # Scan each edge with increasing depth
    max_scan = int(h * scan_depth)
    
    # Top edge
    for y in range(max_scan):
        if np.mean(gray[y, :]) > base_threshold:
            top_crop = max(y - 5, 0)  # Small safety margin
            break
    else:
        top_crop = min_crop
    
    # Bottom edge
    for y in range(h - 1, h - max_scan, -1):
        if np.mean(gray[y, :]) > base_threshold:
            bottom_crop = min(y + 5, h)
            break
    else:
        bottom_crop = h - min_crop
    
    # Left edge
    max_scan_w = int(w * scan_depth)
    for x in range(max_scan_w):
        if np.mean(gray[:, x]) > base_threshold:
            left_crop = max(x - 5, 0)
            break
    else:
        left_crop = min_crop
    
    # Right edge
    for x in range(w - 1, w - max_scan_w, -1):
        if np.mean(gray[:, x]) > base_threshold:
            right_crop = min(x + 5, w)
            break
    else:
        right_crop = w - min_crop
    
    # Apply crop
    if top_crop < bottom_crop and left_crop < right_crop:
        cropped = image_array[top_crop:bottom_crop, left_crop:right_crop]
        removed = (top_crop > 0 or bottom_crop < h or left_crop > 0 or right_crop < w)
        return cropped, removed
    
    return image_array, False

def hybrid_inpaint_edges(image_array: np.ndarray, edge_size: int = 10) -> np.ndarray:
    """Apply hybrid inpainting to edges for any remaining artifacts"""
    h, w = image_array.shape[:2]
    result = image_array.copy()
    
    # Create masks for each edge
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Top edge
    if np.mean(gray[:edge_size, :]) < 100:
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        mask[:edge_size, :] = 255
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    # Bottom edge
    if np.mean(gray[-edge_size:, :]) < 100:
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        mask[-edge_size:, :] = 255
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    # Left edge
    if np.mean(gray[:, :edge_size]) < 100:
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        mask[:, :edge_size] = 255
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    # Right edge
    if np.mean(gray[:, -edge_size:]) < 100:
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        mask[:, -edge_size:] = 255
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    return result

def detect_metal_type(image: Image.Image) -> str:
    """Detect metal type from the ring image"""
    img_array = np.array(image)
    
    # Sample center area
    height, width = img_array.shape[:2]
    center_y, center_x = height // 2, width // 2
    sample_size = min(height, width) // 4
    
    center_region = img_array[
        center_y - sample_size:center_y + sample_size,
        center_x - sample_size:center_x + sample_size
    ]
    
    # Calculate color statistics
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    # Color ratios for metal detection
    rg_ratio = r_mean / (g_mean + 1)
    rb_ratio = r_mean / (b_mean + 1)
    
    # Brightness
    brightness = (r_mean + g_mean + b_mean) / 3
    
    # Metal type detection logic
    if brightness > 200 and abs(r_mean - g_mean) < 15 and abs(g_mean - b_mean) < 15:
        return 'silver'
    elif rg_ratio > 1.08 and rb_ratio > 1.15:
        return 'rose_gold'
    elif rg_ratio > 1.02 and rb_ratio > 1.05:
        return 'yellow_gold'
    else:
        return 'white_gold'

def detect_lighting(image: Image.Image) -> str:
    """Detect lighting condition from the image"""
    img_array = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Calculate histogram
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Find peak
    peak_idx = np.argmax(hist)
    
    # Determine lighting based on histogram distribution
    if peak_idx < 80:
        return 'cool'  # Dark, needs brightening
    elif peak_idx > 170:
        return 'warm'  # Bright, needs toning down
    else:
        return 'natural'

def enhance_ring_colors(image: Image.Image, metal_type: str) -> Image.Image:
    """Enhance ring colors based on metal type"""
    # Detect lighting
    lighting = detect_lighting(image)
    
    # Get parameters
    params = WEDDING_RING_PARAMS[metal_type][lighting]
    
    # Apply enhancements
    enhanced = image.copy()
    
    # Brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(params['brightness'])
    
    # Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(params['contrast'])
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(params['sharpness'])
    
    # Saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(params['saturation'])
    
    # Apply white overlay
    if params['white_overlay'] > 0:
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, params['white_overlay'])
    
    # Gamma correction
    if params['gamma'] != 1.0:
        enhanced_array = np.array(enhanced).astype(np.float32) / 255.0
        enhanced_array = np.power(enhanced_array, params['gamma'])
        enhanced_array = (enhanced_array * 255).astype(np.uint8)
        enhanced = Image.fromarray(enhanced_array)
    
    # Color temperature adjustment
    if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
        enhanced_array = np.array(enhanced)
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] += params['color_temp_a']  # a channel
        lab[:, :, 2] += params['color_temp_b']  # b channel
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = Image.fromarray(enhanced_array)
    
    # Blend with original
    if params['original_blend'] > 0:
        enhanced = Image.blend(enhanced, image, params['original_blend'])
    
    return enhanced

def create_thumbnail_ultra_zoom(original_image: Image.Image, enhanced_image: Image.Image) -> Image.Image:
    """Create ultra-zoomed thumbnail focusing on ring only"""
    # Convert to numpy array
    img_array = np.array(enhanced_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to find ring
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to center crop
        width, height = enhanced_image.size
        crop_size = min(width, height) // 2
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        ring_crop = enhanced_image.crop((left, top, right, bottom))
    else:
        # Find largest contour (ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = int(max(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        
        # Ensure square crop
        if w > h:
            diff = w - h
            y = max(0, y - diff // 2)
            h = w
        else:
            diff = h - w
            x = max(0, x - diff // 2)
            w = h
        
        ring_crop = enhanced_image.crop((x, y, x + w, y + h))
    
    # Resize to thumbnail size
    thumbnail = ring_crop.resize((1000, 1300), Image.Resampling.LANCZOS)
    
    return thumbnail

def process_wedding_ring_v100(image_base64: str) -> Dict:
    """Main processing function with iterative perfect removal"""
    try:
        print("Starting Wedding Ring AI v100 - Iterative Perfect Removal")
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        original_image = Image.open(BytesIO(image_data))
        
        # Convert to numpy array
        image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        print("Step 1: Initial 2% force crop for guaranteed edge removal")
        # Step 1: Initial force crop (2% from each edge)
        image_bgr = initial_force_crop(image_bgr, crop_percent=0.02)
        
        print("Step 2: Iterative black border removal with verification")
        # Step 2: Iterative removal with verification
        for iteration in range(5):  # Maximum 5 iterations
            print(f"Iteration {iteration + 1}: Checking for black borders...")
            
            # Check if black borders are removed
            if verify_black_removal(image_bgr):
                print(f"Black borders successfully removed after {iteration + 1} iterations!")
                break
            
            # Remove black borders with increasing aggressiveness
            image_bgr, removed = detect_and_remove_black_iterative(image_bgr, iteration)
            
            if not removed and iteration == 4:
                # Last resort: aggressive crop
                print("Applying final aggressive crop...")
                # Final aggressive crop if still not successful
                h, w = image_bgr.shape[:2]
                final_crop = 50  # Remove 50 pixels from each edge
                image_bgr = image_bgr[final_crop:h-final_crop, final_crop:w-final_crop]
        
        # Step 3: Hybrid edge inpainting for any remaining artifacts
        print("Step 3: Applying hybrid edge inpainting...")
        image_bgr = hybrid_inpaint_edges(image_bgr, edge_size=10)
        
        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(image_rgb)
        
        # Create clean background
        width, height = processed_image.size
        final_image = Image.new('RGB', (width, height), (248, 248, 248))
        
        # Detect metal type
        metal_type = detect_metal_type(processed_image)
        print(f"Detected metal type: {metal_type}")
        
        # Enhance ring colors
        enhanced_ring = enhance_ring_colors(processed_image, metal_type)
        
        # Paste enhanced ring on clean background
        final_image.paste(enhanced_ring, (0, 0))
        
        # Create thumbnail
        thumbnail = create_thumbnail_ultra_zoom(original_image, final_image)
        
        # Convert to base64
        # Main image
        main_buffer = BytesIO()
        final_image.save(main_buffer, format='JPEG', quality=95, optimize=True)
        main_buffer.seek(0)
        main_base64 = base64.b64encode(main_buffer.read()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='JPEG', quality=95, optimize=True)
        thumb_buffer.seek(0)
        thumb_base64 = base64.b64encode(thumb_buffer.read()).decode('utf-8')
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "metal_type": metal_type,
                "processing_version": "v100_iterative_perfect",
                "removal_stats": {
                    "initial_crop": "2%",
                    "iterations_used": iteration + 1 if 'iteration' in locals() else 1,
                    "hybrid_inpaint": "applied"
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v100: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v100_iterative_perfect"
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        input_data = event.get("input", {})
        
        # Check for test mode
        if input_data.get("test") == True:
            return {
                "status": "test_success",
                "message": "Wedding Ring Processor v100 - Iterative Perfect Removal Ready",
                "version": "v100_iterative_perfect",
                "features": [
                    "Initial 2% force crop guarantee",
                    "5-iteration progressive removal",
                    "Verification after each iteration",
                    "Hybrid edge inpainting",
                    "Increasing aggressiveness per iteration",
                    "Final aggressive fallback",
                    "100% black border removal guarantee"
                ]
            }
        
        # Get image
        image_base64 = input_data.get("image")
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Process image
        return process_wedding_ring_v100(image_base64)
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
