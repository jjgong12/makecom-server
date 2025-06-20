import os
import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
import numpy as np
import cv2
import logging
from typing import Dict, Any, Tuple, Optional, List
import time
import json
import replicate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_rectangular_masking(image: np.ndarray) -> Dict[str, Any]:
    """Detect rectangular black masking with progressive approach and validation"""
    height, width = image.shape[:2]
    logger.info(f"Starting rectangular masking detection on {width}x{height} image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Try multiple detection strategies
    detection_strategies = [
        # Strategy 1: Progressive threshold with different scan depths
        {"method": "threshold", "thresholds": [10, 20, 30, 40, 50, 60, 70, 80], 
         "scan_depths": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]},
        # Strategy 2: Gradient-based edge detection
        {"method": "gradient"},
        # Strategy 3: Morphological operations
        {"method": "morphology"}
    ]
    
    best_mask = None
    best_score = 0
    
    for strategy in detection_strategies:
        if strategy["method"] == "threshold":
            for threshold in strategy["thresholds"]:
                for scan_depth in strategy["scan_depths"]:
                    mask_info = detect_with_threshold(gray, threshold, scan_depth, width, height)
                    if mask_info and mask_info["score"] > best_score:
                        if validate_rectangular_mask(mask_info, width, height):
                            best_mask = mask_info
                            best_score = mask_info["score"]
                            logger.info(f"Found valid mask with threshold {threshold}, depth {scan_depth}")
                            
        elif strategy["method"] == "gradient":
            mask_info = detect_with_gradient(gray, width, height)
            if mask_info and mask_info["score"] > best_score:
                if validate_rectangular_mask(mask_info, width, height):
                    best_mask = mask_info
                    best_score = mask_info["score"]
                    
        elif strategy["method"] == "morphology":
            mask_info = detect_with_morphology(gray, width, height)
            if mask_info and mask_info["score"] > best_score:
                if validate_rectangular_mask(mask_info, width, height):
                    best_mask = mask_info
                    best_score = mask_info["score"]
    
    if best_mask:
        logger.info(f"Best mask found: {best_mask['type']} at boundaries "
                   f"[{best_mask['left']}, {best_mask['top']}, {best_mask['right']}, {best_mask['bottom']}]")
        return best_mask
    
    return {
        'has_masking': False,
        'top': 0,
        'bottom': height,
        'left': 0,
        'right': width,
        'type': None
    }

def detect_with_threshold(gray: np.ndarray, threshold: int, scan_depth: float, 
                         width: int, height: int) -> Optional[Dict[str, Any]]:
    """Detect masking using threshold method"""
    scan_pixels = int(min(width, height) * scan_depth)
    
    # Find black regions from all edges
    edges = {'top': -1, 'bottom': -1, 'left': -1, 'right': -1}
    
    # Top edge scan
    for y in range(min(scan_pixels, height//3)):
        row = gray[y, width//4:3*width//4]  # Check center portion
        if np.mean(row) < threshold and np.max(row) < threshold + 30:
            edges['top'] = y
        else:
            break
    
    # Bottom edge scan
    for y in range(min(scan_pixels, height//3)):
        row = gray[height-1-y, width//4:3*width//4]
        if np.mean(row) < threshold and np.max(row) < threshold + 30:
            edges['bottom'] = height-1-y
        else:
            break
    
    # Left edge scan
    for x in range(min(scan_pixels, width//3)):
        col = gray[height//4:3*height//4, x]
        if np.mean(col) < threshold and np.max(col) < threshold + 30:
            edges['left'] = x
        else:
            break
    
    # Right edge scan
    for x in range(min(scan_pixels, width//3)):
        col = gray[height//4:3*height//4, width-1-x]
        if np.mean(col) < threshold and np.max(col) < threshold + 30:
            edges['right'] = width-1-x
        else:
            break
    
    # Count detected edges
    detected_edges = sum(1 for v in edges.values() if v >= 0)
    
    if detected_edges >= 2:
        # Determine type and boundaries
        mask_info = {
            'has_masking': True,
            'top': edges['top'] if edges['top'] >= 0 else 0,
            'bottom': edges['bottom'] if edges['bottom'] >= 0 else height,
            'left': edges['left'] if edges['left'] >= 0 else 0,
            'right': edges['right'] if edges['right'] >= 0 else width,
            'score': detected_edges * 25 + (100 - threshold) / 2,
            'type': None
        }
        
        # Determine masking type
        if detected_edges == 4:
            mask_info['type'] = 'full_frame'
        elif edges['top'] >= 0 and edges['bottom'] >= 0:
            mask_info['type'] = 'horizontal_bars'
        elif edges['left'] >= 0 and edges['right'] >= 0:
            mask_info['type'] = 'vertical_bars'
        else:
            mask_info['type'] = 'partial'
        
        return mask_info
    
    return None

def detect_with_gradient(gray: np.ndarray, width: int, height: int) -> Optional[Dict[str, Any]]:
    """Detect masking using gradient analysis"""
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Look for strong horizontal and vertical edges
    edges = {'top': -1, 'bottom': -1, 'left': -1, 'right': -1}
    
    # Horizontal edges (top/bottom)
    for y in range(height//3):
        if np.mean(grad_mag[y, :]) > 50:
            if edges['top'] == -1:
                edges['top'] = y
        if np.mean(grad_mag[height-1-y, :]) > 50:
            if edges['bottom'] == -1:
                edges['bottom'] = height-1-y
    
    # Vertical edges (left/right)
    for x in range(width//3):
        if np.mean(grad_mag[:, x]) > 50:
            if edges['left'] == -1:
                edges['left'] = x
        if np.mean(grad_mag[:, width-1-x]) > 50:
            if edges['right'] == -1:
                edges['right'] = width-1-x
    
    detected_edges = sum(1 for v in edges.values() if v >= 0)
    
    if detected_edges >= 2:
        return {
            'has_masking': True,
            'top': edges['top'] if edges['top'] >= 0 else 0,
            'bottom': edges['bottom'] if edges['bottom'] >= 0 else height,
            'left': edges['left'] if edges['left'] >= 0 else 0,
            'right': edges['right'] if edges['right'] >= 0 else width,
            'score': detected_edges * 20,
            'type': 'gradient_detected'
        }
    
    return None

def detect_with_morphology(gray: np.ndarray, width: int, height: int) -> Optional[Dict[str, Any]]:
    """Detect masking using morphological operations"""
    # Apply morphological operations to find rectangular structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Threshold to find dark regions
    _, binary = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest rectangular contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Check if it touches edges
        touches_edge = (x <= 10 or y <= 10 or 
                       x + w >= width - 10 or y + h >= height - 10)
        
        if touches_edge and w * h > width * height * 0.1:
            return {
                'has_masking': True,
                'top': y,
                'bottom': y + h,
                'left': x,
                'right': x + w,
                'score': 60,
                'type': 'morphology_detected'
            }
    
    return None

def validate_rectangular_mask(mask_info: Dict[str, Any], width: int, height: int) -> bool:
    """Validate that the detected mask is truly rectangular"""
    if not mask_info or not mask_info.get('has_masking'):
        return False
    
    # Check aspect ratio is reasonable
    mask_width = mask_info['right'] - mask_info['left']
    mask_height = mask_info['bottom'] - mask_info['top']
    
    if mask_width <= 0 or mask_height <= 0:
        return False
    
    # The masked area should be smaller than the full image
    if mask_width >= width * 0.98 or mask_height >= height * 0.98:
        return False
    
    # The mask should be reasonably rectangular (not too thin)
    aspect_ratio = max(mask_width, mask_height) / min(mask_width, mask_height)
    if aspect_ratio > 20:  # Too elongated
        return False
    
    return True

def remove_masking_with_ai(image: Image.Image, mask_info: Dict[str, Any]) -> Image.Image:
    """Remove masking and fill with clean background"""
    if not mask_info.get('has_masking'):
        return image
    
    logger.info(f"Removing {mask_info['type']} masking")
    
    try:
        # Use Replicate API if available
        if os.environ.get("REPLICATE_API_TOKEN"):
            return remove_with_replicate_api(image, mask_info)
    except Exception as e:
        logger.error(f"Replicate API failed: {e}")
    
    # Fallback: Crop to content area
    return image.crop((
        mask_info['left'],
        mask_info['top'],
        mask_info['right'],
        mask_info['bottom']
    ))

def remove_with_replicate_api(image: Image.Image, mask_info: Dict[str, Any]) -> Image.Image:
    """Use Replicate API for high-quality inpainting"""
    # Create mask
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Mark areas to inpaint (white = inpaint)
    if mask_info['top'] > 0:
        draw.rectangle([0, 0, image.width, mask_info['top']], fill=255)
    if mask_info['bottom'] < image.height:
        draw.rectangle([0, mask_info['bottom'], image.width, image.height], fill=255)
    if mask_info['left'] > 0:
        draw.rectangle([0, 0, mask_info['left'], image.height], fill=255)
    if mask_info['right'] < image.width:
        draw.rectangle([mask_info['right'], 0, image.width, image.height], fill=255)
    
    # Convert to base64
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    mask_buffer = BytesIO()
    mask.save(mask_buffer, format="PNG")
    mask_buffer.seek(0)
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
    
    # Call Replicate
    api = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
    
    output = api.run(
        "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
        input={
            "image": f"data:image/png;base64,{img_base64}",
            "mask": f"data:image/png;base64,{mask_base64}",
            "prompt": "clean white seamless background, product photography",
            "negative_prompt": "black, dark, shadows, borders",
            "num_inference_steps": 25,
            "guidance_scale": 7.5
        }
    )
    
    if output and len(output) > 0:
        response = requests.get(output[0])
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    
    raise Exception("Failed to get result from Replicate")

def detect_metal_type(image: Image.Image) -> str:
    """Detect metal type with champagne gold as default"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Focus on center region where ring is likely to be
    center_size = min(h, w) // 2
    center_y, center_x = h // 2, w // 2
    
    roi = img_array[
        max(0, center_y - center_size//2):min(h, center_y + center_size//2),
        max(0, center_x - center_size//2):min(w, center_x + center_size//2)
    ]
    
    # Calculate average colors
    avg_b = np.mean(roi[:,:,0])
    avg_g = np.mean(roi[:,:,1])
    avg_r = np.mean(roi[:,:,2])
    
    brightness = (avg_r + avg_g + avg_b) / 3
    
    logger.info(f"Metal detection - R:{avg_r:.1f} G:{avg_g:.1f} B:{avg_b:.1f} Brightness:{brightness:.1f}")
    
    # Detection logic
    if avg_r > avg_g + 15 and avg_r > avg_b + 15:
        return "rose_gold"
    elif avg_r > avg_b + 10 and avg_g > avg_b + 5 and brightness < 180:
        return "yellow_gold"
    elif avg_b >= avg_r and brightness > 190:
        return "white_gold"
    else:
        return "champagne_gold"

def apply_light_enhancement(image: Image.Image, metal_type: str) -> Image.Image:
    """Apply light enhancement like in reference image 4"""
    # Light enhancement parameters
    params = {
        "white_gold": {"brightness": 1.15, "contrast": 1.10, "saturation": 0.85},
        "rose_gold": {"brightness": 1.12, "contrast": 1.08, "saturation": 0.90},
        "yellow_gold": {"brightness": 1.10, "contrast": 1.08, "saturation": 0.88},
        "champagne_gold": {"brightness": 1.18, "contrast": 1.12, "saturation": 0.80}
    }
    
    p = params[metal_type]
    enhanced = image.copy()
    
    # Simple light enhancement
    # 1. Brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(p["brightness"])
    
    # 2. Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(p["contrast"])
    
    # 3. Saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(p["saturation"])
    
    # 4. Add slight white overlay for clean look
    white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
    enhanced = Image.blend(enhanced, white_layer, 0.05)
    
    return enhanced

def apply_detail_enhancement(image: Image.Image, metal_type: str) -> Image.Image:
    """Apply strong detail enhancement for ring area only"""
    # Strong enhancement parameters for details
    params = {
        "white_gold": {
            "brightness": 1.25, "contrast": 1.20, "saturation": 0.70,
            "sharpness": 1.40, "white_overlay": 0.15
        },
        "rose_gold": {
            "brightness": 1.20, "contrast": 1.18, "saturation": 0.85,
            "sharpness": 1.35, "white_overlay": 0.10
        },
        "yellow_gold": {
            "brightness": 1.18, "contrast": 1.15, "saturation": 0.80,
            "sharpness": 1.35, "white_overlay": 0.12
        },
        "champagne_gold": {
            "brightness": 1.30, "contrast": 1.25, "saturation": 0.65,
            "sharpness": 1.45, "white_overlay": 0.18
        }
    }
    
    p = params[metal_type]
    enhanced = image.copy()
    
    # Apply full enhancement process
    # 1. Denoise
    enhanced = enhanced.filter(ImageFilter.MedianFilter(3))
    
    # 2. Brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(p["brightness"])
    
    # 3. Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(p["contrast"])
    
    # 4. Sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(p["sharpness"])
    
    # 5. Saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(p["saturation"])
    
    # 6. White overlay
    white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
    enhanced = Image.blend(enhanced, white_layer, p["white_overlay"])
    
    # 7. CLAHE for detail
    img_array = np.array(enhanced)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    enhanced = Image.fromarray(img_array)
    
    return enhanced

def create_thumbnail_from_masked(image: Image.Image, mask_info: Dict[str, Any]) -> Image.Image:
    """Create thumbnail focusing on content within mask"""
    if mask_info.get('has_masking'):
        # Crop to masked area first
        cropped = image.crop((
            mask_info['left'],
            mask_info['top'],
            mask_info['right'],
            mask_info['bottom']
        ))
    else:
        cropped = image
    
    # Detect ring in cropped image
    img_array = np.array(cropped)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Find ring using edge detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add 5% padding
        padding = int(max(w, h) * 0.05)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, y + 2 * padding)
        
        ring_crop = Image.fromarray(img_array[y:y+h, x:x+w])
    else:
        ring_crop = cropped
    
    # Apply detail enhancement to ring
    metal_type = detect_metal_type(ring_crop)
    enhanced_ring = apply_detail_enhancement(ring_crop, metal_type)
    
    # Create 1000x1300 thumbnail
    target_size = (980, 1274)  # 98% of 1000x1300
    scale = min(target_size[0] / enhanced_ring.width, target_size[1] / enhanced_ring.height)
    
    new_size = (int(enhanced_ring.width * scale), int(enhanced_ring.height * scale))
    resized = enhanced_ring.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create final thumbnail
    thumbnail = Image.new('RGB', (1000, 1300), (255, 255, 255))
    offset = ((1000 - new_size[0]) // 2, (1300 - new_size[1]) // 2)
    thumbnail.paste(resized, offset)
    
    return thumbnail

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function"""
    start_time = time.time()
    
    try:
        # Extract input
        job_input = event.get("input", {})
        
        # Get image from various possible locations
        image_base64 = None
        if "image" in job_input:
            image_base64 = job_input["image"]
        elif "image_base64" in job_input:
            image_base64 = job_input["image_base64"]
        elif "data" in job_input and isinstance(job_input["data"], dict):
            image_base64 = job_input["data"].get("image") or job_input["data"].get("image_base64")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Clean and decode
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        original_image = Image.open(BytesIO(image_data))
        
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        logger.info(f"Processing image: {original_image.size}")
        
        # Step 1: Detect rectangular masking
        img_array = np.array(original_image)
        mask_info = detect_rectangular_masking(img_array)
        
        # Step 2: Process based on masking
        if mask_info['has_masking']:
            logger.info(f"Processing with {mask_info['type']} masking")
            
            # Extract content area for metal detection
            content_area = original_image.crop((
                mask_info['left'],
                mask_info['top'],
                mask_info['right'],
                mask_info['bottom']
            ))
            
            # Detect metal type from content
            metal_type = detect_metal_type(content_area)
            
            # Remove masking and get clean image
            clean_image = remove_masking_with_ai(original_image, mask_info)
            
            # Apply light enhancement to whole image
            enhanced_image = apply_light_enhancement(clean_image, metal_type)
            
            # Create thumbnail with detail enhancement
            thumbnail = create_thumbnail_from_masked(original_image, mask_info)
            
        else:
            logger.info("No masking detected, processing normally")
            
            # Detect metal type
            metal_type = detect_metal_type(original_image)
            
            # Apply light enhancement
            enhanced_image = apply_light_enhancement(original_image, metal_type)
            
            # Create thumbnail
            thumbnail = create_thumbnail_from_masked(enhanced_image, mask_info)
        
        # Convert to base64
        # Enhanced image
        enhanced_buffer = BytesIO()
        enhanced_image.save(enhanced_buffer, format="PNG", quality=95)
        enhanced_buffer.seek(0)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode()
        
        # Thumbnail
        thumbnail_buffer = BytesIO()
        thumbnail.save(thumbnail_buffer, format="PNG", quality=95)
        thumbnail_buffer.seek(0)
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "metal_type": metal_type,
                "original_size": list(original_image.size),
                "processing_time": round(processing_time, 2),
                "masking_detected": mask_info['has_masking'],
                "masking_type": mask_info.get('type', 'none'),
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "processing_time": round(time.time() - start_time, 2)
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
