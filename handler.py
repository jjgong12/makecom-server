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

def find_black_masking_progressive(image: np.ndarray) -> Dict[str, Any]:
    """Find black rectangular masking with progressive detection for high-res images"""
    height, width = image.shape[:2]
    logger.info(f"Starting progressive masking detection on {width}x{height} image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Progressive detection parameters
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 2% to 30% of image
    threshold_values = [20, 30, 40, 50, 60, 70, 80]
    
    found_masking = False
    masking_info = {
        'has_masking': False,
        'top': 0,
        'bottom': height,
        'left': 0,
        'right': width,
        'type': None,
        'thickness': 0
    }
    
    for scan_pct, threshold in zip(scan_percentages, threshold_values):
        scan_pixels = int(min(width, height) * scan_pct)
        logger.info(f"Scanning with {scan_pct*100}% ({scan_pixels}px) and threshold {threshold}")
        
        # Check all four edges
        edges_black = {
            'top': False,
            'bottom': False,
            'left': False,
            'right': False
        }
        
        edge_positions = {
            'top': 0,
            'bottom': height,
            'left': 0,
            'right': width
        }
        
        # Scan from edges inward
        # Top edge
        for y in range(min(scan_pixels, height//2)):
            row = gray[y, :]
            if np.mean(row) < threshold and np.std(row) < 20:
                edges_black['top'] = True
                edge_positions['top'] = y
            else:
                if edges_black['top']:
                    edge_positions['top'] = y
                    break
        
        # Bottom edge
        for y in range(min(scan_pixels, height//2)):
            row = gray[height-1-y, :]
            if np.mean(row) < threshold and np.std(row) < 20:
                edges_black['bottom'] = True
                edge_positions['bottom'] = height-1-y
            else:
                if edges_black['bottom']:
                    edge_positions['bottom'] = height-1-y
                    break
        
        # Left edge
        for x in range(min(scan_pixels, width//2)):
            col = gray[:, x]
            if np.mean(col) < threshold and np.std(col) < 20:
                edges_black['left'] = True
                edge_positions['left'] = x
            else:
                if edges_black['left']:
                    edge_positions['left'] = x
                    break
        
        # Right edge
        for x in range(min(scan_pixels, width//2)):
            col = gray[:, width-1-x]
            if np.mean(col) < threshold and np.std(col) < 20:
                edges_black['right'] = True
                edge_positions['right'] = width-1-x
            else:
                if edges_black['right']:
                    edge_positions['right'] = width-1-x
                    break
        
        # Check if we found rectangular masking
        black_edges = sum(edges_black.values())
        
        if black_edges >= 2:  # At least 2 edges have black masking
            found_masking = True
            
            # Determine masking type
            if edges_black['top'] and edges_black['bottom'] and edges_black['left'] and edges_black['right']:
                masking_info['type'] = 'full_frame'
            elif edges_black['top'] and edges_black['bottom']:
                masking_info['type'] = 'horizontal_bars'
            elif edges_black['left'] and edges_black['right']:
                masking_info['type'] = 'vertical_bars'
            else:
                masking_info['type'] = 'partial'
            
            # Update positions
            if edges_black['top']:
                masking_info['top'] = edge_positions['top']
            if edges_black['bottom']:
                masking_info['bottom'] = edge_positions['bottom']
            if edges_black['left']:
                masking_info['left'] = edge_positions['left']
            if edges_black['right']:
                masking_info['right'] = edge_positions['right']
            
            masking_info['has_masking'] = True
            masking_info['thickness'] = scan_pixels
            
            logger.info(f"Found {masking_info['type']} masking at {scan_pct*100}% scan")
            logger.info(f"Masking boundaries: top={masking_info['top']}, bottom={masking_info['bottom']}, "
                       f"left={masking_info['left']}, right={masking_info['right']}")
            break
    
    # Verify the detected area is actually masked
    if found_masking:
        # Check if the detected masked area is significant (at least 5% of image)
        masked_area = (masking_info['bottom'] - masking_info['top']) * (masking_info['right'] - masking_info['left'])
        total_area = width * height
        
        if masked_area / total_area < 0.95:  # If less than 95% is masked
            # Double-check by sampling the masked regions
            sample_points = []
            
            # Sample from detected black regions
            if masking_info['top'] > 0:
                for i in range(5):
                    y = masking_info['top'] // 2
                    x = int(width * (i + 1) / 6)
                    sample_points.append((y, x))
            
            if masking_info['bottom'] < height:
                for i in range(5):
                    y = (masking_info['bottom'] + height) // 2
                    x = int(width * (i + 1) / 6)
                    sample_points.append((y, x))
            
            # Verify samples are actually black
            black_samples = 0
            for y, x in sample_points:
                if 0 <= y < height and 0 <= x < width:
                    if gray[y, x] < 80:
                        black_samples += 1
            
            if black_samples < len(sample_points) * 0.6:
                logger.warning("Detected masking failed verification, resetting")
                masking_info['has_masking'] = False
    
    return masking_info

def remove_masking_with_replicate(image: Image.Image, masking_info: Dict[str, Any]) -> Image.Image:
    """Remove masking using Replicate API for high-quality inpainting"""
    if not masking_info['has_masking']:
        return image
    
    logger.info("Removing masking with Replicate API")
    
    # Create mask for inpainting
    mask = Image.new('L', image.size, 0)  # Black = keep, White = inpaint
    draw = ImageDraw.Draw(mask)
    
    # Mark masked areas as white (to be inpainted)
    if masking_info['top'] > 0:
        draw.rectangle([0, 0, image.width, masking_info['top']], fill=255)
    
    if masking_info['bottom'] < image.height:
        draw.rectangle([0, masking_info['bottom'], image.width, image.height], fill=255)
    
    if masking_info['left'] > 0:
        draw.rectangle([0, 0, masking_info['left'], image.height], fill=255)
    
    if masking_info['right'] < image.width:
        draw.rectangle([masking_info['right'], 0, image.width, image.height], fill=255)
    
    try:
        # Convert images to base64
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format="PNG")
        mask_buffer.seek(0)
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Use Replicate API
        api = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
        
        output = api.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image": f"data:image/png;base64,{image_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "prompt": "clean white background, product photography background, seamless white backdrop",
                "negative_prompt": "black bars, black borders, black masking, dark edges",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        
        # Get result
        if output and len(output) > 0:
            response = requests.get(output[0])
            if response.status_code == 200:
                result_image = Image.open(BytesIO(response.content))
                logger.info("Successfully removed masking with Replicate")
                return result_image
        
    except Exception as e:
        logger.error(f"Replicate API error: {str(e)}")
    
    # Fallback to crop if Replicate fails
    logger.info("Falling back to crop method")
    return image.crop((
        masking_info['left'],
        masking_info['top'],
        masking_info['right'],
        masking_info['bottom']
    ))

def detect_metal_type(image: Image.Image) -> str:
    """Detect wedding ring metal type with champagne gold priority"""
    img_array = np.array(image)
    
    # Get center region (more likely to contain the ring)
    h, w = img_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    size = min(h, w) // 3
    center_region = img_array[
        max(0, center_y - size):min(h, center_y + size),
        max(0, center_x - size):min(w, center_x + size)
    ]
    
    # Calculate color statistics
    b, g, r = cv2.split(center_region)
    
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    
    # Color differences
    rg_diff = abs(avg_r - avg_g)
    gb_diff = abs(avg_g - avg_b)
    rb_diff = abs(avg_r - avg_b)
    
    brightness = (avg_r + avg_g + avg_b) / 3
    
    logger.info(f"Color analysis - R:{avg_r:.1f} G:{avg_g:.1f} B:{avg_b:.1f} Brightness:{brightness:.1f}")
    logger.info(f"Differences - RG:{rg_diff:.1f} GB:{gb_diff:.1f} RB:{rb_diff:.1f}")
    
    # Improved detection logic
    # Rose Gold: Distinctly reddish
    if avg_r > avg_g + 15 and avg_r > avg_b + 15 and brightness < 200:
        return "rose_gold"
    
    # Yellow Gold: Warm tones
    elif rg_diff > 8 and gb_diff > 8 and avg_r > avg_b and brightness < 180:
        return "yellow_gold"
    
    # White Gold: Cool, bright, bluish tint
    elif avg_b >= avg_r and brightness > 180 and rb_diff < 10:
        return "white_gold"
    
    # Champagne Gold: Default for ambiguous cases
    else:
        return "champagne_gold"

def apply_enhancement(image: Image.Image, metal_type: str) -> Image.Image:
    """Apply v13.3 enhancement based on metal type"""
    # Enhancement parameters for each metal type
    params = {
        "white_gold": {
            "brightness": 1.25,
            "contrast": 1.15,
            "saturation": 0.7,
            "sharpness": 1.3,
            "white_overlay": 0.12,
            "color_temp": (-2, -2),
            "gamma": 1.1
        },
        "rose_gold": {
            "brightness": 1.18,
            "contrast": 1.20,
            "saturation": 0.85,
            "sharpness": 1.25,
            "white_overlay": 0.08,
            "color_temp": (2, -1),
            "gamma": 1.05
        },
        "yellow_gold": {
            "brightness": 1.20,
            "contrast": 1.18,
            "saturation": 0.80,
            "sharpness": 1.25,
            "white_overlay": 0.10,
            "color_temp": (1, -2),
            "gamma": 1.08
        },
        "champagne_gold": {
            "brightness": 1.30,
            "contrast": 1.22,
            "saturation": 0.65,
            "sharpness": 1.35,
            "white_overlay": 0.15,
            "color_temp": (-3, -3),
            "gamma": 1.12
        }
    }
    
    p = params[metal_type]
    enhanced = image.copy()
    
    # Apply 10-step enhancement process
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
    if p["white_overlay"] > 0:
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, p["white_overlay"])
    
    # 7. Color temperature
    img_array = np.array(enhanced).astype(np.float32)
    img_array[:,:,0] = np.clip(img_array[:,:,0] + p["color_temp"][1], 0, 255)
    img_array[:,:,2] = np.clip(img_array[:,:,2] + p["color_temp"][0], 0, 255)
    enhanced = Image.fromarray(img_array.astype(np.uint8))
    
    # 8. CLAHE
    img_array = np.array(enhanced)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    enhanced = Image.fromarray(img_array)
    
    # 9. Gamma correction
    img_array = np.array(enhanced).astype(np.float32) / 255.0
    img_array = np.power(img_array, p["gamma"])
    img_array = (img_array * 255).astype(np.uint8)
    enhanced = Image.fromarray(img_array)
    
    # 10. Blend with original
    enhanced = Image.blend(image, enhanced, 0.85)
    
    return enhanced

def create_thumbnail(image: Image.Image) -> Image.Image:
    """Create 1000x1300 thumbnail with ring filling 98% of frame"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Find ring boundaries using progressive edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Progressive threshold to find ring
    ring_mask = None
    for thresh_val in range(250, 50, -10):
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if contour is significant (at least 1% of image)
            if area > img_array.shape[0] * img_array.shape[1] * 0.01:
                ring_mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(ring_mask, [largest_contour], -1, 255, -1)
                break
    
    if ring_mask is None:
        # Fallback: use the whole image
        logger.warning("Could not detect ring, using full image")
        x, y, w, h = 0, 0, img_array.shape[1], img_array.shape[0]
    else:
        # Find bounding box of the ring
        coords = np.column_stack(np.where(ring_mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding (2%)
        padding = int(max(y_max - y_min, x_max - x_min) * 0.02)
        x = max(0, x_min - padding)
        y = max(0, y_min - padding)
        w = min(img_array.shape[1] - x, x_max - x_min + 2 * padding)
        h = min(img_array.shape[0] - y, y_max - y_min + 2 * padding)
    
    # Crop to ring area
    cropped = Image.fromarray(img_array[y:y+h, x:x+w])
    
    # Calculate scale to fit 98% of 1000x1300
    target_w, target_h = 980, 1274  # 98% of 1000x1300
    scale = min(target_w / cropped.width, target_h / cropped.height)
    
    new_w = int(cropped.width * scale)
    new_h = int(cropped.height * scale)
    
    # Resize
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create final thumbnail with white background
    thumbnail = Image.new('RGB', (1000, 1300), (255, 255, 255))
    
    # Paste centered
    x_offset = (1000 - new_w) // 2
    y_offset = (1300 - new_h) // 2
    thumbnail.paste(resized, (x_offset, y_offset))
    
    return thumbnail

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function for RunPod"""
    start_time = time.time()
    
    try:
        # Extract input with multiple fallbacks
        job_input = event.get("input", {})
        
        # Try different ways to get the image
        image_base64 = None
        
        # Method 1: Direct image field
        if "image" in job_input:
            image_base64 = job_input["image"]
        # Method 2: image_base64 field
        elif "image_base64" in job_input:
            image_base64 = job_input["image_base64"]
        # Method 3: Nested in data
        elif "data" in job_input and isinstance(job_input["data"], dict):
            if "image" in job_input["data"]:
                image_base64 = job_input["data"]["image"]
            elif "image_base64" in job_input["data"]:
                image_base64 = job_input["data"]["image_base64"]
        # Method 4: String that needs parsing
        elif isinstance(job_input, str):
            try:
                parsed = json.loads(job_input)
                if isinstance(parsed, dict):
                    image_base64 = parsed.get("image") or parsed.get("image_base64")
            except:
                image_base64 = job_input
        
        if not image_base64:
            logger.error(f"No image found in input. Input structure: {type(job_input)}, keys: {job_input.keys() if isinstance(job_input, dict) else 'Not a dict'}")
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error",
                    "debug_info": {
                        "input_type": str(type(job_input)),
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else "Not a dict"
                    }
                }
            }
        
        # Clean base64 string
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")
        
        # Step 1: Detect and remove masking
        img_array = np.array(image)
        masking_info = find_black_masking_progressive(img_array)
        
        if masking_info['has_masking']:
            logger.info(f"Found {masking_info['type']} masking, removing...")
            image = remove_masking_with_replicate(image, masking_info)
        
        # Step 2: Detect metal type
        metal_type = detect_metal_type(image)
        logger.info(f"Detected metal type: {metal_type}")
        
        # Step 3: Apply enhancement
        enhanced = apply_enhancement(image, metal_type)
        
        # Step 4: Create thumbnail
        thumbnail = create_thumbnail(enhanced)
        
        # Convert results to base64
        # Enhanced image
        enhanced_buffer = BytesIO()
        enhanced.save(enhanced_buffer, format="PNG", quality=95)
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
                "original_size": list(image.size),
                "processing_time": round(processing_time, 2),
                "masking_removed": masking_info['has_masking'],
                "masking_type": masking_info.get('type', 'none'),
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "processing_time": round(time.time() - start_time, 2)
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
