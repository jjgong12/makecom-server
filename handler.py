import runpod
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import os
import requests
import time
import logging
import replicate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wedding ring enhancement parameters (28 pairs of learning data)
WEDDING_RING_PARAMS = {
    'yellow_gold': {
        'low': {
            'brightness': 1.15, 'contrast': 1.10, 'sharpness': 1.2,
            'saturation': 1.15, 'white_overlay': 0.05, 'gamma': 0.95,
            'color_temp_a': 3, 'color_temp_b': 5, 'original_blend': 0.15
        },
        'medium': {
            'brightness': 1.12, 'contrast': 1.08, 'sharpness': 1.15,
            'saturation': 1.10, 'white_overlay': 0.03, 'gamma': 0.97,
            'color_temp_a': 2, 'color_temp_b': 4, 'original_blend': 0.12
        },
        'high': {
            'brightness': 1.08, 'contrast': 1.05, 'sharpness': 1.1,
            'saturation': 1.05, 'white_overlay': 0.02, 'gamma': 1.0,
            'color_temp_a': 1, 'color_temp_b': 2, 'original_blend': 0.1
        }
    },
    'rose_gold': {
        'low': {
            'brightness': 1.18, 'contrast': 1.12, 'sharpness': 1.25,
            'saturation': 1.20, 'white_overlay': 0.08, 'gamma': 0.93,
            'color_temp_a': 5, 'color_temp_b': 2, 'original_blend': 0.12
        },
        'medium': {
            'brightness': 1.15, 'contrast': 1.10, 'sharpness': 1.2,
            'saturation': 1.15, 'white_overlay': 0.05, 'gamma': 0.95,
            'color_temp_a': 4, 'color_temp_b': 1, 'original_blend': 0.1
        },
        'high': {
            'brightness': 1.10, 'contrast': 1.08, 'sharpness': 1.15,
            'saturation': 1.10, 'white_overlay': 0.03, 'gamma': 0.98,
            'color_temp_a': 3, 'color_temp_b': 0, 'original_blend': 0.08
        }
    },
    'white_gold': {
        'low': {
            'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.3,
            'saturation': 0.85, 'white_overlay': 0.12, 'gamma': 0.90,
            'color_temp_a': -3, 'color_temp_b': -5, 'original_blend': 0.1
        },
        'medium': {
            'brightness': 1.20, 'contrast': 1.12, 'sharpness': 1.25,
            'saturation': 0.90, 'white_overlay': 0.08, 'gamma': 0.93,
            'color_temp_a': -2, 'color_temp_b': -3, 'original_blend': 0.08
        },
        'high': {
            'brightness': 1.15, 'contrast': 1.10, 'sharpness': 1.2,
            'saturation': 0.95, 'white_overlay': 0.05, 'gamma': 0.95,
            'color_temp_a': -1, 'color_temp_b': -2, 'original_blend': 0.05
        }
    },
    'plain_white': {
        'low': {
            'brightness': 1.30, 'contrast': 1.18, 'sharpness': 1.35,
            'saturation': 0.70, 'white_overlay': 0.18, 'gamma': 0.88,
            'color_temp_a': -5, 'color_temp_b': -8, 'original_blend': 0.08
        },
        'medium': {
            'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.3,
            'saturation': 0.75, 'white_overlay': 0.12, 'gamma': 0.90,
            'color_temp_a': -4, 'color_temp_b': -6, 'original_blend': 0.06
        },
        'high': {
            'brightness': 1.18, 'contrast': 1.12, 'sharpness': 1.25,
            'saturation': 0.80, 'white_overlay': 0.08, 'gamma': 0.93,
            'color_temp_a': -3, 'color_temp_b': -4, 'original_blend': 0.04
        }
    }
}

def detect_black_borders_advanced(image):
    """Detect black borders with advanced method"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple threshold levels
    threshold_levels = [10, 20, 30]
    borders_list = []
    
    for thresh in threshold_levels:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        # Check each side
        top = 0
        for i in range(height // 2):
            if np.mean(binary[i, :]) > 10:
                top = i
                break
        
        bottom = height
        for i in range(height - 1, height // 2, -1):
            if np.mean(binary[i, :]) > 10:
                bottom = i + 1
                break
        
        left = 0
        for i in range(width // 2):
            if np.mean(binary[:, i]) > 10:
                left = i
                break
        
        right = width
        for i in range(width - 1, width // 2, -1):
            if np.mean(binary[:, i]) > 10:
                right = i + 1
                break
        
        borders_list.append({
            'top': top, 'bottom': bottom,
            'left': left, 'right': right,
            'max_border': max(top, height - bottom, left, width - right)
        })
    
    # Choose the most conservative (smallest crop)
    best_borders = min(borders_list, key=lambda x: x['max_border'])
    
    logger.info(f"Detected borders: {best_borders}")
    return best_borders

def apply_opencv_inpainting(image, mask):
    """Apply OpenCV inpainting for small borders"""
    # Dilate mask slightly
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply inpainting
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return result

def apply_replicate_inpainting(image, mask, model_name):
    """Apply Replicate AI inpainting"""
    try:
        # Convert image to base64
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert mask to base64
        pil_mask = Image.fromarray(mask)
        mask_buffered = BytesIO()
        pil_mask.save(mask_buffered, format="PNG")
        mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode()
        
        if model_name == "ideogram-ai/ideogram-v2-turbo":
            # Ideogram v2-turbo
            output = replicate.run(
                "ideogram-ai/ideogram-v2-turbo",
                input={
                    "prompt": "professional product photography background, clean white studio background, soft lighting",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "mode": "inpaint",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            )
        else:
            # FLUX Fill dev
            output = replicate.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": "professional product photography, clean white seamless background, studio lighting",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            )
        
        # Get result
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = output
            
        # Download result
        response = requests.get(result_url)
        result_image = Image.open(BytesIO(response.content))
        result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        
        # Resize to original size if needed
        if result_array.shape[:2] != image.shape[:2]:
            result_array = cv2.resize(result_array, (image.shape[1], image.shape[0]))
        
        return result_array
        
    except Exception as e:
        logger.error(f"Replicate inpainting failed: {str(e)}")
        # Fallback to OpenCV
        return apply_opencv_inpainting(image, mask)

def remove_black_borders_smart(image):
    """Remove black borders with smart model selection"""
    borders = detect_black_borders_advanced(image)
    
    # Calculate border size
    max_border = borders['max_border']
    
    # If no significant borders, return original
    if max_border < 10:
        logger.info("No significant borders detected")
        return image
    
    # Create mask for borders
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill border areas in mask
    mask[:borders['top'], :] = 255
    mask[borders['bottom']:, :] = 255
    mask[:, :borders['left']] = 255
    mask[:, borders['right']:] = 255
    
    # Select model based on border size
    if max_border < 50:
        logger.info(f"Using OpenCV for small borders ({max_border}px)")
        result = apply_opencv_inpainting(image, mask)
    elif max_border < 100:
        logger.info(f"Using Ideogram v2-turbo for medium borders ({max_border}px)")
        result = apply_replicate_inpainting(image, mask, "ideogram-ai/ideogram-v2-turbo")
    else:
        logger.info(f"Using FLUX Fill for large borders ({max_border}px)")
        result = apply_replicate_inpainting(image, mask, "black-forest-labs/flux-fill-dev")
    
    # Crop to remove processed borders
    cropped = result[borders['top']:borders['bottom'], 
                     borders['left']:borders['right']]
    
    return cropped

def analyze_lighting(image):
    """Analyze image lighting conditions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 85:
        return 'low'
    elif mean_brightness < 170:
        return 'medium'
    else:
        return 'high'

def enhance_wedding_ring(image: Image.Image, metal_type: str, lighting: str) -> Image.Image:
    """Apply v13.3 wedding ring enhancement"""
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
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        lab = lab.astype(np.uint8)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = Image.fromarray(enhanced_array)
    
    # Blend with original
    if params['original_blend'] > 0:
        enhanced = Image.blend(enhanced, image, params['original_blend'])
    
    return enhanced

def create_thumbnail_ultra_zoom(original_image: Image.Image, enhanced_image: Image.Image) -> Image.Image:
    """Create ultra-zoomed thumbnail focusing on ring only"""
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
        padding = int(max(w, h) * 0.3)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        
        # Make square
        if w > h:
            diff = w - h
            y = max(0, y - diff // 2)
            h = min(img_array.shape[0] - y, w)
        else:
            diff = h - w
            x = max(0, x - diff // 2)
            w = min(img_array.shape[1] - x, h)
        
        ring_crop = enhanced_image.crop((x, y, x + w, y + h))
    
    # Resize to target size
    thumbnail = ring_crop.resize((1000, 1300), Image.Resampling.LANCZOS)
    
    # Apply sharpening
    thumbnail = thumbnail.filter(ImageFilter.SHARPEN)
    
    return thumbnail

def handler(job):
    """RunPod handler function"""
    try:
        input_data = job['input']
        
        # Get image from either field
        image_base64 = input_data.get('image') or input_data.get('image_base64')
        
        if not image_base64:
            raise ValueError("No image provided in 'image' or 'image_base64' field")
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Add padding if needed
        padding = 4 - len(image_base64) % 4
        if padding != 4:
            image_base64 += '=' * padding
        
        # Decode image
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_bytes))
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Get metal type
        metal_type = input_data.get('metal_type', 'white_gold').lower()
        if metal_type not in WEDDING_RING_PARAMS:
            metal_type = 'white_gold'
        
        logger.info(f"Processing: metal={metal_type}, size={img_array.shape}")
        
        # Step 1: Remove black borders with smart inpainting
        no_border_img = remove_black_borders_smart(img_array)
        
        # Step 2: Analyze lighting
        lighting = analyze_lighting(no_border_img)
        logger.info(f"Detected lighting: {lighting}")
        
        # Step 3: Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(no_border_img, cv2.COLOR_BGR2RGB))
        
        # Step 4: Apply wedding ring enhancement
        enhanced_image = enhance_wedding_ring(pil_image, metal_type, lighting)
        
        # Step 5: Create thumbnail
        thumbnail = create_thumbnail_ultra_zoom(pil_image, enhanced_image)
        
        # Convert to base64
        # Main image
        main_buffer = BytesIO()
        enhanced_image.save(main_buffer, format='PNG', quality=95)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Nested output structure for Make.com
        return {
            "output": {  # This 'output' wrapper is REQUIRED!
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": f"{img.width}x{img.height}",
                    "enhanced_size": f"{enhanced_image.width}x{enhanced_image.height}",
                    "thumbnail_size": f"{thumbnail.width}x{thumbnail.height}",
                    "status": "success",
                    "version": "v103-replicate"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            "output": {  # Even errors need the output wrapper!
                "error": str(e),
                "status": "failed",
                "version": "v103-replicate"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
