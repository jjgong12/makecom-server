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

def detect_black_borders_simple(image):
    """Simple black border detection using threshold"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a single threshold
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Find the actual content area
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return {
            'top': y,
            'bottom': y + h,
            'left': x,
            'right': x + w,
            'has_border': (y > 10 or x > 10 or (y + h) < height - 10 or (x + w) < width - 10)
        }
    
    return {
        'top': 0,
        'bottom': height,
        'left': 0,
        'right': width,
        'has_border': False
    }

def apply_replicate_inpainting(image, mask, use_flux=False):
    """Apply Replicate AI inpainting - always use this instead of OpenCV"""
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
        
        if use_flux:
            # Use FLUX for better quality
            logger.info("Using FLUX Fill for high-quality inpainting")
            output = replicate.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": "professional product photography, clean white seamless background, studio lighting, no shadows",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "num_inference_steps": 30,
                    "guidance_scale": 8.0
                }
            )
        else:
            # Use Ideogram for speed
            logger.info("Using Ideogram v2-turbo for fast inpainting")
            output = replicate.run(
                "ideogram-ai/ideogram-v2-turbo",
                input={
                    "prompt": "professional product photography background, clean white studio background, soft even lighting",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "mode": "inpaint",
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
        # Return original image if failed
        return image

def remove_black_borders_replicate_only(image):
    """Remove black borders using only Replicate models"""
    borders = detect_black_borders_simple(image)
    
    # If no borders detected, return original
    if not borders['has_border']:
        logger.info("No black borders detected")
        return image
    
    # Create mask for borders
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a larger mask area for better inpainting
    border_expansion = 20  # Expand border area for better results
    
    # Fill border areas in mask with some expansion
    mask[:max(0, borders['top'] + border_expansion), :] = 255
    mask[min(height, borders['bottom'] - border_expansion):, :] = 255
    mask[:, :max(0, borders['left'] + border_expansion)] = 255
    mask[:, min(width, borders['right'] - border_expansion):] = 255
    
    # Calculate total border area
    total_pixels = height * width
    border_pixels = np.sum(mask > 0)
    border_percentage = (border_pixels / total_pixels) * 100
    
    logger.info(f"Border area: {border_percentage:.1f}% of image")
    
    # Always use Replicate, choose model based on complexity
    if border_percentage > 20:
        # Large borders - use FLUX for better quality
        logger.info("Large border area detected, using FLUX for quality")
        result = apply_replicate_inpainting(image, mask, use_flux=True)
    else:
        # Smaller borders - use Ideogram for speed
        logger.info("Small/medium border area, using Ideogram for speed")
        result = apply_replicate_inpainting(image, mask, use_flux=False)
    
    # Crop to content area
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

def create_thumbnail_center_focused(original_image: Image.Image, enhanced_image: Image.Image) -> Image.Image:
    """Create thumbnail with better ring detection"""
    img_array = np.array(enhanced_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # More aggressive threshold for ring detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the contour closest to center (likely the ring)
        height, width = img_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        best_contour = None
        min_distance = float('inf')
        
        for contour in contours:
            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate distance from image center
                distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                # Also check contour area (ring should be substantial)
                area = cv2.contourArea(contour)
                if area > 1000 and distance < min_distance:
                    min_distance = distance
                    best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # Make the crop area square and larger
            size = int(max(w, h) * 2.0)  # 2x padding
            
            # Center the crop on the ring
            center_x = x + w // 2
            center_y = y + h // 2
            
            x = max(0, center_x - size // 2)
            y = max(0, center_y - size // 2)
            
            # Ensure we don't exceed image boundaries
            x = min(x, img_array.shape[1] - size)
            y = min(y, img_array.shape[0] - size)
            
            ring_crop = enhanced_image.crop((x, y, x + size, y + size))
        else:
            # Fallback to center crop
            logger.info("Using center crop for thumbnail")
            width, height = enhanced_image.size
            size = min(width, height) // 2
            left = (width - size) // 2
            top = (height - size) // 2
            ring_crop = enhanced_image.crop((left, top, left + size, top + size))
    else:
        # Fallback to center crop
        logger.info("No contours found, using center crop")
        width, height = enhanced_image.size
        size = min(width, height) // 2
        left = (width - size) // 2
        top = (height - size) // 2
        ring_crop = enhanced_image.crop((left, top, left + size, top + size))
    
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
        
        logger.info(f"Processing v104: metal={metal_type}, size={img_array.shape}")
        
        # Step 1: Remove black borders using only Replicate
        no_border_img = remove_black_borders_replicate_only(img_array)
        
        # Step 2: Analyze lighting
        lighting = analyze_lighting(no_border_img)
        logger.info(f"Detected lighting: {lighting}")
        
        # Step 3: Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(no_border_img, cv2.COLOR_BGR2RGB))
        
        # Step 4: Apply wedding ring enhancement
        enhanced_image = enhance_wedding_ring(pil_image, metal_type, lighting)
        
        # Step 5: Create thumbnail with better centering
        thumbnail = create_thumbnail_center_focused(pil_image, enhanced_image)
        
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
                    "version": "v104-replicate-only"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            "output": {  # Even errors need the output wrapper!
                "error": str(e),
                "status": "failed",
                "version": "v104-replicate-only"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
