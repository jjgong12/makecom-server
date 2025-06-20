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

def detect_black_borders_ultra_precise(image):
    """Ultra-precise black border detection with multiple methods and validation loops"""
    height, width = image.shape[:2]
    
    # Initialize best borders
    best_borders = {
        'top': 0,
        'bottom': height,
        'left': 0,
        'right': width,
        'has_border': False,
        'confidence': 0
    }
    
    # Method 1: Multiple threshold levels
    threshold_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    all_detections = []
    
    for thresh in threshold_levels:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        # Detect from each direction independently
        # TOP
        top = 0
        for i in range(height):
            row_mean = np.mean(binary[i, width//4:3*width//4])  # Check center portion
            if row_mean > 20:  # Found content
                top = i
                break
        
        # BOTTOM
        bottom = height
        for i in range(height - 1, -1, -1):
            row_mean = np.mean(binary[i, width//4:3*width//4])
            if row_mean > 20:
                bottom = i + 1
                break
        
        # LEFT
        left = 0
        for i in range(width):
            col_mean = np.mean(binary[height//4:3*height//4, i])
            if col_mean > 20:
                left = i
                break
        
        # RIGHT
        right = width
        for i in range(width - 1, -1, -1):
            col_mean = np.mean(binary[height//4:3*height//4, i])
            if col_mean > 20:
                right = i + 1
                break
        
        all_detections.append({
            'threshold': thresh,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right
        })
    
    # Method 2: Edge detection
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 30, 100)
    
    # Find first/last edge from each direction
    edge_top = 0
    for i in range(height):
        if np.sum(edges[i, :]) > width * 0.1:  # 10% of width has edges
            edge_top = max(0, i - 5)  # Small margin
            break
    
    edge_bottom = height
    for i in range(height - 1, -1, -1):
        if np.sum(edges[i, :]) > width * 0.1:
            edge_bottom = min(height, i + 6)
            break
    
    edge_left = 0
    for i in range(width):
        if np.sum(edges[:, i]) > height * 0.1:
            edge_left = max(0, i - 5)
            break
    
    edge_right = width
    for i in range(width - 1, -1, -1):
        if np.sum(edges[:, i]) > height * 0.1:
            edge_right = min(width, i + 6)
            break
    
    # Method 3: Gradient-based detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Detect significant gradient changes
    grad_threshold = np.max(gradient) * 0.1
    
    grad_top = 0
    for i in range(height):
        if np.max(gradient[i, :]) > grad_threshold:
            grad_top = max(0, i - 5)
            break
    
    grad_bottom = height
    for i in range(height - 1, -1, -1):
        if np.max(gradient[i, :]) > grad_threshold:
            grad_bottom = min(height, i + 6)
            break
    
    grad_left = 0
    for i in range(width):
        if np.max(gradient[:, i]) > grad_threshold:
            grad_left = max(0, i - 5)
            break
    
    grad_right = width
    for i in range(width - 1, -1, -1):
        if np.max(gradient[:, i]) > grad_threshold:
            grad_right = min(width, i + 6)
            break
    
    # Method 4: Color variance detection
    # Check if borders have very low color variance (indicating solid black)
    border_size = 50  # Check up to 50 pixels
    
    # Top border variance
    var_top = 0
    for i in range(min(border_size, height)):
        row_variance = np.var(image[i, :])
        if row_variance > 100:  # Significant variance found
            var_top = i
            break
    
    # Bottom border variance
    var_bottom = height
    for i in range(max(0, height - border_size), height):
        row_variance = np.var(image[i, :])
        if row_variance > 100:
            var_bottom = i + 1
            break
    
    # Left border variance
    var_left = 0
    for i in range(min(border_size, width)):
        col_variance = np.var(image[:, i])
        if col_variance > 100:
            var_left = i
            break
    
    # Right border variance
    var_right = width
    for i in range(max(0, width - border_size), width):
        col_variance = np.var(image[:, i])
        if col_variance > 100:
            var_right = i + 1
            break
    
    # Consensus algorithm - take the most conservative (largest) border from all methods
    final_top = max(edge_top, grad_top, var_top, max([d['top'] for d in all_detections]))
    final_bottom = min(edge_bottom, grad_bottom, var_bottom, min([d['bottom'] for d in all_detections]))
    final_left = max(edge_left, grad_left, var_left, max([d['left'] for d in all_detections]))
    final_right = min(edge_right, grad_right, var_right, min([d['right'] for d in all_detections]))
    
    # Validation loop - ensure we detected actual borders
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        # Check if detected area is too small (might have over-cropped)
        detected_width = final_right - final_left
        detected_height = final_bottom - final_top
        
        if detected_width < width * 0.5 or detected_height < height * 0.5:
            # Too aggressive, reduce borders
            logger.warning(f"Iteration {iteration}: Detected area too small, adjusting...")
            final_top = max(0, final_top - 10)
            final_bottom = min(height, final_bottom + 10)
            final_left = max(0, final_left - 10)
            final_right = min(width, final_right + 10)
        else:
            # Validate borders are actually black
            top_strip = image[max(0, final_top-10):final_top, :]
            bottom_strip = image[final_bottom:min(height, final_bottom+10), :]
            left_strip = image[:, max(0, final_left-10):final_left]
            right_strip = image[:, final_right:min(width, final_right+10)]
            
            # Check if strips are actually dark
            strips_are_dark = True
            if top_strip.size > 0 and np.mean(top_strip) > 40:
                strips_are_dark = False
            if bottom_strip.size > 0 and np.mean(bottom_strip) > 40:
                strips_are_dark = False
            if left_strip.size > 0 and np.mean(left_strip) > 40:
                strips_are_dark = False
            if right_strip.size > 0 and np.mean(right_strip) > 40:
                strips_are_dark = False
            
            if strips_are_dark:
                break  # Good detection
            else:
                logger.warning(f"Iteration {iteration}: Borders not dark enough, refining...")
                # Refine detection with stricter threshold
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                
                # Re-detect with stricter criteria
                for i in range(final_top, min(final_top + 50, height)):
                    if np.mean(binary[i, width//4:3*width//4]) > 30:
                        final_top = i
                        break
                
                for i in range(final_bottom - 1, max(final_bottom - 50, 0), -1):
                    if np.mean(binary[i, width//4:3*width//4]) > 30:
                        final_bottom = i + 1
                        break
        
        iteration += 1
    
    # Calculate confidence based on how many methods agreed
    has_significant_border = (
        (final_top > 20) or 
        (height - final_bottom > 20) or 
        (final_left > 20) or 
        (width - final_right > 20)
    )
    
    confidence = 100 if has_significant_border else 0
    
    # Add extra margin to ensure complete removal
    margin = 10
    final_top = max(0, final_top - margin)
    final_bottom = min(height, final_bottom + margin)
    final_left = max(0, final_left - margin)
    final_right = min(width, final_right + margin)
    
    result = {
        'top': final_top,
        'bottom': final_bottom,
        'left': final_left,
        'right': final_right,
        'has_border': has_significant_border,
        'confidence': confidence,
        'detected_methods': {
            'threshold': len([d for d in all_detections if d['top'] > 10 or height - d['bottom'] > 10]),
            'edge': (edge_top > 10 or height - edge_bottom > 10),
            'gradient': (grad_top > 10 or height - grad_bottom > 10),
            'variance': (var_top > 10 or height - var_bottom > 10)
        }
    }
    
    logger.info(f"Border detection complete: {result}")
    return result

def apply_replicate_inpainting(image, mask, use_flux=False):
    """Apply Replicate AI inpainting with natural background extension"""
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
            # Use FLUX for highest quality
            logger.info("Using FLUX Fill for maximum quality inpainting")
            output = replicate.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": "extend the existing background naturally, maintain the same lighting and texture, seamless continuation of the original background",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "num_inference_steps": 35,
                    "guidance_scale": 8.5
                }
            )
        else:
            # Use Ideogram for speed
            logger.info("Using Ideogram v2-turbo for fast inpainting")
            output = replicate.run(
                "ideogram-ai/ideogram-v2-turbo",
                input={
                    "prompt": "natural extension of existing background, maintain exact same style and lighting, seamless blend with original",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "mode": "inpaint",
                    "num_inference_steps": 28,
                    "guidance_scale": 8.0
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

def remove_black_borders_with_replicate(image):
    """Remove black borders using ultra-precise OpenCV detection + Replicate inpainting"""
    # Step 1: Ultra-precise border detection
    borders = detect_black_borders_ultra_precise(image)
    
    # If no borders detected, return original
    if not borders['has_border']:
        logger.info("No black borders detected after thorough analysis")
        return image
    
    # Step 2: Create detailed mask for inpainting
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill detected border areas
    # Extra thick mask for better inpainting
    mask[:borders['top'], :] = 255
    mask[borders['bottom']:, :] = 255
    mask[:, :borders['left']] = 255
    mask[:, borders['right']:] = 255
    
    # Add gradient to mask edges for smoother blending
    kernel_size = 15
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask = np.where(mask > 127, 255, mask).astype(np.uint8)
    
    # Calculate inpainting area percentage
    total_pixels = height * width
    mask_pixels = np.sum(mask > 0)
    mask_percentage = (mask_pixels / total_pixels) * 100
    
    logger.info(f"Mask covers {mask_percentage:.1f}% of image")
    
    # Step 3: Apply Replicate inpainting
    if mask_percentage > 25:
        # Large area - use FLUX for quality
        result = apply_replicate_inpainting(image, mask, use_flux=True)
    else:
        # Smaller area - use Ideogram for speed
        result = apply_replicate_inpainting(image, mask, use_flux=False)
    
    # Step 4: Crop to content area
    cropped = result[borders['top']:borders['bottom'], 
                     borders['left']:borders['right']]
    
    logger.info(f"Cropped from {image.shape} to {cropped.shape}")
    
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
    
    # Multiple detection methods for ring
    best_crop = None
    
    # Method 1: Threshold-based
    for thresh_val in [180, 200, 220, 240]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find most centered and substantial contour
            height, width = img_array.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        valid_contours.append((contour, distance, area))
            
            if valid_contours:
                # Sort by distance from center, then by area
                valid_contours.sort(key=lambda x: (x[1], -x[2]))
                best_contour = valid_contours[0][0]
                
                x, y, w, h = cv2.boundingRect(best_contour)
                
                # Create square crop with generous padding
                size = int(max(w, h) * 2.5)
                cx = x + w // 2
                cy = y + h // 2
                
                x = max(0, cx - size // 2)
                y = max(0, cy - size // 2)
                x = min(x, img_array.shape[1] - size)
                y = min(y, img_array.shape[0] - size)
                
                if size > 100:  # Valid detection
                    best_crop = (x, y, x + size, y + size)
                    break
    
    # Use best detection or fallback to center
    if best_crop:
        ring_crop = enhanced_image.crop(best_crop)
        logger.info(f"Ring detected and cropped at {best_crop}")
    else:
        # Fallback: center crop
        width, height = enhanced_image.size
        size = min(width, height) * 2 // 3
        left = (width - size) // 2
        top = (height - size) // 2
        ring_crop = enhanced_image.crop((left, top, left + size, top + size))
        logger.info("Using center crop for thumbnail")
    
    # Resize to target size
    thumbnail = ring_crop.resize((1000, 1300), Image.Resampling.LANCZOS)
    
    # Apply double sharpening
    thumbnail = thumbnail.filter(ImageFilter.SHARPEN)
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
        
        logger.info(f"Processing v105: metal={metal_type}, size={img_array.shape}")
        
        # Step 1: Remove black borders with ultra-precise detection + Replicate inpainting
        no_border_img = remove_black_borders_with_replicate(img_array)
        
        # Step 2: Analyze lighting
        lighting = analyze_lighting(no_border_img)
        logger.info(f"Detected lighting: {lighting}")
        
        # Step 3: Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(no_border_img, cv2.COLOR_BGR2RGB))
        
        # Step 4: Apply wedding ring enhancement
        enhanced_image = enhance_wedding_ring(pil_image, metal_type, lighting)
        
        # Step 5: Create thumbnail with ultra zoom
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
                    "version": "v105-ultra-detection"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            "output": {  # Even errors need the output wrapper!
                "error": str(e),
                "status": "failed",
                "version": "v105-ultra-detection"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
