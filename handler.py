import runpod
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_and_remove_black_border_ultimate(image):
    """
    Ultimate black border removal - v72 Super Supreme Ultra Perfect
    Based on all successful methods from v23.1 ULTRA + v52 + v61
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    original_h, original_w = h, w
    
    # PASS 1: Maximum aggressive scan (60% of image)
    logger.info("PASS 1: Maximum aggressive border scan - 60% area")
    max_border = min(int(h * 0.6), int(w * 0.6), 600)
    
    # Top border - ultra aggressive
    top_crop = 0
    for y in range(max_border):
        row = gray[y, :]
        # Multiple conditions for detection
        if np.mean(row) < 100 or np.median(row) < 90 or np.max(row) < 120:
            top_crop = y + 1
        elif np.sum(row < 100) > len(row) * 0.85:  # 85% dark pixels
            top_crop = y + 1
        else:
            break
    
    # Bottom border
    bottom_crop = h
    for y in range(h - 1, h - max_border, -1):
        row = gray[y, :]
        if np.mean(row) < 100 or np.median(row) < 90 or np.max(row) < 120:
            bottom_crop = y
        elif np.sum(row < 100) > len(row) * 0.85:
            bottom_crop = y
        else:
            break
    
    # Left border
    left_crop = 0
    for x in range(max_border):
        col = gray[:, x]
        if np.mean(col) < 100 or np.median(col) < 90 or np.max(col) < 120:
            left_crop = x + 1
        elif np.sum(col < 100) > len(col) * 0.85:
            left_crop = x + 1
        else:
            break
    
    # Right border
    right_crop = w
    for x in range(w - 1, w - max_border, -1):
        col = gray[:, x]
        if np.mean(col) < 100 or np.median(col) < 90 or np.max(col) < 120:
            right_crop = x
        elif np.sum(col < 100) > len(col) * 0.85:
            right_crop = x
        else:
            break
    
    # Apply PASS 1
    if top_crop > 0 or bottom_crop < h or left_crop > 0 or right_crop < w:
        logger.info(f"PASS 1 removed: top={top_crop}, bottom={h-bottom_crop}, left={left_crop}, right={w-right_crop}")
        image = image[top_crop:bottom_crop, left_crop:right_crop]
        gray = gray[top_crop:bottom_crop, left_crop:right_crop]
        h, w = gray.shape
    
    # PASS 2: Edge gradient detection (find where content starts)
    logger.info("PASS 2: Edge gradient detection")
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Find first significant edge from each direction
    edge_threshold = 50
    check_depth = min(50, h//4, w//4)
    
    # Top
    for y in range(check_depth):
        if np.max(edges[y, :]) > edge_threshold:
            if y > 5:
                image = image[y-5:, :]
                gray = gray[y-5:, :]
                edges = edges[y-5:, :]
                logger.info(f"PASS 2: Removed {y-5}px from top based on edges")
            break
    
    # Update dimensions
    h, w = gray.shape
    
    # Bottom
    for y in range(h-1, h-check_depth, -1):
        if np.max(edges[y, :]) > edge_threshold:
            if y < h-5:
                image = image[:y+5, :]
                gray = gray[:y+5, :]
                edges = edges[:y+5, :]
                logger.info(f"PASS 2: Removed {h-y-5}px from bottom based on edges")
            break
    
    # Update dimensions
    h, w = gray.shape
    
    # Left
    for x in range(check_depth):
        if np.max(edges[:, x]) > edge_threshold:
            if x > 5:
                image = image[:, x-5:]
                gray = gray[:, x-5:]
                logger.info(f"PASS 2: Removed {x-5}px from left based on edges")
            break
    
    # Right
    for x in range(w-1, w-check_depth, -1):
        if np.max(edges[:, x]) > edge_threshold:
            if x < w-5:
                image = image[:, :x+5]
                gray = gray[:, :x+5]
                logger.info(f"PASS 2: Removed {w-x-5}px from right based on edges")
            break
    
    # PASS 3: Final precision cleanup
    logger.info("PASS 3: Final precision cleanup")
    h, w = gray.shape
    
    # Very aggressive final pass
    final_check = 20
    
    # Check each edge and remove if mostly dark
    if h > final_check * 2:
        if np.percentile(gray[:final_check, :], 90) < 100:
            image = image[final_check:, :]
            gray = gray[final_check:, :]
            logger.info(f"PASS 3: Removed {final_check}px from top")
        
        if np.percentile(gray[-final_check:, :], 90) < 100:
            image = image[:-final_check, :]
            gray = gray[:-final_check, :]
            logger.info(f"PASS 3: Removed {final_check}px from bottom")
    
    if w > final_check * 2:
        if np.percentile(gray[:, :final_check], 90) < 100:
            image = image[:, final_check:]
            gray = gray[:, final_check:]
            logger.info(f"PASS 3: Removed {final_check}px from left")
        
        if np.percentile(gray[:, -final_check:], 90) < 100:
            image = image[:, :-final_check]
            logger.info(f"PASS 3: Removed {final_check}px from right")
    
    final_h, final_w = image.shape[:2]
    logger.info(f"Ultimate border removal complete: {original_h}x{original_w} -> {final_h}x{final_w}")
    
    return image

def detect_ring_advanced(image):
    """
    Advanced ring detection with multiple methods
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Binary threshold
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Method 2: Adaptive threshold for better edge detection
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both methods
    combined = cv2.bitwise_or(binary, adaptive)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    if not valid_contours:
        return None
    
    # Get overall bounding box
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def enhance_wedding_ring_perfect(image, metal_type='white_gold', lighting='balanced'):
    """
    Perfect enhancement without color distortion - NO LAB/HSV conversions!
    """
    # Convert to PIL for safe enhancements
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Metal-specific enhancement parameters (refined for no color shift)
    metal_params = {
        'white_gold': {
            'brightness': 1.35,
            'contrast': 1.25,
            'saturation': 0.98,
            'sharpness': 2.0
        },
        'yellow_gold': {
            'brightness': 1.32,
            'contrast': 1.22,
            'saturation': 1.08,
            'sharpness': 1.8
        },
        'rose_gold': {
            'brightness': 1.32,
            'contrast': 1.20,
            'saturation': 1.1,
            'sharpness': 1.8
        },
        'platinum': {
            'brightness': 1.38,
            'contrast': 1.28,
            'saturation': 0.95,
            'sharpness': 2.2
        }
    }
    
    # Lighting adjustments (subtle)
    lighting_params = {
        'studio': {
            'brightness_mult': 1.02,
            'contrast_mult': 1.05
        },
        'natural': {
            'brightness_mult': 1.0,
            'contrast_mult': 0.98
        },
        'mixed': {
            'brightness_mult': 1.01,
            'contrast_mult': 1.0
        },
        'balanced': {
            'brightness_mult': 1.0,
            'contrast_mult': 1.0
        }
    }
    
    # Get parameters
    metal = metal_params.get(metal_type, metal_params['white_gold'])
    light = lighting_params.get(lighting, lighting_params['balanced'])
    
    # Apply enhancements using PIL only (no color space conversions)
    # 1. Brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(metal['brightness'] * light['brightness_mult'])
    
    # 2. Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(metal['contrast'] * light['contrast_mult'])
    
    # 3. Color (subtle saturation)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(metal['saturation'])
    
    # 4. Sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(metal['sharpness'])
    
    # Convert back to OpenCV
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Gentle bilateral filter for smoothness
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Subtle unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # Ensure pure white background (248, 248, 248)
    # Create mask for very bright areas
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    
    # Apply white background
    enhanced[white_mask == 255] = [248, 248, 248]
    
    return enhanced

def create_professional_background_perfect(image):
    """
    Create professional white background without color shifts
    """
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create slightly larger canvas
    width, height = pil_image.size
    new_width = int(width * 1.1)
    new_height = int(height * 1.1)
    
    # Create pure white background (248, 248, 248)
    background = Image.new('RGB', (new_width, new_height), (248, 248, 248))
    
    # Center the image
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2
    background.paste(pil_image, (x_offset, y_offset))
    
    # Very subtle edge softening (no vignette that changes colors)
    background = background.filter(ImageFilter.GaussianBlur(1))
    
    # Ensure edges are white
    draw_background = Image.new('RGB', (new_width, new_height), (248, 248, 248))
    mask = Image.new('L', (new_width, new_height), 0)
    
    # Create center mask
    center_box = (
        int(new_width * 0.05),
        int(new_height * 0.05),
        int(new_width * 0.95),
        int(new_height * 0.95)
    )
    mask.paste(255, center_box)
    mask = mask.filter(ImageFilter.GaussianBlur(20))
    
    # Composite
    background = Image.composite(background, draw_background, mask)
    
    return background

def create_thumbnail_ultimate(image, ring_bbox=None, target_size=(1000, 1300)):
    """
    Ultimate thumbnail with perfect border removal
    """
    logger.info("Creating ultimate thumbnail")
    
    # Apply same ultimate border removal as main image
    image_clean = detect_and_remove_black_border_ultimate(image.copy())
    
    # Detect ring if not provided
    if ring_bbox is None:
        ring_bbox = detect_ring_advanced(image_clean)
    
    if ring_bbox is not None:
        x, y, w, h = ring_bbox
        # Minimal padding (0.5% - ultra tight crop)
        padding = int(max(w, h) * 0.005)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_clean.shape[1] - x, w + 2 * padding)
        h = min(image_clean.shape[0] - y, h + 2 * padding)
        
        # Crop to ring area
        ring_crop = image_clean[y:y+h, x:x+w]
    else:
        logger.warning("No ring detected, using full cleaned image")
        ring_crop = image_clean
    
    # Create white background
    background = np.full((target_size[1], target_size[0], 3), 248, dtype=np.uint8)
    
    # Calculate scale to fill 99.5% of canvas (maximum possible)
    scale_x = (target_size[0] * 0.995) / ring_crop.shape[1]
    scale_y = (target_size[1] * 0.995) / ring_crop.shape[0]
    scale = min(scale_x, scale_y)
    
    new_width = int(ring_crop.shape[1] * scale)
    new_height = int(ring_crop.shape[0] * scale)
    
    # Resize with highest quality
    ring_resized = cv2.resize(ring_crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply perfect enhancement
    ring_resized = enhance_wedding_ring_perfect(ring_resized)
    
    # Center on background
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = ring_resized
    
    # Convert to PIL for final polish
    thumbnail = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # Final sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    logger.info(f"Ultimate thumbnail: {new_width}x{new_height} in {target_size[0]}x{target_size[1]}")
    
    return thumbnail

def handler(job):
    """RunPod handler - v72 Super Supreme Ultra Perfect"""
    logger.info("Starting v72 Super Supreme Ultra Perfect processing")
    
    try:
        job_input = job['input']
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get('image') or job_input.get('image_base64')
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Handle base64 padding
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        logger.info(f"Image loaded: {image.shape}")
        
        # Step 1: Apply ultimate black border removal
        logger.info("Applying ultimate border removal...")
        processed_image = detect_and_remove_black_border_ultimate(image)
        logger.info(f"After border removal: {processed_image.shape}")
        
        # Step 2: Detect ring for better processing
        logger.info("Detecting wedding ring...")
        ring_bbox = detect_ring_advanced(processed_image)
        
        # Step 3: Enhance with metal and lighting (no color distortion)
        metal_type = job_input.get("metal_type", "white_gold")
        lighting = job_input.get("lighting", "balanced")
        
        logger.info(f"Enhancing ring - Metal: {metal_type}, Lighting: {lighting}")
        enhanced_image = enhance_wedding_ring_perfect(processed_image, metal_type, lighting)
        
        # Step 4: Create professional background
        final_image = create_professional_background_perfect(enhanced_image)
        
        # Step 5: Create ultimate thumbnail
        logger.info("Creating ultimate thumbnail...")
        thumbnail = create_thumbnail_ultimate(image, ring_bbox, target_size=(1000, 1300))
        
        # Convert images to base64
        # Main image
        main_buffer = BytesIO()
        # Convert numpy array to PIL if needed
        if isinstance(final_image, np.ndarray):
            final_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        final_image.save(main_buffer, format='PNG', optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding for Make.com
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding
        
        logger.info("Processing completed successfully - v72 Super Supreme Ultra Perfect")
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v72_super_supreme_ultra_perfect",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": list(image.shape[:2]),
                    "processed_size": list(processed_image.shape[:2]),
                    "thumbnail_size": [1000, 1300],
                    "border_removed": True,
                    "removal_passes": 3,
                    "ring_detected": ring_bbox is not None,
                    "ring_bbox": list(ring_bbox) if ring_bbox else None,
                    "enhancements_applied": [
                        "ultimate_border_removal_3pass",
                        "edge_gradient_detection",
                        "metal_specific_correction",
                        "lighting_adjustment",
                        "professional_background",
                        "no_color_space_conversion",
                        "pure_white_background_248"
                    ],
                    "color_accuracy": "perfect_no_lab_hsv",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "processing_info": {
                    "version": "v72_super_supreme_ultra_perfect",
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
