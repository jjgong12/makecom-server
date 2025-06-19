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

def detect_and_remove_black_border_supreme(image):
    """
    Supreme black border removal - Based on v23.1 ULTRA success
    Ultra aggressive 3-pass system with OpenCV
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    original_h, original_w = h, w
    
    # PASS 1: Ultra aggressive scan (50% of image, like v23.1 ULTRA)
    logger.info("PASS 1: Ultra aggressive border scan - 50% area")
    max_border = min(int(h * 0.5), int(w * 0.5), 500)
    
    # Top border - check entire row
    top_crop = 0
    for y in range(max_border):
        row_mean = np.mean(gray[y, :])  # Entire row like v23.1 ULTRA
        if row_mean < 100:  # More aggressive than v70's 120
            top_crop = y + 1
        else:
            # Check if 90% of pixels are dark
            dark_pixels = np.sum(gray[y, :] < 100)
            if dark_pixels > w * 0.9:
                top_crop = y + 1
            else:
                break
    
    # Bottom border
    bottom_crop = h
    for y in range(h - 1, h - max_border, -1):
        row_mean = np.mean(gray[y, :])
        if row_mean < 100:
            bottom_crop = y
        else:
            dark_pixels = np.sum(gray[y, :] < 100)
            if dark_pixels > w * 0.9:
                bottom_crop = y
            else:
                break
    
    # Left border
    left_crop = 0
    for x in range(max_border):
        col_mean = np.mean(gray[:, x])
        if col_mean < 100:
            left_crop = x + 1
        else:
            dark_pixels = np.sum(gray[:, x] < 100)
            if dark_pixels > h * 0.9:
                left_crop = x + 1
            else:
                break
    
    # Right border
    right_crop = w
    for x in range(w - 1, w - max_border, -1):
        col_mean = np.mean(gray[:, x])
        if col_mean < 100:
            right_crop = x
        else:
            dark_pixels = np.sum(gray[:, x] < 100)
            if dark_pixels > h * 0.9:
                right_crop = x
            else:
                break
    
    # Apply PASS 1
    if top_crop > 0 or bottom_crop < h or left_crop > 0 or right_crop < w:
        logger.info(f"PASS 1 removed: top={top_crop}, bottom={h-bottom_crop}, left={left_crop}, right={w-right_crop}")
        image = image[top_crop:bottom_crop, left_crop:right_crop]
        gray = gray[top_crop:bottom_crop, left_crop:right_crop]
        h, w = gray.shape
    
    # PASS 2: Secondary precision crop (like v23.1)
    logger.info("PASS 2: Precision border removal")
    
    # Check smaller area for remaining borders
    check_size = min(30, int(h * 0.1), int(w * 0.1))
    
    # Top edge check
    if np.mean(gray[:check_size, :]) < 80:
        for y in range(check_size):
            if np.mean(gray[y, :]) < 80:
                top_extra = y + 1
            else:
                break
        if top_extra > 0:
            image = image[top_extra:, :]
            gray = gray[top_extra:, :]
            logger.info(f"PASS 2 removed additional {top_extra}px from top")
    
    # Bottom edge check
    h, w = gray.shape
    if np.mean(gray[-check_size:, :]) < 80:
        for y in range(h - 1, h - check_size, -1):
            if np.mean(gray[y, :]) < 80:
                bottom_extra = h - y
            else:
                break
        if bottom_extra > 0:
            image = image[:-bottom_extra, :]
            gray = gray[:-bottom_extra, :]
            logger.info(f"PASS 2 removed additional {bottom_extra}px from bottom")
    
    # Left edge check
    h, w = gray.shape
    if np.mean(gray[:, :check_size]) < 80:
        for x in range(check_size):
            if np.mean(gray[:, x]) < 80:
                left_extra = x + 1
            else:
                break
        if left_extra > 0:
            image = image[:, left_extra:]
            gray = gray[:, left_extra:]
            logger.info(f"PASS 2 removed additional {left_extra}px from left")
    
    # Right edge check
    h, w = gray.shape
    if np.mean(gray[:, -check_size:]) < 80:
        for x in range(w - 1, w - check_size, -1):
            if np.mean(gray[:, x]) < 80:
                right_extra = w - x
            else:
                break
        if right_extra > 0:
            image = image[:, :-right_extra]
            gray = gray[:, :-right_extra]
            logger.info(f"PASS 2 removed additional {right_extra}px from right")
    
    # PASS 3: Final fine-tuning (very aggressive)
    logger.info("PASS 3: Final edge cleaning")
    h, w = gray.shape
    
    # Ultra fine edge removal - even more aggressive
    edge_check = 15  # Fixed 15 pixels like v23.1
    
    if np.mean(gray[:edge_check, :]) < 100:
        image = image[edge_check:, :]
        gray = gray[edge_check:, :]
        logger.info("PASS 3 removed 15px from top")
    
    if gray.shape[0] > edge_check and np.mean(gray[-edge_check:, :]) < 100:
        image = image[:-edge_check, :]
        gray = gray[:-edge_check, :]
        logger.info("PASS 3 removed 15px from bottom")
    
    if gray.shape[1] > edge_check and np.mean(gray[:, :edge_check]) < 100:
        image = image[:, edge_check:]
        gray = gray[:, edge_check:]
        logger.info("PASS 3 removed 15px from left")
    
    if gray.shape[1] > edge_check and np.mean(gray[:, -edge_check:]) < 100:
        image = image[:, :-edge_check]
        logger.info("PASS 3 removed 15px from right")
    
    # Final safety check
    final_h, final_w = image.shape[:2]
    if final_h < original_h * 0.3 or final_w < original_w * 0.3:
        logger.warning("Warning: Removed too much, but proceeding anyway")
    
    logger.info(f"Supreme removal complete: {original_h}x{original_w} -> {final_h}x{final_w}")
    return image

def detect_ring_opencv(image):
    """
    Detect wedding ring(s) in the image using OpenCV
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get bounding box of all significant contours
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

def enhance_wedding_ring(image, metal_type='white_gold', lighting='balanced'):
    """
    Enhanced wedding ring processing with metal-specific adjustments
    """
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Metal-specific enhancement parameters
    metal_params = {
        'white_gold': {
            'brightness': 1.4,
            'contrast': 1.3,
            'saturation': 0.95,
            'temperature': -5
        },
        'yellow_gold': {
            'brightness': 1.35,
            'contrast': 1.25,
            'saturation': 1.15,
            'temperature': 10
        },
        'rose_gold': {
            'brightness': 1.35,
            'contrast': 1.2,
            'saturation': 1.2,
            'temperature': 15
        },
        'platinum': {
            'brightness': 1.45,
            'contrast': 1.35,
            'saturation': 0.9,
            'temperature': -10
        }
    }
    
    # Lighting-specific adjustments
    lighting_params = {
        'studio': {
            'brightness_mult': 1.0,
            'contrast_mult': 1.1,
            'highlights': 1.2
        },
        'natural': {
            'brightness_mult': 1.05,
            'contrast_mult': 0.95,
            'highlights': 1.0
        },
        'mixed': {
            'brightness_mult': 1.02,
            'contrast_mult': 1.0,
            'highlights': 1.1
        },
        'balanced': {
            'brightness_mult': 1.0,
            'contrast_mult': 1.0,
            'highlights': 1.05
        }
    }
    
    # Get parameters
    metal = metal_params.get(metal_type, metal_params['white_gold'])
    light = lighting_params.get(lighting, lighting_params['balanced'])
    
    # Apply brightness with lighting adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(metal['brightness'] * light['brightness_mult'])
    
    # Apply contrast with lighting adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(metal['contrast'] * light['contrast_mult'])
    
    # Apply saturation
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(metal['saturation'])
    
    # Maximum sharpness for details
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(2.0)
    
    # Convert back to OpenCV for advanced processing
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Apply color temperature adjustment
    if metal['temperature'] != 0:
        if metal['temperature'] > 0:  # Warmer
            enhanced[:,:,0] = np.clip(enhanced[:,:,0] * 0.95, 0, 255)  # Less blue
            enhanced[:,:,2] = np.clip(enhanced[:,:,2] * 1.05, 0, 255)  # More red
        else:  # Cooler
            enhanced[:,:,0] = np.clip(enhanced[:,:,0] * 1.05, 0, 255)  # More blue
            enhanced[:,:,2] = np.clip(enhanced[:,:,2] * 0.95, 0, 255)  # Less red
    
    # Bilateral filter for smoothness while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Highlight enhancement for sparkle
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance highlights
    highlight_threshold = 200
    highlights_mask = cv2.inRange(l, highlight_threshold, 255)
    l[highlights_mask > 0] = np.clip(l[highlights_mask > 0] * light['highlights'], 0, 255).astype(np.uint8)
    
    # Merge back
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Final unsharp mask for pop
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    enhanced = cv2.addWeighted(enhanced, 1.7, gaussian, -0.7, 0)
    
    return enhanced

def create_professional_background(image):
    """
    Create a professional white background for the image
    """
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create slightly larger canvas for breathing room
    width, height = pil_image.size
    new_width = int(width * 1.1)
    new_height = int(height * 1.1)
    
    # Create professional white background (248, 248, 248)
    background = Image.new('RGB', (new_width, new_height), (248, 248, 248))
    
    # Center the image
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2
    background.paste(pil_image, (x_offset, y_offset))
    
    # Add very subtle vignette for depth
    vignette = Image.new('L', (new_width, new_height), 255)
    for i in range(3):
        vignette = vignette.filter(ImageFilter.GaussianBlur(50))
    
    # Apply vignette
    background = Image.composite(
        background,
        Image.new('RGB', (new_width, new_height), (240, 240, 240)),
        vignette
    )
    
    return background

def create_thumbnail_supreme(image, ring_bbox=None, target_size=(1000, 1300)):
    """
    Create thumbnail with supreme border removal and ring detection
    """
    logger.info("Creating supreme thumbnail")
    
    # Apply same supreme border removal as main image
    image_clean = detect_and_remove_black_border_supreme(image.copy())
    
    # If no bbox provided, detect it
    if ring_bbox is None:
        ring_bbox = detect_ring_opencv(image_clean)
    
    if ring_bbox is not None:
        x, y, w, h = ring_bbox
        # Minimal padding (1%)
        padding = int(max(w, h) * 0.01)
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
    
    # Calculate scale to fill 99% of canvas
    scale_x = (target_size[0] * 0.99) / ring_crop.shape[1]
    scale_y = (target_size[1] * 0.99) / ring_crop.shape[0]
    scale = min(scale_x, scale_y)
    
    new_width = int(ring_crop.shape[1] * scale)
    new_height = int(ring_crop.shape[0] * scale)
    
    # Resize with high quality
    ring_resized = cv2.resize(ring_crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply enhancement
    ring_resized = enhance_wedding_ring(ring_resized)
    
    # Center on background
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = ring_resized
    
    logger.info(f"Thumbnail created: {new_width}x{new_height} ring in {target_size[0]}x{target_size[1]} canvas")
    
    # Convert to PIL for final polish
    thumbnail = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # Add subtle sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return thumbnail

def handler(job):
    """RunPod handler function with full v70 features + v71 supreme border removal"""
    logger.info("Starting v71 Supreme processing")
    
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
        
        # Step 1: Apply supreme black border removal
        logger.info("Applying supreme border removal...")
        processed_image = detect_and_remove_black_border_supreme(image)
        logger.info(f"After border removal: {processed_image.shape}")
        
        # Step 2: Detect ring for better processing
        logger.info("Detecting wedding ring...")
        ring_bbox = detect_ring_opencv(processed_image)
        
        # Step 3: Enhance with metal and lighting parameters
        metal_type = job_input.get("metal_type", "white_gold")
        lighting = job_input.get("lighting", "balanced")
        
        logger.info(f"Enhancing ring - Metal: {metal_type}, Lighting: {lighting}")
        enhanced_image = enhance_wedding_ring(processed_image, metal_type, lighting)
        
        # Step 4: Create professional background
        final_image = create_professional_background(enhanced_image)
        
        # Step 5: Create supreme thumbnail
        logger.info("Creating supreme thumbnail...")
        thumbnail = create_thumbnail_supreme(image, ring_bbox, target_size=(1000, 1300))
        
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
        
        logger.info("Processing completed successfully - v71 Supreme")
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v71_supreme",
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
                        "supreme_border_removal",
                        "metal_specific_correction",
                        "lighting_adjustment",
                        "professional_background",
                        "highlight_enhancement",
                        "color_temperature_adjustment"
                    ],
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
                    "version": "v71_supreme",
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
