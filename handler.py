import runpod
import base64
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_and_remove_black_border_perfect(image):
    """Perfect black border detection and removal with triple-pass system"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for border detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        h, w = gray.shape
        original_h, original_w = h, w
        
        # PASS 1: Aggressive initial detection
        threshold = 100  # Lower threshold to catch more borders
        max_scan = int(min(h, w) * 0.6)  # Scan up to 60%
        
        # Find cumulative dark pixels from edges
        top = 0
        for i in range(max_scan):
            # Count dark pixels in current row
            dark_count = np.sum(gray[i, :] <= threshold)
            # If more than 70% dark, it's likely a border
            if dark_count >= w * 0.7:
                top = i + 1
            else:
                # Check if we've found a substantial border
                if top > 10:  # At least 10 pixels of border
                    break
        
        bottom = h
        for i in range(max_scan):
            dark_count = np.sum(gray[h-1-i, :] <= threshold)
            if dark_count >= w * 0.7:
                bottom = h - 1 - i
            else:
                if h - bottom > 10:
                    break
        
        left = 0
        for i in range(max_scan):
            dark_count = np.sum(gray[:, i] <= threshold)
            if dark_count >= h * 0.7:
                left = i + 1
            else:
                if left > 10:
                    break
        
        right = w
        for i in range(max_scan):
            dark_count = np.sum(gray[:, w-1-i] <= threshold)
            if dark_count >= h * 0.7:
                right = w - 1 - i
            else:
                if w - right > 10:
                    break
        
        # Minimal safety margin for first pass
        safety_margin = 5
        top = min(top + safety_margin, h // 3)
        bottom = max(bottom - safety_margin, 2 * h // 3)
        left = min(left + safety_margin, w // 3)
        right = max(right - safety_margin, 2 * w // 3)
        
        # First crop
        if top < bottom and left < right:
            if len(img_array.shape) == 3:
                cropped = img_array[top:bottom, left:right]
            else:
                cropped = img_array[top:bottom, left:right]
            
            # PASS 2: Medium precision check
            gray2 = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY) if len(cropped.shape) == 3 else cropped
            h2, w2 = gray2.shape
            
            # Check for any remaining dark edges with stricter threshold
            edge_threshold = 80
            edge_cut = 0
            
            # Check each edge more carefully
            if np.mean(gray2[:30, :]) < edge_threshold:
                for i in range(min(30, h2//4)):
                    if np.mean(gray2[i, :]) < edge_threshold:
                        edge_cut = i + 1
                    else:
                        break
                if edge_cut > 0:
                    cropped = cropped[edge_cut+10:, :]
            
            # Bottom edge
            edge_cut = 0
            if np.mean(gray2[-30:, :]) < edge_threshold:
                for i in range(min(30, h2//4)):
                    if np.mean(gray2[-(i+1), :]) < edge_threshold:
                        edge_cut = i + 1
                    else:
                        break
                if edge_cut > 0:
                    cropped = cropped[:-(edge_cut+10), :]
            
            # Left edge
            edge_cut = 0
            if np.mean(gray2[:, :30]) < edge_threshold:
                for i in range(min(30, w2//4)):
                    if np.mean(gray2[:, i]) < edge_threshold:
                        edge_cut = i + 1
                    else:
                        break
                if edge_cut > 0:
                    cropped = cropped[:, edge_cut+10:]
            
            # Right edge
            edge_cut = 0
            if np.mean(gray2[:, -30:]) < edge_threshold:
                for i in range(min(30, w2//4)):
                    if np.mean(gray2[:, -(i+1)]) < edge_threshold:
                        edge_cut = i + 1
                    else:
                        break
                if edge_cut > 0:
                    cropped = cropped[:, :-(edge_cut+10)]
            
            # PASS 3: Ultra precision final check
            gray3 = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY) if len(cropped.shape) == 3 else cropped
            h3, w3 = gray3.shape
            
            # Final aggressive cleanup
            final_threshold = 60
            
            # Check corners and edges one more time
            if np.mean(gray3[:10, :]) < final_threshold:
                cropped = cropped[15:, :]
            if np.mean(gray3[-10:, :]) < final_threshold:
                cropped = cropped[:-15, :]
            if np.mean(gray3[:, :10]) < final_threshold:
                cropped = cropped[:, 15:]
            if np.mean(gray3[:, -10:]) < final_threshold:
                cropped = cropped[:, :-15]
            
            final_h, final_w = cropped.shape[:2]
            logger.info(f"Perfect border removal: {original_w}x{original_h} → {final_w}x{final_h}")
            
            return Image.fromarray(cropped), (left, top, final_w, final_h)
        else:
            logger.warning("Invalid crop dimensions, returning original")
            return image, (0, 0, w, h)
            
    except Exception as e:
        logger.error(f"Error in border removal: {str(e)}")
        return image, (0, 0, image.width, image.height)

def detect_ring(image):
    """Detect wedding ring in the image"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Validate detection
        min_size = min(image.width, image.height) * 0.1
        if w < min_size or h < min_size:
            return None
            
        return (x, y, w, h)
        
    except Exception as e:
        logger.error(f"Error in ring detection: {str(e)}")
        return None

def enhance_wedding_ring(image, metal_type='white_gold', lighting='balanced'):
    """Enhance wedding ring with metal-specific adjustments"""
    try:
        # Metal-specific enhancement parameters
        metal_params = {
            'white_gold': {
                'brightness': 1.05,
                'contrast': 1.1,
                'saturation': 0.95,
                'highlights': 1.1
            },
            'yellow_gold': {
                'brightness': 1.08,
                'contrast': 1.15,
                'saturation': 1.05,
                'highlights': 1.15
            },
            'rose_gold': {
                'brightness': 1.06,
                'contrast': 1.12,
                'saturation': 1.02,
                'highlights': 1.12
            },
            'platinum': {
                'brightness': 1.03,
                'contrast': 1.08,
                'saturation': 0.9,
                'highlights': 1.08
            }
        }
        
        params = metal_params.get(metal_type, metal_params['white_gold'])
        
        # Apply enhancements
        enhanced = image.copy()
        
        # Brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(params['brightness'])
        
        # Contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # Saturation
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(params['saturation'])
        
        # Sharpness for detail
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in enhancement: {str(e)}")
        return image

def create_professional_background(image, background_color=(248, 248, 248)):
    """Create professional jewelry photography background"""
    try:
        # Create gradient background
        width, height = image.size
        background = Image.new('RGB', (width, height), background_color)
        
        # Create subtle gradient
        img_array = np.array(background)
        for y in range(height):
            fade = 1.0 - (y / height) * 0.1  # 10% gradient
            img_array[y, :] = img_array[y, :] * fade
        
        background = Image.fromarray(img_array.astype(np.uint8))
        
        # Composite the enhanced image onto background
        if image.mode == 'RGBA':
            background.paste(image, (0, 0), image)
        else:
            # Create mask for better blending
            mask = Image.new('L', image.size, 255)
            background.paste(image, (0, 0), mask)
        
        return background
        
    except Exception as e:
        logger.error(f"Error creating background: {str(e)}")
        return image

def create_thumbnail_perfect(image, ring_bbox, target_size=(1000, 1300)):
    """Create perfect thumbnail with complete border removal"""
    try:
        # Apply perfect border removal first
        cleaned_image, _ = detect_and_remove_black_border_perfect(image)
        
        # Re-detect ring in cleaned image
        new_ring_bbox = detect_ring(cleaned_image)
        if new_ring_bbox:
            x, y, w, h = new_ring_bbox
        else:
            # Use center area if detection fails
            w, h = cleaned_image.size
            x, y = w//4, h//4
            w, h = w//2, h//2
        
        # Ultra tight crop - only 1% margin for maximum zoom
        margin = 0.01
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        # Calculate crop area
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(cleaned_image.width, x + w + margin_w)
        y2 = min(cleaned_image.height, y + h + margin_h)
        
        # Crop the ring area
        cropped = cleaned_image.crop((x1, y1, x2, y2))
        
        # Calculate scaling to fill entire canvas (99% coverage)
        crop_w, crop_h = cropped.size
        scale_w = target_size[0] / crop_w * 0.99
        scale_h = target_size[1] / crop_h * 0.99
        scale = max(scale_w, scale_h)  # Use max to ensure full coverage
        
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        # Resize with high quality
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas with pure white background
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        
        # Center the resized image
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        
        canvas.paste(resized, (paste_x, paste_y))
        
        # Apply final enhancement to thumbnail
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(1.02)  # Slight brightness boost
        
        return canvas
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        # Fallback to simple resize
        return image.resize(target_size, Image.Resampling.LANCZOS)

def handler(event):
    """RunPod Serverless Handler with perfect border removal"""
    try:
        logger.info("Handler started - v70 Perfect")
        input_data = event.get("input", {})
        
        # Handle test connection
        if "test" in input_data or "prompt" in input_data:
            return {
                "output": {
                    "status": "connected",
                    "message": "Wedding Ring AI v70 Perfect Ready",
                    "version": "v70"
                }
            }
        
        # Get image data - support both field names for compatibility
        image_data = input_data.get("image") or input_data.get("image_base64")
        
        if not image_data:
            logger.error("No image data provided")
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "processing_info": {
                        "version": "v70",
                        "error_details": "Missing 'image' or 'image_base64' in input"
                    }
                }
            }
        
        # Process base64 padding if needed
        if isinstance(image_data, str):
            # Remove any whitespace
            image_data = image_data.strip()
            # Add padding if missing
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"Loaded image: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "error",
                    "processing_info": {"version": "v70"}
                }
            }
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (248, 248, 248))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 1: Perfect triple-pass border removal
        logger.info("Applying perfect border removal with triple-pass system...")
        processed_image, border_crop = detect_and_remove_black_border_perfect(image)
        
        # Step 2: Detect ring in the processed image
        logger.info("Detecting wedding ring...")
        ring_bbox = detect_ring(processed_image)
        
        if not ring_bbox:
            logger.warning("No ring detected, using center crop")
            w, h = processed_image.size
            ring_bbox = (w//4, h//4, w//2, h//2)
        else:
            logger.info(f"Ring detected at: {ring_bbox}")
        
        # Step 3: Enhance the ring
        metal_type = input_data.get("metal_type", "white_gold")
        lighting = input_data.get("lighting", "balanced")
        
        logger.info(f"Enhancing ring - Metal: {metal_type}, Lighting: {lighting}")
        enhanced_image = enhance_wedding_ring(processed_image, metal_type, lighting)
        
        # Step 4: Create professional background
        final_image = create_professional_background(enhanced_image)
        
        # Step 5: Create perfect thumbnail with complete border removal
        logger.info("Creating perfect thumbnail...")
        thumbnail = create_thumbnail_perfect(image, ring_bbox, target_size=(1000, 1300))
        
        # Convert images to base64
        # Main image
        main_buffer = BytesIO()
        final_image.save(main_buffer, format='JPEG', quality=98, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding for Make.com
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='JPEG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding for Make.com
        
        logger.info("Processing completed successfully - v70 Perfect")
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": list(image.size),
                    "processed_size": list(final_image.size),
                    "thumbnail_size": [1000, 1300],
                    "border_removed": True,
                    "removal_passes": 3,
                    "ring_detected": ring_bbox is not None,
                    "status": "success",
                    "version": "v70-perfect"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "processing_info": {
                    "version": "v70",
                    "error_type": type(e).__name__
                }
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
