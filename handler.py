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

def detect_and_remove_black_border_ultra(image):
    """Ultra aggressive black border detection and removal"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for border detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        h, w = gray.shape
        
        # v23.1 ULTRA style parameters
        threshold = 120  # Aggressive threshold
        max_scan = int(min(h, w) * 0.5)  # 50% scan range
        
        # Find borders - check for continuous dark pixels
        top = 0
        for i in range(max_scan):
            # Check if 90% of the row is dark (more flexible than all)
            dark_pixels = np.sum(gray[i, :] <= threshold)
            if dark_pixels < w * 0.9:  # If less than 90% dark, stop
                break
            top = i + 1
        
        bottom = h
        for i in range(max_scan):
            dark_pixels = np.sum(gray[h-1-i, :] <= threshold)
            if dark_pixels < w * 0.9:
                break
            bottom = h - 1 - i
        
        left = 0
        for i in range(max_scan):
            dark_pixels = np.sum(gray[:, i] <= threshold)
            if dark_pixels < h * 0.9:
                break
            left = i + 1
        
        right = w
        for i in range(max_scan):
            dark_pixels = np.sum(gray[:, w-1-i] <= threshold)
            if dark_pixels < h * 0.9:
                break
            right = w - 1 - i
        
        # Minimal safety margin (v52 style)
        safety_margin = 10
        top = min(top + safety_margin, h // 4)
        bottom = max(bottom - safety_margin, 3 * h // 4)
        left = min(left + safety_margin, w // 4)
        right = max(right - safety_margin, 3 * w // 4)
        
        # First crop
        if top < bottom and left < right:
            if len(img_array.shape) == 3:
                cropped = img_array[top:bottom, left:right]
            else:
                cropped = img_array[top:bottom, left:right]
            
            # Second pass - v52 style ultra precision crop
            gray2 = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY) if len(cropped.shape) == 3 else cropped
            h2, w2 = gray2.shape
            
            # Check edges again for any remaining dark borders
            edge_cut = 25  # v52 ULTRA style
            
            # Top edge
            if np.mean(gray2[:20, :]) < 100:
                cropped = cropped[edge_cut:, :]
            
            # Bottom edge
            if np.mean(gray2[-20:, :]) < 100:
                cropped = cropped[:-edge_cut, :]
            
            # Left edge
            if np.mean(gray2[:, :20]) < 100:
                cropped = cropped[:, edge_cut:]
            
            # Right edge
            if np.mean(gray2[:, -20:]) < 100:
                cropped = cropped[:, :-edge_cut]
            
            logger.info(f"Ultra border removal complete. Original: {w}x{h}, Final: {cropped.shape[1]}x{cropped.shape[0]}")
            
            # Calculate total crop coordinates
            final_left = left + (edge_cut if np.mean(gray2[:, :20]) < 100 else 0)
            final_top = top + (edge_cut if np.mean(gray2[:20, :]) < 100 else 0)
            final_width = cropped.shape[1]
            final_height = cropped.shape[0]
            
            return Image.fromarray(cropped), (final_left, final_top, final_width, final_height)
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

def create_thumbnail_ultra_zoom(image, ring_bbox, target_size=(1000, 1300)):
    """Create thumbnail with v64 style - ultra zoom to fill canvas"""
    try:
        # Apply same border removal to ensure no black frames
        cleaned_image, _ = detect_and_remove_black_border_ultra(image)
        
        # Re-detect ring in cleaned image
        new_ring_bbox = detect_ring(cleaned_image)
        if new_ring_bbox:
            x, y, w, h = new_ring_bbox
        else:
            # Use original bbox adjusted for cleaning
            x, y, w, h = ring_bbox
            # Adjust for potential size change
            scale_x = cleaned_image.width / image.width
            scale_y = cleaned_image.height / image.height
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
        
        # Ultra tight crop - only 2% margin (v64 style)
        margin = 0.02
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        # Calculate crop area
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(cleaned_image.width, x + w + margin_w)
        y2 = min(cleaned_image.height, y + h + margin_h)
        
        # Crop the ring area
        cropped = cleaned_image.crop((x1, y1, x2, y2))
        
        # Calculate scaling to fill entire canvas (98% coverage)
        crop_w, crop_h = cropped.size
        scale_w = target_size[0] / crop_w * 0.98
        scale_h = target_size[1] / crop_h * 0.98
        scale = max(scale_w, scale_h)  # Use max to ensure full coverage
        
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        # Resize with high quality
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas with background color
        canvas = Image.new('RGB', target_size, (248, 248, 248))
        
        # Center the resized image (it might overflow, but that's okay)
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        
        canvas.paste(resized, (paste_x, paste_y))
        
        return canvas
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        # Fallback to simple resize
        return image.resize(target_size, Image.Resampling.LANCZOS)

def handler(event):
    """RunPod Serverless Handler with complete jewelry processing"""
    try:
        logger.info("Handler started - v69 Ultra")
        input_data = event.get("input", {})
        
        # Handle test connection
        if "test" in input_data or "prompt" in input_data:
            return {
                "output": {
                    "status": "connected",
                    "message": "Wedding Ring AI v69 Ultra Ready",
                    "version": "v69"
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
                        "version": "v69",
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
                    "processing_info": {"version": "v69"}
                }
            }
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (248, 248, 248))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 1: Ultra aggressive border removal
        logger.info("Detecting and removing black border with ULTRA mode...")
        processed_image, border_crop = detect_and_remove_black_border_ultra(image)
        
        # Step 2: Detect ring in the processed image
        logger.info("Detecting wedding ring...")
        ring_bbox = detect_ring(processed_image)
        
        if not ring_bbox:
            logger.warning("No ring detected, using center crop")
            w, h = processed_image.size
            ring_bbox = (w//4, h//4, w//2, h//2)
        
        # Step 3: Enhance the ring
        metal_type = input_data.get("metal_type", "white_gold")
        lighting = input_data.get("lighting", "balanced")
        
        logger.info(f"Enhancing ring - Metal: {metal_type}, Lighting: {lighting}")
        enhanced_image = enhance_wedding_ring(processed_image, metal_type, lighting)
        
        # Step 4: Create professional background
        final_image = create_professional_background(enhanced_image)
        
        # Step 5: Create thumbnail using v64 ultra zoom style with border removal
        logger.info("Creating ultra zoom thumbnail with border removal...")
        thumbnail = create_thumbnail_ultra_zoom(image, ring_bbox, target_size=(1000, 1300))
        
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
        
        logger.info("Processing completed successfully - v69 Ultra")
        
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
                    "border_removed": border_crop != (0, 0, image.width, image.height),
                    "ring_detected": ring_bbox is not None,
                    "status": "success",
                    "version": "v69-ultra"
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
                    "version": "v69",
                    "error_type": type(e).__name__
                }
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
