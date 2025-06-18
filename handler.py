import runpod
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter

def handler(event):
    try:
        # Input validation
        if "input" not in event:
            return {"error": "No input provided"}
        
        image_base64 = event["input"].get("image_base64")
        if not image_base64:
            return {"error": "No image_base64 provided"}
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image"}
        
        # Step 1: Remove black border completely
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the actual content area (non-black area)
        # Use multiple thresholds to ensure complete black removal
        _, thresh1 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        _, thresh3 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Combine all thresholds
        combined_thresh = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, thresh3))
        
        # Apply morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, use the whole image
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        else:
            # Find the largest contour (should be the main content)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small padding to avoid cutting edges
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop to remove black border
        cropped = image[y:y+h, x:x+w]
        
        # Step 2: Create clean white background
        # Get the original aspect ratio
        original_h, original_w = cropped.shape[:2]
        aspect_ratio = original_w / original_h
        
        # Create new white background with original aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            new_w = max(original_w, 2048)
            new_h = int(new_w / aspect_ratio)
        else:  # Taller than wide
            new_h = max(original_h, 2048)
            new_w = int(new_h * aspect_ratio)
        
        # Create white background
        white_bg = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        
        # Resize cropped image to fit
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 3: Apply edge smoothing to blend with white background
        # Create mask for edge blending
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_resized, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate mask slightly for smoother edges
        kernel_small = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        
        # Apply Gaussian blur to mask for smooth blending
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend image with white background using the mask
        result = resized * mask_3ch + white_bg * (1 - mask_3ch)
        result = result.astype(np.uint8)
        
        # Step 4: Detect ring area for enhancement
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        # Create mask for metallic/shiny areas (rings)
        # Lower saturation areas with high value (brightness)
        lower_metal = np.array([0, 0, 150])
        upper_metal = np.array([180, 50, 255])
        metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
        
        # Clean up the mask
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
        
        # Find ring contours
        ring_contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create ring enhancement mask
        ring_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        if ring_contours:
            # Filter contours by area to get only significant ones
            min_area = 1000
            valid_contours = [c for c in ring_contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                cv2.drawContours(ring_mask, valid_contours, -1, 255, -1)
                # Dilate slightly for better coverage
                ring_mask = cv2.dilate(ring_mask, kernel, iterations=2)
                # Blur for smooth transition
                ring_mask = cv2.GaussianBlur(ring_mask, (15, 15), 0)
        
        # Step 5: Apply v13.3 wedding ring enhancement parameters
        # Detect metal type and lighting
        metal_type, lighting = detect_metal_and_lighting(result, ring_mask)
        
        # Get enhancement parameters
        params = get_v13_3_params(metal_type, lighting)
        
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements selectively
        if np.any(ring_mask > 0):
            # Create enhanced version
            enhanced = pil_image.copy()
            
            # Brightness
            if params['brightness'] != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(params['brightness'])
            
            # Contrast
            if params['contrast'] != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(params['contrast'])
            
            # Sharpness
            if params['sharpness'] != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(params['sharpness'])
            
            # Convert back to numpy
            enhanced_np = np.array(enhanced)
            original_np = np.array(pil_image)
            
            # Apply ring mask for selective enhancement
            ring_mask_3ch = np.stack([ring_mask/255.0]*3, axis=-1)
            final_enhanced = (enhanced_np * ring_mask_3ch + original_np * (1 - ring_mask_3ch)).astype(np.uint8)
            
            # Apply white overlay if needed
            if params['white_overlay'] > 0:
                white_layer = np.ones_like(final_enhanced) * 255
                overlay_mask = ring_mask_3ch * params['white_overlay']
                final_enhanced = (final_enhanced * (1 - overlay_mask) + white_layer * overlay_mask).astype(np.uint8)
            
            # Apply color temperature adjustment to ring area only
            if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
                # Convert to LAB
                lab = cv2.cvtColor(final_enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Apply color temperature to ring area
                lab[:,:,1] = np.where(ring_mask > 0, 
                                     np.clip(lab[:,:,1] + params['color_temp_a'], 0, 255),
                                     lab[:,:,1])
                lab[:,:,2] = np.where(ring_mask > 0,
                                     np.clip(lab[:,:,2] + params['color_temp_b'], 0, 255),
                                     lab[:,:,2])
                
                # Convert back to RGB
                final_enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            # Blend with original based on blend parameter
            if params['original_blend'] > 0:
                final_enhanced = (final_enhanced * (1 - params['original_blend']) + 
                                original_np * params['original_blend']).astype(np.uint8)
            
            final_pil = Image.fromarray(final_enhanced)
        else:
            final_pil = pil_image
        
        # Step 6: Final quality enhancement
        # Apply subtle unsharp mask for crisp details
        final_pil = final_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
        
        # Step 7: Create thumbnail (1000x1300)
        thumb = create_thumbnail(final_pil, ring_mask if np.any(ring_mask > 0) else None)
        
        # Convert to base64
        # Main image
        main_buffer = io.BytesIO()
        final_pil.save(main_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # Thumbnail
        thumb_buffer = io.BytesIO()
        thumb.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        # Return with output nesting structure (CRITICAL!)
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": f"{image.shape[1]}x{image.shape[0]}",
                    "cropped_size": f"{original_w}x{original_h}",
                    "final_size": f"{final_pil.width}x{final_pil.height}",
                    "thumbnail_size": "1000x1300",
                    "black_border_removed": True,
                    "enhancement_applied": True,
                    "version": "v20.0",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "failed",
                "version": "v20.0"
            }
        }

def detect_metal_and_lighting(image_np, ring_mask=None):
    """Detect metal type and lighting condition"""
    try:
        # Convert to RGB if needed
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Use ring area if mask provided, otherwise use center area
        if ring_mask is not None and np.any(ring_mask > 0):
            mask_bool = ring_mask > 128
            masked_pixels = image_np[mask_bool]
            if len(masked_pixels) > 0:
                avg_color = np.mean(masked_pixels, axis=0)
            else:
                avg_color = np.mean(image_np, axis=(0, 1))
        else:
            # Use center region
            h, w = image_np.shape[:2]
            center_y, center_x = h // 2, w // 2
            region_size = min(h, w) // 4
            center_region = image_np[center_y-region_size:center_y+region_size,
                                   center_x-region_size:center_x+region_size]
            avg_color = np.mean(center_region, axis=(0, 1))
        
        r, g, b = avg_color
        
        # Calculate color characteristics
        brightness = np.mean(avg_color)
        rg_ratio = r / (g + 1)
        rb_ratio = r / (b + 1)
        
        # Metal type detection
        if rg_ratio > 1.15 and rb_ratio > 1.2:
            metal_type = "rose_gold"
        elif brightness > 200 and abs(r - g) < 10 and abs(g - b) < 10:
            metal_type = "white_gold"
        elif rg_ratio > 1.05 and brightness > 180:
            metal_type = "yellow_gold"
        else:
            metal_type = "white_gold"  # Default
        
        # Lighting detection
        if brightness > 220:
            lighting = "bright"
        elif brightness < 180:
            lighting = "ambient"
        else:
            lighting = "natural"
        
        return metal_type, lighting
        
    except:
        return "white_gold", "natural"

def get_v13_3_params(metal_type, lighting):
    """Get v13.3 enhancement parameters"""
    params_v13_3 = {
        'white_gold': {
            'bright': {
                'brightness': 1.22,
                'contrast': 1.18,
                'white_overlay': 0.11,
                'sharpness': 1.20,
                'color_temp_a': -5,
                'color_temp_b': -5,
                'original_blend': 0.10
            },
            'natural': {
                'brightness': 1.18,
                'contrast': 1.12,
                'white_overlay': 0.09,
                'sharpness': 1.15,
                'color_temp_a': -3,
                'color_temp_b': -3,
                'original_blend': 0.15
            },
            'ambient': {
                'brightness': 1.25,
                'contrast': 1.20,
                'white_overlay': 0.13,
                'sharpness': 1.18,
                'color_temp_a': -4,
                'color_temp_b': -4,
                'original_blend': 0.12
            }
        },
        'yellow_gold': {
            'bright': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.08,
                'sharpness': 1.18,
                'color_temp_a': 3,
                'color_temp_b': 2,
                'original_blend': 0.12
            },
            'natural': {
                'brightness': 1.18,
                'contrast': 1.10,
                'white_overlay': 0.06,
                'sharpness': 1.15,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.18
            },
            'ambient': {
                'brightness': 1.23,
                'contrast': 1.18,
                'white_overlay': 0.10,
                'sharpness': 1.17,
                'color_temp_a': 3,
                'color_temp_b': 2,
                'original_blend': 0.15
            }
        },
        'rose_gold': {
            'bright': {
                'brightness': 1.18,
                'contrast': 1.12,
                'white_overlay': 0.07,
                'sharpness': 1.17,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.15
            },
            'natural': {
                'brightness': 1.15,
                'contrast': 1.08,
                'white_overlay': 0.06,
                'sharpness': 1.15,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.20
            },
            'ambient': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.09,
                'sharpness': 1.16,
                'color_temp_a': 3,
                'color_temp_b': 1,
                'original_blend': 0.18
            }
        },
        'mixed_metal': {
            'bright': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.09,
                'sharpness': 1.18,
                'color_temp_a': -1,
                'color_temp_b': -1,
                'original_blend': 0.12
            },
            'natural': {
                'brightness': 1.17,
                'contrast': 1.10,
                'white_overlay': 0.07,
                'sharpness': 1.15,
                'color_temp_a': 0,
                'color_temp_b': 0,
                'original_blend': 0.17
            },
            'ambient': {
                'brightness': 1.22,
                'contrast': 1.17,
                'white_overlay': 0.11,
                'sharpness': 1.17,
                'color_temp_a': -1,
                'color_temp_b': -1,
                'original_blend': 0.15
            }
        }
    }
    
    # Return parameters with fallback
    if metal_type in params_v13_3 and lighting in params_v13_3[metal_type]:
        return params_v13_3[metal_type][lighting]
    else:
        # Default fallback
        return params_v13_3['white_gold']['natural']

def create_thumbnail(pil_image, ring_mask=None):
    """Create 1000x1300 thumbnail with ring centered"""
    target_width = 1000
    target_height = 1300
    
    # If we have a ring mask, use it to find the ring area
    if ring_mask is not None and np.any(ring_mask > 0):
        # Find bounding box of the ring
        y_indices, x_indices = np.where(ring_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Add padding
            padding = 50
            y_min = max(0, y_min - padding)
            y_max = min(pil_image.height, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(pil_image.width, x_max + padding)
            
            # Crop to ring area
            ring_crop = pil_image.crop((x_min, y_min, x_max, y_max))
        else:
            ring_crop = pil_image
    else:
        ring_crop = pil_image
    
    # Calculate scale to fit in 1000x1300 while maintaining aspect ratio
    crop_w, crop_h = ring_crop.size
    scale_w = target_width / crop_w
    scale_h = target_height / crop_h
    scale = min(scale_w, scale_h) * 0.8  # 80% to leave some margin
    
    # Resize the cropped area
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = ring_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background
    thumbnail = Image.new('RGB', (target_width, target_height), 'white')
    
    # Paste centered
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    thumbnail.paste(resized, (x_offset, y_offset))
    
    # Apply subtle sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=0))
    
    return thumbnail

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
