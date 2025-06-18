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
        
        # Step 1: Aggressive black border removal
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find non-black pixels (anything above threshold is content)
        # Use very low threshold to catch all black areas
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours of the content
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours combined
            x_min, y_min = image.shape[1], image.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add small padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            # Crop image
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image
        
        # Step 2: Create pure white background
        h, w = cropped.shape[:2]
        
        # Scale up if image is too small
        min_dimension = 2048
        if w < min_dimension or h < min_dimension:
            scale = min_dimension / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            h, w = new_h, new_w
        
        # Convert to RGB for PIL
        rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Step 3: Ring detection and isolation
        # Convert to HSV for better detection
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # Create multiple masks for different ring materials
        # For metallic surfaces (low saturation, high value)
        mask1 = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 60, 255]))
        
        # For gold tones
        mask2 = cv2.inRange(hsv, np.array([10, 20, 100]), np.array([30, 255, 255]))
        
        # For silver/white gold
        mask3 = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
        
        # Combine masks
        ring_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (main ring area)
        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ring_mask_clean = np.zeros(ring_mask.shape, dtype=np.uint8)
            cv2.drawContours(ring_mask_clean, [largest_contour], -1, 255, -1)
            
            # Expand mask slightly for better coverage
            ring_mask_clean = cv2.dilate(ring_mask_clean, kernel, iterations=2)
            
            # Get ring bounding box for processing
            x, y, w, h = cv2.boundingRect(largest_contour)
            ring_region = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w//2, 'cy': y + h//2
            }
        else:
            ring_mask_clean = np.ones(ring_mask.shape, dtype=np.uint8) * 255
            ring_region = {
                'x': 0, 'y': 0, 'w': w, 'h': h,
                'cx': w//2, 'cy': h//2
            }
        
        # Step 4: Detect metal type and lighting
        metal_type, lighting = detect_metal_and_lighting(cropped, ring_mask_clean)
        
        # Step 5: Apply v13.3 enhancement
        params = get_v13_3_params(metal_type, lighting)
        
        # Create enhanced version
        enhanced = pil_image.copy()
        
        # Apply brightness
        if params['brightness'] != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(params['brightness'])
        
        # Apply contrast
        if params['contrast'] != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
        
        # Apply sharpness
        if params['sharpness'] != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
        
        # Convert to numpy for advanced processing
        enhanced_np = np.array(enhanced)
        original_np = np.array(pil_image)
        
        # Apply color temperature adjustment
        if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Apply adjustments
            lab[:,:,1] += params['color_temp_a']
            lab[:,:,2] += params['color_temp_b']
            
            # Clip values
            lab[:,:,1] = np.clip(lab[:,:,1], 0, 255)
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            
            # Convert back to RGB
            enhanced_np = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Apply white overlay to ring area only
        if params['white_overlay'] > 0:
            # Create white overlay
            white_overlay = np.ones_like(enhanced_np) * 255
            
            # Apply overlay using ring mask
            ring_mask_3ch = cv2.cvtColor(ring_mask_clean, cv2.COLOR_GRAY2BGR) / 255.0
            overlay_strength = params['white_overlay'] * ring_mask_3ch
            
            enhanced_np = (enhanced_np * (1 - overlay_strength) + white_overlay * overlay_strength).astype(np.uint8)
        
        # Blend with original
        if params['original_blend'] > 0:
            enhanced_np = (enhanced_np * (1 - params['original_blend']) + original_np * params['original_blend']).astype(np.uint8)
        
        # Step 6: Create clean white background composite
        # Create pure white background
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Create smooth mask for blending
        ring_mask_blur = cv2.GaussianBlur(ring_mask_clean, (21, 21), 0)
        ring_mask_3ch = cv2.cvtColor(ring_mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Composite enhanced ring onto white background
        final_image = (enhanced_np * ring_mask_3ch + white_bg * (1 - ring_mask_3ch)).astype(np.uint8)
        
        # Convert to PIL
        final_pil = Image.fromarray(final_image)
        
        # Apply final sharpening
        final_pil = final_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=0))
        
        # Step 7: Create thumbnail
        thumbnail = create_thumbnail(final_pil, ring_region)
        
        # Convert to base64
        # Main image
        main_buffer = io.BytesIO()
        final_pil.save(main_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # Thumbnail
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        # Return with output nesting
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": f"{image.shape[1]}x{image.shape[0]}",
                    "cropped_size": f"{w}x{h}",
                    "final_size": f"{final_pil.width}x{final_pil.height}",
                    "thumbnail_size": "1000x1300",
                    "black_border_removed": True,
                    "white_background": True,
                    "enhancement_applied": True,
                    "version": "v20.1",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "failed",
                "version": "v20.1"
            }
        }

def detect_metal_and_lighting(image_np, ring_mask):
    """Detect metal type and lighting condition"""
    try:
        # Get ring pixels only
        ring_pixels = image_np[ring_mask > 128]
        
        if len(ring_pixels) == 0:
            # Fallback to center region
            h, w = image_np.shape[:2]
            center_region = image_np[h//3:2*h//3, w//3:2*w//3]
            avg_color = np.mean(center_region, axis=(0, 1))
        else:
            avg_color = np.mean(ring_pixels, axis=0)
        
        # BGR to RGB
        b, g, r = avg_color
        
        # Calculate characteristics
        brightness = np.mean(avg_color)
        
        # Color ratios
        max_color = max(r, g, b)
        if max_color > 0:
            r_ratio = r / max_color
            g_ratio = g / max_color
            b_ratio = b / max_color
        else:
            r_ratio = g_ratio = b_ratio = 1
        
        # Detect metal type
        if r_ratio > 0.95 and g_ratio > 0.85 and g_ratio < 0.95:
            metal_type = "rose_gold"
        elif r_ratio > 0.95 and g_ratio > 0.9 and b_ratio < 0.8:
            metal_type = "yellow_gold"
        elif abs(r_ratio - g_ratio) < 0.05 and abs(g_ratio - b_ratio) < 0.05:
            metal_type = "white_gold"
        else:
            metal_type = "mixed_metal"
        
        # Detect lighting
        if brightness > 200:
            lighting = "bright"
        elif brightness < 150:
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

def create_thumbnail(pil_image, ring_region):
    """Create 1000x1300 thumbnail with ring centered"""
    target_width = 1000
    target_height = 1300
    
    # Get ring center and dimensions
    cx = ring_region['cx']
    cy = ring_region['cy']
    rw = ring_region['w']
    rh = ring_region['h']
    
    # Calculate crop area to center the ring
    # Make crop area larger than ring to include some background
    crop_factor = 1.5
    crop_w = int(rw * crop_factor)
    crop_h = int(rh * crop_factor)
    
    # Calculate crop coordinates
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(pil_image.width, x1 + crop_w)
    y2 = min(pil_image.height, y1 + crop_h)
    
    # Adjust if crop went out of bounds
    if x2 - x1 < crop_w:
        if x1 == 0:
            x2 = min(pil_image.width, crop_w)
        else:
            x1 = max(0, pil_image.width - crop_w)
    
    if y2 - y1 < crop_h:
        if y1 == 0:
            y2 = min(pil_image.height, crop_h)
        else:
            y1 = max(0, pil_image.height - crop_h)
    
    # Crop to ring area
    ring_crop = pil_image.crop((x1, y1, x2, y2))
    
    # Calculate scale to fit in target size
    crop_w, crop_h = ring_crop.size
    scale_w = target_width / crop_w
    scale_h = target_height / crop_h
    scale = min(scale_w, scale_h) * 0.8  # 80% to leave margin
    
    # Resize
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = ring_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background
    thumbnail = Image.new('RGB', (target_width, target_height), 'white')
    
    # Paste centered
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    thumbnail.paste(resized, (x_offset, y_offset))
    
    # Apply sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
    
    return thumbnail

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
