import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def handler(event):
    """Wedding Ring Enhancement v19.0 - Complete Production Version"""
    try:
        # Get input
        input_data = event.get("input", {})
        image_base64 = input_data.get("image_base64", "")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "failed"
                }
            }
        
        # Decode image
        try:
            image_data = base64.b64decode(image_base64)
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_array = np.array(image_pil)
        except Exception as e:
            return {
                "output": {
                    "error": f"Image decode error: {str(e)}",
                    "status": "failed"
                }
            }
        
        # 1. Detect metal type and lighting
        metal_type, lighting = detect_metal_and_lighting(image_array)
        
        # 2. Detect and remove black frame
        cleaned_image = remove_black_frame_advanced(image_array)
        
        # 3. Apply v13.3 wedding ring enhancement
        enhanced_image = apply_wedding_ring_enhancement(cleaned_image, metal_type, lighting)
        
        # 4. Apply 6-stage quality improvement
        final_image = apply_quality_improvements(enhanced_image)
        
        # 5. Create thumbnail (1000x1300)
        thumbnail = create_thumbnail(final_image)
        
        # Convert to base64
        # Main image
        main_pil = Image.fromarray(final_image)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # Thumbnail
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        # Return with proper output structure
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "status": "success",
                    "version": "v19.0",
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "failed"
            }
        }

def detect_metal_and_lighting(image):
    """Detect metal type and lighting condition"""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate average values in center region
        h, w = image.shape[:2]
        center_y1, center_y2 = h//3, 2*h//3
        center_x1, center_x2 = w//3, 2*w//3
        center_region = hsv[center_y1:center_y2, center_x1:center_x2]
        
        avg_hue = np.mean(center_region[:, :, 0])
        avg_sat = np.mean(center_region[:, :, 1])
        avg_val = np.mean(center_region[:, :, 2])
        
        # Detect metal type based on hue and saturation
        if avg_sat < 30:  # Low saturation = white/silver metals
            if avg_val > 180:
                metal_type = 'white_gold'
            else:
                metal_type = 'champagne_gold'
        elif 10 <= avg_hue <= 25:  # Orange/pink hue
            metal_type = 'rose_gold'
        elif 25 <= avg_hue <= 40:  # Yellow hue
            metal_type = 'yellow_gold'
        else:
            metal_type = 'champagne_gold'  # Default
        
        # Detect lighting based on color temperature
        b_mean = np.mean(image[:, :, 2])  # Blue channel
        r_mean = np.mean(image[:, :, 0])  # Red channel
        
        if b_mean > r_mean * 1.1:
            lighting = 'cool'
        elif r_mean > b_mean * 1.1:
            lighting = 'warm'
        else:
            lighting = 'natural'
        
        return metal_type, lighting
        
    except:
        return 'champagne_gold', 'natural'

def remove_black_frame_advanced(image):
    """Remove black frame using advanced detection"""
    try:
        h, w = image.shape[:2]
        
        # Method 1: Adaptive threshold detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try multiple thresholds
        best_mask = None
        best_thickness = 0
        
        for threshold in [10, 20, 30]:
            # Create binary mask
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Detect frame thickness
            thickness = detect_frame_thickness(binary)
            
            if thickness > best_thickness and thickness < min(w, h) * 0.3:
                best_thickness = thickness
                best_mask = binary
        
        if best_thickness > 0:
            # Method 2: Ultra removal - remove detected frame + margin
            margin = 20
            removal_thickness = min(best_thickness + margin, min(w, h) // 4)
            
            # Get inner region
            inner_x1 = removal_thickness
            inner_y1 = removal_thickness
            inner_x2 = w - removal_thickness
            inner_y2 = h - removal_thickness
            
            if inner_x2 > inner_x1 and inner_y2 > inner_y1:
                # Crop inner region
                inner_region = image[inner_y1:inner_y2, inner_x1:inner_x2]
                
                # Create smooth transition
                result = create_smooth_transition(image, inner_region, removal_thickness)
                return result
        
        return image
        
    except:
        return image

def detect_frame_thickness(binary_mask):
    """Detect black frame thickness"""
    try:
        h, w = binary_mask.shape
        
        # Check from edges inward
        max_thickness = min(w, h) // 3
        
        # Top edge
        for i in range(max_thickness):
            if np.mean(binary_mask[i, w//4:3*w//4]) > 128:
                return i
        
        # Left edge
        for i in range(max_thickness):
            if np.mean(binary_mask[h//4:3*h//4, i]) > 128:
                return i
        
        return 0
        
    except:
        return 0

def create_smooth_transition(original, inner, thickness):
    """Create smooth transition between inner region and background"""
    try:
        h, w = original.shape[:2]
        inner_h, inner_w = inner.shape[:2]
        
        # Resize inner region to fill the frame
        scale_x = w / inner_w
        scale_y = h / inner_h
        scale = max(scale_x, scale_y)
        
        new_w = int(inner_w * scale)
        new_h = int(inner_h * scale)
        
        # Resize with high quality
        resized = cv2.resize(inner, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop if needed
        if new_w > w or new_h > h:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            resized = resized[start_y:start_y+h, start_x:start_x+w]
        
        return resized
        
    except:
        return original

def apply_wedding_ring_enhancement(image, metal_type='champagne_gold', lighting='natural'):
    """Apply v13.3 wedding ring enhancement parameters"""
    try:
        # Base parameters for champagne gold (most common)
        params = {
            'brightness': 1.22,
            'contrast': 1.18,
            'white_overlay': 0.15,
            'sharpness': 1.25,
            'saturation': 0.85,
            'gamma': 1.05
        }
        
        # Adjust for different metal types
        if metal_type == 'white_gold':
            params['brightness'] = 1.18
            params['white_overlay'] = 0.09
            params['saturation'] = 0.90
        elif metal_type == 'rose_gold':
            params['brightness'] = 1.15
            params['white_overlay'] = 0.06
            params['saturation'] = 1.05
        elif metal_type == 'yellow_gold':
            params['brightness'] = 1.12
            params['white_overlay'] = 0.05
            params['saturation'] = 1.10
        
        # Adjust for lighting
        if lighting == 'warm':
            params['brightness'] *= 0.98
            params['saturation'] *= 1.05
        elif lighting == 'cool':
            params['brightness'] *= 1.02
            params['saturation'] *= 0.95
        
        # Convert to PIL for enhancements
        pil_image = Image.fromarray(image)
        
        # Apply brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(params['brightness'])
        
        # Apply contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(params['contrast'])
        
        # Apply sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(params['sharpness'])
        
        # Convert back to numpy
        enhanced = np.array(pil_image)
        
        # Apply white overlay for champagne effect
        if params['white_overlay'] > 0:
            white_layer = np.ones_like(enhanced) * 255
            enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                                     white_layer, params['white_overlay'], 0)
        
        # Apply gamma correction
        inv_gamma = 1.0 / params['gamma']
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # Adjust saturation in HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * params['saturation']
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return enhanced
        
    except Exception as e:
        return image

def apply_quality_improvements(image):
    """Apply 6-stage quality improvement process"""
    try:
        result = image.copy()
        
        # Stage 1: Noise reduction
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
        
        # Stage 2: Additional sharpening
        pil_img = Image.fromarray(result)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)
        result = np.array(pil_img)
        
        # Stage 3: Brightness/Contrast fine-tuning
        # Convert to LAB for better color preservation
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Enhance L channel
        lab[:, :, 0] = lab[:, :, 0] * 1.05  # Slight brightness boost
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        # Enhance contrast in L channel
        l_mean = np.mean(lab[:, :, 0])
        lab[:, :, 0] = (lab[:, :, 0] - l_mean) * 1.1 + l_mean
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        # Stage 4: Color saturation enhancement
        lab[:, :, 1] = lab[:, :, 1] * 1.05  # Enhance a channel
        lab[:, :, 2] = lab[:, :, 2] * 1.05  # Enhance b channel
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        # Stage 5: White overlay for champagne gold effect
        # Reduce blue channel slightly for warmer tone
        lab[:, :, 2] = np.clip(lab[:, :, 2] - 5, 0, 255)
        
        # Stage 6: Convert back and final adjustments
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Final edge enhancement
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9
        result = cv2.filter2D(result, -1, kernel)
        
        # Ensure no clipping
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except:
        return image

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create centered thumbnail with exact size"""
    try:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit
        scale_w = target_w / w
        scale_h = target_h / h
        scale = max(scale_w, scale_h)  # Use max to fill the frame
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop to exact size
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        
        # Handle cases where resized is smaller than target
        if new_w < target_w or new_h < target_h:
            # Create canvas and center the image
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            paste_x = (target_w - new_w) // 2 if new_w < target_w else 0
            paste_y = (target_h - new_h) // 2 if new_h < target_h else 0
            canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
            return canvas
        else:
            # Crop to exact size
            thumbnail = resized[start_y:start_y+target_h, start_x:start_x+target_w]
            return thumbnail
        
    except:
        # Fallback: simple resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

# RunPod serverless handler registration
runpod.serverless.start({"handler": handler})
