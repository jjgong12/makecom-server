import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 Wedding Ring Parameters (28 pairs learning data based)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.10,
            'sharpness': 1.10, 'color_temp_a': 0, 'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.03,
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 3,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.15,
            'sharpness': 1.16, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.15, 'saturation': 1.02
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.18,
            'sharpness': 1.20, 'color_temp_a': -8, 'color_temp_b': -8,
            'original_blend': 0.18, 'saturation': 1.00
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.12,
            'sharpness': 1.25, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.12, 'saturation': 1.05
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.06, 'white_overlay': 0.08,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 1,
            'original_blend': 0.28
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.02,
            'sharpness': 1.25, 'color_temp_a': 6, 'color_temp_b': 4,
            'original_blend': 0.18
        }
    }
}

# 28 pairs AFTER background colors (learning data based)
AFTER_BACKGROUND_COLORS = {
    'natural': {'light': [250, 248, 245], 'medium': [242, 240, 237], 'dark': [235, 232, 228]},
    'warm': {'light': [252, 248, 240], 'medium': [245, 240, 230], 'dark': [238, 232, 220]},
    'cool': {'light': [248, 250, 252], 'medium': [240, 242, 245], 'dark': [232, 235, 238]}
}

class PerfectWeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        self.bg_colors = AFTER_BACKGROUND_COLORS

    def detect_actual_line_thickness_safe(self, mask, bbox):
        """Adaptive line thickness detection with complete safety"""
        try:
            if bbox is None or len(bbox) != 4:
                return 70  # Default safe thickness
            
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                return 70
            
            # Safe boundary check
            if x >= mask.shape[1] or y >= mask.shape[0]:
                return 70
            if x + w > mask.shape[1] or y + h > mask.shape[0]:
                return 70
            
            measurements = []
            
            # Top edge thickness measurement (safe)
            if y + 10 < mask.shape[0] and x + w <= mask.shape[1]:
                top_line = mask[y:y+10, x:x+w]
                if top_line.size > 0:
                    thickness = np.sum(top_line > 0) // max(w, 1)
                    if thickness > 0:
                        measurements.append(thickness)
            
            # Bottom edge thickness measurement (safe)
            if y + h - 10 >= 0 and y + h <= mask.shape[0] and x + w <= mask.shape[1]:
                bottom_line = mask[y+h-10:y+h, x:x+w]
                if bottom_line.size > 0:
                    thickness = np.sum(bottom_line > 0) // max(w, 1)
                    if thickness > 0:
                        measurements.append(thickness)
            
            # Left edge thickness measurement (safe)
            if x + 10 < mask.shape[1] and y + h <= mask.shape[0]:
                left_line = mask[y:y+h, x:x+10]
                if left_line.size > 0:
                    thickness = np.sum(left_line > 0) // max(h, 1)
                    if thickness > 0:
                        measurements.append(thickness)
            
            # Right edge thickness measurement (safe)
            if x + w - 10 >= 0 and x + w <= mask.shape[1] and y + h <= mask.shape[0]:
                right_line = mask[y:y+h, x+w-10:x+w]
                if right_line.size > 0:
                    thickness = np.sum(right_line > 0) // max(h, 1)
                    if thickness > 0:
                        measurements.append(thickness)
            
            if len(measurements) > 0:
                median_thickness = int(np.median(measurements))
                # Add 50% safety margin
                final_thickness = int(median_thickness * 1.5)
                return max(50, min(final_thickness, 200))  # 50-200 pixel range
            else:
                return 70  # Default safe value
                
        except Exception:
            return 70  # Always return safe default

    def detect_black_border_safe(self, image):
        """Detect black border with AGGRESSIVE detection"""
        try:
            if image is None or image.size == 0:
                return None, None, None
            
            # Much more aggressive threshold detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Try VERY LOW thresholds to catch any dark lines
            masks = []
            for threshold in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                masks.append(binary)
            
            # Combine all masks - if ANY threshold detects it, include it
            combined_mask = np.zeros_like(gray)
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, None
            
            # Find largest rectangular contour (VERY RELAXED criteria)
            best_contour = None
            best_bbox = None
            best_area = 0
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Much more relaxed criteria - accept almost anything
                if w > 50 and h > 50 and area > best_area:
                    aspect_ratio = w / h if h > 0 else 1
                    if 0.2 <= aspect_ratio <= 5.0:  # Very relaxed aspect ratio
                        best_contour = contour
                        best_bbox = (x, y, w, h)
                        best_area = area
            
            if best_bbox is None:
                return None, None, None
            
            # Create precise mask for the detected border
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [best_contour], 255)
            
            return mask, best_contour, best_bbox
            
        except Exception:
            return None, None, None

    def detect_metal_type_safe(self, image, mask=None):
        """Safe metal type detection"""
        try:
            if image is None or image.size == 0:
                return 'white_gold'
            
            if mask is not None:
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) > 0:
                    rgb_values = image[mask_indices[0], mask_indices[1], :]
                    hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                    avg_hue = np.mean(hsv_values[:, 0])
                    avg_sat = np.mean(hsv_values[:, 1])
                else:
                    return 'white_gold'
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
            
            # Metal type classification
            if avg_hue < 15 or avg_hue > 165:
                if avg_sat > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_hue <= 35:
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'white_gold'
                
        except Exception:
            return 'white_gold'

    def detect_lighting_safe(self, image):
        """Safe lighting detection"""
        try:
            if image is None or image.size == 0:
                return 'natural'
            
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_channel = lab[:, :, 2]
            b_mean = np.mean(b_channel)
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
                
        except Exception:
            return 'natural'

    def extract_and_enhance_ring_safe(self, image, mask, bbox, border_thickness):
        """Safe ring extraction and enhancement with absolute protection"""
        try:
            if image is None or mask is None or bbox is None:
                return image
            
            x, y, w, h = bbox
            
            # Calculate safe inner margin (very conservative)
            inner_margin = border_thickness + max(50, border_thickness // 2)
            
            # Ensure inner region exists
            inner_x = x + inner_margin
            inner_y = y + inner_margin
            inner_w = w - 2 * inner_margin
            inner_h = h - 2 * inner_margin
            
            # Safety check for minimum size
            if inner_w <= 100 or inner_h <= 100:
                inner_margin = border_thickness + 30  # Minimum safety margin
                inner_x = x + inner_margin
                inner_y = y + inner_margin
                inner_w = w - 2 * inner_margin
                inner_h = h - 2 * inner_margin
            
            # Final boundary check
            if (inner_x < 0 or inner_y < 0 or 
                inner_x + inner_w > image.shape[1] or 
                inner_y + inner_h > image.shape[0] or
                inner_w <= 50 or inner_h <= 50):
                return image  # Return original if extraction is unsafe
            
            # Extract ring region safely
            ring_region = image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w].copy()
            
            if ring_region.size == 0:
                return image
            
            # Auto-detect metal type and lighting from ring region
            metal_type = self.detect_metal_type_safe(ring_region)
            lighting = self.detect_lighting_safe(ring_region)
            
            # Apply v13.3 enhancement
            enhanced_ring = self.enhance_wedding_ring_v13_3(ring_region, metal_type, lighting)
            
            # Scale up ring region (2x expansion)
            scale_factor = 2.0
            new_w = int(inner_w * scale_factor)
            new_h = int(inner_h * scale_factor)
            expanded_ring = cv2.resize(enhanced_ring, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Scale down to original size
            final_ring = cv2.resize(expanded_ring, (inner_w, inner_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Place back into original image
            result = image.copy()
            result[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = final_ring
            
            return result
            
        except Exception:
            return image  # Return original image on any error

    def enhance_wedding_ring_v13_3(self, image, metal_type, lighting):
        """v13.3 Wedding ring enhancement (28 pairs learning data based)"""
        try:
            if image is None or image.size == 0:
                return image
            
            # Get parameters for this metal and lighting combination
            params = self.params.get(metal_type, {}).get(lighting, 
                     self.params['white_gold']['natural'])
            
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image)
            
            # 1. Brightness adjustment
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 2. Contrast adjustment
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 3. Sharpness adjustment
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # 4. Convert back to numpy for advanced processing
            enhanced_array = np.array(enhanced)
            
            # 5. White overlay ("slightly white tinted feeling")
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 6. LAB color space temperature adjustment
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)  # A channel
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)  # B channel
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 7. Saturation adjustment (for champagne gold)
            if 'saturation' in params:
                hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)
                enhanced_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 8. Noise reduction
            enhanced_array = cv2.bilateralFilter(enhanced_array, 9, 75, 75)
            
            # 9. CLAHE (limited)
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 10. Gamma correction
            gamma = params.get('gamma', 1.0)
            if gamma != 1.0:
                enhanced_array = np.power(enhanced_array / 255.0, gamma) * 255.0
                enhanced_array = enhanced_array.astype(np.uint8)
            
            # 11. Blend with original (naturalness guarantee)
            original_blend = params['original_blend']
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                image, original_blend, 0
            )
            
            return final.astype(np.uint8)
            
        except Exception:
            return image

    def get_after_background_color_safe(self, lighting, uniformity='medium'):
        """Get 28 pairs AFTER background color safely"""
        try:
            colors = self.bg_colors.get(lighting, self.bg_colors['natural'])
            color = colors.get(uniformity, colors['medium'])
            return np.array(color, dtype=np.uint8)
        except Exception:
            return np.array([240, 240, 240], dtype=np.uint8)  # Safe default

    def remove_black_border_advanced_safe(self, image, mask):
        """Advanced black border removal with perfect safety"""
        try:
            if image is None or mask is None or image.size == 0 or mask.size == 0:
                return image
            
            # Try advanced inpainting first
            try:
                inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
                
                # Quality check: if inpainting result looks good, use it
                non_mask_area = image[mask == 0]
                inpainted_area = inpainted[mask > 0]
                
                if len(non_mask_area) > 0 and len(inpainted_area) > 0:
                    color_diff = np.mean(np.abs(np.mean(non_mask_area, axis=0) - np.mean(inpainted_area, axis=0)))
                    if color_diff < 30:  # Good inpainting result
                        return inpainted
            except Exception:
                pass
            
            # Fallback: use background color replacement
            lighting = self.detect_lighting_safe(image)
            bg_color = self.get_after_background_color_safe(lighting)
            
            result = image.copy()
            result[mask > 0] = bg_color
            
            # Smooth the edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)
            edge_mask = dilated_mask - mask
            
            if np.any(edge_mask):
                blurred = cv2.GaussianBlur(result, (7, 7), 0)
                result = np.where(edge_mask[..., None] > 0, blurred, result)
            
            return result.astype(np.uint8)
            
        except Exception:
            return image

    def basic_upscale_safe(self, image, scale=2):
        """Safe 2x upscaling"""
        try:
            if image is None or image.size == 0:
                return image
            
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width <= 0 or new_height <= 0:
                return image
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        except Exception:
            return image

    def create_thumbnail_safe(self, image, bbox, target_size=(1000, 1300)):
        """Safe thumbnail creation with exact 1000x1300 cropping"""
        try:
            if image is None or bbox is None or image.size == 0:
                # Return a default thumbnail if input is invalid
                canvas = np.full((target_size[1], target_size[0], 3), 240, dtype=np.uint8)
                return canvas
            
            x, y, w, h = bbox
            
            # Add 30% margin around ring
            margin_w = int(w * 0.3)
            margin_h = int(h * 0.3)
            
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image.shape[1], x + w + margin_w)
            y2 = min(image.shape[0], y + h + margin_h)
            
            # Safety check
            if x2 <= x1 or y2 <= y1:
                canvas = np.full((target_size[1], target_size[0], 3), 240, dtype=np.uint8)
                return canvas
            
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                canvas = np.full((target_size[1], target_size[0], 3), 240, dtype=np.uint8)
                return canvas
            
            # Calculate scaling to fit 1000x1300 while maintaining aspect ratio
            target_w, target_h = target_size
            crop_h, crop_w = cropped.shape[:2]
            
            # Calculate ratios
            ratio_w = target_w / crop_w
            ratio_h = target_h / crop_h
            ratio = min(ratio_w, ratio_h)
            
            # Resize cropped image
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            if new_w <= 0 or new_h <= 0:
                canvas = np.full((target_size[1], target_size[0], 3), 240, dtype=np.uint8)
                return canvas
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create 1000x1300 canvas and center the resized image
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            
            # Calculate centering position
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            # Ensure positioning is safe
            if (start_y >= 0 and start_x >= 0 and 
                start_y + new_h <= target_h and 
                start_x + new_w <= target_w):
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            return canvas
            
        except Exception:
            # Always return a valid thumbnail even on error
            canvas = np.full((target_size[1], target_size[0], 3), 240, dtype=np.uint8)
            return canvas

def handler(event):
    """RunPod Serverless main handler with complete error safety"""
    
    # Initialize ALL variables at the very beginning
    final_image = None
    thumbnail = None
    enhanced_image_b64 = None
    thumbnail_b64 = None
    
    try:
        input_data = event.get("input", {})
        
        # Handle connection test
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"Wedding Ring AI v15.3.5 Ready: {input_data['prompt']}",
                "status": "ready_for_processing",
                "capabilities": ["adaptive_thickness_detection", "v13.3_enhancement", "perfect_thumbnails"]
            }
        
        # Handle image processing
        if "image_base64" not in input_data:
            return {"error": "No image_base64 provided"}
        
        # Decode image safely
        try:
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return {"error": f"Image decoding failed: {str(e)}"}
        
        # Initialize processor
        processor = PerfectWeddingRingProcessor()
        
        # Step 1: Detect black border
        mask, contour, bbox = processor.detect_black_border_safe(image_array)
        
        if mask is None or bbox is None:
            return {"error": "Black border not detected"}
        
        # Step 2: Measure actual line thickness
        border_thickness = processor.detect_actual_line_thickness_safe(mask, bbox)
        
        # Step 3: Extract and enhance ring with v13.3
        enhanced = processor.extract_and_enhance_ring_safe(image_array, mask, bbox, border_thickness)
        
        # Step 4: 2x upscaling
        upscaled = processor.basic_upscale_safe(enhanced, scale=2)
        
        # Scale mask accordingly
        upscaled_mask = processor.basic_upscale_safe(mask, scale=2)
        upscaled_mask = np.where(upscaled_mask > 127, 255, 0).astype(np.uint8)
        
        # Update bbox for upscaled image
        upscaled_bbox = tuple(int(x * 2) for x in bbox)
        
        # Step 5: Remove black border
        final_image = processor.remove_black_border_advanced_safe(upscaled, upscaled_mask)
        
        # Step 6: Create 1000x1300 thumbnail
        thumbnail = processor.create_thumbnail_safe(final_image, upscaled_bbox)
        
        # Ensure we have valid results
        if final_image is None:
            final_image = upscaled  # Use upscaled as fallback
        
        if thumbnail is None:
            # Create emergency thumbnail
            thumbnail = np.full((1300, 1000, 3), 240, dtype=np.uint8)
        
        # Step 7: Encode results safely
        try:
            # Main image encoding
            main_pil = Image.fromarray(final_image.astype(np.uint8))
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            enhanced_image_b64 = base64.b64encode(main_buffer.getvalue()).decode()
        except Exception as e:
            return {"error": f"Main image encoding failed: {str(e)}"}
        
        try:
            # Thumbnail encoding
            thumb_pil = Image.fromarray(thumbnail.astype(np.uint8))
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        except Exception as e:
            return {"error": f"Thumbnail encoding failed: {str(e)}"}
        
        # Metal detection for info
        metal_type = processor.detect_metal_type_safe(image_array, mask)
        lighting = processor.detect_lighting_safe(image_array)
        
        return {
            "enhanced_image": enhanced_image_b64,
            "thumbnail": thumbnail_b64,
            "processing_info": {
                "metal_type": metal_type,
                "lighting": lighting,
                "border_thickness": border_thickness,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                "thumbnail_size": "1000x1300",
                "v13_3_applied": True,
                "adaptive_thickness": True
            }
        }
        
    except Exception as e:
        # Emergency fallback: ensure we never return without final_image defined
        if final_image is None:
            # Create a minimal valid response
            try:
                # Try to create a basic image if possible
                if 'image_array' in locals():
                    final_image = image_array
                else:
                    # Create emergency image
                    final_image = np.full((800, 600, 3), 240, dtype=np.uint8)
                
                main_pil = Image.fromarray(final_image.astype(np.uint8))
                main_buffer = io.BytesIO()
                main_pil.save(main_buffer, format='JPEG', quality=95)
                enhanced_image_b64 = base64.b64encode(main_buffer.getvalue()).decode()
                
                return {
                    "enhanced_image": enhanced_image_b64,
                    "thumbnail": enhanced_image_b64,  # Use same image
                    "error": f"Processing failed but emergency result provided: {str(e)}"
                }
            except:
                pass
        
        return {"error": f"Complete processing failure: {str(e)}"}

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
