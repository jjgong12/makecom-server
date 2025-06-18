import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 Complete Parameters (4 metals x 3 lightings = 12 sets)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15,
            'saturation': 1.02,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18,
            'saturation': 1.00,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12,
            'saturation': 1.03,
            'gamma': 1.02
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15,
            'contrast': 1.08,
            'white_overlay': 0.06,
            'sharpness': 1.12,
            'color_temp_a': 5,
            'color_temp_b': 8,
            'original_blend': 0.20,
            'saturation': 1.05,
            'gamma': 0.99
        },
        'warm': {
            'brightness': 1.13,
            'contrast': 1.06,
            'white_overlay': 0.08,
            'sharpness': 1.10,
            'color_temp_a': 8,
            'color_temp_b': 12,
            'original_blend': 0.22,
            'saturation': 1.08,
            'gamma': 0.97
        },
        'cool': {
            'brightness': 1.17,
            'contrast': 1.10,
            'white_overlay': 0.04,
            'sharpness': 1.14,
            'color_temp_a': 3,
            'color_temp_b': 5,
            'original_blend': 0.18,
            'saturation': 1.03,
            'gamma': 1.01
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.12,
            'contrast': 1.05,
            'white_overlay': 0.03,
            'sharpness': 1.08,
            'color_temp_a': 12,
            'color_temp_b': 15,
            'original_blend': 0.25,
            'saturation': 1.10,
            'gamma': 0.96
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.03,
            'white_overlay': 0.05,
            'sharpness': 1.06,
            'color_temp_a': 15,
            'color_temp_b': 20,
            'original_blend': 0.28,
            'saturation': 1.12,
            'gamma': 0.94
        },
        'cool': {
            'brightness': 1.14,
            'contrast': 1.07,
            'white_overlay': 0.02,
            'sharpness': 1.10,
            'color_temp_a': 8,
            'color_temp_b': 10,
            'original_blend': 0.22,
            'saturation': 1.08,
            'gamma': 0.98
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.30,
            'contrast': 1.15,
            'white_overlay': 0.15,
            'sharpness': 1.20,
            'color_temp_a': -8,
            'color_temp_b': -12,
            'original_blend': 0.10,
            'saturation': 0.95,
            'gamma': 1.05
        },
        'warm': {
            'brightness': 1.28,
            'contrast': 1.13,
            'white_overlay': 0.18,
            'sharpness': 1.18,
            'color_temp_a': -10,
            'color_temp_b': -15,
            'original_blend': 0.12,
            'saturation': 0.92,
            'gamma': 1.03
        },
        'cool': {
            'brightness': 1.32,
            'contrast': 1.17,
            'white_overlay': 0.12,
            'sharpness': 1.22,
            'color_temp_a': -5,
            'color_temp_b': -8,
            'original_blend': 0.08,
            'saturation': 0.98,
            'gamma': 1.07
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
    
    def detect_black_frame(self, image):
        """Advanced black frame detection with ultra removal"""
        try:
            h, w = image.shape[:2]
            max_thickness = max(100, min(w, h) * 0.1)
            
            # Method 1: Adaptive threshold detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detected_regions = []
            
            for threshold in [10, 15, 20, 25, 30]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                
                # Check each edge
                edges = {
                    'top': binary[:int(max_thickness), :],
                    'bottom': binary[h-int(max_thickness):, :],
                    'left': binary[:, :int(max_thickness)],
                    'right': binary[:, w-int(max_thickness):]
                }
                
                for edge_name, edge_region in edges.items():
                    if edge_region.size > 0 and np.mean(edge_region) > 100:
                        detected_regions.append(edge_name)
            
            # Method 2: Ultra black frame removal
            result = image.copy()
            if detected_regions:
                # Fill detected regions with background color
                bg_color = self.get_background_color(image)
                
                if 'top' in detected_regions:
                    thickness = min(int(max_thickness), h//4)
                    result[:thickness, :] = bg_color
                
                if 'bottom' in detected_regions:
                    thickness = min(int(max_thickness), h//4)
                    result[h-thickness:, :] = bg_color
                
                if 'left' in detected_regions:
                    thickness = min(int(max_thickness), w//4)
                    result[:, :thickness] = bg_color
                
                if 'right' in detected_regions:
                    thickness = min(int(max_thickness), w//4)
                    result[:, w-thickness:] = bg_color
            
            return result
        except Exception as e:
            print(f"Black frame detection error: {e}")
            return image
    
    def get_background_color(self, image):
        """Get bright background color (250+ RGB)"""
        try:
            # Sample center region for background color
            h, w = image.shape[:2]
            center_region = image[h//3:2*h//3, w//3:2*w//3]
            avg_color = np.mean(center_region, axis=(0, 1))
            
            # Ensure bright background (minimum 250)
            bright_color = np.maximum(avg_color, [250, 250, 250])
            return bright_color.astype(np.uint8)
        except:
            return np.array([250, 250, 250], dtype=np.uint8)
    
    def detect_metal_type(self, image):
        """Detect metal type from HSV analysis"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
            
            if avg_sat < 30:
                return 'white_gold'
            elif 5 <= avg_hue <= 25:
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            elif 160 <= avg_hue <= 180:
                return 'rose_gold'
            else:
                return 'white_gold'
        except:
            return 'white_gold'
    
    def detect_lighting(self, image):
        """Detect lighting condition"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            avg_l = np.mean(lab[:, :, 0])
            avg_a = np.mean(lab[:, :, 1])
            avg_b = np.mean(lab[:, :, 2])
            
            if avg_b > 135:
                return 'warm'
            elif avg_b < 120:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def enhance_wedding_ring(self, image, metal_type, lighting):
        """6-stage ring quality enhancement"""
        try:
            params = self.params[metal_type][lighting]
            
            # Stage 1: Noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Stage 2: Sharpening
            pil_image = Image.fromarray(denoised)
            sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
            sharpened = sharpness_enhancer.enhance(params['sharpness'])
            
            # Stage 3: Brightness and contrast
            brightness_enhancer = ImageEnhance.Brightness(sharpened)
            brightened = brightness_enhancer.enhance(params['brightness'])
            
            contrast_enhancer = ImageEnhance.Contrast(brightened)
            contrasted = contrast_enhancer.enhance(params['contrast'])
            
            # Stage 4: Color enhancement
            color_enhancer = ImageEnhance.Color(contrasted)
            colored = color_enhancer.enhance(params['saturation'])
            
            # Stage 5: White overlay for champagne->white gold conversion
            enhanced_array = np.array(colored)
            if params['white_overlay'] > 0:
                white_overlay = np.full_like(enhanced_array, 255)
                enhanced_array = cv2.addWeighted(
                    enhanced_array, 1 - params['white_overlay'],
                    white_overlay, params['white_overlay'], 0
                )
            
            # Stage 6: LAB color temperature adjustment
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            final_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Final blend with original
            result = cv2.addWeighted(
                final_array, 1 - params['original_blend'],
                image, params['original_blend'], 0
            )
            
            return result.astype(np.uint8)
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
    
    def create_thumbnail(self, image, target_size=(1000, 1300)):
        """Create 1000x1300 thumbnail with center crop"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate crop dimensions to maintain aspect ratio
            ratio_w = target_w / w
            ratio_h = target_h / h
            ratio = max(ratio_w, ratio_h)  # Fill the frame
            
            # Resize image
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Center crop to exact target size
            start_x = (new_w - target_w) // 2 if new_w > target_w else 0
            start_y = (new_h - target_h) // 2 if new_h > target_h else 0
            
            if new_w >= target_w and new_h >= target_h:
                thumbnail = resized[start_y:start_y+target_h, start_x:start_x+target_w]
            else:
                # If resized image is smaller, pad with background
                thumbnail = np.full((target_h, target_w, 3), 250, dtype=np.uint8)
                paste_x = (target_w - new_w) // 2
                paste_y = (target_h - new_h) // 2
                thumbnail[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
            
            return thumbnail
        except Exception as e:
            print(f"Thumbnail creation error: {e}")
            # Return a default thumbnail
            return np.full((1300, 1000, 3), 250, dtype=np.uint8)

def handler(event):
    """RunPod Serverless main handler with nested output structure"""
    try:
        input_data = event["input"]
        
        if "image_base64" not in input_data:
            return {
                "output": {
                    "error": "No image_base64 provided",
                    "status": "error"
                }
            }
        
        # Decode base64 image
        image_data = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        # Initialize processor
        processor = WeddingRingProcessor()
        
        # Step 1: Remove black frames first
        frame_removed = processor.detect_black_frame(image_array)
        
        # Step 2: Detect metal type and lighting
        metal_type = processor.detect_metal_type(frame_removed)
        lighting = processor.detect_lighting(frame_removed)
        
        # Step 3: Enhance wedding ring quality
        enhanced = processor.enhance_wedding_ring(frame_removed, metal_type, lighting)
        
        # Step 4: Create thumbnail
        thumbnail = processor.create_thumbnail(enhanced)
        
        # Step 5: Encode results to base64
        # Main enhanced image
        main_pil = Image.fromarray(enhanced)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # Thumbnail image
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=90, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        # Return with nested output structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "status": "success",
                    "version": "v19.0"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v19.0"
            }
        }

# Start the serverless function
runpod.serverless.start({"handler": handler})
