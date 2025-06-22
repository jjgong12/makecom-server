import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
import io
import os
import json
import re

# Import Replicate only when available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

class WeddingRingEnhancer:
    """v152 Wedding Ring Enhancement System - Speed Optimized"""
    
    def __init__(self):
        print("[v152] Initializing - Speed Optimized")
        self.replicate_client = None
        
        # 38 pairs learning data parameters - same as v151
        self.enhancement_params = {
            'yellow_gold': {
                'brightness': 1.15,
                'contrast': 1.08,
                'sharpness': 1.10,
                'saturation': 1.06,
                'white_overlay': 0.03,
                'temperature': 1.02,
                'clahe_limit': 2.5,
                'gamma': 0.98,
                'blend_original': 0.15,
                'h_shift': -2,
                's_mult': 0.95,
                'v_mult': 1.05,
                'warmth': 1.03,
                'target_rgb': (255, 235, 190)
            },
            'rose_gold': {
                'brightness': 1.12,
                'contrast': 1.10,
                'sharpness': 1.12,
                'saturation': 1.05,
                'white_overlay': 0.02,
                'temperature': 1.00,
                'clahe_limit': 2.8,
                'gamma': 0.97,
                'blend_original': 0.12,
                'h_shift': 0,
                's_mult': 0.92,
                'v_mult': 1.06,
                'warmth': 1.01,
                'target_rgb': (245, 220, 200)
            },
            'white_gold': {
                'brightness': 1.18,
                'contrast': 1.06,
                'sharpness': 1.08,
                'saturation': 0.98,
                'white_overlay': 0.08,
                'temperature': 0.98,
                'clahe_limit': 2.2,
                'gamma': 0.96,
                'blend_original': 0.10,
                'h_shift': 0,
                's_mult': 0.85,
                'v_mult': 1.08,
                'warmth': 0.99,
                'target_rgb': (250, 250, 245)
            },
            'plain_white': {
                'brightness': 1.35,
                'contrast': 1.02,
                'sharpness': 1.05,
                'saturation': 0.90,
                'white_overlay': 0.20,
                'temperature': 0.95,
                'clahe_limit': 2.0,
                'gamma': 0.92,
                'blend_original': 0.05,
                'h_shift': 2,
                's_mult': 0.75,
                'v_mult': 1.15,
                'warmth': 0.98,
                'target_rgb': (255, 253, 250)
            }
        }
        
        # 28 pairs AFTER background colors
        self.after_bg_colors = {
            'yellow_gold': np.array([248, 243, 238]),
            'rose_gold': np.array([245, 240, 235]),
            'white_gold': np.array([250, 248, 245]),
            'plain_white': np.array([252, 250, 248])
        }

    def _init_replicate_client(self):
        """Initialize Replicate client only when needed"""
        if self.replicate_client is None and REPLICATE_AVAILABLE:
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if api_token:
                try:
                    self.replicate_client = replicate.Client(api_token=api_token)
                except Exception as e:
                    self.replicate_client = None

    def decode_base64_image(self, base64_string):
        """Optimized base64 decoding"""
        try:
            # Clean the string
            base64_string = base64_string.strip()
            
            # Remove data URL prefix
            if 'base64,' in base64_string:
                base64_string = base64_string.split('base64,')[1]
            elif base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]
            
            # Direct decode first (fastest)
            try:
                image_data = base64.b64decode(base64_string, validate=True)
            except:
                # Add padding if needed
                base64_string = re.sub(r'[^A-Za-z0-9+/]', '', base64_string)
                missing_padding = len(base64_string) % 4
                if missing_padding:
                    base64_string += '=' * (4 - missing_padding)
                image_data = base64.b64decode(base64_string)
            
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
                
        except Exception as e:
            raise ValueError(f"Unable to decode image: {e}")

    def resize_for_processing(self, image, max_size=1500):
        """Resize image for faster processing if too large"""
        w, h = image.size
        if w > max_size or h > max_size:
            ratio = min(max_size/w, max_size/h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            print(f"[v152] Resizing from {w}x{h} to {new_w}x{new_h} for speed")
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS), (w, h)
        return image, None

    def detect_and_remove_black_borders_fast(self, image_np, metal_type):
        """Fast black border removal with natural gradient"""
        h, w = image_np.shape[:2]
        result = image_np.copy()
        
        bg_color = self.after_bg_colors.get(metal_type, np.array([250, 248, 245]))
        
        # Single threshold for speed
        threshold = 50
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        
        # Fast border detection - check only edges
        borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Top edge (check every 5 pixels for speed)
        for i in range(0, min(100, h//3), 5):
            if np.mean(gray[i, ::5] < threshold) > 0.8:
                borders['top'] = i + 5
            else:
                break
        
        # Bottom edge
        for i in range(0, min(100, h//3), 5):
            if np.mean(gray[h-i-1, ::5] < threshold) > 0.8:
                borders['bottom'] = i + 5
            else:
                break
        
        # Left edge
        for i in range(0, min(100, w//3), 5):
            if np.mean(gray[::5, i] < threshold) > 0.8:
                borders['left'] = i + 5
            else:
                break
        
        # Right edge
        for i in range(0, min(100, w//3), 5):
            if np.mean(gray[::5, w-i-1] < threshold) > 0.8:
                borders['right'] = i + 5
            else:
                break
        
        # Apply gradient transition (simplified for speed)
        if borders['top'] > 0:
            for i in range(borders['top']):
                alpha = i / borders['top']
                result[i, :] = result[i, :] * alpha + bg_color * (1 - alpha)
        
        if borders['bottom'] > 0:
            for i in range(borders['bottom']):
                alpha = i / borders['bottom']
                row = h - borders['bottom'] + i
                result[row, :] = result[row, :] * alpha + bg_color * (1 - alpha)
        
        if borders['left'] > 0:
            for i in range(borders['left']):
                alpha = i / borders['left']
                result[:, i] = result[:, i] * alpha + bg_color.reshape(1, 3) * (1 - alpha)
        
        if borders['right'] > 0:
            for i in range(borders['right']):
                alpha = i / borders['right']
                col = w - borders['right'] + i
                result[:, col] = result[:, col] * alpha + bg_color.reshape(1, 3) * (1 - alpha)
        
        return result

    def detect_masking_fast(self, image_np):
        """Fast masking detection"""
        h, w = image_np.shape[:2]
        
        # Quick grayscale conversion
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Check center region only (faster)
        center_x_start = int(w * 0.3)
        center_x_end = int(w * 0.7)
        center_y_start = int(h * 0.3)
        center_y_end = int(h * 0.7)
        
        center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        
        # Single gray range check
        gray_mask = ((center_region > 120) & (center_region < 160)).astype(np.uint8) * 255
        
        # Quick area check
        white_pixels = np.sum(gray_mask > 0)
        total_pixels = gray_mask.size
        
        if white_pixels > total_pixels * 0.2:
            # Find bounding box quickly
            coords = np.where(gray_mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                return {
                    'has_masking': True,
                    'type': 'central_box',
                    'bounds': {
                        'x': center_x_start + x_min,
                        'y': center_y_start + y_min,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    }
                }
        
        return {'has_masking': False, 'type': None}

    def remove_masking_with_replicate(self, image, masking_info):
        """Remove masking using Replicate API - only if really needed"""
        # Skip if masking area is too small
        bounds = masking_info['bounds']
        if bounds['width'] < 50 or bounds['height'] < 50:
            return image
        
        try:
            self._init_replicate_client()
            
            if not self.replicate_client:
                return image
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"
            
            # Create simple mask
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([
                bounds['x'] - 5,  # Small margin
                bounds['y'] - 5, 
                bounds['x'] + bounds['width'] + 5,
                bounds['y'] + bounds['height'] + 5
            ], fill=255)
            
            # Convert mask to base64
            mask_buffered = io.BytesIO()
            mask.save(mask_buffered, format="PNG", optimize=True)
            mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode('utf-8')
            mask_data_url = f"data:image/png;base64,{mask_base64}"
            
            # Run Replicate with timeout
            output = self.replicate_client.run(
                "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
                input={
                    "image": img_data_url,
                    "mask": mask_data_url,
                    "alpha_matting": True,
                    "alpha_matting_foreground_threshold": 240,
                    "alpha_matting_background_threshold": 50,
                    "alpha_matting_erode_size": 10
                }
            )
            
            if output:
                result_base64 = output.split(',')[1] if ',' in output else output
                result_data = base64.b64decode(result_base64)
                return Image.open(io.BytesIO(result_data))
            
        except Exception as e:
            pass
        
        return image

    def detect_metal_type_fast(self, image_np):
        """Fast metal type detection"""
        h, w = image_np.shape[:2]
        
        # Sample only center region for speed
        center_y, center_x = h // 2, w // 2
        sample_size = min(w, h) // 4
        
        roi = image_np[
            max(0, center_y - sample_size):min(h, center_y + sample_size),
            max(0, center_x - sample_size):min(w, center_x + sample_size)
        ]
        
        # Quick color analysis
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        # Quick HSV check
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        h_val, s, v = avg_hsv
        
        # Simple detection logic
        gold_score = (r - b) / (r + b + 1e-5)
        warm_score = (r + g) / (2 * b + 1e-5)
        white_score = min(r, g, b) / (max(r, g, b) + 1e-5)
        
        # Quick decision
        if gold_score > 0.15 and warm_score > 1.8:
            return 'yellow_gold'
        elif gold_score > 0.05 and s > 20 and 10 < h_val < 25:
            return 'rose_gold'
        elif white_score > 0.85 and s < 15 and v > 210:
            return 'plain_white'
        else:
            return 'white_gold'

    def enhance_wedding_ring_fast(self, image, metal_type):
        """Optimized 10-step enhancement"""
        params = self.enhancement_params[metal_type]
        enhanced = image.copy()
        
        # Steps 1-4: PIL enhancements (fast)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(params['brightness'])
        enhanced = ImageEnhance.Contrast(enhanced).enhance(params['contrast'])
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(params['sharpness'])
        enhanced = ImageEnhance.Color(enhanced).enhance(params['saturation'])
        
        # Step 5: White overlay
        if params['white_overlay'] > 0:
            white = Image.new('RGB', enhanced.size, (255, 255, 255))
            enhanced = Image.blend(enhanced, white, params['white_overlay'])
        
        # Convert to numpy once
        enhanced_np = np.array(enhanced)
        
        # Steps 6-7: Combined LAB operations
        lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:,:,1] *= params['temperature']
        lab[:,:,2] *= params['warmth']
        
        # Apply CLAHE on L channel
        lab[:,:,0] = lab[:,:,0].astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=params['clahe_limit'], tileGridSize=(4,4))  # Smaller grid
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        enhanced_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Step 8: Gamma correction
        enhanced_np = (255 * np.power(enhanced_np / 255.0, params['gamma'])).astype(np.uint8)
        
        # Step 9: HSV adjustment
        hsv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + params['h_shift']) % 180
        hsv[:,:,1] *= params['s_mult']
        hsv[:,:,2] *= params['v_mult']
        hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
        enhanced_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Step 10: Final blend
        enhanced = Image.fromarray(enhanced_np)
        final = Image.blend(enhanced, image, params['blend_original'])
        
        return final

    def apply_natural_blending_fast(self, image_np, metal_type):
        """Fast natural blending"""
        h, w = image_np.shape[:2]
        
        # Create simple edge mask
        mask = np.ones((h, w), dtype=np.float32)
        edge_width = 40  # Reduced from 60
        
        # Simple linear gradient (faster than quadratic)
        for i in range(edge_width):
            alpha = i / edge_width
            mask[i, :] *= alpha
            mask[h-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, w-i-1] *= alpha
        
        # Smaller Gaussian blur for speed
        mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Reduced from (31, 31)
        
        # Apply blending
        bg_color = self.after_bg_colors[metal_type]
        background = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        mask_3d = mask[:,:,np.newaxis]
        result = (image_np * mask_3d + background * (1 - mask_3d)).astype(np.uint8)
        
        return result

    def create_perfect_thumbnail_fast(self, image, masking_info=None):
        """Fast thumbnail creation"""
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Use masking bounds if available
        if masking_info and masking_info.get('has_masking'):
            bounds = masking_info['bounds']
            ring_center_x = bounds['x'] + bounds['width'] // 2
            ring_center_y = bounds['y'] + bounds['height'] // 2
            ring_size = max(bounds['width'], bounds['height'])
        else:
            # Simple center crop
            ring_center_x = w // 2
            ring_center_y = h // 2
            ring_size = min(w, h) * 0.5
        
        # Calculate crop region
        target_w, target_h = 1000, 1300
        scale = 0.85
        
        crop_w = int(ring_size / scale)
        crop_h = int(crop_w * 1.3)  # 1000:1300 ratio
        
        # Center crop
        x1 = max(0, ring_center_x - crop_w // 2)
        y1 = max(0, ring_center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # Adjust if out of bounds
        if x2 - x1 < crop_w:
            x1 = max(0, w - crop_w)
            x2 = w
        if y2 - y1 < crop_h:
            y1 = max(0, h - crop_h)
            y2 = h
        
        # Crop and resize
        cropped = image_np[y1:y2, x1:x2]
        thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # Faster
        
        return Image.fromarray(thumbnail)

    def process_image(self, base64_image):
        """Main processing pipeline - optimized for speed"""
        try:
            # Decode image
            image = self.decode_base64_image(base64_image)
            original_image = image.copy()
            
            # Resize if too large
            image, original_size = self.resize_for_processing(image)
            
            # Convert to numpy
            image_np = np.array(image)
            
            # Fast metal type detection
            metal_type = self.detect_metal_type_fast(image_np)
            
            # Fast border removal
            image_np = self.detect_and_remove_black_borders_fast(image_np, metal_type)
            
            # Fast masking detection
            masking_info = self.detect_masking_fast(image_np)
            
            # Remove masking only if significant
            if masking_info['has_masking'] and masking_info['type'] == 'central_box':
                bounds = masking_info['bounds']
                if bounds['width'] > 100 and bounds['height'] > 100:  # Only for large masks
                    image = self.remove_masking_with_replicate(Image.fromarray(image_np), masking_info)
                    image_np = np.array(image)
            
            # Fast natural blending
            image_np = self.apply_natural_blending_fast(image_np, metal_type)
            image = Image.fromarray(image_np)
            
            # Fast enhancement
            enhanced = self.enhance_wedding_ring_fast(image, metal_type)
            
            # Restore original size if needed
            if original_size:
                enhanced = enhanced.resize(original_size, Image.Resampling.LANCZOS)
            
            # Fast thumbnail creation
            thumbnail = self.create_perfect_thumbnail_fast(enhanced, masking_info)
            
            # Convert to base64 with optimization
            # Enhanced image
            buffer = io.BytesIO()
            enhanced.save(buffer, format='PNG', optimize=True, compress_level=1)  # Fast compression
            enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Thumbnail
            buffer_thumb = io.BytesIO()
            thumbnail.save(buffer_thumb, format='PNG', optimize=True, compress_level=1)
            thumbnail_base64 = base64.b64encode(buffer_thumb.getvalue()).decode('utf-8').rstrip('=')
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": "v152",
                    "metal_type": metal_type,
                    "masking_detected": masking_info['has_masking'],
                    "optimized": True
                }
            }
            
        except Exception as e:
            import traceback
            print(f"[v152] Error: {str(e)}")
            
            # Fast fallback
            try:
                buffer = io.BytesIO()
                original_image.save(buffer, format='PNG', compress_level=1)
                fallback_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
                
                # Quick thumbnail
                thumb = original_image.resize((1000, 1300), Image.Resampling.BILINEAR)
                buffer_thumb = io.BytesIO()
                thumb.save(buffer_thumb, format='PNG', compress_level=1)
                thumb_base64 = base64.b64encode(buffer_thumb.getvalue()).decode('utf-8').rstrip('=')
                
                return {
                    "enhanced_image": fallback_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "version": "v152",
                        "error": str(e),
                        "fallback": True
                    }
                }
            except:
                return {
                    "enhanced_image": base64_image.rstrip('='),
                    "thumbnail": base64_image.rstrip('='),
                    "processing_info": {
                        "version": "v152",
                        "error": "Critical error",
                        "fallback": True
                    }
                }

def handler(event):
    """RunPod handler function - optimized"""
    try:
        # Get input
        image_input = event.get("input", {})
        
        # Find image quickly
        base64_image = None
        
        # Check common keys only
        if isinstance(image_input, dict):
            for key in ['image', 'image_base64', 'base64']:
                if key in image_input and image_input[key]:
                    base64_image = image_input[key]
                    break
        
        # Check event directly if needed
        if not base64_image and isinstance(event, dict):
            for key in ['image', 'image_base64']:
                if key in event and event[key]:
                    base64_image = event[key]
                    break
        
        # Check if string
        if not base64_image:
            if isinstance(event, str) and len(event) > 100:
                base64_image = event
            elif isinstance(image_input, str) and len(image_input) > 100:
                base64_image = image_input
        
        if not base64_image:
            return {
                "output": {
                    "error": "No image provided",
                    "processing_info": {
                        "version": "v152",
                        "error": "No input image found"
                    }
                }
            }
        
        # Remove data URL prefix if present
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Process image
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(base64_image)
        
        return {
            "output": result
        }
        
    except Exception as e:
        import traceback
        return {
            "output": {
                "error": str(e),
                "processing_info": {
                    "version": "v152",
                    "error": "Handler exception",
                    "traceback": traceback.format_exc()
                }
            }
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
