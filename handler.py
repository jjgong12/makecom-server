import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging
from typing import Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeddingRingEnhancerV23_2:
    def __init__(self):
        """v23.2 EXTREME - Complete black border removal + full thumbnail"""
        self.version = "v23.2_EXTREME"
        
        # v13.3 parameters from 28 Before/After learning data
        self.v13_3_params = {
            'white_gold': {
                'natural': {
                    'brightness': 1.29, 'contrast': 1.28, 'exposure': 0.42,
                    'highlights': 0.18, 'shadows': -0.24, 'vibrance': 0.28,
                    'saturation': 1.03, 'warmth': 1.03, 'white_overlay': 0.26,
                    'gamma': 1.13, 'color_temp_a': -6, 'color_temp_b': -4
                },
                'warm': {
                    'brightness': 1.31, 'contrast': 1.29, 'exposure': 0.44,
                    'highlights': 0.20, 'shadows': -0.26, 'vibrance': 0.29,
                    'saturation': 1.04, 'warmth': 1.06, 'white_overlay': 0.28,
                    'gamma': 1.14, 'color_temp_a': -8, 'color_temp_b': -5
                },
                'cool': {
                    'brightness': 1.27, 'contrast': 1.26, 'exposure': 0.40,
                    'highlights': 0.17, 'shadows': -0.23, 'vibrance': 0.26,
                    'saturation': 1.01, 'warmth': 0.99, 'white_overlay': 0.24,
                    'gamma': 1.12, 'color_temp_a': -4, 'color_temp_b': -3
                }
            },
            'yellow_gold': {
                'natural': {
                    'brightness': 1.24, 'contrast': 1.22, 'exposure': 0.36,
                    'highlights': 0.14, 'shadows': -0.19, 'vibrance': 0.21,
                    'saturation': 0.97, 'warmth': 0.95, 'white_overlay': 0.19,
                    'gamma': 1.10, 'color_temp_a': -3, 'color_temp_b': -2
                },
                'warm': {
                    'brightness': 1.26, 'contrast': 1.24, 'exposure': 0.38,
                    'highlights': 0.16, 'shadows': -0.21, 'vibrance': 0.23,
                    'saturation': 0.99, 'warmth': 0.98, 'white_overlay': 0.21,
                    'gamma': 1.11, 'color_temp_a': -5, 'color_temp_b': -3
                },
                'cool': {
                    'brightness': 1.22, 'contrast': 1.20, 'exposure': 0.34,
                    'highlights': 0.13, 'shadows': -0.18, 'vibrance': 0.19,
                    'saturation': 0.95, 'warmth': 0.92, 'white_overlay': 0.17,
                    'gamma': 1.09, 'color_temp_a': -2, 'color_temp_b': -1
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.26, 'contrast': 1.24, 'exposure': 0.38,
                    'highlights': 0.15, 'shadows': -0.20, 'vibrance': 0.23,
                    'saturation': 0.99, 'warmth': 0.97, 'white_overlay': 0.21,
                    'gamma': 1.11, 'color_temp_a': -4, 'color_temp_b': -3
                },
                'warm': {
                    'brightness': 1.28, 'contrast': 1.26, 'exposure': 0.40,
                    'highlights': 0.17, 'shadows': -0.22, 'vibrance': 0.25,
                    'saturation': 1.01, 'warmth': 1.00, 'white_overlay': 0.23,
                    'gamma': 1.12, 'color_temp_a': -6, 'color_temp_b': -4
                },
                'cool': {
                    'brightness': 1.24, 'contrast': 1.22, 'exposure': 0.36,
                    'highlights': 0.14, 'shadows': -0.19, 'vibrance': 0.21,
                    'saturation': 0.97, 'warmth': 0.94, 'white_overlay': 0.19,
                    'gamma': 1.10, 'color_temp_a': -3, 'color_temp_b': -2
                }
            },
            'champagne_gold': {
                'natural': {
                    'brightness': 1.25, 'contrast': 1.23, 'exposure': 0.37,
                    'highlights': 0.15, 'shadows': -0.20, 'vibrance': 0.22,
                    'saturation': 0.98, 'warmth': 0.96, 'white_overlay': 0.20,
                    'gamma': 1.11, 'color_temp_a': -4, 'color_temp_b': -3
                },
                'warm': {
                    'brightness': 1.27, 'contrast': 1.25, 'exposure': 0.39,
                    'highlights': 0.17, 'shadows': -0.22, 'vibrance': 0.24,
                    'saturation': 1.00, 'warmth': 0.99, 'white_overlay': 0.22,
                    'gamma': 1.12, 'color_temp_a': -6, 'color_temp_b': -4
                },
                'cool': {
                    'brightness': 1.23, 'contrast': 1.21, 'exposure': 0.35,
                    'highlights': 0.14, 'shadows': -0.19, 'vibrance': 0.20,
                    'saturation': 0.96, 'warmth': 0.93, 'white_overlay': 0.18,
                    'gamma': 1.10, 'color_temp_a': -3, 'color_temp_b': -2
                }
            }
        }
        
        # Background colors from 28 AFTER files
        self.after_bg_colors = {
            'white_gold': {
                'natural': [252, 250, 248], 'warm': [254, 252, 250], 'cool': [250, 248, 245]
            },
            'yellow_gold': {
                'natural': [251, 249, 246], 'warm': [253, 251, 248], 'cool': [249, 247, 244]
            },
            'rose_gold': {
                'natural': [252, 248, 246], 'warm': [254, 250, 248], 'cool': [250, 246, 244]
            },
            'champagne_gold': {
                'natural': [253, 251, 249], 'warm': [255, 253, 251], 'cool': [251, 249, 247]
            }
        }

    def detect_and_remove_black_border_v23_2(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """v23.2 EXTREME: Maximum black border removal (70% scan + threshold 150)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # v23.2 EXTREME: Scan up to 70% of image
        max_border = min(int(h * 0.7), int(w * 0.7), 700)
        
        borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        found_border = False
        
        # EXTREME top border detection
        for y in range(max_border):
            row_mean = np.mean(gray[y, :])
            if row_mean < 150:  # EXTREME threshold
                borders['top'] = y + 1
                found_border = True
            else:
                break
        
        # EXTREME bottom border detection
        for y in range(h-1, h-max_border-1, -1):
            row_mean = np.mean(gray[y, :])
            if row_mean < 150:
                borders['bottom'] = h - y
                found_border = True
            else:
                break
        
        # EXTREME left border detection
        for x in range(max_border):
            col_mean = np.mean(gray[:, x])
            if col_mean < 150:
                borders['left'] = x + 1
                found_border = True
            else:
                break
        
        # EXTREME right border detection
        for x in range(w-1, w-max_border-1, -1):
            col_mean = np.mean(gray[:, x])
            if col_mean < 150:
                borders['right'] = w - x
                found_border = True
            else:
                break
        
        if found_border:
            # v23.2: EXTREME safety margin 15 pixels
            safety_margin = 15
            top = max(0, borders['top'] + safety_margin)
            bottom = max(0, borders['bottom'] + safety_margin)
            left = max(0, borders['left'] + safety_margin)
            right = max(0, borders['right'] + safety_margin)
            
            # Crop with extreme margins
            cropped = image[top:h-bottom, left:w-right]
            
            # Secondary precision crop - EXTREME
            gray2 = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            h2, w2 = cropped.shape[:2]
            
            # Check edges and remove more aggressively
            edge_cut = 30  # EXTREME edge cut
            
            # Check all edges for any remaining dark pixels
            if h2 > 60 and w2 > 60:
                if np.mean(gray2[:30, :]) < 120:  # Top edge
                    cropped = cropped[edge_cut:, :]
                if np.mean(gray2[-30:, :]) < 120:  # Bottom edge
                    cropped = cropped[:-edge_cut, :]
                if np.mean(gray2[:, :30]) < 120:  # Left edge
                    cropped = cropped[:, edge_cut:]
                if np.mean(gray2[:, -30:]) < 120:  # Right edge
                    cropped = cropped[:, :-edge_cut]
            
            return cropped, True
        
        return image, False

    def detect_ring_bounds(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect wedding ring boundaries for perfect thumbnail cropping"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection to find ring
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding around the ring
            padding = 50
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return x, y, w, h
        
        # If no ring detected, return center crop
        h, w = image.shape[:2]
        crop_size = min(h, w) * 0.8
        x = int((w - crop_size) / 2)
        y = int((h - crop_size) / 2)
        return x, y, int(crop_size), int(crop_size)

    def create_perfect_thumbnail_v23_2(self, image: np.ndarray) -> np.ndarray:
        """v23.2: Wedding ring centered and filling 1000x1300 completely"""
        try:
            target_width, target_height = 1000, 1300
            
            # Detect ring boundaries
            x, y, w, h = self.detect_ring_bounds(image)
            
            # Crop to ring area
            ring_crop = image[y:y+h, x:x+w]
            
            # Calculate scale to fill target dimensions
            scale_w = target_width / w
            scale_h = target_height / h
            scale = max(scale_w, scale_h) * 0.95  # 95% to ensure ring fills frame
            
            # Calculate new dimensions
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize ring
            pil_ring = Image.fromarray(ring_crop)
            resized_ring = pil_ring.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create white canvas
            canvas = Image.new('RGB', (target_width, target_height), color=(248, 248, 248))
            
            # Center the ring on canvas
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # If resized image is larger than canvas, crop it
            if new_width > target_width or new_height > target_height:
                # Crop from center
                crop_x = max(0, (new_width - target_width) // 2)
                crop_y = max(0, (new_height - target_height) // 2)
                
                resized_ring = resized_ring.crop((
                    crop_x, 
                    crop_y, 
                    crop_x + min(new_width, target_width),
                    crop_y + min(new_height, target_height)
                ))
                paste_x = 0
                paste_y = 0
            
            canvas.paste(resized_ring, (paste_x, paste_y))
            
            return np.array(canvas)
            
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            # Fallback to simple resize
            return self.simple_thumbnail_fallback(image)
    
    def simple_thumbnail_fallback(self, image: np.ndarray) -> np.ndarray:
        """Simple fallback thumbnail creation"""
        target_width, target_height = 1000, 1300
        h, w = image.shape[:2]
        
        # Simple center crop
        if w > h:
            new_w = int(h * target_width / target_height)
            x_start = (w - new_w) // 2
            cropped = image[:, x_start:x_start + new_w]
        else:
            new_h = int(w * target_height / target_width)
            y_start = (h - new_h) // 2
            cropped = image[y_start:y_start + new_h, :]
        
        # Resize to target
        pil_image = Image.fromarray(cropped)
        resized = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return np.array(resized)

    def detect_metal_type(self, image: np.ndarray) -> str:
        """Detect metal type using color analysis"""
        try:
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            sample_size = min(h, w) // 4
            
            y1 = max(0, center_y - sample_size // 2)
            y2 = min(h, center_y + sample_size // 2)
            x1 = max(0, center_x - sample_size // 2)
            x2 = min(w, center_x + sample_size // 2)
            
            center_region = image[y1:y2, x1:x2]
            
            # Calculate color metrics
            avg_color = np.mean(center_region, axis=(0, 1))
            r, g, b = avg_color
            
            brightness = np.mean(avg_color)
            rg_ratio = r / (g + 1)
            gb_ratio = g / (b + 1)
            
            if brightness > 180:
                if abs(r - g) < 10 and abs(g - b) < 10:
                    return 'white_gold'
                elif r > g and rg_ratio > 1.08:
                    return 'rose_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'yellow_gold'
                
        except Exception as e:
            logger.error(f"Metal detection error: {e}")
            return 'white_gold'

    def detect_lighting_condition(self, image: np.ndarray) -> str:
        """Detect lighting condition"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Calculate distribution metrics
            mean_brightness = np.average(np.arange(256), weights=hist)
            
            # Analyze color temperature
            avg_color = np.mean(image, axis=(0, 1))
            r, g, b = avg_color
            
            warm_cool_ratio = (r - b) / (r + b + 1)
            
            if warm_cool_ratio > 0.05:
                return 'warm'
            elif warm_cool_ratio < -0.05:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            logger.error(f"Lighting detection error: {e}")
            return 'natural'

    def apply_v13_3_enhancement(self, image: np.ndarray, metal_type: str, lighting: str) -> np.ndarray:
        """Apply v13.3 10-step enhancement process"""
        params = self.v13_3_params.get(metal_type, {}).get(lighting, self.v13_3_params['white_gold']['natural'])
        
        # Convert to PIL for processing
        pil_image = Image.fromarray(image)
        
        # Step 1-2: Brightness and Contrast
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(params['brightness'])
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(params['contrast'])
        
        # Convert back to numpy for advanced processing
        img_array = np.array(pil_image).astype(np.float32)
        
        # Step 3: Exposure adjustment
        exposure_factor = 1 + params['exposure']
        img_array = np.clip(img_array * exposure_factor, 0, 255)
        
        # Step 4-5: Highlights and Shadows
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Highlights adjustment
        highlight_mask = (gray > 180).astype(np.float32)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
        highlight_adjustment = 1 + (params['highlights'] * highlight_mask[:, :, np.newaxis])
        img_array = np.clip(img_array * highlight_adjustment, 0, 255)
        
        # Shadows adjustment
        shadow_mask = (gray < 80).astype(np.float32)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        shadow_adjustment = 1 - (params['shadows'] * shadow_mask[:, :, np.newaxis])
        img_array = np.clip(img_array * shadow_adjustment, 0, 255)
        
        # Step 6-7: Vibrance and Saturation (safe method)
        pil_temp = Image.fromarray(img_array.astype(np.uint8))
        
        # Saturation
        enhancer = ImageEnhance.Color(pil_temp)
        pil_temp = enhancer.enhance(params['saturation'])
        img_array = np.array(pil_temp).astype(np.float32)
        
        # Step 8: Temperature adjustment
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + params['color_temp_a'], 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + params['color_temp_b'], 0, 255)
        
        # Step 9: White overlay
        white_overlay = np.ones_like(img_array) * 255
        alpha = params['white_overlay']
        img_array = (1 - alpha) * img_array + alpha * white_overlay
        
        # Step 10: Gamma correction
        img_array = np.clip(255 * np.power(img_array / 255, 1 / params['gamma']), 0, 255)
        
        return img_array.astype(np.uint8)

    def create_white_background(self, image: np.ndarray) -> np.ndarray:
        """Create perfect white background (248, 248, 248)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Multiple threshold approach for better masking
            _, mask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            _, mask2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Combine masks
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find largest contour (the ring)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Smooth mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 2)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
            
            # Create white background
            background = np.full_like(image, 248, dtype=np.uint8)
            
            # Composite
            result = image.astype(np.float32) * mask_3ch + background.astype(np.float32) * (1 - mask_3ch)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Background creation failed: {e}")
            return image

    def process_image(self, image_data: str, output_format: str = "enhanced") -> Dict:
        """v23.2 main processing function"""
        try:
            # Base64 decoding
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image.convert('RGB'))
            
            results = {}
            
            # 1. v23.2 EXTREME black border removal
            processed_image, border_removed = self.detect_and_remove_black_border_v23_2(image_np)
            logger.info(f"Border removal: {border_removed}")
            
            # 2. Detect metal type and lighting
            metal_type = self.detect_metal_type(processed_image)
            lighting = self.detect_lighting_condition(processed_image)
            logger.info(f"Detected: {metal_type} in {lighting} lighting")
            
            # 3. Apply v13.3 enhancement
            enhanced = self.apply_v13_3_enhancement(processed_image, metal_type, lighting)
            
            # 4. Create white background
            with_background = self.create_white_background(enhanced)
            
            # 5. 2x upscale
            h, w = with_background.shape[:2]
            pil_img = Image.fromarray(with_background)
            upscaled = pil_img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
            final_image = np.array(upscaled)
            
            # 6. Create perfect thumbnail (v23.2 - full frame)
            thumbnail = self.create_perfect_thumbnail_v23_2(with_background)
            
            # Convert to base64
            # Main image
            main_buffer = io.BytesIO()
            Image.fromarray(final_image).save(main_buffer, format='PNG', quality=95)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
            
            # Thumbnail
            thumb_buffer = io.BytesIO()
            Image.fromarray(thumbnail).save(thumb_buffer, format='PNG', quality=95)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
            
            return {
                "output": {
                    "enhanced_image": main_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "metal_type": metal_type,
                        "lighting": lighting,
                        "border_removed": border_removed,
                        "version": self.version,
                        "status": "success",
                        "enhancements": {
                            "black_border_removal": "v23.2_EXTREME",
                            "thumbnail": "ring_detection_full_frame",
                            "background": "white_248"
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                "output": {
                    "enhanced_image": image_data,
                    "thumbnail": image_data,
                    "processing_info": {
                        "status": "error",
                        "error": str(e),
                        "version": self.version
                    }
                }
            }

def handler(event):
    """RunPod handler function for v23.2"""
    try:
        # Initialize processor
        processor = WeddingRingEnhancerV23_2()
        
        # Get input
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Process image
        result = processor.process_image(image_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "output": {
                "enhanced_image": "",
                "thumbnail": "",
                "processing_info": {
                    "status": "error",
                    "error": str(e),
                    "version": "v23.2_EXTREME"
                }
            }
        }

if __name__ == "__main__":
    # Test mode
    print("Wedding Ring Enhancer v23.2 EXTREME - Ready")
