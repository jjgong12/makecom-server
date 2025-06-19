import base64
import cv2
import numpy as np
from PIL import Image
import io
import runpod
from typing import Dict, Tuple, List, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeddingRingProcessor:
    def __init__(self):
        """Initialize the wedding ring processor"""
        self.initialize_after_colors()
        
    def initialize_after_colors(self):
        """Initialize AFTER file background colors from 28 pairs"""
        self.after_bg_colors = {
            'gold': {
                'studio': (252, 250, 248),    # Very bright cream
                'natural': (254, 252, 250),   # Almost white
                'warm': (253, 251, 249)       # Bright warm white
            },
            'silver': {
                'studio': (251, 251, 251),    # Pure bright gray
                'natural': (253, 253, 253),   # Near white
                'cool': (252, 252, 254)       # Cool white
            },
            'rose_gold': {
                'studio': (254, 252, 250),    # Warm white
                'natural': (253, 251, 249),   # Natural white
                'warm': (255, 253, 251)       # Warmest white
            }
        }
    
    def base64_to_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy array"""
        try:
            # Remove any whitespace
            base64_str = base64_str.strip()
            
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Add padding if missing (for proper decode)
            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += '=' * (4 - missing_padding)
            
            # Decode base64
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return np.array(img)
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            raise

    def image_to_base64(self, image: np.ndarray, quality: int = 95) -> str:
        """Convert numpy array to base64 string without padding"""
        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            # Encode to base64 and remove padding
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_str = base64_str.rstrip('=')  # Remove padding
            
            return base64_str
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise

    def detect_black_frame(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Detect black frame around the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Check for black frame by examining edges
            edge_threshold = 50
            frame_detected = False
            
            # Check top edge
            top_edge = gray[:100, :]
            if np.mean(top_edge) < edge_threshold:
                frame_detected = True
                
            # Check bottom edge
            bottom_edge = gray[-100:, :]
            if np.mean(bottom_edge) < edge_threshold:
                frame_detected = True
                
            # Check left edge
            left_edge = gray[:, :100]
            if np.mean(left_edge) < edge_threshold:
                frame_detected = True
                
            # Check right edge
            right_edge = gray[:, -100:]
            if np.mean(right_edge) < edge_threshold:
                frame_detected = True
            
            if not frame_detected:
                return False, {}
            
            # Find the inner content boundaries
            # Scan from edges inward to find where black frame ends
            top = 0
            for i in range(h // 2):
                if np.mean(gray[i, w//4:3*w//4]) > edge_threshold:
                    top = i
                    break
                    
            bottom = h
            for i in range(h - 1, h // 2, -1):
                if np.mean(gray[i, w//4:3*w//4]) > edge_threshold:
                    bottom = i + 1
                    break
                    
            left = 0
            for i in range(w // 2):
                if np.mean(gray[h//4:3*h//4, i]) > edge_threshold:
                    left = i
                    break
                    
            right = w
            for i in range(w - 1, w // 2, -1):
                if np.mean(gray[h//4:3*h//4, i]) > edge_threshold:
                    right = i + 1
                    break
            
            info = {
                'has_frame': True,
                'inner_bbox': (left, top, right - left, bottom - top),
                'frame_thickness': {
                    'top': top,
                    'bottom': h - bottom,
                    'left': left,
                    'right': w - right
                }
            }
            
            return True, info
            
        except Exception as e:
            logger.error(f"Error detecting frame: {e}")
            return False, {}

    def remove_black_frame(self, image: np.ndarray, frame_info: Dict) -> np.ndarray:
        """Remove black frame and keep only the inner content"""
        try:
            if 'inner_bbox' not in frame_info:
                return image
                
            x, y, w, h = frame_info['inner_bbox']
            
            # Crop to inner content
            cropped = image[y:y+h, x:x+w]
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error removing frame: {e}")
            return image

    def detect_metal_type(self, image: np.ndarray) -> str:
        """Detect metal type from ring color"""
        try:
            # Get center region
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            size = min(h, w) // 4
            
            roi = image[center_y-size:center_y+size, center_x-size:center_x+size]
            
            # Calculate average color
            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            # Determine metal type based on color
            if r > g and r > b and (r - b) > 20:
                return 'rose_gold'
            elif abs(r - g) < 10 and abs(g - b) < 10:
                return 'silver'
            else:
                return 'gold'
                
        except Exception:
            return 'gold'  # Default

    def enhance_image(self, image: np.ndarray, metal_type: str) -> np.ndarray:
        """Apply v13.3 style enhancements"""
        try:
            result = image.copy()
            
            # Subtle brightness adjustment
            brightness_factor = 1.03
            result = np.clip(result * brightness_factor, 0, 255)
            
            # Gentle contrast enhancement
            result = result.astype(np.float32)
            result = (result - 128) * 1.02 + 128
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Slight sharpening
            kernel = np.array([[-0.5, -0.5, -0.5],
                             [-0.5,  5.0, -0.5],
                             [-0.5, -0.5, -0.5]]) / 2.0
            sharpened = cv2.filter2D(result, -1, kernel)
            result = cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image

    def create_thumbnail(self, image: np.ndarray) -> np.ndarray:
        """Create perfect thumbnail with extended background"""
        try:
            h, w = image.shape[:2]
            
            # Detect ring region by finding non-background pixels
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Get background color from image edges
            bg_samples = []
            # Top edge
            bg_samples.extend(image[0:10, :].reshape(-1, 3))
            # Bottom edge
            bg_samples.extend(image[-10:, :].reshape(-1, 3))
            # Left edge
            bg_samples.extend(image[:, 0:10].reshape(-1, 3))
            # Right edge
            bg_samples.extend(image[:, -10:].reshape(-1, 3))
            
            bg_color = np.median(bg_samples, axis=0).astype(np.uint8)
            
            # Find ring bounds
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (should be the ring)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest_contour)
                
                # Add margin
                margin = int(max(cw, ch) * 0.3)
                x = max(0, x - margin)
                y = max(0, y - margin)
                cw = min(w - x, cw + 2 * margin)
                ch = min(h - y, ch + 2 * margin)
            else:
                # Fallback to center crop
                x, y = w // 4, h // 4
                cw, ch = w // 2, h // 2
            
            # Target dimensions
            target_w, target_h = 1000, 1300
            
            # Create canvas with detected background color
            canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
            
            # Calculate scale to fit ring nicely (60% of width)
            desired_ring_width = int(target_w * 0.6)
            scale = desired_ring_width / cw
            
            # Check if height fits
            if ch * scale > target_h * 0.8:
                scale = (target_h * 0.8) / ch
            
            # Resize the cropped region
            cropped = image[y:y+ch, x:x+cw]
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Center placement
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            # Place on canvas
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            # Apply subtle brightness to match AFTER style
            metal_type = self.detect_metal_type(image)
            target_bg = self.after_bg_colors[metal_type]['studio']
            
            # Adjust background to target color
            for c in range(3):
                canvas[:, :, c] = np.where(
                    np.all(canvas == bg_color, axis=2),
                    target_bg[c],
                    canvas[:, :, c]
                )
            
            # Subtle brightness boost
            canvas = np.clip(canvas * 1.05, 0, 255).astype(np.uint8)
            
            return canvas
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            # Return a default thumbnail
            return np.full((1300, 1000, 3), (253, 251, 249), dtype=np.uint8)

    def process_image(self, base64_image: str) -> Dict:
        """Main processing pipeline"""
        try:
            # Convert base64 to image
            image = self.base64_to_image(base64_image)
            logger.info(f"Loaded image: {image.shape}")
            
            # Detect and remove black frame
            has_frame, frame_info = self.detect_black_frame(image)
            
            if has_frame:
                logger.info("Black frame detected, removing...")
                image = self.remove_black_frame(image, frame_info)
                logger.info(f"After frame removal: {image.shape}")
            
            # Detect metal type
            metal_type = self.detect_metal_type(image)
            logger.info(f"Detected metal type: {metal_type}")
            
            # Enhance image
            enhanced = self.enhance_image(image, metal_type)
            
            # Create thumbnail
            thumbnail = self.create_thumbnail(enhanced)
            logger.info(f"Created thumbnail: {thumbnail.shape}")
            
            # Convert to base64 without padding
            enhanced_base64 = self.image_to_base64(enhanced, quality=95)
            thumbnail_base64 = self.image_to_base64(thumbnail, quality=90)
            
            # Return with correct structure for Make.com
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                        "had_frame": has_frame,
                        "metal_type": metal_type,
                        "original_size": f"{image.shape[1]}x{image.shape[0]}",
                        "thumbnail_size": "1000x1300"
                    }
                }
            }
                    "thumbnail_size": "1000x1300"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

def handler(event):
    """RunPod handler function"""
    try:
        logger.info("Starting handler...")
        
        # Simple extraction like v64
        input_data = event.get("input", {})
        image_base64 = input_data.get("image_base64", "")
        
        if not image_base64:
            raise ValueError("No image_base64 provided in input")
        
        # Process image
        processor = WeddingRingProcessor()
        result = processor.process_image(image_base64)
        
        logger.info("Processing completed successfully")
        
        # Return result - RunPod will wrap this in {"output": ...}
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
