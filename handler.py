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

    def detect_masking(self, image: np.ndarray) -> Tuple[bool, np.ndarray, Dict]:
        """Detect black masking in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Multi-threshold approach
            masks = []
            for threshold in [30, 40, 50, 60, 70, 80]:
                mask = gray < threshold
                masks.append(mask)
            
            # Combine all masks
            combined_mask = np.logical_or.reduce(masks)
            
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                            cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, 
                                            cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, None, {}
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if significant masking
            total_area = image.shape[0] * image.shape[1]
            if area < total_area * 0.01:  # Less than 1%
                return False, None, {}
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Create detailed mask including edges
            detailed_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(detailed_mask, [largest_contour], -1, 255, -1)
            
            # Expand mask to catch edges
            dilated_mask = cv2.dilate(detailed_mask, kernel, iterations=2)
            
            info = {
                'bbox': (x, y, w, h),
                'area': area,
                'mask_ratio': area / total_area
            }
            
            return True, dilated_mask, info
            
        except Exception as e:
            logger.error(f"Error in detect_masking: {e}")
            return False, None, {}

    def remove_black_masking(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove black masking and replace with appropriate background"""
        try:
            result = image.copy()
            
            # Detect metal type
            metal_type = self.detect_metal_type(image)
            
            # Get appropriate background color
            bg_color = self.after_bg_colors[metal_type]['studio']
            
            # Apply background color to masked areas
            result[mask > 0] = bg_color
            
            # Smooth edges
            # Create edge mask
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            edge_mask = dilated - mask
            
            # Blend edges
            for c in range(3):
                result[:, :, c] = np.where(
                    edge_mask > 0,
                    image[:, :, c] * 0.3 + bg_color[c] * 0.7,
                    result[:, :, c]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing masking: {e}")
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

    def create_thumbnail(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Create perfect thumbnail like in project knowledge"""
        try:
            x, y, w, h = bbox
            
            # Calculate crop area with margin
            margin = 0.3  # 30% margin around ring
            cx = x + w // 2
            cy = y + h // 2
            
            # Make square crop first
            size = int(max(w, h) * (1 + margin * 2))
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(image.shape[1], x1 + size)
            y2 = min(image.shape[0], y1 + size)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            # Target dimensions
            target_w, target_h = 1000, 1300
            
            # Create canvas with bright background
            canvas = np.full((target_h, target_w, 3), (253, 251, 249), dtype=np.uint8)
            
            # Calculate placement
            # Ring should occupy about 60% of width
            desired_ring_width = int(target_w * 0.6)
            scale = desired_ring_width / w
            
            new_w = int(cropped.shape[1] * scale)
            new_h = int(cropped.shape[0] * scale)
            
            # Ensure it fits
            if new_h > target_h * 0.8:
                scale = (target_h * 0.8) / cropped.shape[0]
                new_w = int(cropped.shape[1] * scale)
                new_h = int(cropped.shape[0] * scale)
            
            # Resize
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Center placement
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            # Place on canvas
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            # Subtle brightness boost for thumbnail
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
            
            # Detect masking
            has_masking, mask, masking_info = self.detect_masking(image)
            
            if has_masking:
                logger.info("Black masking detected, removing...")
                image = self.remove_black_masking(image, mask)
                bbox = masking_info['bbox']
            else:
                # Use center crop if no masking
                h, w = image.shape[:2]
                size = min(h, w) // 2
                cx, cy = w // 2, h // 2
                bbox = (cx - size // 2, cy - size // 2, size, size)
            
            # Detect metal type
            metal_type = self.detect_metal_type(image)
            logger.info(f"Detected metal type: {metal_type}")
            
            # Enhance image
            enhanced = self.enhance_image(image, metal_type)
            
            # Create thumbnail
            thumbnail = self.create_thumbnail(enhanced, bbox)
            logger.info(f"Created thumbnail: {thumbnail.shape}")
            
            # Convert to base64 without padding
            enhanced_base64 = self.image_to_base64(enhanced, quality=95)
            thumbnail_base64 = self.image_to_base64(thumbnail, quality=90)
            
            # Return with correct structure for Make.com
            return {
                "output": {
                    "enhanced_image": enhanced_base64,
                    "thumbnail": thumbnail_base64,
                    "processing_info": {
                        "had_masking": has_masking,
                        "metal_type": metal_type,
                        "original_size": f"{image.shape[1]}x{image.shape[0]}",
                        "thumbnail_size": "1000x1300"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

def handler(event):
    """RunPod handler function"""
    try:
        logger.info("Starting handler...")
        
        # Extract input
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
