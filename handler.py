import runpod
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import logging
from typing import Dict, Any, Tuple, Optional
import traceback
from datetime import datetime
import json
import os
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replicate API configuration
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

class WeddingRingEnhancer:
    def __init__(self):
        """Initialize the wedding ring enhancer with reference images"""
        self.setup_references()
        
    def setup_references(self):
        """Setup reference images for different gold types"""
        self.gold_references = {
            'yellow': {
                'rgb': np.array([255, 215, 0]),
                'lab': np.array([87, -2, 86]),
                'highlights': np.array([255, 245, 200]),
                'shadows': np.array([184, 134, 11])
            },
            'rose': {
                'rgb': np.array([183, 110, 121]),
                'lab': np.array([55, 25, 10]),
                'highlights': np.array([229, 186, 192]),
                'shadows': np.array([130, 70, 75])
            },
            'white': {
                'rgb': np.array([250, 250, 250]),
                'lab': np.array([98, 0, 2]),
                'highlights': np.array([255, 255, 255]),
                'shadows': np.array([200, 200, 200])
            },
            'unplated_white': {
                'rgb': np.array([240, 240, 235]),
                'lab': np.array([95, 0, 3]),
                'highlights': np.array([252, 252, 248]),
                'shadows': np.array([190, 190, 185])
            }
        }

    def detect_gold_type(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
        """Detect gold type from image"""
        if mask is not None:
            # Use mask to focus on ring area
            ring_pixels = image[mask > 0]
            if len(ring_pixels) == 0:
                ring_pixels = image.reshape(-1, 3)
        else:
            # Use center region if no mask
            h, w = image.shape[:2]
            center_region = image[h//3:2*h//3, w//3:2*w//3]
            ring_pixels = center_region.reshape(-1, 3)
        
        # Convert to LAB for better color analysis
        lab_pixels = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        
        # Calculate median color
        median_lab = np.median(lab_pixels, axis=0)
        
        # Determine gold type based on LAB values
        a_channel = median_lab[1]
        b_channel = median_lab[2]
        
        if b_channel > 20:  # Yellow tones
            return 'yellow'
        elif a_channel > 10:  # Red/pink tones
            return 'rose'
        elif median_lab[0] > 93:  # Very bright white
            return 'white'
        else:
            return 'unplated_white'

    def enhance_metallic_properties(self, image: np.ndarray, gold_type: str) -> np.ndarray:
        """Enhance metallic shine and reflections"""
        # Get reference colors
        ref = self.gold_references[gold_type]
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Enhance luminance for metallic effect
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.1, 0, 255)
        
        # Adjust color channels based on gold type
        if gold_type == 'yellow':
            lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.2, 0, 255)  # Enhance yellow
        elif gold_type == 'rose':
            lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.15, 0, 255)  # Enhance red
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Add specular highlights
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, highlights = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        highlights_colored = cv2.cvtColor(highlights, cv2.COLOR_GRAY2BGR)
        
        # Blend highlights
        highlight_color = ref['highlights'].reshape(1, 1, 3)
        highlights_colored = (highlights_colored / 255.0 * highlight_color).astype(np.uint8)
        
        return cv2.addWeighted(enhanced, 0.9, highlights_colored, 0.1, 0)

    def enhance_details(self, image: np.ndarray) -> np.ndarray:
        """Enhance fine details and textures"""
        # Unsharp masking for detail enhancement
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Edge enhancement
        edges = cv2.Canny(image, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Subtle edge addition
        result = cv2.addWeighted(sharpened, 0.95, edges_colored, 0.05, 0)
        
        return result

    def apply_color_correction(self, image: np.ndarray, gold_type: str) -> np.ndarray:
        """Apply color correction based on gold type"""
        ref = self.gold_references[gold_type]
        
        # Create color correction matrix
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Target LAB values
        target_lab = ref['lab']
        
        # Calculate current average
        mask = cv2.inRange(image_lab[:, :, 0], 30, 250)  # Avoid very dark/bright areas
        current_avg = cv2.mean(image_lab, mask=mask)[:3]
        
        # Calculate correction factors
        l_factor = 1.0 + (target_lab[0] - current_avg[0]) * 0.003
        a_factor = 1.0 + (target_lab[1] - current_avg[1]) * 0.005
        b_factor = 1.0 + (target_lab[2] - current_avg[2]) * 0.005
        
        # Apply corrections
        image_lab[:, :, 0] = np.clip(image_lab[:, :, 0] * l_factor, 0, 255)
        image_lab[:, :, 1] = np.clip(image_lab[:, :, 1] * a_factor + (target_lab[1] - current_avg[1]) * 0.1, 0, 255)
        image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] * b_factor + (target_lab[2] - current_avg[2]) * 0.1, 0, 255)
        
        return cv2.cvtColor(image_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality with CLAHE and sharpening"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

    def remove_background_with_replicate(self, image_base64: str) -> Dict[str, Any]:
        """Remove background using Replicate API"""
        try:
            headers = {
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }
            
            data = {
                "version": "fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                "input": {
                    "image": f"data:image/png;base64,{image_base64}"
                }
            }
            
            # Create prediction
            response = requests.post(REPLICATE_API_URL, headers=headers, json=data)
            response.raise_for_status()
            prediction = response.json()
            
            # Poll for result
            prediction_url = f"{REPLICATE_API_URL}/{prediction['id']}"
            max_attempts = 30
            
            for _ in range(max_attempts):
                time.sleep(2)
                response = requests.get(prediction_url, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                if result['status'] == 'succeeded':
                    output_url = result['output']
                    
                    # Download result
                    img_response = requests.get(output_url)
                    img_response.raise_for_status()
                    
                    # Convert to numpy array
                    img = Image.open(BytesIO(img_response.content)).convert('RGBA')
                    img_array = np.array(img)
                    
                    # Extract alpha channel as mask
                    mask = img_array[:, :, 3]
                    
                    # Convert RGBA to BGR
                    bgr_image = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
                    
                    return {
                        'success': True,
                        'image': bgr_image,
                        'mask': mask
                    }
                elif result['status'] == 'failed':
                    raise Exception(f"Replicate processing failed: {result.get('error', 'Unknown error')}")
            
            raise Exception("Timeout waiting for Replicate result")
            
        except Exception as e:
            logger.error(f"Replicate API error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def process_ring(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Main processing pipeline for ring enhancement"""
        # Detect gold type
        gold_type = self.detect_gold_type(image, mask)
        logger.info(f"Detected gold type: {gold_type}")
        
        # Apply enhancements
        enhanced = self.enhance_image(image)
        enhanced = self.enhance_metallic_properties(enhanced, gold_type)
        enhanced = self.apply_color_correction(enhanced, gold_type)
        enhanced = self.enhance_details(enhanced)
        
        return enhanced
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (1000, 1300)) -> str:
        """Create thumbnail with ring filling most of the frame"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # If image has alpha channel, find ring bounds
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
            coords = np.column_stack(np.where(alpha > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Add 2% padding
                h, w = y_max - y_min, x_max - x_min
                padding = int(max(h, w) * 0.02)
                
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding)
                y_max = min(img_array.shape[0], y_max + padding)
                x_max = min(img_array.shape[1], x_max + padding)
                
                # Crop to ring area
                cropped = image.crop((x_min, y_min, x_max, y_max))
            else:
                # Fallback: use center 98%
                h, w = img_array.shape[:2]
                margin = int(min(h, w) * 0.01)
                cropped = image.crop((margin, margin, w-margin, h-margin))
        else:
            # No alpha: use center crop
            h, w = img_array.shape[:2]
            margin = int(min(h, w) * 0.01)
            cropped = image.crop((margin, margin, w-margin, h-margin))
        
        # Create new image with target size
        thumb = Image.new('RGBA', size, (255, 255, 255, 0))
        
        # Resize cropped image to fit
        cropped_ratio = cropped.width / cropped.height
        target_ratio = size[0] / size[1]
        
        if cropped_ratio > target_ratio:
            # Fit to width
            new_width = size[0]
            new_height = int(size[0] / cropped_ratio)
        else:
            # Fit to height
            new_height = size[1]
            new_width = int(size[1] * cropped_ratio)
        
        # Resize with high quality
        resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Paste centered
        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2
        thumb.paste(resized, (x_offset, y_offset))
        
        # Convert to base64
        buffered = BytesIO()
        thumb.save(buffered, format="PNG", optimize=True, quality=95)
        thumb_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        return thumb_base64.rstrip('=')

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job["input"]
        logger.info(f"Processing job with input keys: {list(job_input.keys())}")
        
        # Get base64 image
        base64_image = job_input.get("image", "")
        if not base64_image:
            raise ValueError("No image provided in input")
        
        # Clean base64 string
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Initialize enhancer
        enhancer = WeddingRingEnhancer()
        
        # Try to remove background with Replicate
        bg_result = enhancer.remove_background_with_replicate(base64_image)
        
        if bg_result['success']:
            # Process with mask
            processed_image = enhancer.process_ring(bg_result['image'], bg_result['mask'])
            
            # Apply mask to final result
            mask = bg_result['mask']
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            processed_image = cv2.bitwise_and(processed_image, mask_3channel)
            
            # Create transparent background
            b, g, r = cv2.split(processed_image)
            rgba = cv2.merge([b, g, r, mask])
            
            # Convert to PIL for saving
            pil_image = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        else:
            # Fallback: process without background removal
            logger.warning(f"Background removal failed: {bg_result['error']}")
            
            # Decode and process image
            img_data = base64.b64decode(base64_image)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Process without mask
            processed_image = enhancer.process_ring(img_bgr)
            
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG", optimize=True, quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        # Create thumbnail
        thumbnail_base64 = enhancer.create_thumbnail(pil_image)
        
        # Get gold type for info
        if bg_result['success']:
            img_for_detection = bg_result['image']
            mask_for_detection = bg_result['mask']
        else:
            img_for_detection = img_bgr
            mask_for_detection = None
        
        detected_gold_type = enhancer.detect_gold_type(img_for_detection, mask_for_detection)
        
        # Prepare response
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "background_removed": bg_result['success'],
                    "detected_gold_type": detected_gold_type,
                    "timestamp": datetime.now().isoformat(),
                    "version": "v152-fixed"
                }
            }
        }
        
        logger.info("Successfully processed wedding ring image")
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_result = {
            "output": {
                "error": str(e),
                "fallback": True,
                "version": "v152-fixed"
            }
        }
        
        return error_result

# RunPod endpoint
runpod.serverless.start({"handler": handler})
