import runpod
import requests
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
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
        """Initialize the wedding ring enhancer with v13.3 parameters"""
        self.setup_parameters()
        
    def setup_parameters(self):
        """Setup v13.3 complete parameters - 4 metals × 3 lightings = 12 sets"""
        self.enhancement_params = {
            'yellow_gold': {
                'natural': {
                    'brightness': 1.25, 'saturation': 1.15, 'contrast': 1.05,
                    'sharpness': 1.35, 'noise_reduction': 8,
                    'highlight_boost': 0.12, 'shadow_lift': 0.08,
                    'white_overlay': 0.15, 's_mult': 0.85, 'v_mult': 1.10
                },
                'warm': {
                    'brightness': 1.30, 'saturation': 1.20, 'contrast': 1.08,
                    'sharpness': 1.40, 'noise_reduction': 10,
                    'highlight_boost': 0.15, 'shadow_lift': 0.10,
                    'white_overlay': 0.18, 's_mult': 0.88, 'v_mult': 1.12
                },
                'cool': {
                    'brightness': 1.20, 'saturation': 1.10, 'contrast': 1.02,
                    'sharpness': 1.30, 'noise_reduction': 7,
                    'highlight_boost': 0.10, 'shadow_lift': 0.06,
                    'white_overlay': 0.12, 's_mult': 0.82, 'v_mult': 1.08
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.22, 'saturation': 1.12, 'contrast': 1.04,
                    'sharpness': 1.32, 'noise_reduction': 9,
                    'highlight_boost': 0.11, 'shadow_lift': 0.07,
                    'white_overlay': 0.13, 's_mult': 0.83, 'v_mult': 1.09
                },
                'warm': {
                    'brightness': 1.28, 'saturation': 1.18, 'contrast': 1.07,
                    'sharpness': 1.38, 'noise_reduction': 11,
                    'highlight_boost': 0.14, 'shadow_lift': 0.09,
                    'white_overlay': 0.16, 's_mult': 0.86, 'v_mult': 1.11
                },
                'cool': {
                    'brightness': 1.18, 'saturation': 1.08, 'contrast': 1.01,
                    'sharpness': 1.28, 'noise_reduction': 8,
                    'highlight_boost': 0.09, 'shadow_lift': 0.05,
                    'white_overlay': 0.10, 's_mult': 0.80, 'v_mult': 1.07
                }
            },
            'white_gold': {
                'natural': {
                    'brightness': 1.28, 'saturation': 0.95, 'contrast': 1.06,
                    'sharpness': 1.45, 'noise_reduction': 7,
                    'highlight_boost': 0.14, 'shadow_lift': 0.06,
                    'white_overlay': 0.20, 's_mult': 0.75, 'v_mult': 1.15
                },
                'warm': {
                    'brightness': 1.32, 'saturation': 0.98, 'contrast': 1.08,
                    'sharpness': 1.50, 'noise_reduction': 8,
                    'highlight_boost': 0.16, 'shadow_lift': 0.08,
                    'white_overlay': 0.22, 's_mult': 0.78, 'v_mult': 1.18
                },
                'cool': {
                    'brightness': 1.25, 'saturation': 0.92, 'contrast': 1.04,
                    'sharpness': 1.40, 'noise_reduction': 6,
                    'highlight_boost': 0.12, 'shadow_lift': 0.05,
                    'white_overlay': 0.18, 's_mult': 0.72, 'v_mult': 1.12
                }
            },
            'plain_white': {
                'natural': {
                    'brightness': 1.35, 'saturation': 0.90, 'contrast': 1.10,
                    'sharpness': 1.55, 'noise_reduction': 5,
                    'highlight_boost': 0.18, 'shadow_lift': 0.04,
                    'white_overlay': 0.28, 's_mult': 0.75, 'v_mult': 1.15
                },
                'warm': {
                    'brightness': 1.40, 'saturation': 0.93, 'contrast': 1.12,
                    'sharpness': 1.60, 'noise_reduction': 6,
                    'highlight_boost': 0.20, 'shadow_lift': 0.05,
                    'white_overlay': 0.30, 's_mult': 0.78, 'v_mult': 1.18
                },
                'cool': {
                    'brightness': 1.30, 'saturation': 0.88, 'contrast': 1.08,
                    'sharpness': 1.50, 'noise_reduction': 4,
                    'highlight_boost': 0.16, 'shadow_lift': 0.03,
                    'white_overlay': 0.25, 's_mult': 0.72, 'v_mult': 1.12
                }
            }
        }
        
        # Gold type reference colors
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
        """Detect gold type from image with white_score evaluation"""
        if mask is not None:
            ring_pixels = image[mask > 0]
            if len(ring_pixels) == 0:
                ring_pixels = image.reshape(-1, 3)
        else:
            h, w = image.shape[:2]
            center_region = image[h//3:2*h//3, w//3:2*w//3]
            ring_pixels = center_region.reshape(-1, 3)
        
        # Convert to LAB
        lab_pixels = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        median_lab = np.median(lab_pixels, axis=0)
        
        # Calculate white_score for additional detection
        white_score = np.mean([
            median_lab[0] / 100.0,  # Lightness
            1.0 - (abs(median_lab[1]) / 128.0),  # Low a*
            1.0 - (abs(median_lab[2]) / 128.0)   # Low b*
        ])
        
        # Determine gold type
        a_channel = median_lab[1]
        b_channel = median_lab[2]
        
        if b_channel > 20:
            return 'yellow'
        elif a_channel > 10:
            return 'rose'
        elif white_score > 0.85:
            return 'plain_white' if median_lab[0] > 93 else 'white'
        else:
            return 'unplated_white'

    def detect_lighting(self, image: np.ndarray) -> str:
        """Detect lighting condition"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_b = np.mean(lab[:, :, 2])
        
        if avg_b > 135:
            return 'warm'
        elif avg_b < 115:
            return 'cool'
        else:
            return 'natural'

    def apply_gradient_transition(self, image: np.ndarray, mask: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """Apply smooth gradient transition from ring to background"""
        result = image.copy()
        h, w = image.shape[:2]
        
        # Create distance transform from mask
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        
        if max_dist > 0:
            # Normalize distance for smooth transition
            transition_width = min(100, int(max_dist))
            
            for i in range(h):
                for j in range(w):
                    if dist_transform[i, j] > 0 and dist_transform[i, j] <= transition_width:
                        # Calculate blend factor
                        alpha = dist_transform[i, j] / transition_width
                        # Smooth transition using cosine interpolation
                        alpha = 0.5 * (1 + np.cos(np.pi * (1 - alpha)))
                        
                        # Blend colors
                        result[i, j] = (1 - alpha) * image[i, j] + alpha * bg_color
        
        return result

    def enhance_ring_details(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply v13.3 6-stage enhancement process"""
        # Stage 1: Noise reduction
        denoised = cv2.bilateralFilter(image, params['noise_reduction'], 50, 50)
        
        # Stage 2: Convert to PIL for enhancement
        pil_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        
        # Brightness adjustment
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(params['brightness'])
        
        # Contrast adjustment
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(params['contrast'])
        
        # Saturation adjustment
        saturation_enhancer = ImageEnhance.Color(pil_img)
        pil_img = saturation_enhancer.enhance(params['saturation'])
        
        # Sharpness adjustment
        sharpness_enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = sharpness_enhancer.enhance(params['sharpness'])
        
        # Convert back to numpy
        enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Stage 3: LAB adjustments
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * (1 + params['highlight_boost']), 0, 255)
        
        # Stage 4: Color channel adjustments
        lab[:, :, 1] = lab[:, :, 1] * params.get('s_mult', 1.0)
        lab[:, :, 2] = lab[:, :, 2] * params.get('v_mult', 1.0)
        
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Stage 5: White overlay for shine
        white_overlay = np.ones_like(enhanced) * 255
        enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                                 white_overlay, params['white_overlay'], 0)
        
        # Stage 6: Final edge enhancement
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality with CLAHE and sharpening"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Ensure L channel is uint8
        l = l.astype(np.uint8)
        
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

    def process_ring(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Main processing pipeline with v13.3 full enhancement"""
        # Detect gold type and lighting
        gold_type_key = self.detect_gold_type(image, mask)
        lighting = self.detect_lighting(image)
        
        # Map gold types to parameter keys
        gold_type_map = {
            'yellow': 'yellow_gold',
            'rose': 'rose_gold',
            'white': 'white_gold',
            'unplated_white': 'white_gold',
            'plain_white': 'plain_white'
        }
        
        param_key = gold_type_map.get(gold_type_key, 'white_gold')
        params = self.enhancement_params[param_key][lighting]
        
        logger.info(f"Processing with: {param_key} under {lighting} lighting")
        
        # Apply v13.3 6-stage enhancement
        enhanced = self.enhance_ring_details(image, params)
        
        # Apply gradient transition if mask available
        if mask is not None:
            # Get background color from edges
            h, w = image.shape[:2]
            edge_pixels = []
            edge_pixels.extend(image[0, :].reshape(-1, 3))
            edge_pixels.extend(image[-1, :].reshape(-1, 3))
            edge_pixels.extend(image[:, 0].reshape(-1, 3))
            edge_pixels.extend(image[:, -1].reshape(-1, 3))
            
            bg_color = np.median(edge_pixels, axis=0)
            enhanced = self.apply_gradient_transition(enhanced, mask, bg_color)
        
        return enhanced

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
        
        # Get processing info
        if bg_result['success']:
            gold_type = enhancer.detect_gold_type(bg_result['image'], bg_result['mask'])
            lighting = enhancer.detect_lighting(bg_result['image'])
        else:
            gold_type = enhancer.detect_gold_type(img_bgr)
            lighting = enhancer.detect_lighting(img_bgr)
        
        # Prepare response
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "background_removed": bg_result['success'],
                    "detected_gold_type": gold_type,
                    "detected_lighting": lighting,
                    "timestamp": datetime.now().isoformat(),
                    "version": "v152-complete"
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
                "version": "v152-complete"
            }
        }
        
        return error_result

# RunPod endpoint
runpod.serverless.start({"handler": handler})
