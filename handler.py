import os
import sys
import json
import requests
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from datetime import datetime
import time
import traceback
import logging
from io import BytesIO
import runpod
from diffusers import StableDiffusionInpaintPipeline
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance - initialized once
global_model = None

class WeddingRingEnhancerFinal:
    def __init__(self):
        """Wedding Ring AI Enhancement System v17 - Final Version"""
        self.version = "17.0"
        logger.info(f"Initializing Wedding Ring Enhancer v{self.version}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Only use inpainting model
        self.model_id = "stabilityai/stable-diffusion-2-inpainting"
        self.pipe = None
        
    def load_model(self):
        """Load only the inpainting model - lightweight"""
        if self.pipe is not None:
            logger.info("Model already loaded")
            return
            
        try:
            logger.info("Loading Stable Diffusion Inpainting model...")
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Optimizations
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_vae_slicing()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def detect_and_remove_black_frame(self, image):
        """Simple and effective black frame detection and removal"""
        logger.info("Detecting black frame...")
        
        try:
            # Convert to numpy
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Simple threshold-based detection
            threshold = 20  # Pixels darker than this are considered black
            
            # Find crop boundaries
            # Top
            top = 0
            for y in range(height // 3):  # Check top third only
                if np.mean(img_array[y, :]) > threshold:
                    break
                top = y + 1
                
            # Bottom
            bottom = height
            for y in range(height - 1, height * 2 // 3, -1):  # Check bottom third only
                if np.mean(img_array[y, :]) > threshold:
                    break
                bottom = y
                
            # Left
            left = 0
            for x in range(width // 3):  # Check left third only
                if np.mean(img_array[:, x]) > threshold:
                    break
                left = x + 1
                
            # Right
            right = width
            for x in range(width - 1, width * 2 // 3, -1):  # Check right third only
                if np.mean(img_array[:, x]) > threshold:
                    break
                right = x
            
            # Check if black frame detected
            if top > 10 or bottom < height - 10 or left > 10 or right < width - 10:
                logger.info(f"Black frame detected: top={top}, bottom={bottom}, left={left}, right={right}")
                
                # Add small margin to ensure complete removal
                margin = 5
                crop_box = (
                    left + margin,
                    top + margin,
                    right - margin,
                    bottom - margin
                )
                
                # Crop image
                cropped = image.crop(crop_box)
                
                # Resize back to original size
                result = cropped.resize((width, height), Image.Resampling.LANCZOS)
                
                logger.info("Black frame removed and image resized")
                return result, True
            else:
                logger.info("No black frame detected")
                return image, False
                
        except Exception as e:
            logger.error(f"Error in black frame removal: {str(e)}")
            return image, False

    def create_ring_mask(self, image):
        """Create simple center-focused mask for ring"""
        logger.info("Creating ring mask...")
        
        try:
            # Create center-focused circular mask
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            
            # Ring is usually in center
            center_x, center_y = width // 2, height // 2
            
            # Create circular gradient mask
            mask_array = np.zeros((height, width), dtype=np.uint8)
            
            # Ring area radius (adjust based on typical ring size)
            radius = min(width, height) // 3
            
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        # Gradient from center
                        intensity = int(255 * (1 - dist / radius))
                        mask_array[y, x] = intensity
            
            # Apply gaussian blur for smooth edges
            mask_array = cv2.GaussianBlur(mask_array, (21, 21), 0)
            
            # Convert back to PIL
            mask = Image.fromarray(mask_array)
            
            logger.info("Ring mask created")
            return mask
            
        except Exception as e:
            logger.error(f"Error creating mask: {str(e)}")
            # Return simple center mask as fallback
            mask = Image.new('L', image.size, 0)
            center = (image.width // 2, image.height // 2)
            radius = min(image.size) // 3
            
            draw = ImageDraw.Draw(mask)
            draw.ellipse([center[0] - radius, center[1] - radius,
                         center[0] + radius, center[1] + radius], fill=255)
            
            return mask.filter(ImageFilter.GaussianBlur(radius=10))

    def enhance_image(self, image, mask, prompt="", num_inference_steps=30):
        """Enhance ring using inpainting"""
        logger.info("Enhancing image with inpainting...")
        
        try:
            if not prompt:
                prompt = "ultra high quality luxury wedding ring, diamond ring, professional jewelry photography, perfect lighting, sharp focus, high detail, reflective surface"
            
            negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad geometry"
            
            # Run inpainting
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                strength=0.8
            ).images[0]
            
            logger.info("Enhancement complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhancement: {str(e)}")
            return image

    def apply_post_processing(self, image, original):
        """Simple post-processing for better quality"""
        logger.info("Applying post-processing...")
        
        try:
            # Ensure same size
            if image.size != original.size:
                image = image.resize(original.size, Image.Resampling.LANCZOS)
            
            # Slight sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Adjust contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Adjust color vibrancy
            color_enhancer = ImageEnhance.Color(image)
            image = color_enhancer.enhance(1.05)
            
            logger.info("Post-processing complete")
            return image
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return image

    def process(self, image, settings=None):
        """Main processing function"""
        logger.info("Starting image processing...")
        
        try:
            original_image = image.copy()
            
            # Step 1: Remove black frame
            image, black_frame_removed = self.detect_and_remove_black_frame(image)
            
            # Step 2: Create mask
            mask = self.create_ring_mask(image)
            
            # Step 3: Enhance with inpainting
            prompt = settings.get('prompt', '') if settings else ''
            steps = settings.get('steps', 30) if settings else 30
            
            enhanced = self.enhance_image(image, mask, prompt, steps)
            
            # Step 4: Post-processing
            final = self.apply_post_processing(enhanced, original_image)
            
            logger.info("Processing complete!")
            
            return final, {
                'success': True,
                'black_frame_removed': black_frame_removed,
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            return image, {
                'success': False,
                'error': str(e)
            }

# Initialize model globally
def init_global_model():
    """Initialize model once globally"""
    global global_model
    if global_model is None:
        logger.info("Initializing global model...")
        global_model = WeddingRingEnhancerFinal()
        global_model.load_model()
        logger.info("Global model initialized!")
    return global_model

# RunPod Handler
def download_image(url):
    """Download image from URL"""
    logger.info(f"Downloading image from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Image downloaded, size: {img.size}")
        return img
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

def handler(job):
    """RunPod handler function"""
    try:
        logger.info(f"Wedding Ring AI Enhancement v17.0 - Handler started")
        
        # Get input
        job_input = job.get('input', {})
        if not job_input:
            return {"error": "No input provided"}
        
        image_url = job_input.get('image_url')
        if not image_url:
            return {"error": "No image_url provided"}
        
        # Get enhancer (already loaded globally)
        enhancer = init_global_model()
        
        # Download image
        input_image = download_image(image_url)
        
        # Process settings
        settings = {
            'prompt': job_input.get('prompt', ''),
            'steps': job_input.get('steps', 30)
        }
        
        # Process image
        enhanced_image, metadata = enhancer.process(input_image, settings)
        
        # Save result
        output_path = "/tmp/enhanced_ring.jpg"
        enhanced_image.save(output_path, 'JPEG', quality=95)
        
        return {
            "status": "success",
            "output_path": output_path,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }

# Import fix for ImageDraw
try:
    from PIL import ImageDraw
except:
    pass

# Initialize model on startup (not in handler)
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    
    # Pre-load model
    init_global_model()
    
    # Start RunPod
    runpod.serverless.start({"handler": handler})
