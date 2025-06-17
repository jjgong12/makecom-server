import os
import sys
import json
import requests
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from datetime import datetime
import time
import traceback
import logging
from io import BytesIO
import runpod
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from controlnet_aux import CannyDetector, MidasDetector
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from compel import Compel
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WeddingRingEnhancer:
    def __init__(self):
        """Wedding Ring AI Enhancement System v16.1 - Debug Version"""
        self.version = "16.1"
        logger.info(f"Initializing Wedding Ring Enhancer v{self.version}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.models_loaded = False
        self.base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
        
        # Controlnet models
        self.controlnet_models = {
            'canny': "lllyasviel/control_v11p_sd15_canny",
            'depth': "lllyasviel/control_v11f1p_sd15_depth"
        }
        
        # Initialize processing history
        self.processing_history = []
        
        logger.info("WeddingRingEnhancer initialized successfully")
    
    def load_models(self):
        """Load all required models"""
        if self.models_loaded:
            logger.info("Models already loaded")
            return
            
        try:
            logger.info("Starting model loading...")
            
            # Load Canny ControlNet
            logger.info("Loading Canny ControlNet...")
            canny_controlnet = ControlNetModel.from_pretrained(
                self.controlnet_models['canny'],
                torch_dtype=torch.float16
            )
            
            self.canny_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=canny_controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load Depth ControlNet
            logger.info("Loading Depth ControlNet...")
            depth_controlnet = ControlNetModel.from_pretrained(
                self.controlnet_models['depth'],
                torch_dtype=torch.float16
            )
            
            self.depth_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=depth_controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load Inpainting Pipeline
            logger.info("Loading Inpainting Pipeline...")
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load BLIP for image captioning
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Initialize detectors
            logger.info("Initializing detectors...")
            self.canny_detector = CannyDetector()
            self.midas_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            
            # Initialize Compel for all pipelines
            logger.info("Initializing Compel...")
            self.canny_compel = Compel(tokenizer=self.canny_pipe.tokenizer, text_encoder=self.canny_pipe.text_encoder)
            self.depth_compel = Compel(tokenizer=self.depth_pipe.tokenizer, text_encoder=self.depth_pipe.text_encoder)
            self.inpaint_compel = Compel(tokenizer=self.inpaint_pipe.tokenizer, text_encoder=self.inpaint_pipe.text_encoder)
            
            # Enable optimizations
            for pipe in [self.canny_pipe, self.depth_pipe, self.inpaint_pipe]:
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            
            self.models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def detect_black_frame_advanced(self, image):
        """Advanced black frame detection with multiple methods"""
        logger.info("Starting advanced black frame detection")
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            logger.info(f"Image dimensions: {width}x{height}")
            
            # Multiple detection methods
            detection_results = {
                'has_black_frame': False,
                'frame_regions': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
                'confidence': 0
            }
            
            # Method 1: Check pure black pixels at edges
            logger.info("Method 1: Checking pure black pixels at edges")
            edge_threshold = 10
            
            # Check each edge
            top_black = 0
            for y in range(min(height//4, 100)):
                row = img_array[y, :, :]
                if np.all(row < edge_threshold):
                    top_black = y + 1
                else:
                    break
            logger.info(f"Top black rows: {top_black}")
            
            bottom_black = 0
            for y in range(min(height//4, 100)):
                row = img_array[height-1-y, :, :]
                if np.all(row < edge_threshold):
                    bottom_black = y + 1
                else:
                    break
            logger.info(f"Bottom black rows: {bottom_black}")
            
            left_black = 0
            for x in range(min(width//4, 100)):
                col = img_array[:, x, :]
                if np.all(col < edge_threshold):
                    left_black = x + 1
                else:
                    break
            logger.info(f"Left black columns: {left_black}")
            
            right_black = 0
            for x in range(min(width//4, 100)):
                col = img_array[:, width-1-x, :]
                if np.all(col < edge_threshold):
                    right_black = x + 1
                else:
                    break
            logger.info(f"Right black columns: {right_black}")
            
            # Method 2: Gradient-based detection
            logger.info("Method 2: Gradient-based detection")
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect horizontal gradients (top/bottom borders)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_y_abs = np.abs(grad_y)
            
            # Find strong horizontal edges
            for y in range(min(height//3, 150)):
                if np.mean(grad_y_abs[y, :]) > 30:
                    top_black = max(top_black, y)
                    break
            
            for y in range(min(height//3, 150)):
                if np.mean(grad_y_abs[height-1-y, :]) > 30:
                    bottom_black = max(bottom_black, y)
                    break
            
            # Detect vertical gradients (left/right borders)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_x_abs = np.abs(grad_x)
            
            for x in range(min(width//3, 150)):
                if np.mean(grad_x_abs[:, x]) > 30:
                    left_black = max(left_black, x)
                    break
            
            for x in range(min(width//3, 150)):
                if np.mean(grad_x_abs[:, width-1-x]) > 30:
                    right_black = max(right_black, x)
                    break
            
            # Method 3: Statistical analysis
            logger.info("Method 3: Statistical analysis")
            border_size = 20
            
            # Analyze borders
            if top_black == 0:
                top_border = gray[:border_size, :]
                if np.mean(top_border) < 15 and np.std(top_border) < 10:
                    top_black = border_size
            
            if bottom_black == 0:
                bottom_border = gray[-border_size:, :]
                if np.mean(bottom_border) < 15 and np.std(bottom_border) < 10:
                    bottom_black = border_size
            
            if left_black == 0:
                left_border = gray[:, :border_size]
                if np.mean(left_border) < 15 and np.std(left_border) < 10:
                    left_black = border_size
            
            if right_black == 0:
                right_border = gray[:, -border_size:]
                if np.mean(right_border) < 15 and np.std(right_border) < 10:
                    right_black = border_size
            
            # Update detection results
            detection_results['frame_regions'] = {
                'top': int(top_black),
                'bottom': int(bottom_black),
                'left': int(left_black),
                'right': int(right_black)
            }
            
            # Calculate confidence
            total_border = top_black + bottom_black + left_black + right_black
            if total_border > 0:
                detection_results['has_black_frame'] = True
                detection_results['confidence'] = min(total_border / (width + height) * 100, 100)
            
            logger.info(f"Black frame detection results: {detection_results}")
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in black frame detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {'has_black_frame': False, 'frame_regions': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}, 'confidence': 0}

    def remove_black_frame_completely(self, image, detection_result):
        """Remove black frame and resize to original dimensions"""
        logger.info("Starting black frame removal")
        
        try:
            if not detection_result['has_black_frame']:
                logger.info("No black frame detected, returning original image")
                return image
            
            regions = detection_result['frame_regions']
            logger.info(f"Removing frame regions: {regions}")
            
            # Add safety margins
            margin = 5
            crop_box = (
                max(0, regions['left'] + margin),
                max(0, regions['top'] + margin),
                image.width - regions['right'] - margin,
                image.height - regions['bottom'] - margin
            )
            
            logger.info(f"Crop box: {crop_box}")
            
            # Ensure valid crop dimensions
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                logger.warning("Invalid crop dimensions, returning original")
                return image
            
            # Crop the image
            cropped = image.crop(crop_box)
            logger.info(f"Cropped size: {cropped.size}")
            
            # Resize back to original dimensions with high quality
            original_size = (image.width, image.height)
            resized = cropped.resize(original_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized back to: {resized.size}")
            
            # Apply slight sharpening to compensate for resize
            enhancer = ImageEnhance.Sharpness(resized)
            final = enhancer.enhance(1.1)
            
            return final
            
        except Exception as e:
            logger.error(f"Error removing black frame: {str(e)}")
            logger.error(traceback.format_exc())
            return image

    def generate_caption(self, image):
        """Generate detailed caption for the image"""
        logger.info("Generating image caption")
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return "a wedding ring"

    def create_enhanced_mask(self, image, mask_image):
        """Create enhanced mask with better edge handling"""
        logger.info("Creating enhanced mask")
        
        try:
            # Convert mask to numpy
            mask_np = np.array(mask_image.convert('L'))
            
            # Apply multiple dilations for better coverage
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(mask_np, kernel, iterations=3)
            
            # Gaussian blur for smooth edges
            blurred = cv2.GaussianBlur(dilated, (9,9), 0)
            
            # Create gradient mask
            gradient = np.zeros_like(blurred)
            center = 128
            for i in range(blurred.shape[0]):
                for j in range(blurred.shape[1]):
                    if blurred[i,j] > 0:
                        dist_to_edge = min(i, j, blurred.shape[0]-i-1, blurred.shape[1]-j-1)
                        gradient[i,j] = min(255, blurred[i,j] + dist_to_edge)
            
            # Convert back to PIL
            enhanced_mask = Image.fromarray(gradient).convert('L')
            
            # Apply slight blur to final mask
            enhanced_mask = enhanced_mask.filter(ImageFilter.GaussianBlur(radius=2))
            
            return enhanced_mask
            
        except Exception as e:
            logger.error(f"Error creating enhanced mask: {str(e)}")
            return mask_image

    def create_composite_image(self, original, generated, mask, blend_mode='overlay'):
        """Advanced compositing with multiple blend modes"""
        logger.info(f"Creating composite with blend mode: {blend_mode}")
        
        try:
            # Ensure same size
            if original.size != generated.size:
                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            orig_array = np.array(original).astype(float) / 255.0
            gen_array = np.array(generated).astype(float) / 255.0
            mask_array = np.array(mask.convert('L')).astype(float) / 255.0
            
            # Expand mask to 3 channels
            mask_3ch = np.stack([mask_array] * 3, axis=-1)
            
            if blend_mode == 'overlay':
                # Overlay blend mode
                def overlay_blend(base, blend):
                    return np.where(base < 0.5, 
                                   2 * base * blend, 
                                   1 - 2 * (1 - base) * (1 - blend))
                
                blended = overlay_blend(orig_array, gen_array)
                result = orig_array * (1 - mask_3ch) + blended * mask_3ch
                
            elif blend_mode == 'soft_light':
                # Soft light blend mode
                def soft_light_blend(base, blend):
                    return np.where(blend < 0.5,
                                   2 * base * blend + base * base * (1 - 2 * blend),
                                   2 * base * (1 - blend) + np.sqrt(base) * (2 * blend - 1))
                
                blended = soft_light_blend(orig_array, gen_array)
                result = orig_array * (1 - mask_3ch) + blended * mask_3ch
                
            else:  # normal
                result = orig_array * (1 - mask_3ch) + gen_array * mask_3ch
            
            # Convert back to PIL Image
            result = (result * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"Error creating composite: {str(e)}")
            return Image.composite(generated, original, mask)

    def process_single_image(self, image, settings):
        """Process a single image with ring enhancement"""
        logger.info("Starting single image processing")
        
        try:
            # Check and remove black frame first
            logger.info("Checking for black frame...")
            black_frame_result = self.detect_black_frame_advanced(image)
            
            if black_frame_result['has_black_frame']:
                logger.info(f"Black frame detected with confidence: {black_frame_result['confidence']}%")
                image = self.remove_black_frame_completely(image, black_frame_result)
                logger.info("Black frame removed successfully")
            else:
                logger.info("No black frame detected")
            
            original_size = image.size
            logger.info(f"Processing image of size: {original_size}")
            
            # Generate caption
            caption = self.generate_caption(image)
            
            # Prepare prompts
            positive_prompt = f"{settings['prompt']}, {caption}, professional product photography, luxury jewelry, perfect lighting, sharp focus, high detail"
            negative_prompt = f"{settings['negative_prompt']}, blurry, low quality, artifacts, distorted"
            
            # Create control images
            logger.info("Creating control images...")
            canny_image = self.canny_detector(image, low_threshold=50, high_threshold=150)
            depth_image = self.midas_detector(image)
            
            # Process with ControlNets
            logger.info("Processing with Canny ControlNet...")
            canny_conditioning = self.canny_compel(positive_prompt)
            canny_result = self.canny_pipe(
                prompt_embeds=canny_conditioning,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=settings['steps'],
                guidance_scale=settings['guidance_scale'],
                controlnet_conditioning_scale=settings['controlnet_scale'],
                generator=torch.Generator(device=self.device).manual_seed(settings['seed'])
            ).images[0]
            
            logger.info("Processing with Depth ControlNet...")
            depth_conditioning = self.depth_compel(positive_prompt)
            depth_result = self.depth_pipe(
                prompt_embeds=depth_conditioning,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=settings['steps'],
                guidance_scale=settings['guidance_scale'],
                controlnet_conditioning_scale=settings['controlnet_scale'] * 0.7,
                generator=torch.Generator(device=self.device).manual_seed(settings['seed'] + 1)
            ).images[0]
            
            # Create mask for inpainting
            logger.info("Creating inpainting mask...")
            mask = self.create_ring_mask(image)
            enhanced_mask = self.create_enhanced_mask(image, mask)
            
            # Inpainting
            logger.info("Processing with inpainting...")
            inpaint_conditioning = self.inpaint_compel(positive_prompt)
            inpaint_result = self.inpaint_pipe(
                prompt_embeds=inpaint_conditioning,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=enhanced_mask,
                num_inference_steps=settings['steps'],
                guidance_scale=settings['guidance_scale'],
                generator=torch.Generator(device=self.device).manual_seed(settings['seed'] + 2)
            ).images[0]
            
            # Combine results
            logger.info("Combining results...")
            combined = self.combine_results_advanced(
                original=image,
                results={
                    'canny': canny_result,
                    'depth': depth_result,
                    'inpaint': inpaint_result
                },
                mask=enhanced_mask,
                weights={'canny': 0.3, 'depth': 0.3, 'inpaint': 0.4}
            )
            
            # Final enhancements
            logger.info("Applying final enhancements...")
            final = self.apply_final_enhancements(combined, settings['enhancement_strength'])
            
            # Ensure correct size
            if final.size != original_size:
                final = final.resize(original_size, Image.Resampling.LANCZOS)
            
            # Add to processing history
            self.processing_history.append({
                'timestamp': datetime.now().isoformat(),
                'original_size': original_size,
                'black_frame_detected': black_frame_result['has_black_frame'],
                'settings': settings
            })
            
            logger.info("Image processing completed successfully")
            return final, {
                'success': True,
                'black_frame_removed': black_frame_result['has_black_frame'],
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return image, {'success': False, 'error': str(e)}

    def create_ring_mask(self, image):
        """Create mask for ring area using edge detection"""
        logger.info("Creating ring mask")
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Edge detection
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask
            mask = np.zeros_like(gray)
            
            # Filter contours by area and circularity
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Ring-like shapes
                            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Dilate mask
            kernel = np.ones((7,7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Convert to PIL
            return Image.fromarray(mask)
            
        except Exception as e:
            logger.error(f"Error creating ring mask: {str(e)}")
            # Return center-focused mask as fallback
            mask = Image.new('L', image.size, 0)
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
            radius = min(image.size) // 3
            
            for x in range(image.size[0]):
                for y in range(image.size[1]):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        mask.putpixel((x, y), 255)
            
            return mask

    def combine_results_advanced(self, original, results, mask, weights):
        """Advanced combination of multiple results"""
        logger.info("Combining multiple results")
        
        try:
            # Ensure all images are same size
            target_size = original.size
            for key in results:
                if results[key].size != target_size:
                    results[key] = results[key].resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to arrays
            orig_array = np.array(original).astype(float)
            mask_array = np.array(mask.convert('L')).astype(float) / 255.0
            mask_3ch = np.stack([mask_array] * 3, axis=-1)
            
            # Weighted combination
            combined = np.zeros_like(orig_array)
            total_weight = sum(weights.values())
            
            for key, img in results.items():
                weight = weights.get(key, 1.0) / total_weight
                img_array = np.array(img).astype(float)
                combined += img_array * weight
            
            # Blend with original
            result = orig_array * (1 - mask_3ch) + combined * mask_3ch
            
            # Convert back to PIL
            result = result.clip(0, 255).astype(np.uint8)
            return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return original

    def apply_final_enhancements(self, image, strength=1.0):
        """Apply final enhancements to the image"""
        logger.info(f"Applying final enhancements with strength: {strength}")
        
        try:
            # Adjust sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1 + 0.3 * strength)
            
            # Adjust contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1 + 0.2 * strength)
            
            # Adjust color
            color_enhancer = ImageEnhance.Color(image)
            image = color_enhancer.enhance(1 + 0.1 * strength)
            
            # Apply unsharp mask
            image_np = np.array(image)
            gaussian = cv2.GaussianBlur(image_np, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(image_np, 1.5, gaussian, -0.5, 0)
            
            # Blend with original
            result = cv2.addWeighted(image_np, 1 - 0.3 * strength, unsharp_mask, 0.3 * strength, 0)
            
            return Image.fromarray(result.clip(0, 255).astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Error applying enhancements: {str(e)}")
            return image

# RunPod Handler
def download_image(url):
    """Download image from URL"""
    logger.info(f"Downloading image from: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Image downloaded successfully, size: {img.size}")
        return img
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

def handler(job):
    """RunPod handler function"""
    print(f"Wedding Ring AI Enhancement v16.1")
    logger.info(f"Handler called with job: {job}")
    
    try:
        # Parse input
        logger.info("Parsing job input...")
        job_input = job.get('input', {})
        
        if not job_input:
            logger.error("No input provided")
            return {"error": "No input provided"}
        
        logger.info(f"Job input: {job_input}")
        
        # Get image URL
        image_url = job_input.get('image_url')
        if not image_url:
            logger.error("No image_url provided")
            return {"error": "No image_url provided"}
        
        # Get settings
        settings = {
            'prompt': job_input.get('prompt', 'ultra-high quality wedding ring, professional jewelry photography'),
            'negative_prompt': job_input.get('negative_prompt', 'low quality, blurry, distorted'),
            'steps': job_input.get('steps', 30),
            'guidance_scale': job_input.get('guidance_scale', 7.5),
            'controlnet_scale': job_input.get('controlnet_scale', 1.0),
            'enhancement_strength': job_input.get('enhancement_strength', 1.0),
            'seed': job_input.get('seed', 42)
        }
        
        logger.info(f"Settings: {settings}")
        
        # Initialize enhancer
        logger.info("Initializing enhancer...")
        enhancer = WeddingRingEnhancer()
        
        # Load models
        logger.info("Loading models...")
        enhancer.load_models()
        
        # Download image
        logger.info("Downloading input image...")
        input_image = download_image(image_url)
        
        # Process image
        logger.info("Processing image...")
        enhanced_image, metadata = enhancer.process_single_image(input_image, settings)
        
        # Save result
        logger.info("Saving result...")
        output_path = "/tmp/enhanced_ring.jpg"
        enhanced_image.save(output_path, 'JPEG', quality=95)
        
        # Upload to temporary storage (you need to implement this based on your setup)
        # For now, we'll return the local path
        
        result = {
            "status": "success",
            "output_path": output_path,
            "metadata": metadata,
            "processing_history": enhancer.processing_history
        }
        
        logger.info(f"Processing complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
