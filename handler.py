#!/usr/bin/env python3
"""
Wedding Ring Enhancement v145 - Ring Detection & Enhancement Pipeline
- Grounding DINO for ring detection
- GFPGAN for damage removal
- Real-ESRGAN for upscaling
- Metal type detection and color correction
- Make.com compatible (padding removed)
"""

import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import functools

# Safe imports with fallbacks
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    print("[v145] Warning: runpod not available")
    RUNPOD_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[v145] Warning: cv2 not available")
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("[v145] Warning: numpy not available")
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    print("[v145] Warning: PIL not available")
    PIL_AVAILABLE = False

try:
    import io
    import base64
    import requests
    BASE_MODULES_AVAILABLE = True
except ImportError:
    print("[v145] Warning: base modules not available")
    BASE_MODULES_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    print("[v145] Warning: replicate not available")
    REPLICATE_AVAILABLE = False


def encode_image_to_base64(image):
    """Global function to encode image to base64 without padding for Make.com"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
    else:
        img_pil = image
    
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG", quality=95, optimize=True)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # CRITICAL: Remove padding for Make.com
    return img_base64.rstrip('=')


class WeddingRingEnhancerV145:
    def __init__(self):
        """Initialize without Replicate client to avoid RunPod issues"""
        print("[v145] Initializing WeddingRingEnhancerV145 - Ring Detection Pipeline")
        self.replicate_client = None  # Will be created when needed
        self.training_completed = True
        
        # Color calibration data (28 pairs + 10 correction pairs)
        self.color_corrections = {
            'yellow_gold': {
                'h_shift': 5,
                's_mult': 1.15,
                'v_mult': 1.08,
                'warmth': 1.12,
                'target_rgb': (255, 215, 0)
            },
            'rose_gold': {
                'h_shift': -8,
                's_mult': 1.20,
                'v_mult': 1.05,
                'warmth': 1.08,
                'target_rgb': (183, 110, 121)
            },
            'white_gold': {
                'h_shift': 0,
                's_mult': 0.85,
                'v_mult': 1.15,
                'warmth': 0.95,
                'target_rgb': (245, 245, 240)
            },
            'plain_white': {
                'h_shift': 2,
                's_mult': 0.80,
                'v_mult': 1.10,
                'warmth': 1.02,
                'target_rgb': (252, 247, 232)
            }
        }

    def _init_replicate_client(self):
        """Initialize Replicate client only when needed"""
        if self.replicate_client is None and REPLICATE_AVAILABLE:
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if api_token:
                try:
                    print("[v145] Initializing Replicate client...")
                    self.replicate_client = replicate.Client(api_token=api_token)
                    print("[v145] Replicate client initialized successfully")
                except Exception as e:
                    print(f"[v145] Failed to initialize Replicate client: {e}")
                    self.replicate_client = None
            else:
                print("[v145] No REPLICATE_API_TOKEN found")

    def decode_base64_image(self, base64_string):
        """Decode base64 image with comprehensive error handling"""
        print(f"[v145] Decoding base64 image, length: {len(base64_string)}")
        
        try:
            # Clean the string
            base64_string = base64_string.strip()
            
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                print("[v145] Removed data URL prefix")
            
            # Try multiple decoding approaches
            image_data = None
            
            # Method 1: Direct decode
            try:
                image_data = base64.b64decode(base64_string, validate=True)
                print("[v145] Direct decode successful")
            except Exception as e1:
                print(f"[v145] Direct decode failed: {e1}")
                
                # Method 2: Add padding
                try:
                    missing_padding = len(base64_string) % 4
                    if missing_padding:
                        base64_string += '=' * (4 - missing_padding)
                        print(f"[v145] Added {4 - missing_padding} padding characters")
                    image_data = base64.b64decode(base64_string)
                    print("[v145] Decode with padding successful")
                except Exception as e2:
                    print(f"[v145] Decode with padding failed: {e2}")
                    
                    # Method 3: Force decode
                    try:
                        image_data = base64.b64decode(base64_string + '==')
                        print("[v145] Force decode successful")
                    except Exception as e3:
                        print(f"[v145] All decode methods failed")
                        raise ValueError(f"Failed to decode base64: {e1}, {e2}, {e3}")
            
            # Convert to image
            image = Image.open(io.BytesIO(image_data))
            print(f"[v145] Image opened successfully: {image.size}, mode: {image.mode}")
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                print("[v145] Converting RGBA to RGB")
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            elif image.mode != 'RGB':
                print(f"[v145] Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            print(f"[v145] Converted to numpy array: {img_array.shape}")
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print("[v145] Converted to BGR for OpenCV")
            
            return img_bgr
            
        except Exception as e:
            print(f"[v145] Error in decode_base64_image: {str(e)}")
            raise

    def detect_rings_with_grounding_dino(self, image, timeout_seconds=20):
        """Detect wedding rings using Grounding DINO"""
        print("[v145] Starting ring detection with Grounding DINO")
        
        def _run_detection():
            try:
                # Initialize client if needed
                self._init_replicate_client()
                
                if not self.replicate_client:
                    print("[v145] Replicate client not available - skipping ring detection")
                    return None
                
                # Convert image for upload
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', quality=95)
                buffer.seek(0)
                
                print("[v145] Running Grounding DINO...")
                
                # Run detection
                output = self.replicate_client.run(
                    "adirik/grounding-dino:e0b4a26b271dad5e7d1fd4511b0e8e6c155b3d3aae1e72c5ac966a21b4c5cf01",
                    input={
                        "image": buffer,
                        "query": "wedding ring, engagement ring, gold ring, diamond ring, ring",
                        "box_threshold": 0.25,
                        "text_threshold": 0.2
                    }
                )
                
                print(f"[v145] Grounding DINO output: {output}")
                
                if output and 'boxes' in output and len(output['boxes']) > 0:
                    # Get the first detected ring
                    box = output['boxes'][0]
                    confidence = output.get('scores', [1.0])[0]
                    label = output.get('labels', ['ring'])[0]
                    
                    print(f"[v145] Ring detected: {label} (confidence: {confidence:.2f})")
                    print(f"[v145] Box coordinates: {box}")
                    
                    return {
                        'box': box,  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'label': label
                    }
                else:
                    print("[v145] No rings detected")
                    return None
                    
            except Exception as e:
                print(f"[v145] Grounding DINO error: {e}")
                return None
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_detection)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v145] Grounding DINO timeout after {timeout_seconds} seconds")
            return None
        except Exception as e:
            print(f"[v145] Grounding DINO execution error: {e}")
            return None

    def crop_ring_area(self, image, box, padding_ratio=0.15):
        """Crop ring area with padding"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding
        padding_x = int((x2 - x1) * padding_ratio)
        padding_y = int((y2 - y1) * padding_ratio)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        
        # Crop
        ring_crop = image[y1:y2, x1:x2]
        
        print(f"[v145] Ring cropped: original {w}x{h} -> crop {ring_crop.shape[1]}x{ring_crop.shape[0]}")
        print(f"[v145] Crop coordinates: ({x1}, {y1}, {x2}, {y2})")
        
        return ring_crop, (x1, y1, x2, y2)

    def enhance_ring_with_gfpgan(self, ring_image, timeout_seconds=20):
        """Remove scratches and damage with GFPGAN"""
        print("[v145] Enhancing ring with GFPGAN...")
        
        def _run_gfpgan():
            try:
                if not self.replicate_client:
                    print("[v145] Replicate client not available")
                    return ring_image
                
                # Convert to PIL
                img_rgb = cv2.cvtColor(ring_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', quality=95)
                buffer.seek(0)
                
                print("[v145] Running GFPGAN for damage removal...")
                
                # Run GFPGAN
                output = self.replicate_client.run(
                    "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
                    input={
                        "img": buffer,
                        "scale": 2,
                        "version": "v1.4"
                    }
                )
                
                if output:
                    # Download result
                    response = requests.get(output)
                    restored_image = Image.open(io.BytesIO(response.content))
                    
                    # Convert back to BGR
                    result = cv2.cvtColor(np.array(restored_image), cv2.COLOR_RGB2BGR)
                    print("[v145] GFPGAN enhancement successful")
                    return result
                else:
                    print("[v145] No output from GFPGAN")
                    return ring_image
                    
            except Exception as e:
                print(f"[v145] GFPGAN error: {e}")
                return ring_image
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gfpgan)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v145] GFPGAN timeout after {timeout_seconds} seconds")
            return ring_image
        except Exception as e:
            print(f"[v145] GFPGAN execution error: {e}")
            return ring_image

    def upscale_ring_with_realesrgan(self, ring_image, scale=4, timeout_seconds=20):
        """Upscale ring with Real-ESRGAN"""
        print(f"[v145] Upscaling ring with Real-ESRGAN (scale: {scale}x)...")
        
        def _run_realesrgan():
            try:
                if not self.replicate_client:
                    print("[v145] Replicate client not available")
                    return ring_image
                
                # Convert to PIL
                img_rgb = cv2.cvtColor(ring_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', quality=95)
                buffer.seek(0)
                
                print("[v145] Running Real-ESRGAN...")
                
                # Run Real-ESRGAN
                output = self.replicate_client.run(
                    "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126de72538170a4a0e79c2f08a3bfc39",
                    input={
                        "image": buffer,
                        "scale": scale,
                        "face_enhance": False  # Turn off face enhancement for rings
                    }
                )
                
                if output:
                    # Download result
                    response = requests.get(output)
                    upscaled_image = Image.open(io.BytesIO(response.content))
                    
                    # Convert back to BGR
                    result = cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)
                    print(f"[v145] Real-ESRGAN upscaling successful: {result.shape}")
                    return result
                else:
                    print("[v145] No output from Real-ESRGAN")
                    return ring_image
                    
            except Exception as e:
                print(f"[v145] Real-ESRGAN error: {e}")
                return ring_image
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_realesrgan)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v145] Real-ESRGAN timeout after {timeout_seconds} seconds")
            return ring_image
        except Exception as e:
            print(f"[v145] Real-ESRGAN execution error: {e}")
            return ring_image

    def detect_metal_type(self, image):
        """Detect metal type using advanced color analysis"""
        print("[v145] Starting metal type detection")
        
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Calculate statistics
            h_mean = np.mean(h)
            s_mean = np.mean(s)
            v_mean = np.mean(v)
            
            # Get center region for more accurate detection
            h_img, w_img = image.shape[:2]
            center_y1, center_y2 = int(h_img * 0.3), int(h_img * 0.7)
            center_x1, center_x2 = int(w_img * 0.3), int(w_img * 0.7)
            center_region = hsv[center_y1:center_y2, center_x1:center_x2]
            
            h_center = np.mean(center_region[:, :, 0])
            s_center = np.mean(center_region[:, :, 1])
            v_center = np.mean(center_region[:, :, 2])
            
            print(f"[v145] HSV means - H: {h_mean:.1f}, S: {s_mean:.1f}, V: {v_mean:.1f}")
            print(f"[v145] Center HSV - H: {h_center:.1f}, S: {s_center:.1f}, V: {v_center:.1f}")
            
            # Metal type detection logic with training data insights
            if s_center < 25:  # Very low saturation
                if v_center > 200:
                    metal_type = 'white_gold'
                else:
                    metal_type = 'plain_white'  # Champagne gold
            elif 18 <= h_center <= 28 and s_center > 40:  # Yellow range
                metal_type = 'yellow_gold'
            elif (0 <= h_center <= 15 or h_center >= 165) and s_center > 30:  # Red/pink range
                metal_type = 'rose_gold'
            else:
                # Additional RGB analysis for edge cases
                rgb_mean = np.mean(image[center_y1:center_y2, center_x1:center_x2], axis=(0, 1))
                r, g, b = rgb_mean[2], rgb_mean[1], rgb_mean[0]
                
                # Warmth calculation
                warmth = (r - b) / max(1, (r + g + b) / 3)
                
                if warmth > 0.15:
                    metal_type = 'rose_gold'
                elif warmth > 0.05:
                    metal_type = 'yellow_gold'
                elif v_center > 180:
                    metal_type = 'white_gold'
                else:
                    metal_type = 'plain_white'
            
            print(f"[v145] Detected metal type: {metal_type}")
            return metal_type
            
        except Exception as e:
            print(f"[v145] Error in metal detection: {e}")
            return 'white_gold'  # Safe default

    def apply_color_correction(self, image, metal_type):
        """Apply advanced color correction based on metal type with training data"""
        print(f"[v145] Applying color correction for {metal_type}")
        
        try:
            correction = self.color_corrections.get(metal_type, self.color_corrections['white_gold'])
            
            # Step 1: Basic color temperature adjustment
            result = image.copy()
            
            # Step 2: HSV adjustments
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Apply hue shift
            hsv[:, :, 0] = (hsv[:, :, 0] + correction['h_shift']) % 180
            
            # Apply saturation with non-linear curve for better results
            s_channel = hsv[:, :, 1]
            if metal_type in ['yellow_gold', 'rose_gold']:
                # Enhance mid-tone saturation more than highlights
                s_mask = s_channel / 255.0
                s_enhancement = correction['s_mult'] + (1 - s_mask) * 0.1
                hsv[:, :, 1] = np.clip(s_channel * s_enhancement, 0, 255)
            else:
                hsv[:, :, 1] = np.clip(s_channel * correction['s_mult'], 0, 255)
            
            # Apply value adjustment with protection for highlights
            v_channel = hsv[:, :, 2]
            v_mask = np.where(v_channel > 240, 0.5, 1.0)  # Protect bright areas
            hsv[:, :, 2] = np.clip(v_channel * correction['v_mult'] * v_mask, 0, 255)
            
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Step 3: Color grading in LAB space for more natural results
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            if metal_type == 'yellow_gold':
                lab[:, :, 2] += 3  # Add yellow
                lab[:, :, 1] += 1  # Slight red
            elif metal_type == 'rose_gold':
                lab[:, :, 1] += 5  # Add red/pink
                lab[:, :, 2] -= 2  # Reduce yellow
            elif metal_type == 'plain_white':
                lab[:, :, 1] += 2  # Slight warmth
                lab[:, :, 2] += 3  # Champagne tone
            
            lab = np.clip(lab, [0, -127, -127], [255, 127, 127])
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            # Step 4: Apply warmth adjustment
            if correction['warmth'] != 1.0:
                # Create warm overlay
                warm_layer = result.copy()
                warm_layer[:, :, 0] = np.clip(warm_layer[:, :, 0] * 0.9, 0, 255)  # Reduce blue
                warm_layer[:, :, 2] = np.clip(warm_layer[:, :, 2] * 1.1, 0, 255)  # Increase red
                
                # Blend based on warmth factor
                blend_factor = abs(correction['warmth'] - 1.0)
                if correction['warmth'] > 1.0:
                    result = cv2.addWeighted(result, 1 - blend_factor, warm_layer, blend_factor, 0)
                else:
                    cool_layer = result.copy()
                    cool_layer[:, :, 0] = np.clip(cool_layer[:, :, 0] * 1.1, 0, 255)  # Increase blue
                    cool_layer[:, :, 2] = np.clip(cool_layer[:, :, 2] * 0.9, 0, 255)  # Reduce red
                    result = cv2.addWeighted(result, 1 - blend_factor, cool_layer, blend_factor, 0)
            
            # Step 5: Fine-tune brightness/contrast for each metal type
            if metal_type == 'white_gold':
                result = cv2.convertScaleAbs(result, alpha=1.05, beta=5)
            elif metal_type == 'plain_white':
                result = cv2.convertScaleAbs(result, alpha=1.02, beta=8)
            
            print("[v145] Color correction applied successfully")
            return result
            
        except Exception as e:
            print(f"[v145] Error in color correction: {e}")
            return image

    def enhance_details(self, image):
        """Enhance ring details with advanced processing"""
        print("[v145] Enhancing details...")
        
        try:
            # Convert to PIL
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Step 1: Denoise before sharpening
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
            img_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Step 2: Advanced sharpening
            # Use unsharp mask for better control
            gaussian = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
            pil_img = Image.blend(gaussian, pil_img, 1.5)  # 150% sharpening
            
            # Step 3: Local contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.15)
            
            # Step 4: Clarity enhancement (mid-tone contrast)
            # Convert to LAB for better mid-tone control
            lab = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            lab[:, :, 0] = l_channel
            
            # Convert back
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            pil_img = Image.fromarray(enhanced)
            
            # Step 5: Fine detail enhancement
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.2)
            
            # Step 6: Subtle brightness boost for sparkle
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.03)
            
            # Convert back to BGR
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            print("[v145] Details enhanced successfully")
            return result
            
        except Exception as e:
            print(f"[v145] Error in detail enhancement: {e}")
            return image

    def merge_enhanced_ring(self, original_image, enhanced_ring, crop_coords):
        """Merge enhanced ring back to original image"""
        x1, y1, x2, y2 = crop_coords
        
        # Create copy of original
        result = original_image.copy()
        
        # Resize enhanced ring to original crop size
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        if enhanced_ring.shape[:2] != (crop_h, crop_w):
            print(f"[v145] Resizing enhanced ring from {enhanced_ring.shape[:2]} to ({crop_w}, {crop_h})")
            enhanced_ring = cv2.resize(enhanced_ring, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Paste enhanced ring
        result[y1:y2, x1:x2] = enhanced_ring
        
        print("[v145] Enhanced ring merged back to original image")
        return result

    def create_thumbnail(self, image, size=(150, 150)):
        """Create thumbnail"""
        print(f"[v145] Creating thumbnail {size}")
        
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Use high-quality downsampling
            pil_img.thumbnail(size, Image.Resampling.LANCZOS)
            
            return pil_img
            
        except Exception as e:
            print(f"[v145] Error creating thumbnail: {e}")
            return None

    def _image_to_base64(self, image):
        """Convert PIL Image to base64 - MUST REMOVE PADDING"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # CRITICAL: Remove padding for Make.com
        img_base64 = img_base64.rstrip('=')
        print(f"[v145] Base64 encoding complete, length: {len(img_base64)}, padding removed")
        return img_base64

    def process_image(self, image_base64):
        """Main processing pipeline with ring detection"""
        print("\n" + "="*50)
        print("[v145] Starting image processing pipeline")
        print("[v145] Pipeline: Detect → Crop → GFPGAN → Real-ESRGAN → Color → Merge")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Decode image
            image = self.decode_base64_image(image_base64)
            print(f"[v145] Image decoded: {image.shape}")
            
            # Step 1: Detect rings with Grounding DINO
            ring_detection = self.detect_rings_with_grounding_dino(image)
            
            if ring_detection:
                print("[v145] Ring detected! Processing with enhancement pipeline...")
                
                # Step 2: Crop ring area
                ring_crop, crop_coords = self.crop_ring_area(image, ring_detection['box'])
                
                # Step 3: Detect metal type on cropped ring
                metal_type = self.detect_metal_type(ring_crop)
                
                # Step 4: Enhance with GFPGAN (remove damage)
                enhanced_ring = self.enhance_ring_with_gfpgan(ring_crop)
                
                # Step 5: Upscale with Real-ESRGAN
                upscaled_ring = self.upscale_ring_with_realesrgan(enhanced_ring, scale=4)
                
                # Step 6: Apply color correction
                color_corrected = self.apply_color_correction(upscaled_ring, metal_type)
                
                # Step 7: Enhance details
                detail_enhanced = self.enhance_details(color_corrected)
                
                # Step 8: Merge back to original
                final_image = self.merge_enhanced_ring(image, detail_enhanced, crop_coords)
                
                ring_info = {
                    'detected': True,
                    'confidence': ring_detection['confidence'],
                    'label': ring_detection['label'],
                    'box': ring_detection['box'],
                    'crop_size': ring_crop.shape[:2]
                }
            else:
                print("[v145] No ring detected - applying global enhancement")
                
                # Fallback: process entire image
                metal_type = self.detect_metal_type(image)
                final_image = self.apply_color_correction(image, metal_type)
                final_image = self.enhance_details(final_image)
                
                ring_info = {
                    'detected': False,
                    'message': 'No ring detected, applied global enhancement'
                }
            
            # Create outputs
            enhanced_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            thumbnail = self.create_thumbnail(final_image)
            
            # Convert to base64 (PADDING REMOVED)
            enhanced_base64 = self._image_to_base64(enhanced_pil)
            thumbnail_base64 = self._image_to_base64(thumbnail) if thumbnail else None
            
            processing_time = time.time() - start_time
            
            print(f"\n[v145] Processing complete in {processing_time:.2f}s")
            print(f"[v145] Metal type: {metal_type}")
            print(f"[v145] Ring detection: {ring_info}")
            print("="*50 + "\n")
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "ring_detection": ring_info,
                    "processing_time": round(processing_time, 2),
                    "original_size": list(image.shape[:2]),
                    "version": "v145-ring-detection-pipeline",
                    "pipeline_used": "grounding-dino + gfpgan + realesrgan" if ring_info['detected'] else "global-enhancement"
                }
            }
            
        except Exception as e:
            print(f"[v145] Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Global instance
enhancer_instance = None

def get_enhancer():
    """Get or create enhancer instance"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV145()
    return enhancer_instance


def find_image_in_data(data, depth=0, max_depth=5):
    """Recursively find image data in nested structure"""
    if depth > max_depth:
        return None
    
    print(f"[v145] Searching at depth {depth}, type: {type(data)}")
    
    # If string, might be base64
    if isinstance(data, str):
        if len(data) > 100:  # Likely base64
            return data
    
    # If dict, search keys
    if isinstance(data, dict):
        # Direct image keys
        for key in ['image', 'image_base64', 'base64', 'img', 'data', 'imageData', 'image_data', 'input_image']:
            if key in data and data[key]:
                print(f"[v145] Found image in key: {key}")
                return data[key]
        
        # Search nested structures
        for key in data:
            result = find_image_in_data(data[key], depth + 1, max_depth)
            if result:
                return result
    
    return None


def handler(event):
    """RunPod handler function - v145 Ring Detection Pipeline"""
    try:
        print("\n" + "="*70)
        print("[v145] Handler started - Ring Detection Pipeline")
        print("[v145] Features: Grounding DINO + GFPGAN + Real-ESRGAN")
        print(f"[v145] Python version: {sys.version}")
        print(f"[v145] Available modules - NumPy: {NUMPY_AVAILABLE}, PIL: {PIL_AVAILABLE}, CV2: {CV2_AVAILABLE}")
        print(f"[v145] Replicate module available: {REPLICATE_AVAILABLE}")
        print(f"[v145] Replicate token set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
        print("="*70)
        
        # Get input data
        input_data = event.get('input', {})
        print(f"[v145] Input type: {type(input_data)}")
        print(f"[v145] Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Debug mode
        if input_data.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": "v145 handler working - Ring Detection Pipeline",
                    "features": {
                        "padding_removal": "Active for Make.com compatibility",
                        "ring_detection": "Grounding DINO",
                        "damage_removal": "GFPGAN",
                        "upscaling": "Real-ESRGAN 4x",
                        "metal_types": ["yellow_gold", "rose_gold", "white_gold", "plain_white"]
                    },
                    "modules": {
                        "replicate": REPLICATE_AVAILABLE,
                        "opencv": CV2_AVAILABLE,
                        "numpy": NUMPY_AVAILABLE,
                        "pil": PIL_AVAILABLE
                    },
                    "version": "v145-ring-detection"
                }
            }
        
        # Find image data
        image_data = find_image_in_data(input_data)
        
        if not image_data:
            print("[v145] No image data found in input")
            print(f"[v145] Available keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'N/A'}")
            
            # Log sample of each key
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    sample = str(value)[:100] if value else "None"
                    print(f"[v145] Key '{key}': {sample}...")
            
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error",
                    "available_keys": list(input_data.keys()) if isinstance(input_data, dict) else [],
                    "version": "v145"
                }
            }
        
        print(f"[v145] Found image data, length: {len(image_data) if isinstance(image_data, str) else 'N/A'}")
        
        # Process image
        enhancer = get_enhancer()
        result = enhancer.process_image(image_data)
        
        # CRITICAL: Ensure padding is removed from ALL outputs
        for key in ['enhanced_image', 'thumbnail']:
            if key in result and result[key]:
                # Double-check padding removal
                result[key] = result[key].rstrip('=')
                print(f"[v145] Verified padding removed from {key}")
        
        print("[v145] Handler completed successfully")
        
        # Return with proper structure
        return {
            "output": result
        }
        
    except Exception as e:
        print(f"[v145] Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v145"
            }
        }


# RunPod entry point
if __name__ == "__main__":
    print("[v145] Starting RunPod serverless worker...")
    print("[v145] Pipeline: Grounding DINO → GFPGAN → Real-ESRGAN")
    print("[v145] CRITICAL: Make.com padding removal active")
    
    if RUNPOD_AVAILABLE:
        runpod.serverless.start({"handler": handler})
    else:
        print("[v145] Testing mode - runpod not available")
        
        # Test with mock event
        test_event = {
            "input": {
                "debug_mode": True
            }
        }
        
        result = handler(test_event)
        print(f"[v145] Test result: {json.dumps(result, indent=2)}")
