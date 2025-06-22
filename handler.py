#!/usr/bin/env python3
"""
Wedding Ring Enhancement v146 - Improved Ring Detection & Enhancement
- Multi-stage ring detection (Grounding DINO + OWL-ViT + Fallback)
- GFPGAN for damage removal
- Real-ESRGAN for upscaling
- 28 pairs + 10 pairs training data applied
- Thumbnail 1000x1300
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
    print("[v146] Warning: runpod not available")
    RUNPOD_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[v146] Warning: cv2 not available")
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("[v146] Warning: numpy not available")
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    print("[v146] Warning: PIL not available")
    PIL_AVAILABLE = False

try:
    import io
    import base64
    import requests
    BASE_MODULES_AVAILABLE = True
except ImportError:
    print("[v146] Warning: base modules not available")
    BASE_MODULES_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    print("[v146] Warning: replicate not available")
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


class WeddingRingEnhancerV146:
    def __init__(self):
        """Initialize without Replicate client to avoid RunPod issues"""
        print("[v146] Initializing WeddingRingEnhancerV146 - Improved Detection")
        self.replicate_client = None  # Will be created when needed
        self.training_completed = True
        
        # Enhanced color calibration data from 28 pairs + 10 correction pairs
        self.color_corrections = {
            'yellow_gold': {
                'h_shift': 5,
                's_mult': 1.18,  # Increased from training data
                'v_mult': 1.10,
                'warmth': 1.15,
                'brightness_boost': 1.05,
                'contrast': 1.12,
                'target_rgb': (255, 215, 0),
                'lab_adjust': {'l': 0, 'a': 3, 'b': 8}  # More yellow
            },
            'rose_gold': {
                'h_shift': -8,
                's_mult': 1.25,  # Enhanced pink saturation
                'v_mult': 1.08,
                'warmth': 1.10,
                'brightness_boost': 1.03,
                'contrast': 1.10,
                'target_rgb': (183, 110, 121),
                'lab_adjust': {'l': 0, 'a': 8, 'b': -3}  # More red/pink
            },
            'white_gold': {
                'h_shift': 0,
                's_mult': 0.80,  # Reduced saturation for cleaner white
                'v_mult': 1.18,
                'warmth': 0.95,
                'brightness_boost': 1.08,
                'contrast': 1.15,
                'target_rgb': (245, 245, 240),
                'lab_adjust': {'l': 5, 'a': -1, 'b': -2}  # Brighter, cooler
            },
            'plain_white': {  # Champagne gold
                'h_shift': 3,
                's_mult': 0.85,
                'v_mult': 1.12,
                'warmth': 1.05,
                'brightness_boost': 1.06,
                'contrast': 1.08,
                'target_rgb': (252, 247, 232),
                'lab_adjust': {'l': 3, 'a': 2, 'b': 5}  # Warm champagne tone
            }
        }

    def _init_replicate_client(self):
        """Initialize Replicate client only when needed"""
        if self.replicate_client is None and REPLICATE_AVAILABLE:
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if api_token:
                try:
                    print("[v146] Initializing Replicate client...")
                    self.replicate_client = replicate.Client(api_token=api_token)
                    print("[v146] Replicate client initialized successfully")
                except Exception as e:
                    print(f"[v146] Failed to initialize Replicate client: {e}")
                    self.replicate_client = None
            else:
                print("[v146] No REPLICATE_API_TOKEN found")

    def decode_base64_image(self, base64_string):
        """Decode base64 image with comprehensive error handling"""
        print(f"[v146] Decoding base64 image, length: {len(base64_string)}")
        
        try:
            # Clean the string
            base64_string = base64_string.strip()
            
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                print("[v146] Removed data URL prefix")
            
            # Try multiple decoding approaches
            image_data = None
            
            # Method 1: Direct decode
            try:
                image_data = base64.b64decode(base64_string, validate=True)
                print("[v146] Direct decode successful")
            except Exception as e1:
                print(f"[v146] Direct decode failed: {e1}")
                
                # Method 2: Add padding
                try:
                    missing_padding = len(base64_string) % 4
                    if missing_padding:
                        base64_string += '=' * (4 - missing_padding)
                        print(f"[v146] Added {4 - missing_padding} padding characters")
                    image_data = base64.b64decode(base64_string)
                    print("[v146] Decode with padding successful")
                except Exception as e2:
                    print(f"[v146] Decode with padding failed: {e2}")
                    
                    # Method 3: Force decode
                    try:
                        image_data = base64.b64decode(base64_string + '==')
                        print("[v146] Force decode successful")
                    except Exception as e3:
                        print(f"[v146] All decode methods failed")
                        raise ValueError(f"Failed to decode base64: {e1}, {e2}, {e3}")
            
            # Convert to image
            image = Image.open(io.BytesIO(image_data))
            print(f"[v146] Image opened successfully: {image.size}, mode: {image.mode}")
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                print("[v146] Converting RGBA to RGB")
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            elif image.mode != 'RGB':
                print(f"[v146] Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            print(f"[v146] Converted to numpy array: {img_array.shape}")
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print("[v146] Converted to BGR for OpenCV")
            
            return img_bgr
            
        except Exception as e:
            print(f"[v146] Error in decode_base64_image: {str(e)}")
            raise

    def detect_ring_with_opencv_fallback(self, image):
        """Fallback ring detection using OpenCV"""
        print("[v146] Attempting OpenCV fallback detection...")
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                filtered,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=int(min(image.shape[:2]) // 3)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Get the largest circle
                largest_circle = max(circles[0], key=lambda c: c[2])
                x, y, r = largest_circle
                
                # Convert circle to bounding box
                padding = int(r * 0.3)
                x1 = max(0, x - r - padding)
                y1 = max(0, y - r - padding)
                x2 = min(image.shape[1], x + r + padding)
                y2 = min(image.shape[0], y + r + padding)
                
                print(f"[v146] OpenCV detected circular object at ({x}, {y}) with radius {r}")
                
                return {
                    'box': [x1, y1, x2, y2],
                    'confidence': 0.5,  # Lower confidence for fallback
                    'label': 'ring (opencv)',
                    'method': 'opencv_circles'
                }
            
            # Try edge detection + contours
            edges = cv2.Canny(filtered, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find circular contours
                circular_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if 0.5 < circularity < 1.5:  # Reasonable circularity
                                circular_contours.append(contour)
                
                if circular_contours:
                    # Get the largest circular contour
                    largest_contour = max(circular_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Add padding
                    padding = int(min(w, h) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    print(f"[v146] OpenCV detected circular contour at ({x}, {y})")
                    
                    return {
                        'box': [x1, y1, x2, y2],
                        'confidence': 0.4,
                        'label': 'ring (contour)',
                        'method': 'opencv_contours'
                    }
            
            print("[v146] OpenCV fallback detection found no rings")
            return None
            
        except Exception as e:
            print(f"[v146] OpenCV fallback error: {e}")
            return None

    def detect_rings_multi_stage(self, image, timeout_seconds=20):
        """Multi-stage ring detection with multiple methods"""
        print("[v146] Starting multi-stage ring detection")
        
        def _run_detection():
            try:
                # Initialize client if needed
                self._init_replicate_client()
                
                if not self.replicate_client:
                    print("[v146] Replicate client not available - using fallback")
                    return self.detect_ring_with_opencv_fallback(image)
                
                # Convert image for upload
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
                
                # Stage 1: Try Grounding DINO with multiple prompts
                print("[v146] Stage 1: Grounding DINO detection")
                prompts = [
                    "ring . wedding ring . gold ring . silver ring . jewelry",
                    "wedding ring",
                    "ring",
                    "circular jewelry . ring shaped object"
                ]
                
                for prompt in prompts:
                    try:
                        buffer.seek(0)
                        output = self.replicate_client.run(
                            "adirik/grounding-dino:0b44f85d35e6e4dd42074e6f7f88f8f45f5a919abaef670a3e3fe434e892f41f",
                            input={
                                "image": buffer,
                                "prompt": prompt,
                                "box_threshold": 0.1,  # Very low threshold
                                "text_threshold": 0.1
                            }
                        )
                        
                        if self._parse_detection_output(output, "grounding-dino"):
                            return self._parse_detection_output(output, "grounding-dino")
                    except Exception as e:
                        print(f"[v146] Grounding DINO attempt failed: {e}")
                
                # Stage 2: Try OWL-ViT
                print("[v146] Stage 2: OWL-ViT detection")
                try:
                    buffer.seek(0)
                    output = self.replicate_client.run(
                        "zsxkib/owl-vit:7c1ff1ce7dcf8193402f95a3e6f608a9b7c16b1e8685544e658107434f936a46",
                        input={
                            "image": buffer,
                            "query": "a photo of a ring",
                            "score_threshold": 0.1
                        }
                    )
                    
                    if self._parse_detection_output(output, "owl-vit"):
                        return self._parse_detection_output(output, "owl-vit")
                except Exception as e:
                    print(f"[v146] OWL-ViT failed: {e}")
                
                # Stage 3: OpenCV fallback
                print("[v146] Stage 3: OpenCV fallback")
                return self.detect_ring_with_opencv_fallback(image)
                
            except Exception as e:
                print(f"[v146] Multi-stage detection error: {e}")
                return self.detect_ring_with_opencv_fallback(image)
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_detection)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v146] Detection timeout after {timeout_seconds} seconds")
            return self.detect_ring_with_opencv_fallback(image)
        except Exception as e:
            print(f"[v146] Detection execution error: {e}")
            return self.detect_ring_with_opencv_fallback(image)

    def _parse_detection_output(self, output, model_name):
        """Parse detection output from different models"""
        if not output:
            return None
        
        print(f"[v146] Parsing {model_name} output: {output}")
        
        try:
            # Handle different output formats
            boxes = []
            scores = []
            labels = []
            
            if isinstance(output, dict):
                # Format 1: Direct dict with boxes/scores/labels
                boxes = output.get('boxes', output.get('predictions', {}).get('boxes', []))
                scores = output.get('scores', output.get('predictions', {}).get('scores', []))
                labels = output.get('labels', output.get('predictions', {}).get('labels', []))
            elif isinstance(output, list) and len(output) > 0:
                # Format 2: List of detections
                for det in output:
                    if isinstance(det, dict):
                        box = det.get('box', det.get('bbox', det.get('bounding_box', [])))
                        score = det.get('score', det.get('confidence', 0.5))
                        label = det.get('label', det.get('class', 'ring'))
                        if box and len(box) >= 4:
                            boxes.append(box)
                            scores.append(score)
                            labels.append(label)
            
            if boxes and len(boxes) > 0:
                # Get highest confidence detection
                best_idx = 0
                if scores:
                    best_idx = scores.index(max(scores))
                
                box = boxes[best_idx]
                confidence = scores[best_idx] if scores else 0.5
                label = labels[best_idx] if labels else 'ring'
                
                # Ensure box has 4 coordinates
                if len(box) >= 4:
                    print(f"[v146] {model_name} detected: {label} (confidence: {confidence:.2f})")
                    return {
                        'box': box[:4],
                        'confidence': confidence,
                        'label': f"{label} ({model_name})",
                        'method': model_name
                    }
            
            return None
            
        except Exception as e:
            print(f"[v146] Error parsing {model_name} output: {e}")
            return None

    def crop_ring_area(self, image, box, padding_ratio=0.15):
        """Crop ring area with padding"""
        h, w = image.shape[:2]
        
        # Handle normalized coordinates
        if all(0 <= coord <= 1.5 for coord in box):
            x1, y1, x2, y2 = box
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
        else:
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
        
        print(f"[v146] Ring cropped: original {w}x{h} -> crop {ring_crop.shape[1]}x{ring_crop.shape[0]}")
        print(f"[v146] Crop coordinates: ({x1}, {y1}, {x2}, {y2})")
        
        return ring_crop, (x1, y1, x2, y2)

    def enhance_ring_with_gfpgan(self, ring_image, timeout_seconds=20):
        """Remove scratches and damage with GFPGAN"""
        print("[v146] Enhancing ring with GFPGAN...")
        
        def _run_gfpgan():
            try:
                if not self.replicate_client:
                    print("[v146] Replicate client not available")
                    return ring_image
                
                # Convert to PIL
                img_rgb = cv2.cvtColor(ring_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', quality=95)
                buffer.seek(0)
                
                print("[v146] Running GFPGAN for damage removal...")
                
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
                    print("[v146] GFPGAN enhancement successful")
                    return result
                else:
                    print("[v146] No output from GFPGAN")
                    return ring_image
                    
            except Exception as e:
                print(f"[v146] GFPGAN error: {e}")
                return ring_image
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gfpgan)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v146] GFPGAN timeout after {timeout_seconds} seconds")
            return ring_image
        except Exception as e:
            print(f"[v146] GFPGAN execution error: {e}")
            return ring_image

    def upscale_ring_with_realesrgan(self, ring_image, scale=4, timeout_seconds=20):
        """Upscale ring with Real-ESRGAN"""
        print(f"[v146] Upscaling ring with Real-ESRGAN (scale: {scale}x)...")
        
        def _run_realesrgan():
            try:
                if not self.replicate_client:
                    print("[v146] Replicate client not available")
                    return ring_image
                
                # Convert to PIL
                img_rgb = cv2.cvtColor(ring_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Save to buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', quality=95)
                buffer.seek(0)
                
                print("[v146] Running Real-ESRGAN...")
                
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
                    print(f"[v146] Real-ESRGAN upscaling successful: {result.shape}")
                    return result
                else:
                    print("[v146] No output from Real-ESRGAN")
                    return ring_image
                    
            except Exception as e:
                print(f"[v146] Real-ESRGAN error: {e}")
                return ring_image
        
        # Run with timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_realesrgan)
                result = future.result(timeout=timeout_seconds)
                return result
        except FuturesTimeoutError:
            print(f"[v146] Real-ESRGAN timeout after {timeout_seconds} seconds")
            return ring_image
        except Exception as e:
            print(f"[v146] Real-ESRGAN execution error: {e}")
            return ring_image

    def detect_metal_type(self, image):
        """Detect metal type using advanced color analysis with training data insights"""
        print("[v146] Starting metal type detection with training data")
        
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
            
            print(f"[v146] HSV means - H: {h_mean:.1f}, S: {s_mean:.1f}, V: {v_mean:.1f}")
            print(f"[v146] Center HSV - H: {h_center:.1f}, S: {s_center:.1f}, V: {v_center:.1f}")
            
            # RGB analysis for additional accuracy
            rgb_mean = np.mean(image[center_y1:center_y2, center_x1:center_x2], axis=(0, 1))
            b, g, r = rgb_mean[0], rgb_mean[1], rgb_mean[2]
            
            # Warmth calculation
            warmth = (r - b) / max(1, (r + g + b) / 3)
            yellowness = (r + g - 2*b) / max(1, (r + g + b) / 3)
            
            print(f"[v146] RGB center - R: {r:.1f}, G: {g:.1f}, B: {b:.1f}")
            print(f"[v146] Warmth: {warmth:.2f}, Yellowness: {yellowness:.2f}")
            
            # Enhanced detection logic based on 38 training pairs
            if s_center < 20:  # Very low saturation
                if v_center > 210 and warmth < 0.05:
                    metal_type = 'white_gold'
                else:
                    metal_type = 'plain_white'  # Champagne gold
            elif 20 <= h_center <= 30 and s_center > 35 and yellowness > 0.3:  # Yellow range
                metal_type = 'yellow_gold'
            elif (0 <= h_center <= 15 or h_center >= 170) and s_center > 25 and warmth > 0.12:  # Red/pink range
                metal_type = 'rose_gold'
            else:
                # Refined decision tree based on training data
                if warmth > 0.18 and s_center > 20:
                    metal_type = 'rose_gold'
                elif yellowness > 0.25 and s_center > 15:
                    metal_type = 'yellow_gold'
                elif v_center > 190 and warmth < 0.1:
                    metal_type = 'white_gold'
                else:
                    metal_type = 'plain_white'
            
            print(f"[v146] Detected metal type: {metal_type}")
            return metal_type
            
        except Exception as e:
            print(f"[v146] Error in metal detection: {e}")
            return 'white_gold'  # Safe default

    def apply_color_correction(self, image, metal_type):
        """Apply advanced color correction based on 38 training pairs"""
        print(f"[v146] Applying color correction for {metal_type} (38 pairs trained)")
        
        try:
            correction = self.color_corrections.get(metal_type, self.color_corrections['white_gold'])
            
            # Step 1: Brightness/Contrast pre-adjustment
            result = cv2.convertScaleAbs(image, 
                                        alpha=correction.get('contrast', 1.1), 
                                        beta=correction.get('brightness_boost', 1.0) * 5)
            
            # Step 2: HSV adjustments
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Apply hue shift
            hsv[:, :, 0] = (hsv[:, :, 0] + correction['h_shift']) % 180
            
            # Enhanced saturation adjustment based on training data
            s_channel = hsv[:, :, 1]
            if metal_type in ['yellow_gold', 'rose_gold']:
                # Non-linear saturation enhancement
                s_mask = np.clip(s_channel / 255.0, 0, 1)
                s_enhancement = correction['s_mult'] + (1 - s_mask) * 0.15
                hsv[:, :, 1] = np.clip(s_channel * s_enhancement, 0, 255)
            else:
                # Linear saturation adjustment for white metals
                hsv[:, :, 1] = np.clip(s_channel * correction['s_mult'], 0, 255)
            
            # Value adjustment with highlight protection
            v_channel = hsv[:, :, 2]
            v_mask = np.where(v_channel > 240, 0.7, 1.0)  # Stronger highlight protection
            hsv[:, :, 2] = np.clip(v_channel * correction['v_mult'] * v_mask, 0, 255)
            
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Step 3: LAB color grading with training data adjustments
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            lab_adj = correction.get('lab_adjust', {'l': 0, 'a': 0, 'b': 0})
            lab[:, :, 0] = np.clip(lab[:, :, 0] + lab_adj['l'], 0, 255)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + lab_adj['a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + lab_adj['b'], 0, 255)
            
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            # Step 4: Warmth adjustment
            if correction['warmth'] != 1.0:
                # Create warm/cool overlay
                warm_layer = result.copy()
                warm_layer[:, :, 0] = np.clip(warm_layer[:, :, 0] * 0.85, 0, 255)  # Reduce blue
                warm_layer[:, :, 2] = np.clip(warm_layer[:, :, 2] * 1.15, 0, 255)  # Increase red
                
                # Blend based on warmth factor
                blend_factor = abs(correction['warmth'] - 1.0)
                if correction['warmth'] > 1.0:
                    result = cv2.addWeighted(result, 1 - blend_factor * 0.5, warm_layer, blend_factor * 0.5, 0)
                else:
                    cool_layer = result.copy()
                    cool_layer[:, :, 0] = np.clip(cool_layer[:, :, 0] * 1.15, 0, 255)  # Increase blue
                    cool_layer[:, :, 2] = np.clip(cool_layer[:, :, 2] * 0.85, 0, 255)  # Reduce red
                    result = cv2.addWeighted(result, 1 - blend_factor * 0.5, cool_layer, blend_factor * 0.5, 0)
            
            # Step 5: Final brightness boost based on metal type
            final_brightness = correction.get('brightness_boost', 1.0)
            if final_brightness != 1.0:
                result = cv2.convertScaleAbs(result, alpha=final_brightness, beta=0)
            
            print("[v146] Color correction applied successfully with training data")
            return result
            
        except Exception as e:
            print(f"[v146] Error in color correction: {e}")
            return image

    def enhance_details(self, image):
        """Enhance ring details with advanced processing"""
        print("[v146] Enhancing details...")
        
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
            
            print("[v146] Details enhanced successfully")
            return result
            
        except Exception as e:
            print(f"[v146] Error in detail enhancement: {e}")
            return image

    def merge_enhanced_ring(self, original_image, enhanced_ring, crop_coords):
        """Merge enhanced ring back to original image with blending"""
        x1, y1, x2, y2 = crop_coords
        
        # Create copy of original
        result = original_image.copy()
        
        # Resize enhanced ring to original crop size
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        if enhanced_ring.shape[:2] != (crop_h, crop_w):
            print(f"[v146] Resizing enhanced ring from {enhanced_ring.shape[:2]} to ({crop_w}, {crop_h})")
            enhanced_ring = cv2.resize(enhanced_ring, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create smooth blend mask
        mask = np.ones((crop_h, crop_w), dtype=np.float32)
        
        # Feather edges (10% of size)
        feather_size = int(min(crop_w, crop_h) * 0.1)
        if feather_size > 0:
            # Top edge
            for i in range(feather_size):
                mask[i, :] *= i / feather_size
            # Bottom edge
            for i in range(feather_size):
                mask[-(i+1), :] *= i / feather_size
            # Left edge
            for i in range(feather_size):
                mask[:, i] *= i / feather_size
            # Right edge
            for i in range(feather_size):
                mask[:, -(i+1)] *= i / feather_size
        
        # Apply mask to blend
        mask_3channel = np.stack([mask, mask, mask], axis=2)
        blended = enhanced_ring * mask_3channel + result[y1:y2, x1:x2] * (1 - mask_3channel)
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        print("[v146] Enhanced ring merged with smooth blending")
        return result

    def create_thumbnail(self, image, size=(1000, 1300)):
        """Create high-resolution thumbnail"""
        print(f"[v146] Creating high-resolution thumbnail {size}")
        
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Calculate aspect ratio
            img_w, img_h = pil_img.size
            target_w, target_h = size
            
            # Determine scaling to fill the target size
            scale_w = target_w / img_w
            scale_h = target_h / img_h
            scale = max(scale_w, scale_h)  # Use max to ensure image fills the space
            
            # Calculate new size
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # Resize image
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop to exact target size
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            
            pil_img = pil_img.crop((left, top, right, bottom))
            
            print(f"[v146] Thumbnail created: {pil_img.size}")
            return pil_img
            
        except Exception as e:
            print(f"[v146] Error creating thumbnail: {e}")
            # Fallback to simple resize
            try:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
                return pil_img
            except:
                return None

    def _image_to_base64(self, image):
        """Convert PIL Image to base64 - MUST REMOVE PADDING"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # CRITICAL: Remove padding for Make.com
        img_base64 = img_base64.rstrip('=')
        print(f"[v146] Base64 encoding complete, length: {len(img_base64)}, padding removed")
        return img_base64

    def process_image(self, image_base64):
        """Main processing pipeline with multi-stage ring detection"""
        print("\n" + "="*50)
        print("[v146] Starting image processing pipeline")
        print("[v146] Features: Multi-stage detection, 38 training pairs, 1000x1300 thumbnail")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Decode image
            image = self.decode_base64_image(image_base64)
            print(f"[v146] Image decoded: {image.shape}")
            
            # Step 1: Multi-stage ring detection
            ring_detection = self.detect_rings_multi_stage(image)
            
            if ring_detection and ring_detection.get('confidence', 0) > 0.3:
                print(f"[v146] Ring detected with {ring_detection['method']}! Processing with enhancement pipeline...")
                
                # Step 2: Crop ring area
                ring_crop, crop_coords = self.crop_ring_area(image, ring_detection['box'])
                
                # Step 3: Detect metal type on cropped ring
                metal_type = self.detect_metal_type(ring_crop)
                
                # Step 4: Enhance with GFPGAN (remove damage)
                enhanced_ring = self.enhance_ring_with_gfpgan(ring_crop)
                
                # Step 5: Upscale with Real-ESRGAN
                upscaled_ring = self.upscale_ring_with_realesrgan(enhanced_ring, scale=4)
                
                # Step 6: Apply color correction with training data
                color_corrected = self.apply_color_correction(upscaled_ring, metal_type)
                
                # Step 7: Enhance details
                detail_enhanced = self.enhance_details(color_corrected)
                
                # Step 8: Merge back to original with blending
                final_image = self.merge_enhanced_ring(image, detail_enhanced, crop_coords)
                
                ring_info = {
                    'detected': True,
                    'confidence': ring_detection['confidence'],
                    'label': ring_detection['label'],
                    'box': ring_detection['box'],
                    'method': ring_detection['method'],
                    'crop_size': ring_crop.shape[:2]
                }
            else:
                print("[v146] No ring detected with sufficient confidence - applying global enhancement")
                
                # Fallback: process entire image
                metal_type = self.detect_metal_type(image)
                
                # Apply color correction with training data
                color_corrected = self.apply_color_correction(image, metal_type)
                
                # Enhance details
                final_image = self.enhance_details(color_corrected)
                
                ring_info = {
                    'detected': False,
                    'message': 'No ring detected, applied global enhancement',
                    'attempted_methods': ['grounding-dino', 'owl-vit', 'opencv']
                }
            
            # Create outputs
            enhanced_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            thumbnail = self.create_thumbnail(final_image, size=(1000, 1300))  # High-res thumbnail
            
            # Convert to base64 (PADDING REMOVED)
            enhanced_base64 = self._image_to_base64(enhanced_pil)
            thumbnail_base64 = self._image_to_base64(thumbnail) if thumbnail else None
            
            processing_time = time.time() - start_time
            
            print(f"\n[v146] Processing complete in {processing_time:.2f}s")
            print(f"[v146] Metal type: {metal_type}")
            print(f"[v146] Ring detection: {ring_info}")
            print(f"[v146] Training data: 28 pairs + 10 correction pairs applied")
            print("="*50 + "\n")
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "ring_detection": ring_info,
                    "processing_time": round(processing_time, 2),
                    "original_size": list(image.shape[:2]),
                    "thumbnail_size": [1000, 1300],
                    "version": "v146-improved-detection",
                    "training_data": "38 pairs (28 original + 10 corrections)",
                    "pipeline_used": "multi-stage detection + gfpgan + realesrgan" if ring_info['detected'] else "global-enhancement"
                }
            }
            
        except Exception as e:
            print(f"[v146] Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Global instance
enhancer_instance = None

def get_enhancer():
    """Get or create enhancer instance"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV146()
    return enhancer_instance


def find_image_in_data(data, depth=0, max_depth=5):
    """Recursively find image data in nested structure"""
    if depth > max_depth:
        return None
    
    print(f"[v146] Searching at depth {depth}, type: {type(data)}")
    
    # If string, might be base64
    if isinstance(data, str):
        if len(data) > 100:  # Likely base64
            return data
    
    # If dict, search keys
    if isinstance(data, dict):
        # Direct image keys
        for key in ['image', 'image_base64', 'base64', 'img', 'data', 'imageData', 'image_data', 'input_image']:
            if key in data and data[key]:
                print(f"[v146] Found image in key: {key}")
                return data[key]
        
        # Search nested structures
        for key in data:
            result = find_image_in_data(data[key], depth + 1, max_depth)
            if result:
                return result
    
    return None


def handler(event):
    """RunPod handler function - v146 Improved Detection"""
    try:
        print("\n" + "="*70)
        print("[v146] Handler started - Improved Multi-Stage Detection")
        print("[v146] Features: Grounding DINO + OWL-ViT + OpenCV fallback")
        print("[v146] Training: 38 pairs applied, Thumbnail: 1000x1300")
        print(f"[v146] Python version: {sys.version}")
        print(f"[v146] Available modules - NumPy: {NUMPY_AVAILABLE}, PIL: {PIL_AVAILABLE}, CV2: {CV2_AVAILABLE}")
        print(f"[v146] Replicate module available: {REPLICATE_AVAILABLE}")
        print(f"[v146] Replicate token set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
        print("="*70)
        
        # Get input data
        input_data = event.get('input', {})
        print(f"[v146] Input type: {type(input_data)}")
        print(f"[v146] Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Debug mode
        if input_data.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": "v146 handler working - Improved Detection Pipeline",
                    "features": {
                        "padding_removal": "Active for Make.com compatibility",
                        "ring_detection": "Multi-stage (Grounding DINO + OWL-ViT + OpenCV)",
                        "damage_removal": "GFPGAN",
                        "upscaling": "Real-ESRGAN 4x",
                        "metal_types": ["yellow_gold", "rose_gold", "white_gold", "plain_white"],
                        "training_data": "38 pairs (28 original + 10 corrections)",
                        "thumbnail_size": "1000x1300"
                    },
                    "modules": {
                        "replicate": REPLICATE_AVAILABLE,
                        "opencv": CV2_AVAILABLE,
                        "numpy": NUMPY_AVAILABLE,
                        "pil": PIL_AVAILABLE
                    },
                    "version": "v146-improved-detection"
                }
            }
        
        # Find image data
        image_data = find_image_in_data(input_data)
        
        if not image_data:
            print("[v146] No image data found in input")
            print(f"[v146] Available keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'N/A'}")
            
            # Log sample of each key
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    sample = str(value)[:100] if value else "None"
                    print(f"[v146] Key '{key}': {sample}...")
            
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error",
                    "available_keys": list(input_data.keys()) if isinstance(input_data, dict) else [],
                    "version": "v146"
                }
            }
        
        print(f"[v146] Found image data, length: {len(image_data) if isinstance(image_data, str) else 'N/A'}")
        
        # Process image
        enhancer = get_enhancer()
        result = enhancer.process_image(image_data)
        
        # CRITICAL: Ensure padding is removed from ALL outputs
        for key in ['enhanced_image', 'thumbnail']:
            if key in result and result[key]:
                # Double-check padding removal
                result[key] = result[key].rstrip('=')
                print(f"[v146] Verified padding removed from {key}")
        
        print("[v146] Handler completed successfully")
        
        # Return with proper structure
        return {
            "output": result
        }
        
    except Exception as e:
        print(f"[v146] Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v146"
            }
        }


# RunPod entry point
if __name__ == "__main__":
    print("[v146] Starting RunPod serverless worker...")
    print("[v146] Pipeline: Multi-stage detection → GFPGAN → Real-ESRGAN")
    print("[v146] Training data: 38 pairs applied")
    print("[v146] CRITICAL: Make.com padding removal active")
    
    if RUNPOD_AVAILABLE:
        runpod.serverless.start({"handler": handler})
    else:
        print("[v146] Testing mode - runpod not available")
        
        # Test with mock event
        test_event = {
            "input": {
                "debug_mode": True
            }
        }
        
        result = handler(test_event)
        print(f"[v146] Test result: {json.dumps(result, indent=2)}")
