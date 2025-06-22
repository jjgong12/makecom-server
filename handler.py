import runpod
import os
import sys
import json
import base64
import io
import time
import traceback
import threading
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Safe imports with fallbacks
print("[v141] Starting imports...")

try:
    import numpy as np
    print("[v141] NumPy imported successfully")
    NUMPY_AVAILABLE = True
except ImportError as e:
    print(f"[v141] NumPy import failed: {e}")
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image, ImageEnhance, ImageDraw, ImageFilter, ImageOps
    print("[v141] PIL imported successfully")
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"[v141] PIL import failed: {e}")
    PIL_AVAILABLE = False

try:
    import cv2
    print("[v141] OpenCV imported successfully")
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"[v141] OpenCV import failed: {e}")
    CV2_AVAILABLE = False
    cv2 = None

# DO NOT import replicate globally - causes crashes in RunPod
REPLICATE_AVAILABLE = False
try:
    import importlib.util
    spec = importlib.util.find_spec("replicate")
    if spec is not None:
        print("[v141] Replicate module found but NOT imported globally (safety)")
        REPLICATE_AVAILABLE = True
except:
    print("[v141] Replicate module not found")

# Metal types
class MetalType(Enum):
    YELLOW_GOLD = "yellow_gold"
    ROSE_GOLD = "rose_gold"
    WHITE_GOLD = "white_gold"
    PLATINUM = "platinum"
    UNPLATED_WHITE = "unplated_white"

# Detection result
@dataclass
class RingDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    metal_type: Optional[MetalType] = None
    has_masking: bool = False

# 38-pair trained enhancement parameters
ENHANCEMENT_PARAMS = {
    'yellow_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.12, 'saturation': 1.20, 
            'sharpness': 1.25, 'white_overlay': 0.08, 'color_temp': 5,
            'gamma': 1.05, 'noise_reduction': 0.3
        },
        'studio': {
            'brightness': 1.20, 'contrast': 1.15, 'saturation': 1.25,
            'sharpness': 1.30, 'white_overlay': 0.10, 'color_temp': 3,
            'gamma': 1.08, 'noise_reduction': 0.2
        },
        'outdoor': {
            'brightness': 1.10, 'contrast': 1.10, 'saturation': 1.15,
            'sharpness': 1.20, 'white_overlay': 0.06, 'color_temp': 7,
            'gamma': 1.02, 'noise_reduction': 0.4
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.12, 'contrast': 1.10, 'saturation': 1.15,
            'sharpness': 1.22, 'white_overlay': 0.06, 'color_temp': 2,
            'gamma': 1.03, 'noise_reduction': 0.3
        },
        'studio': {
            'brightness': 1.18, 'contrast': 1.13, 'saturation': 1.18,
            'sharpness': 1.28, 'white_overlay': 0.08, 'color_temp': 0,
            'gamma': 1.05, 'noise_reduction': 0.2
        },
        'outdoor': {
            'brightness': 1.08, 'contrast': 1.08, 'saturation': 1.12,
            'sharpness': 1.18, 'white_overlay': 0.04, 'color_temp': 4,
            'gamma': 1.00, 'noise_reduction': 0.4
        }
    },
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.15, 'saturation': 1.05,
            'sharpness': 1.28, 'white_overlay': 0.12, 'color_temp': -3,
            'gamma': 1.02, 'noise_reduction': 0.2
        },
        'studio': {
            'brightness': 1.22, 'contrast': 1.18, 'saturation': 1.08,
            'sharpness': 1.32, 'white_overlay': 0.15, 'color_temp': -5,
            'gamma': 1.05, 'noise_reduction': 0.1
        },
        'outdoor': {
            'brightness': 1.15, 'contrast': 1.12, 'saturation': 1.03,
            'sharpness': 1.25, 'white_overlay': 0.10, 'color_temp': -2,
            'gamma': 1.00, 'noise_reduction': 0.3
        }
    },
    'platinum': {
        'natural': {
            'brightness': 1.20, 'contrast': 1.17, 'saturation': 1.02,
            'sharpness': 1.30, 'white_overlay': 0.14, 'color_temp': -4,
            'gamma': 1.03, 'noise_reduction': 0.2
        },
        'studio': {
            'brightness': 1.25, 'contrast': 1.20, 'saturation': 1.05,
            'sharpness': 1.35, 'white_overlay': 0.17, 'color_temp': -6,
            'gamma': 1.06, 'noise_reduction': 0.1
        },
        'outdoor': {
            'brightness': 1.17, 'contrast': 1.15, 'saturation': 1.00,
            'sharpness': 1.28, 'white_overlay': 0.12, 'color_temp': -3,
            'gamma': 1.01, 'noise_reduction': 0.3
        }
    },
    'unplated_white': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.13, 'saturation': 1.08,
            'sharpness': 1.26, 'white_overlay': 0.10, 'color_temp': -2,
            'gamma': 1.04, 'noise_reduction': 0.25
        },
        'studio': {
            'brightness': 1.20, 'contrast': 1.16, 'saturation': 1.10,
            'sharpness': 1.30, 'white_overlay': 0.13, 'color_temp': -4,
            'gamma': 1.07, 'noise_reduction': 0.15
        },
        'outdoor': {
            'brightness': 1.13, 'contrast': 1.10, 'saturation': 1.05,
            'sharpness': 1.23, 'white_overlay': 0.08, 'color_temp': -1,
            'gamma': 1.02, 'noise_reduction': 0.35
        }
    }
}

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 image with multiple fallback methods"""
    try:
        print(f"[v141] Base64 string length: {len(base64_string)}")
        
        # Clean the string first
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Try different decoding methods
        decoded = None
        
        # Method 1: Direct decode (no padding modification)
        try:
            decoded = base64.b64decode(base64_string, validate=True)
            print("[v141] Method 1 (direct with validation) successful")
        except:
            pass
        
        # Method 2: Add padding if needed
        if decoded is None:
            try:
                missing_padding = len(base64_string) % 4
                if missing_padding:
                    base64_string += '=' * (4 - missing_padding)
                decoded = base64.b64decode(base64_string)
                print(f"[v141] Method 2 (added {4-missing_padding} padding) successful")
            except:
                pass
        
        # Method 3: URL-safe decode
        if decoded is None:
            try:
                decoded = base64.urlsafe_b64decode(base64_string)
                print("[v141] Method 3 (URL-safe) successful")
            except:
                pass
        
        # Method 4: Force decode with padding variations
        if decoded is None:
            for pad in ['', '=', '==', '===']:
                try:
                    decoded = base64.b64decode(base64_string + pad)
                    print(f"[v141] Method 4 (padding: '{pad}') successful")
                    break
                except:
                    continue
        
        if decoded is None:
            raise ValueError("All base64 decoding methods failed")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(decoded))
        print(f"[v141] Image decoded successfully: {image.size}")
        return image
        
    except Exception as e:
        print(f"[v141] Failed to decode base64: {str(e)}")
        raise

class WeddingRingEnhancerV141:
    """Wedding ring enhancer with GAS compatibility and timeout protection"""
    
    def __init__(self):
        print("[v141] Initializing WeddingRingEnhancerV141...")
        self.replicate_enabled = False
        self.replicate_timeout = 20  # 20 second timeout
        self.replicate_client = None
        
        # Check if Replicate should be enabled
        if REPLICATE_AVAILABLE and os.environ.get('REPLICATE_API_TOKEN'):
            print("[v141] Replicate token found, will initialize on first use")
            self.replicate_enabled = True
        else:
            print("[v141] Replicate disabled (no token or module)")
            
    def _init_replicate_lazy(self):
        """Lazy initialization of Replicate to avoid startup crashes"""
        if self.replicate_client is not None:
            return True
            
        try:
            print("[v141] Lazy loading Replicate module...")
            import replicate
            self.replicate_client = replicate.Client(
                api_token=os.environ.get('REPLICATE_API_TOKEN')
            )
            print("[v141] Replicate client initialized successfully")
            return True
        except Exception as e:
            print(f"[v141] Failed to initialize Replicate: {e}")
            self.replicate_enabled = False
            return False
    
    def enhance_image(self, image: Image.Image) -> Dict[str, Any]:
        """Main enhancement pipeline"""
        try:
            start_time = time.time()
            print(f"[v141] Starting enhancement for image size: {image.size}")
            
            # Step 1: Detect rings and analyze
            detections = self._detect_rings(image)
            print(f"[v141] Found {len(detections)} rings")
            
            # Step 2: Detect masking if any rings found
            has_masking = False
            if detections:
                has_masking = self._detect_masking(image, detections)
                print(f"[v141] Masking detected: {has_masking}")
            
            # Step 3: Determine metal type and lighting
            metal_type = self._detect_metal_type(image, detections)
            lighting = self._detect_lighting(image)
            print(f"[v141] Metal: {metal_type}, Lighting: {lighting}")
            
            # Step 4: Apply enhancement
            enhanced = self._apply_enhancement(image, metal_type, lighting)
            
            # Step 5: Remove masking if detected and Replicate available
            if has_masking and self.replicate_enabled:
                print("[v141] Attempting masking removal with Replicate...")
                enhanced = self._remove_masking_safe(enhanced, detections)
            
            # Step 6: Create thumbnail
            thumbnail = self._create_thumbnail(enhanced)
            
            # Convert to base64 - DO NOT REMOVE PADDING FOR GAS COMPATIBILITY
            enhanced_base64 = self._image_to_base64(enhanced)
            thumbnail_base64 = self._image_to_base64(thumbnail) if thumbnail else None
            
            processing_time = time.time() - start_time
            print(f"[v141] Enhancement completed in {processing_time:.2f}s")
            
            response = {
                "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
                "status": "success",
                "metal_type": metal_type,
                "lighting": lighting,
                "rings_detected": len(detections),
                "masking_removed": has_masking and self.replicate_enabled,
                "processing_time": f"{processing_time:.2f}s",
                "version": "v141-gas-compatible"
            }
            
            if thumbnail_base64:
                response["thumbnail"] = f"data:image/png;base64,{thumbnail_base64}"
                
            return response
            
        except Exception as e:
            print(f"[v141] Enhancement error: {str(e)}")
            print(traceback.format_exc())
            
            # Return original with error info
            try:
                original_base64 = self._image_to_base64(image)
                return {
                    "enhanced_image": f"data:image/png;base64,{original_base64}",
                    "status": "error",
                    "error": str(e),
                    "version": "v141-gas-compatible"
                }
            except:
                return {
                    "status": "error",
                    "error": "Critical error in processing",
                    "version": "v141-gas-compatible"
                }
    
    def _detect_rings(self, image: Image.Image) -> List[RingDetection]:
        """Detect wedding rings using fallback methods"""
        detections = []
        
        try:
            if CV2_AVAILABLE and NUMPY_AVAILABLE:
                # Convert to OpenCV format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Detect circles (rings)
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=50,
                    param1=50,
                    param2=30,
                    minRadius=20,
                    maxRadius=300
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        x, y, r = circle
                        # Create bounding box from circle
                        bbox = (
                            max(0, x - r),
                            max(0, y - r),
                            min(image.width, x + r),
                            min(image.height, y + r)
                        )
                        detection = RingDetection(
                            bbox=bbox,
                            confidence=0.8
                        )
                        detections.append(detection)
                        
            # If no circles found, assume center region
            if not detections:
                # Default to center region
                w, h = image.size
                center_bbox = (w//4, h//4, 3*w//4, 3*h//4)
                detections.append(RingDetection(
                    bbox=center_bbox,
                    confidence=0.5
                ))
                
        except Exception as e:
            print(f"[v141] Ring detection error: {e}")
            # Fallback to center detection
            w, h = image.size
            center_bbox = (w//4, h//4, 3*w//4, 3*h//4)
            detections.append(RingDetection(
                bbox=center_bbox,
                confidence=0.3
            ))
            
        return detections
    
    def _detect_masking(self, image: Image.Image, detections: List[RingDetection]) -> bool:
        """Detect if rings have masking/censoring"""
        if not detections:
            return False
            
        try:
            # Check first detection area for masking patterns
            det = detections[0]
            x1, y1, x2, y2 = det.bbox
            
            # Crop ring area
            ring_area = image.crop((x1, y1, x2, y2))
            
            # Convert to grayscale
            gray = ring_area.convert('L')
            
            # Check for solid color blocks (masking indicator)
            pixels = list(gray.getdata())
            unique_colors = len(set(pixels))
            
            # If very few unique colors in ring area, likely masked
            if unique_colors < 50:  # Threshold
                return True
                
            # Check for blur patterns
            if CV2_AVAILABLE and NUMPY_AVAILABLE:
                ring_cv = cv2.cvtColor(np.array(ring_area), cv2.COLOR_RGB2BGR)
                laplacian_var = cv2.Laplacian(ring_cv, cv2.CV_64F).var()
                if laplacian_var < 100:  # Blurry = likely masked
                    return True
                    
        except Exception as e:
            print(f"[v141] Masking detection error: {e}")
            
        return False
    
    def _detect_metal_type(self, image: Image.Image, detections: List[RingDetection]) -> str:
        """Detect metal type from ring color"""
        if not detections:
            return 'white_gold'  # Default
            
        try:
            # Analyze first detection
            det = detections[0]
            x1, y1, x2, y2 = det.bbox
            ring_area = image.crop((x1, y1, x2, y2))
            
            # Get average color
            ring_array = np.array(ring_area)
            avg_color = ring_array.mean(axis=(0, 1))
            r, g, b = avg_color
            
            # Color-based detection
            yellow_score = r + g - b
            rose_score = r - g
            white_score = min(r, g, b)
            
            if yellow_score > 300:
                return 'yellow_gold'
            elif rose_score > 30:
                return 'rose_gold'
            elif white_score > 200:
                return 'platinum'
            else:
                return 'white_gold'
                
        except Exception as e:
            print(f"[v141] Metal detection error: {e}")
            return 'white_gold'
    
    def _detect_lighting(self, image: Image.Image) -> str:
        """Detect lighting conditions"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            histogram = gray.histogram()
            
            # Calculate brightness distribution
            total_pixels = sum(histogram)
            mean_brightness = sum(i * histogram[i] for i in range(256)) / total_pixels
            
            # Determine lighting
            if mean_brightness > 200:
                return 'studio'
            elif mean_brightness < 100:
                return 'outdoor'
            else:
                return 'natural'
                
        except:
            return 'natural'
    
    def _apply_enhancement(self, image: Image.Image, metal_type: str, lighting: str) -> Image.Image:
        """Apply 38-pair trained enhancements"""
        try:
            params = ENHANCEMENT_PARAMS.get(metal_type, {}).get(lighting, {})
            if not params:
                params = ENHANCEMENT_PARAMS['white_gold']['natural']
            
            enhanced = image.copy()
            
            # Apply enhancements in order
            if params.get('brightness', 1.0) != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(params['brightness'])
            
            if params.get('contrast', 1.0) != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(params['contrast'])
            
            if params.get('saturation', 1.0) != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(params['saturation'])
            
            if params.get('sharpness', 1.0) != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(params['sharpness'])
            
            # Apply white overlay if specified
            if params.get('white_overlay', 0) > 0:
                overlay = Image.new('RGB', enhanced.size, (255, 255, 255))
                enhanced = Image.blend(enhanced, overlay, params['white_overlay'])
            
            # Apply color temperature adjustment
            if params.get('color_temp', 0) != 0 and NUMPY_AVAILABLE:
                enhanced = self._adjust_color_temperature(enhanced, params['color_temp'])
            
            return enhanced
            
        except Exception as e:
            print(f"[v141] Enhancement error: {e}")
            return image
    
    def _adjust_color_temperature(self, image: Image.Image, temp_adjust: int) -> Image.Image:
        """Adjust color temperature"""
        try:
            # Convert to array
            img_array = np.array(image, dtype=np.float32)
            
            # Adjust color channels
            if temp_adjust > 0:  # Warmer
                img_array[:, :, 0] *= 1 + (temp_adjust * 0.01)  # More red
                img_array[:, :, 2] *= 1 - (temp_adjust * 0.005)  # Less blue
            else:  # Cooler
                img_array[:, :, 0] *= 1 + (temp_adjust * 0.005)  # Less red
                img_array[:, :, 2] *= 1 - (temp_adjust * 0.01)  # More blue
            
            # Clip values
            img_array = np.clip(img_array, 0, 255)
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except:
            return image
    
    def _remove_masking_safe(self, image: Image.Image, detections: List[RingDetection]) -> Image.Image:
        """Remove masking using Replicate with timeout protection"""
        
        def run_replicate():
            """Run Replicate in separate thread"""
            try:
                # Initialize Replicate if needed
                if not self._init_replicate_lazy():
                    return None
                
                # Convert image to base64
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Create mask for inpainting
                mask = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask)
                
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    # Expand bbox slightly
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(image.width, x2 + margin)
                    y2 = min(image.height, y2 + margin)
                    draw.ellipse([x1, y1, x2, y2], fill=255)
                
                # Convert mask to base64
                mask_buffer = io.BytesIO()
                mask.save(mask_buffer, format='PNG')
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
                
                # Run inpainting
                output = self.replicate_client.run(
                    "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                    input={
                        "image": f"data:image/png;base64,{img_base64}",
                        "mask": f"data:image/png;base64,{mask_base64}",
                        "prompt": "professional wedding ring photo, high quality jewelry photography",
                        "negative_prompt": "blur, censored, masked, low quality",
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5
                    }
                )
                
                if output and len(output) > 0:
                    # Get result image
                    result_url = output[0]
                    import requests
                    response = requests.get(result_url, timeout=10)
                    
                    if response.status_code == 200:
                        result_image = Image.open(io.BytesIO(response.content))
                        return result_image
                        
                return None
                
            except Exception as e:
                print(f"[v141] Replicate processing error: {e}")
                return None
        
        try:
            # Run with timeout using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_replicate)
                
                try:
                    result = future.result(timeout=self.replicate_timeout)
                    if result is not None:
                        print("[v141] Masking removed successfully")
                        return result
                    else:
                        print("[v141] Replicate returned no result")
                        
                except TimeoutError:
                    print(f"[v141] Replicate timed out after {self.replicate_timeout}s")
                    future.cancel()
                    
        except Exception as e:
            print(f"[v141] Safe masking removal error: {e}")
        
        # Return original if Replicate fails
        print("[v141] Returning original image (Replicate failed)")
        return image
    
    def _create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (1000, 1300)) -> Optional[Image.Image]:
        """Create thumbnail with upscaling"""
        try:
            # Calculate aspect ratios
            img_ratio = image.width / image.height
            target_ratio = size[0] / size[1]
            
            if img_ratio > target_ratio:
                # Image is wider
                new_width = size[0]
                new_height = int(size[0] / img_ratio)
            else:
                # Image is taller
                new_height = size[1]
                new_width = int(size[1] * img_ratio)
            
            # Resize
            thumbnail = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create white background
            background = Image.new('RGB', size, (255, 255, 255))
            
            # Paste centered
            x = (size[0] - new_width) // 2
            y = (size[1] - new_height) // 2
            background.paste(thumbnail, (x, y))
            
            return background
            
        except Exception as e:
            print(f"[v141] Thumbnail creation error: {e}")
            return None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 string - KEEP PADDING FOR GAS"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        # DO NOT REMOVE PADDING - Google Apps Script needs it!
        print(f"[v141] Base64 encoding complete, length: {len(img_base64)}, has padding: {img_base64[-2:] == '=='}")
        return img_base64

# Global instance (safe because no Replicate init)
enhancer_instance = None

def get_enhancer():
    """Get or create enhancer instance"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV141()
    return enhancer_instance

def handler(event):
    """RunPod handler function - v141 GAS Compatible"""
    try:
        print("="*70)
        print("[v141] Handler started - Google Apps Script Compatible Version")
        print(f"[v141] Python version: {sys.version}")
        print(f"[v141] Available modules - NumPy: {NUMPY_AVAILABLE}, PIL: {PIL_AVAILABLE}, CV2: {CV2_AVAILABLE}")
        print(f"[v141] Replicate module available: {REPLICATE_AVAILABLE}")
        print(f"[v141] Replicate token set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
        print("="*70)
        
        # Get input data
        input_data = event.get('input', {})
        print(f"[v141] Input type: {type(input_data)}")
        print(f"[v141] Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Debug mode
        if input_data.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": "v141 handler is working - GAS Compatible",
                    "replicate_available": REPLICATE_AVAILABLE,
                    "replicate_token_set": bool(os.environ.get('REPLICATE_API_TOKEN')),
                    "opencv_available": CV2_AVAILABLE,
                    "numpy_available": NUMPY_AVAILABLE,
                    "pil_available": PIL_AVAILABLE,
                    "version": "v141-gas-compatible"
                }
            }
        
        # Extract image data with multiple fallbacks
        image_data = None
        
        # Try different keys
        for key in ['image', 'image_base64', 'input_image', 'base64', 'data']:
            if key in input_data:
                image_data = input_data[key]
                if image_data:
                    print(f"[v141] Found image data in key: {key}")
                    break
        
        # Try nested structures
        if not image_data and isinstance(input_data, dict):
            for pattern in ['data', 'body', 'payload']:
                if pattern in input_data and isinstance(input_data[pattern], dict):
                    for key in ['image', 'image_base64']:
                        if key in input_data[pattern]:
                            image_data = input_data[pattern][key]
                            if image_data:
                                print(f"[v141] Found image data in {pattern}.{key}")
                                break
        
        if not image_data:
            print("[v141] No image data found in input")
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "debug_info": {
                        "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict",
                        "input_type": str(type(input_data))
                    },
                    "version": "v141-gas-compatible"
                }
            }
        
        # Decode image
        print(f"[v141] Decoding image data (length: {len(str(image_data))})")
        image = decode_base64_image(image_data)
        print(f"[v141] Image decoded successfully: {image.size}")
        
        # Get enhancer and process
        enhancer = get_enhancer()
        
        # Process with timeout protection
        print("[v141] Starting image enhancement...")
        result = enhancer.enhance_image(image)
        
        # Return in RunPod format
        print("[v141] Processing complete, returning result")
        return {"output": result}
        
    except Exception as e:
        print(f"[v141] Handler error: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "version": "v141-gas-compatible"
            }
        }

# RunPod entry point
if __name__ == "__main__":
    print("="*70)
    print("Wedding Ring Enhancement v141 Starting...")
    print("Google Apps Script Compatible Version")
    print("IMPORTANT: Base64 padding is PRESERVED for GAS compatibility")
    print(f"Replicate Module Check: {REPLICATE_AVAILABLE}")
    print(f"Replicate Token Set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
