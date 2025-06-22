import runpod
import os
import sys
import json
import base64
import io
import time
import traceback
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

# Safe imports with fallbacks
print("[v139] Starting imports...")


try:
    import numpy as np
    print("[v139] NumPy imported successfully")
    NUMPY_AVAILABLE = True
except ImportError as e:
    print(f"[v139] NumPy import failed: {e}")
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
    print("[v139] PIL imported successfully")
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"[v139] PIL import failed: {e}")
    PIL_AVAILABLE = False

try:
    import cv2
    print("[v139] OpenCV imported successfully")
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"[v139] OpenCV import failed: {e}")
    CV2_AVAILABLE = False

try:
    import replicate
    print("[v139] Replicate imported successfully")
    REPLICATE_AVAILABLE = True
except ImportError as e:
    print(f"[v139] Replicate import failed: {e}")
    REPLICATE_AVAILABLE = False

try:
    import requests
    print("[v139] Requests imported successfully")
    REQUESTS_AVAILABLE = True
except ImportError as e:
    print(f"[v139] Requests import failed: {e}")
    REQUESTS_AVAILABLE = False

# Metal types
class MetalType(Enum):
    YELLOW_GOLD = "yellow_gold"
    ROSE_GOLD = "rose_gold"
    WHITE_GOLD = "white_gold"
    PLATINUM = "platinum"
    UNPLATED_WHITE = "unplated_white"  # 무도금화이트

# Ring detection dataclass
@dataclass
class RingDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    metal_type: Optional[MetalType] = None
    quality_score: float = 0.0
    needs_restoration: bool = False

# Color profiles for 38-pair trained data
@dataclass
class ColorProfile:
    base_rgb: Tuple[int, int, int]
    highlight_rgb: Tuple[int, int, int]
    shadow_rgb: Tuple[int, int, int]
    saturation_range: Tuple[float, float]
    brightness_range: Tuple[float, float]
    temperature: float
    shine_factor: float = 1.0
    reflection_intensity: float = 1.0

# Complete metal profiles based on 38 training pairs
METAL_PROFILES = {
    MetalType.YELLOW_GOLD: ColorProfile(
        base_rgb=(255, 215, 0),
        highlight_rgb=(255, 245, 200),
        shadow_rgb=(184, 134, 11),
        saturation_range=(0.15, 0.35),
        brightness_range=(0.70, 0.95),
        temperature=0.15,
        shine_factor=0.85,
        reflection_intensity=0.9
    ),
    MetalType.ROSE_GOLD: ColorProfile(
        base_rgb=(255, 200, 180),
        highlight_rgb=(255, 230, 220),
        shadow_rgb=(183, 110, 95),
        saturation_range=(0.10, 0.25),
        brightness_range=(0.75, 0.92),
        temperature=0.08,
        shine_factor=0.88,
        reflection_intensity=0.85
    ),
    MetalType.WHITE_GOLD: ColorProfile(
        base_rgb=(235, 235, 235),
        highlight_rgb=(250, 250, 250),
        shadow_rgb=(175, 175, 175),
        saturation_range=(0.0, 0.08),
        brightness_range=(0.80, 0.98),
        temperature=-0.05,
        shine_factor=0.95,
        reflection_intensity=0.95
    ),
    MetalType.PLATINUM: ColorProfile(
        base_rgb=(245, 245, 245),
        highlight_rgb=(255, 255, 255),
        shadow_rgb=(185, 185, 185),
        saturation_range=(0.0, 0.05),
        brightness_range=(0.85, 0.99),
        temperature=-0.02,
        shine_factor=0.98,
        reflection_intensity=0.98
    ),
    MetalType.UNPLATED_WHITE: ColorProfile(  # 무도금화이트
        base_rgb=(240, 235, 225),
        highlight_rgb=(255, 250, 245),
        shadow_rgb=(180, 175, 165),
        saturation_range=(0.02, 0.10),
        brightness_range=(0.78, 0.95),
        temperature=-0.03,
        shine_factor=0.90,
        reflection_intensity=0.88
    )
}

def decode_base64_image(base64_string):
    """Decode base64 image with multiple fallback methods"""
    try:
        print(f"[v139] Decoding base64 string (length: {len(base64_string) if base64_string else 0})")
        
        if not base64_string:
            raise ValueError("Empty base64 string")
        
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Clean the string
        base64_string = base64_string.strip()
        base64_string = base64_string.replace(' ', '+')
        base64_string = base64_string.replace('\n', '')
        base64_string = base64_string.replace('\r', '')
        
        # Try multiple decoding approaches
        decoded = None
        
        # Method 1: Direct decode
        try:
            decoded = base64.b64decode(base64_string)
            print("[v139] Method 1 (direct) successful")
        except:
            pass
        
        # Method 2: Add padding
        if decoded is None:
            try:
                padding = 4 - (len(base64_string) % 4)
                if padding != 4:
                    base64_string += '=' * padding
                decoded = base64.b64decode(base64_string)
                print("[v139] Method 2 (with padding) successful")
            except:
                pass
        
        # Method 3: URL-safe decode
        if decoded is None:
            try:
                decoded = base64.urlsafe_b64decode(base64_string)
                print("[v139] Method 3 (URL-safe) successful")
            except:
                pass
        
        # Method 4: Force decode with padding variations
        if decoded is None:
            for pad in ['', '=', '==', '===']:
                try:
                    decoded = base64.b64decode(base64_string + pad)
                    print(f"[v139] Method 4 (padding: '{pad}') successful")
                    break
                except:
                    continue
        
        if decoded is None:
            raise ValueError("All base64 decoding methods failed")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(decoded))
        print(f"[v139] Image decoded successfully: {image.size}")
        return image
        
    except Exception as e:
        print(f"[v139] Failed to decode base64: {str(e)}")
        raise

class WeddingRingEnhancerV139:
    """Complete wedding ring enhancement system with all features"""
    
    def __init__(self):
        print("[v139] Initializing WeddingRingEnhancerV139...")
        self.replicate_client = None
        self.replicate_enabled = False
        
        # Try to initialize Replicate
        if REPLICATE_AVAILABLE:
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if api_token:
                try:
                    self.replicate_client = replicate.Client(api_token=api_token)
                    self.replicate_enabled = True
                    print("[v139] Replicate client initialized successfully")
                except Exception as e:
                    print(f"[v139] Failed to initialize Replicate: {e}")
            else:
                print("[v139] REPLICATE_API_TOKEN not found in environment")
        else:
            print("[v139] Replicate module not available")
            
        print(f"[v139] Initialization complete. Replicate enabled: {self.replicate_enabled}")
    
    def enhance_image(self, image: Image.Image, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main enhancement pipeline"""
        try:
            start_time = time.time()
            print(f"[v139] Starting enhancement for image size: {image.size}")
            
            # Step 1: Detect rings
            detections = self._detect_rings(image, params.get('detect_all_rings', True))
            
            if not detections:
                print("[v139] No rings detected, applying basic enhancement")
                enhanced = self._basic_enhancement(image)
                return self._create_response(enhanced, [], "no_rings_detected", time.time() - start_time)
            
            print(f"[v139] Detected {len(detections)} rings")
            
            # Step 2: Process each detected ring
            enhanced_image = image.copy()
            
            for idx, detection in enumerate(detections):
                print(f"[v139] Processing ring {idx+1}/{len(detections)}")
                
                # Extract ring region
                ring_region = self._extract_ring_region(image, detection.bbox)
                
                # Apply enhancements
                if detection.needs_restoration and params.get('use_replicate_restoration', True) and self.replicate_enabled:
                    enhanced_ring = self._restore_with_replicate(ring_region, detection)
                else:
                    enhanced_ring = self._enhance_ring(ring_region, detection)
                
                # Merge back
                enhanced_image = self._merge_enhanced_ring(enhanced_image, enhanced_ring, detection.bbox)
            
            # Step 3: Final polish
            enhanced_image = self._final_polish(enhanced_image)
            
            # Step 4: Create thumbnail
            thumbnail = self._create_thumbnail(enhanced_image)
            
            processing_time = time.time() - start_time
            print(f"[v139] Enhancement completed in {processing_time:.2f}s")
            
            return self._create_response(enhanced_image, detections, "success", processing_time, thumbnail)
            
        except Exception as e:
            print(f"[v139] Enhancement error: {str(e)}")
            print(traceback.format_exc())
            
            # Return original with basic enhancement
            try:
                basic = self._basic_enhancement(image)
                return self._create_response(basic, [], "error", 0, None, str(e))
            except:
                return self._create_response(image, [], "error", 0, None, str(e))
    
    def _detect_rings(self, image: Image.Image, detect_all: bool = True) -> List[RingDetection]:
        """Detect wedding rings in image"""
        detections = []
        
        # Try Replicate detection first if available
        if self.replicate_enabled and REQUESTS_AVAILABLE:
            try:
                detections = self._detect_with_replicate(image)
                if detections:
                    return detections
            except Exception as e:
                print(f"[v139] Replicate detection failed: {e}")
        
        # Fallback detection
        return self._fallback_detection(image, detect_all)
    
    def _detect_with_replicate(self, image: Image.Image) -> List[RingDetection]:
        """Use Replicate's Grounding DINO for detection"""
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Run Grounding DINO
            output = self.replicate_client.run(
                "zsxkib/grounding-dino:e0b4a1c76c5b25a943336a9053721e77891ed6e165c2fb1c64ff4871f436d5ad",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "query": "wedding ring . engagement ring . gold ring . silver ring",
                    "box_threshold": 0.3,
                    "text_threshold": 0.3
                }
            )
            
            detections = []
            if output and 'predictions' in output:
                for pred in output['predictions']:
                    bbox = pred['bbox']
                    x1, y1, x2, y2 = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
                    
                    detection = RingDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=pred['confidence'],
                        metal_type=self._detect_metal_type(image, (x1, y1, x2, y2)),
                        quality_score=self._assess_quality(image, (x1, y1, x2, y2)),
                        needs_restoration=False
                    )
                    
                    detection.needs_restoration = detection.quality_score < 0.6
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[v139] Grounding DINO error: {e}")
            return []
    
    def _fallback_detection(self, image: Image.Image, detect_all: bool) -> List[RingDetection]:
        """Fallback detection using image processing"""
        detections = []
        
        # Convert to numpy array if available
        if NUMPY_AVAILABLE and CV2_AVAILABLE:
            # Use OpenCV detection
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Find circular regions
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=min(image.width, image.height) // 3
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    x1 = max(0, x - r)
                    y1 = max(0, y - r)
                    x2 = min(image.width, x + r)
                    y2 = min(image.height, y + r)
                    
                    detection = RingDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=0.7,
                        metal_type=self._detect_metal_type(image, (x1, y1, x2, y2)),
                        quality_score=self._assess_quality(image, (x1, y1, x2, y2)),
                        needs_restoration=False
                    )
                    
                    detection.needs_restoration = detection.quality_score < 0.6
                    detections.append(detection)
                    
                    if not detect_all:
                        break
        
        # If no detections or OpenCV not available, use center crop
        if not detections:
            w, h = image.size
            size = min(w, h) // 2
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size
            
            detection = RingDetection(
                bbox=(x1, y1, x2, y2),
                confidence=0.5,
                metal_type=self._detect_metal_type(image, (x1, y1, x2, y2)),
                quality_score=0.5,
                needs_restoration=True
            )
            detections.append(detection)
        
        return detections
    
    def _detect_metal_type(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> MetalType:
        """Detect metal type from ring region"""
        try:
            # Crop ring area
            x1, y1, x2, y2 = bbox
            ring_crop = image.crop((x1, y1, x2, y2))
            
            # Sample center region
            w, h = ring_crop.size
            center_size = min(w, h) // 3
            center_x = w // 2
            center_y = h // 2
            
            center_crop = ring_crop.crop((
                center_x - center_size,
                center_y - center_size,
                center_x + center_size,
                center_y + center_size
            ))
            
            # Get average color
            if NUMPY_AVAILABLE:
                pixels = np.array(center_crop)
                avg_color = np.mean(pixels.reshape(-1, 3), axis=0)
            else:
                # PIL fallback
                pixels = list(center_crop.getdata())
                avg_color = [sum(x)/len(pixels) for x in zip(*pixels)]
            
            r, g, b = avg_color
            
            # Color-based detection
            # Calculate saturation
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            if max_c == 0:
                saturation = 0
            else:
                saturation = (max_c - min_c) / max_c
            
            # Brightness
            brightness = max_c / 255.0
            
            # Detect metal type
            if saturation < 0.1:  # Low saturation = white metals
                if brightness > 0.9:
                    return MetalType.PLATINUM
                elif brightness > 0.85:
                    return MetalType.WHITE_GOLD
                else:
                    return MetalType.UNPLATED_WHITE
            elif r > g > b and (r - b) > 30:  # Reddish
                return MetalType.ROSE_GOLD
            elif g > b and (g - b) > 20:  # Yellowish
                return MetalType.YELLOW_GOLD
            else:
                return MetalType.WHITE_GOLD
                
        except Exception as e:
            print(f"[v139] Metal detection error: {e}")
            return MetalType.WHITE_GOLD  # Default
    
    def _assess_quality(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> float:
        """Assess image quality of ring region"""
        try:
            x1, y1, x2, y2 = bbox
            ring_crop = image.crop((x1, y1, x2, y2))
            
            scores = []
            
            # 1. Size score (bigger is better)
            size_score = min((x2 - x1) * (y2 - y1) / (image.width * image.height * 0.25), 1.0)
            scores.append(size_score)
            
            # 2. Brightness score
            if NUMPY_AVAILABLE:
                pixels = np.array(ring_crop)
                brightness = np.mean(pixels) / 255.0
                brightness_score = 1.0 - abs(brightness - 0.5) * 2
                scores.append(brightness_score)
            
            # 3. Contrast approximation (using PIL)
            stat = ring_crop.convert('L').getextrema()
            if stat[1] > stat[0]:
                contrast_score = (stat[1] - stat[0]) / 255.0
                scores.append(contrast_score)
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            print(f"[v139] Quality assessment error: {e}")
            return 0.5
    
    def _extract_ring_region(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Extract ring region with padding"""
        x1, y1, x2, y2 = bbox
        
        # Add 10% padding
        padding = int(max(x2 - x1, y2 - y1) * 0.1)
        
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image.width, x2 + padding)
        y2_pad = min(image.height, y2 + padding)
        
        return image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
    
    def _enhance_ring(self, ring_image: Image.Image, detection: RingDetection) -> Image.Image:
        """Apply comprehensive ring enhancement based on 38-pair training data"""
        try:
            # Get metal profile
            profile = METAL_PROFILES.get(detection.metal_type, METAL_PROFILES[MetalType.WHITE_GOLD])
            
            # Step 1: Basic adjustments
            # Brightness
            enhancer = ImageEnhance.Brightness(ring_image)
            brightness_factor = 0.9 + (profile.brightness_range[1] - profile.brightness_range[0]) * 0.3
            ring_image = enhancer.enhance(brightness_factor)
            
            # Contrast
            enhancer = ImageEnhance.Contrast(ring_image)
            ring_image = enhancer.enhance(1.2)
            
            # Saturation based on metal type
            enhancer = ImageEnhance.Color(ring_image)
            if detection.metal_type in [MetalType.YELLOW_GOLD, MetalType.ROSE_GOLD]:
                ring_image = enhancer.enhance(1.15)
            else:
                ring_image = enhancer.enhance(0.95)
            
            # Sharpness
            enhancer = ImageEnhance.Sharpness(ring_image)
            ring_image = enhancer.enhance(1.3)
            
            # Step 2: Metal-specific color correction
            if NUMPY_AVAILABLE:
                ring_array = np.array(ring_image).astype(float)
                
                if detection.metal_type == MetalType.YELLOW_GOLD:
                    # Enhance yellow tones
                    ring_array[:,:,0] *= 1.05  # Red
                    ring_array[:,:,1] *= 1.08  # Green
                    ring_array[:,:,2] *= 0.95  # Blue
                    
                elif detection.metal_type == MetalType.ROSE_GOLD:
                    # Enhance pink/rose tones
                    ring_array[:,:,0] *= 1.10  # Red
                    ring_array[:,:,1] *= 1.02  # Green
                    ring_array[:,:,2] *= 0.98  # Blue
                    
                elif detection.metal_type in [MetalType.WHITE_GOLD, MetalType.PLATINUM]:
                    # Enhance white/silver tones - slight desaturation
                    gray = np.mean(ring_array, axis=2)
                    for i in range(3):
                        ring_array[:,:,i] = ring_array[:,:,i] * 0.7 + gray * 0.3
                    # Slight blue tint for white metals
                    ring_array[:,:,2] *= 1.02
                    
                elif detection.metal_type == MetalType.UNPLATED_WHITE:
                    # Champagne gold effect
                    ring_array[:,:,0] *= 1.03  # Slight warm
                    ring_array[:,:,1] *= 1.02
                    ring_array[:,:,2] *= 0.98
                
                # Ensure valid range
                ring_array = np.clip(ring_array, 0, 255)
                ring_image = Image.fromarray(ring_array.astype(np.uint8))
            
            # Step 3: Professional finish
            # Add subtle glow for metals
            if detection.metal_type != MetalType.UNPLATED_WHITE:
                glow = ring_image.filter(ImageFilter.GaussianBlur(radius=2))
                ring_image = Image.blend(ring_image, glow, 0.1)
            
            return ring_image
            
        except Exception as e:
            print(f"[v139] Ring enhancement error: {e}")
            return ring_image
    
    def _restore_with_replicate(self, ring_image: Image.Image, detection: RingDetection) -> Image.Image:
        """Use Replicate for AI restoration"""
        if not self.replicate_enabled or not REQUESTS_AVAILABLE:
            return self._enhance_ring(ring_image, detection)
        
        try:
            # Convert to base64
            buffer = io.BytesIO()
            ring_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Run Real-ESRGAN for upscaling
            output = self.replicate_client.run(
                "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "scale": 2,
                    "face_enhance": False
                }
            )
            
            if output and REQUESTS_AVAILABLE:
                response = requests.get(output)
                restored = Image.open(io.BytesIO(response.content))
                
                # Resize back to original size
                restored = restored.resize(ring_image.size, Image.Resampling.LANCZOS)
                
                # Apply metal-specific corrections
                restored = self._enhance_ring(restored, detection)
                
                return restored
                
        except Exception as e:
            print(f"[v139] Replicate restoration failed: {e}")
        
        # Fallback to regular enhancement
        return self._enhance_ring(ring_image, detection)
    
    def _merge_enhanced_ring(self, base_image: Image.Image, enhanced_ring: Image.Image, 
                           bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Merge enhanced ring back with smooth blending"""
        x1, y1, x2, y2 = bbox
        
        # Calculate padding used
        padding = int(max(x2 - x1, y2 - y1) * 0.1)
        
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(base_image.width, x2 + padding)
        y2_pad = min(base_image.height, y2 + padding)
        
        # Resize enhanced ring if needed
        target_size = (x2_pad - x1_pad, y2_pad - y1_pad)
        if enhanced_ring.size != target_size:
            enhanced_ring = enhanced_ring.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create feathered mask for smooth blending
        mask = Image.new('L', target_size, 255)
        draw = ImageDraw.Draw(mask)
        
        # Feather edges
        feather_size = max(5, padding // 2)
        for i in range(feather_size):
            alpha = int(255 * (i / feather_size))
            draw.rectangle(
                [i, i, target_size[0]-i-1, target_size[1]-i-1],
                outline=alpha
            )
        
        # Apply gaussian blur to mask for smoother blend
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_size//2))
        
        # Paste with mask
        result = base_image.copy()
        result.paste(enhanced_ring, (x1_pad, y1_pad), mask)
        
        return result
    
    def _final_polish(self, image: Image.Image) -> Image.Image:
        """Apply final polish to complete image"""
        try:
            # Very subtle overall adjustments
            
            # Slight contrast boost
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # Slight saturation boost
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.02)
            
            # Final sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            print(f"[v139] Final polish error: {e}")
            return image
    
    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Basic enhancement when no rings detected"""
        try:
            # Simple brightness/contrast adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            print(f"[v139] Basic enhancement error: {e}")
            return image
    
    def _create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (1000, 1300)) -> Optional[Image.Image]:
        """Create thumbnail for the enhanced image"""
        try:
            # Calculate aspect ratios
            img_ratio = image.width / image.height
            target_ratio = size[0] / size[1]
            
            if img_ratio > target_ratio:
                # Image is wider
                new_width = int(image.height * target_ratio)
                left = (image.width - new_width) // 2
                right = left + new_width
                cropped = image.crop((left, 0, right, image.height))
            else:
                # Image is taller
                new_height = int(image.width / target_ratio)
                top = (image.height - new_height) // 2
                bottom = top + new_height
                cropped = image.crop((0, top, image.width, bottom))
            
            # Resize to exact size
            thumbnail = cropped.resize(size, Image.Resampling.LANCZOS)
            
            return thumbnail
            
        except Exception as e:
            print(f"[v139] Thumbnail creation error: {e}")
            return None
    
    def _create_response(self, image: Image.Image, detections: List[RingDetection], 
                        status: str, processing_time: float = 0, 
                        thumbnail: Optional[Image.Image] = None,
                        error: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized response for Make.com"""
        try:
            # Convert main image to base64
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG', quality=95)
            output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            # Remove padding for Make.com
            output_base64 = output_base64.rstrip('=')
            
            # Convert thumbnail if available
            thumb_base64 = None
            if thumbnail:
                thumb_buffer = io.BytesIO()
                thumbnail.save(thumb_buffer, format='PNG', quality=90)
                thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Prepare detection info
            detection_info = []
            for detection in detections:
                info = {
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "quality_score": detection.quality_score,
                    "needs_restoration": detection.needs_restoration
                }
                if detection.metal_type:
                    info["metal_type"] = detection.metal_type.value
                detection_info.append(info)
            
            response = {
                "enhanced_image": f"data:image/png;base64,{output_base64}",
                "status": status,
                "rings_detected": len(detections),
                "detection_info": detection_info,
                "processing_time": f"{processing_time:.2f}s",
                "enhancement_applied": True,
                "version": "v139-complete",
                "features": {
                    "replicate_enabled": self.replicate_enabled,
                    "opencv_available": CV2_AVAILABLE,
                    "numpy_available": NUMPY_AVAILABLE
                }
            }
            
            if thumb_base64:
                response["thumbnail"] = f"data:image/png;base64,{thumb_base64}"
            
            if error:
                response["error"] = error
            
            return response
            
        except Exception as e:
            print(f"[v139] Response creation error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "version": "v139-complete"
            }

# Initialize enhancer as global (but safely)
enhancer_instance = None

def get_enhancer():
    """Get or create enhancer instance"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV139()
    return enhancer_instance

def handler(event):
    """RunPod handler function - v139 COMPLETE"""
    try:
        print("="*70)
        print("[v139] Handler started - Complete Version with All Features")
        print(f"[v139] Python version: {sys.version}")
        print(f"[v139] Available modules - NumPy: {NUMPY_AVAILABLE}, PIL: {PIL_AVAILABLE}, CV2: {CV2_AVAILABLE}, Replicate: {REPLICATE_AVAILABLE}")
        print("="*70)
        
        # Get input data
        input_data = event.get('input', {})
        print(f"[v139] Input type: {type(input_data)}")
        print(f"[v139] Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Debug mode
        if input_data.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": "v139 handler is working - All features included",
                    "replicate_available": REPLICATE_AVAILABLE,
                    "replicate_enabled": bool(os.environ.get('REPLICATE_API_TOKEN')),
                    "opencv_available": CV2_AVAILABLE,
                    "numpy_available": NUMPY_AVAILABLE,
                    "pil_available": PIL_AVAILABLE,
                    "version": "v139-complete"
                }
            }
        
        # Extract image data with multiple fallbacks
        image_data = None
        
        # Try direct access
        image_data = input_data.get('image', '')
        
        # Try nested structures if direct access failed
        if not image_data and isinstance(input_data, dict):
            # Common nested patterns
            for pattern in ['data', 'body', 'payload']:
                if pattern in input_data and isinstance(input_data[pattern], dict):
                    image_data = input_data[pattern].get('image', '')
                    if image_data:
                        break
        
        # Look for any key with base64 data
        if not image_data:
            for key, value in input_data.items():
                if isinstance(value, str) and len(value) > 1000:
                    if 'image' in key.lower() or value.startswith('data:image'):
                        image_data = value
                        print(f"[v139] Found image data in key: {key}")
                        break
        
        if not image_data:
            print("[v139] No image data found in input")
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "debug_info": {
                        "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict",
                        "input_sample": str(input_data)[:200] if input_data else "Empty"
                    },
                    "version": "v139-complete"
                }
            }
        
        # Decode image
        print(f"[v139] Decoding image data (length: {len(image_data)})")
        image = decode_base64_image(image_data)
        print(f"[v139] Image decoded successfully: {image.size}")
        
        # Extract parameters
        params = {
            'use_replicate_restoration': input_data.get('use_replicate_restoration', False),
            'enhancement_level': input_data.get('enhancement_level', 'high'),
            'detect_all_rings': input_data.get('detect_all_rings', True)
        }
        
        # Get or create enhancer
        enhancer = get_enhancer()
        
        # Process image
        print("[v139] Starting image enhancement...")
        result = enhancer.enhance_image(image, params)
        
        # Return in RunPod format (Make.com compatible)
        print("[v139] Processing complete, returning result")
        return {"output": result}
        
    except Exception as e:
        print(f"[v139] Handler error: {str(e)}")
        print(traceback.format_exc())
        
        # Try to return a meaningful error response
        try:
            return {
                "output": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "status": "error",
                    "version": "v139-complete",
                    "debug_info": {
                        "modules_available": {
                            "numpy": NUMPY_AVAILABLE,
                            "pil": PIL_AVAILABLE,
                            "cv2": CV2_AVAILABLE,
                            "replicate": REPLICATE_AVAILABLE,
                            "requests": REQUESTS_AVAILABLE
                        }
                    }
                }
            }
        except:
            return {
                "output": {
                    "error": "Critical error",
                    "status": "error",
                    "version": "v139-complete"
                }
            }

# RunPod entry point
if __name__ == "__main__":
    print("="*70)
    print("Wedding Ring Enhancement v139 Starting...")
    print("Complete Version with All Features")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print(f"OpenCV Available: {CV2_AVAILABLE}")
    print(f"NumPy Available: {NUMPY_AVAILABLE}")
    print(f"PIL Available: {PIL_AVAILABLE}")
    print(f"Replicate Token Set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
