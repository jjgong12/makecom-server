import runpod
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import base64
import io
import os
import logging
import traceback
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check Replicate availability
try:
    import replicate
    REPLICATE_AVAILABLE = True
    logger.info("Replicate module imported successfully")
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate module not available - will use fallback methods")

# Metal types
class MetalType(Enum):
    YELLOW_GOLD = "yellow_gold"
    ROSE_GOLD = "rose_gold"
    WHITE_GOLD = "white_gold"
    PLATINUM = "platinum"

# Ring detection dataclass
@dataclass
class RingDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    metal_type: Optional[MetalType] = None
    quality_score: float = 0.0
    needs_restoration: bool = False

# Color profiles
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
    )
}

def decode_base64_image(base64_string):
    """Decode base64 image with Make.com compatibility"""
    try:
        # Log the input
        logger.info(f"[v138] Base64 string length: {len(base64_string) if base64_string else 0}")
        
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
            logger.info("[v138] Method 1 (direct) successful")
        except:
            pass
        
        # Method 2: Add padding
        if decoded is None:
            try:
                padding = 4 - (len(base64_string) % 4)
                if padding != 4:
                    base64_string += '=' * padding
                decoded = base64.b64decode(base64_string)
                logger.info("[v138] Method 2 (with padding) successful")
            except:
                pass
        
        # Method 3: URL-safe decode
        if decoded is None:
            try:
                decoded = base64.urlsafe_b64decode(base64_string)
                logger.info("[v138] Method 3 (URL-safe) successful")
            except:
                pass
        
        # Method 4: Force decode
        if decoded is None:
            try:
                # Remove non-base64 characters
                import re
                clean_string = re.sub(r'[^A-Za-z0-9+/]', '', base64_string)
                padding = 4 - (len(clean_string) % 4)
                if padding != 4:
                    clean_string += '=' * padding
                decoded = base64.b64decode(clean_string)
                logger.info("[v138] Method 4 (force clean) successful")
            except:
                pass
        
        if decoded is None:
            raise ValueError("Failed to decode base64 string with all methods")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(decoded))
        logger.info(f"[v138] Successfully decoded image: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"[v138] Error decoding base64: {str(e)}")
        raise

class HybridWeddingRingEnhancer:
    def __init__(self):
        self.replicate_client = None
        if REPLICATE_AVAILABLE and os.environ.get('REPLICATE_API_TOKEN'):
            try:
                self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
                logger.info("[v138] Replicate client initialized")
            except Exception as e:
                logger.error(f"[v138] Failed to initialize Replicate: {e}")
    
    def enhance_image(self, image: Image.Image, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main enhancement pipeline"""
        try:
            # Detect rings
            detections = self._detect_rings(image, params.get('detect_all_rings', True))
            
            if not detections:
                logger.warning("[v138] No rings detected, applying basic enhancement")
                enhanced = self._basic_enhancement(image)
                return self._create_response(enhanced, [], "no_rings_detected")
            
            # Process each detected ring
            enhanced_image = image.copy()
            for detection in detections:
                # Determine if restoration needed
                if detection.needs_restoration and params.get('use_replicate_restoration', True):
                    enhanced_ring = self._restore_with_replicate(image, detection)
                else:
                    enhanced_ring = self._process_ring(image, detection)
                
                # Merge back
                enhanced_image = self._merge_enhanced_ring(enhanced_image, enhanced_ring, detection.bbox)
            
            # Final polish
            enhanced_image = self._final_polish(enhanced_image)
            
            return self._create_response(enhanced_image, detections, "success")
            
        except Exception as e:
            logger.error(f"[v138] Enhancement error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return original with error
            return self._create_response(image, [], "error", str(e))
    
    def _detect_rings(self, image: Image.Image, detect_all: bool = True) -> List[RingDetection]:
        """Detect wedding rings in image"""
        detections = []
        
        # Simple detection based on metallic regions
        img_array = np.array(image)
        
        # Convert to grayscale for edge detection
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
                # Create bounding box
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(image.width, x + r)
                y2 = min(image.height, y + r)
                
                # Check if it's likely a ring
                roi = img_array[y1:y2, x1:x2]
                if self._is_likely_ring(roi):
                    detection = RingDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=0.8,
                        metal_type=self._detect_metal_type(image, (x1, y1, x2, y2)),
                        quality_score=self._assess_quality(roi),
                        needs_restoration=self._needs_restoration(roi)
                    )
                    detections.append(detection)
                    
                    if not detect_all:
                        break
        
        # If no circles found, try simpler approach
        if not detections:
            # Center crop assumption
            w, h = image.size
            size = min(w, h) // 2
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size
            
            roi = img_array[y1:y2, x1:x2]
            detection = RingDetection(
                bbox=(x1, y1, x2, y2),
                confidence=0.5,
                metal_type=self._detect_metal_type(image, (x1, y1, x2, y2)),
                quality_score=self._assess_quality(roi),
                needs_restoration=self._needs_restoration(roi)
            )
            detections.append(detection)
        
        return detections
    
    def _is_likely_ring(self, roi: np.ndarray) -> bool:
        """Check if region likely contains a ring"""
        # Simple heuristics
        # Check for metallic colors
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        
        # Check if colors are in metallic range
        is_metallic = (
            (avg_color[0] > 150) or  # High red (gold)
            (np.std(avg_color) < 30)  # Low variation (silver/white)
        )
        
        return is_metallic
    
    def _assess_quality(self, roi: np.ndarray) -> float:
        """Assess image quality of ring region"""
        # Simple quality metrics
        sharpness = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        brightness = np.mean(roi)
        contrast = np.std(roi)
        
        # Normalize scores
        sharpness_score = min(sharpness / 1000, 1.0)
        brightness_score = min(brightness / 200, 1.0)
        contrast_score = min(contrast / 50, 1.0)
        
        return (sharpness_score + brightness_score + contrast_score) / 3
    
    def _needs_restoration(self, roi: np.ndarray) -> bool:
        """Determine if ring needs AI restoration"""
        quality_score = self._assess_quality(roi)
        return quality_score < 0.5
    
    def _restore_with_replicate(self, image: Image.Image, detection: RingDetection) -> Image.Image:
        """Use Replicate API for restoration"""
        if not self.replicate_client:
            return self._process_ring(image, detection)
        
        try:
            # Crop ring area
            ring_crop = image.crop(detection.bbox)
            
            # Convert to base64
            buffer = io.BytesIO()
            ring_crop.save(buffer, format='PNG')
            input_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Run GFPGAN
            output = self.replicate_client.run(
                "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
                input={
                    "img": f"data:image/png;base64,{input_base64}",
                    "scale": 2,
                    "version": "v1.4"
                }
            )
            
            # Get result
            if output:
                response = requests.get(output)
                restored = Image.open(io.BytesIO(response.content))
                
                # Apply metal-specific corrections
                restored = self._correct_metal_color(restored, detection.metal_type)
                restored = self._apply_professional_finish(restored, detection.metal_type)
                
                return restored
                
        except Exception as e:
            logger.error(f"[v138] Replicate restoration failed: {e}")
        
        # Fallback to regular processing
        return self._process_ring(image, detection)
    
    def _process_ring(self, image: Image.Image, detection: RingDetection, 
                     use_replicate_restoration: bool = True) -> Image.Image:
        """Process a single detected ring"""
        # Crop ring area with padding
        x1, y1, x2, y2 = detection.bbox
        padding = int((x2 - x1) * 0.1)
        
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image.width, x2 + padding)
        y2_pad = min(image.height, y2 + padding)
        
        ring_crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
        
        # Apply v136-style enhancements
        ring_crop = self._correct_metal_color(ring_crop, detection.metal_type)
        ring_crop = self._enhance_details(ring_crop, detection.metal_type)
        ring_crop = self._apply_professional_finish(ring_crop, detection.metal_type)
        
        return ring_crop
    
    def _detect_metal_type(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> MetalType:
        """Detect metal type from ring region"""
        # Crop ring area
        ring_crop = image.crop(bbox)
        img_array = np.array(ring_crop)
        
        # Sample center region
        h, w = img_array.shape[:2]
        center_region = img_array[h//3:2*h//3, w//3:2*w//3]
        
        # Calculate average color
        avg_color = np.mean(center_region.reshape(-1, 3), axis=0)
        
        # Convert to HSV for analysis
        avg_hsv = cv2.cvtColor(np.array([[avg_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0,0]
        
        hue = avg_hsv[0]
        saturation = avg_hsv[1] / 255.0
        value = avg_hsv[2] / 255.0
        
        # Enhanced decision logic with more precise thresholds
        if saturation < 0.1:  # Low saturation = white metals
            if value > 0.9:
                return MetalType.PLATINUM
            else:
                return MetalType.WHITE_GOLD
        elif 20 < hue < 40:  # Yellow/gold hue range
            return MetalType.YELLOW_GOLD
        elif (0 <= hue < 20) or (340 < hue <= 360):  # Red/pink hue range
            return MetalType.ROSE_GOLD
        else:
            # Additional analysis using RGB ratios
            r, g, b = avg_color
            
            if r > g > b and (r - b) > 30:
                return MetalType.ROSE_GOLD
            elif g > b and (g - b) > 20:
                return MetalType.YELLOW_GOLD
            else:
                return MetalType.WHITE_GOLD
    
    def _correct_metal_color(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Apply metal-specific color correction"""
        img_array = np.array(image).astype(float)
        profile = METAL_PROFILES[metal_type]
        
        # Simple color adjustment based on metal type
        if metal_type == MetalType.YELLOW_GOLD:
            # Enhance yellow tones
            img_array[:,:,0] *= 1.05  # Red
            img_array[:,:,1] *= 1.08  # Green
            img_array[:,:,2] *= 0.95  # Blue
        elif metal_type == MetalType.ROSE_GOLD:
            # Enhance pink/rose tones
            img_array[:,:,0] *= 1.1   # Red
            img_array[:,:,1] *= 1.02  # Green
            img_array[:,:,2] *= 0.98  # Blue
        elif metal_type in [MetalType.WHITE_GOLD, MetalType.PLATINUM]:
            # Enhance white/silver tones
            # Slight desaturation
            gray = np.mean(img_array, axis=2)
            for i in range(3):
                img_array[:,:,i] = img_array[:,:,i] * 0.7 + gray * 0.3
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _enhance_details(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Enhance ring details"""
        # Simple sharpening
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        # Slight contrast boost
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def _apply_professional_finish(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Apply professional jewelry photography finish"""
        profile = METAL_PROFILES[metal_type]
        
        # Brightness adjustment based on metal type
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = 0.95 + (profile.brightness_range[1] - 0.8) * 0.2
        image = enhancer.enhance(brightness_factor)
        
        return image
    
    def _merge_enhanced_ring(self, base_image: Image.Image, enhanced_ring: Image.Image, 
                           bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Merge enhanced ring back to original image with smooth blending"""
        x1, y1, x2, y2 = bbox
        
        # Add padding used during processing
        padding = int((x2 - x1) * 0.1)
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(base_image.width, x2 + padding)
        y2_pad = min(base_image.height, y2 + padding)
        
        # Resize enhanced ring if needed
        target_size = (x2_pad - x1_pad, y2_pad - y1_pad)
        if enhanced_ring.size != target_size:
            enhanced_ring = enhanced_ring.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create smooth blend mask
        blend_mask = Image.new('L', target_size, 255)
        draw = ImageDraw.Draw(blend_mask)
        
        # Create feathered edges
        feather_size = padding // 2
        for i in range(feather_size):
            alpha = int(255 * (i / feather_size))
            draw.rectangle(
                [i, i, target_size[0]-i-1, target_size[1]-i-1],
                outline=alpha
            )
        
        # Paste with blend mask
        base_copy = base_image.copy()
        base_copy.paste(enhanced_ring, (x1_pad, y1_pad), blend_mask)
        
        return base_copy
    
    def _final_polish(self, image: Image.Image) -> Image.Image:
        """Apply final polish to complete image"""
        # Subtle overall enhancements
        
        # 1. Slight contrast boost
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # 2. Slight saturation boost
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.02)
        
        # 3. Final sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Basic enhancement when no rings detected"""
        # Simple brightness/contrast adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def _create_response(self, image: Image.Image, detections: List[RingDetection], 
                        status: str, error: str = None) -> Dict[str, Any]:
        """Create standardized response"""
        # Convert image to base64
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG', quality=95)
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        output_base64 = output_base64.rstrip('=')
        
        # Create thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((300, 300), Image.Resampling.LANCZOS)
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
            "thumbnail": f"data:image/png;base64,{thumb_base64}",
            "status": status,
            "rings_detected": len(detections),
            "detection_info": detection_info,
            "enhancement_applied": True,
            "version": "v138-make-fixed"
        }
        
        if error:
            response["error"] = error
        
        return response

def handler(event):
    """RunPod handler function - FIXED FOR MAKE.COM"""
    try:
        logger.info("="*50)
        logger.info("[v138] Handler started - Make.com Fixed Version")
        
        # Log event structure for debugging
        logger.info(f"[v138] Event type: {type(event)}")
        logger.info(f"[v138] Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
        
        # Get input data
        input_data = event.get('input', {})
        logger.info(f"[v138] Input type: {type(input_data)}")
        logger.info(f"[v138] Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Debug mode
        if input_data.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": "v138 handler is working",
                    "replicate_available": REPLICATE_AVAILABLE,
                    "replicate_enabled": bool(os.environ.get('REPLICATE_API_TOKEN')),
                    "version": "v138-make-fixed"
                }
            }
        
        # Try multiple ways to get image data
        image_data = input_data.get('image', '')
        
        # Check for nested structures
        if not image_data and isinstance(input_data, dict):
            # Try common nested patterns
            if 'data' in input_data:
                image_data = input_data['data'].get('image', '')
            elif 'body' in input_data:
                image_data = input_data['body'].get('image', '')
            elif 'payload' in input_data:
                image_data = input_data['payload'].get('image', '')
        
        # Check if image_data is in a different key
        if not image_data:
            # Look for any key containing base64 data
            for key, value in input_data.items():
                if isinstance(value, str) and len(value) > 1000:
                    if 'image' in key.lower() or value.startswith('data:image'):
                        image_data = value
                        break
        
        if not image_data:
            logger.error("[v138] No image data found in input")
            # Log all available data for debugging
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    if isinstance(value, str):
                        logger.info(f"[v138] Key '{key}' sample: {value[:50]}...")
                    else:
                        logger.info(f"[v138] Key '{key}' type: {type(value)}")
            
            return {
                "output": {
                    "error": "No image data provided",
                    "status": "error",
                    "debug_info": {
                        "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict",
                        "input_type": str(type(input_data))
                    },
                    "version": "v138-make-fixed"
                }
            }
        
        # Decode image
        logger.info(f"[v138] Attempting to decode image data (length: {len(image_data)})")
        image = decode_base64_image(image_data)
        logger.info(f"[v138] Image decoded successfully: {image.size}")
        
        # Extract parameters
        params = {
            'use_replicate_restoration': input_data.get('use_replicate_restoration', False),
            'enhancement_level': input_data.get('enhancement_level', 'high'),
            'detect_all_rings': input_data.get('detect_all_rings', True)
        }
        
        # Initialize enhancer
        enhancer = HybridWeddingRingEnhancer()
        
        # Process image
        result = enhancer.enhance_image(image, params)
        
        # Return in RunPod format
        logger.info("[v138] Processing complete, returning result")
        return {"output": result}
        
    except Exception as e:
        logger.error(f"[v138] Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "version": "v138-make-fixed"
            }
        }

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Wedding Ring Enhancement v138 Starting...")
    logger.info("Make.com Fixed Version")
    logger.info(f"Replicate Available: {REPLICATE_AVAILABLE}")
    logger.info(f"Replicate Token Set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    logger.info("="*50)
    runpod.serverless.start({"handler": handler})
