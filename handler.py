import runpod
import os
import base64
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import io
import cv2
import json
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional import for Replicate
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate package not installed, some features will be disabled")

class MetalType(Enum):
    YELLOW_GOLD = "yellow_gold"
    ROSE_GOLD = "rose_gold"
    WHITE_GOLD = "white_gold"
    PLATINUM = "platinum"

@dataclass
class ColorProfile:
    base_rgb: Tuple[int, int, int]
    highlight_rgb: Tuple[int, int, int]
    shadow_rgb: Tuple[int, int, int]
    saturation_range: Tuple[float, float]
    brightness_range: Tuple[float, float]
    temperature: float
    shine_factor: float
    reflection_intensity: float

@dataclass
class RingDetection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    metal_type: Optional[MetalType] = None
    needs_restoration: bool = False
    quality_score: float = 0.0

# Enhanced metal color profiles based on 38 training samples
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
        base_rgb=(234, 179, 162),
        highlight_rgb=(255, 220, 210),
        shadow_rgb=(183, 110, 95),
        saturation_range=(0.20, 0.40),
        brightness_range=(0.65, 0.90),
        temperature=0.08,
        shine_factor=0.80,
        reflection_intensity=0.85
    ),
    MetalType.WHITE_GOLD: ColorProfile(
        base_rgb=(235, 235, 235),
        highlight_rgb=(255, 255, 255),
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
                decoded = base64.b64decode(base64_string + '==')
                logger.info("[v138] Method 4 (force padding) successful")
            except:
                pass
        
        if decoded is None:
            raise ValueError("All base64 decoding methods failed")
        
        # Convert to image
        image = Image.open(io.BytesIO(decoded))
        logger.info(f"[v138] Image decoded successfully: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"[v138] Failed to decode base64: {str(e)}")
        raise

class ReplicateRingDetector:
    """Use Replicate's Grounding DINO for accurate ring detection"""
    
    def __init__(self):
        self.replicate_enabled = False
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        
        if REPLICATE_AVAILABLE and api_token:
            try:
                self.client = replicate.Client(api_token=api_token)
                self.replicate_enabled = True
                logger.info("Replicate API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Replicate: {e}")
        else:
            if not REPLICATE_AVAILABLE:
                logger.warning("Replicate package not installed")
            else:
                logger.warning("REPLICATE_API_TOKEN not found")
            logger.info("Using fallback detection only")
            
        self.ring_queries = [
            "wedding ring",
            "engagement ring", 
            "gold ring",
            "silver ring",
            "diamond ring",
            "wedding band",
            "metal ring",
            "jewelry ring"
        ]
        
    def detect_rings(self, image: Image.Image) -> List[RingDetection]:
        """Detect all rings in the image using Grounding DINO"""
        try:
            # If Replicate not enabled, use fallback immediately
            if not self.replicate_enabled:
                logger.info("Using fallback detection (Replicate not available)")
                return self._fallback_detection(image)
                
            # Save image temporarily
            temp_buffer = io.BytesIO()
            image.save(temp_buffer, format='PNG')
            temp_buffer.seek(0)
            
            # Run Grounding DINO with multiple queries
            detections = []
            
            for query in self.ring_queries[:3]:  # Use top 3 queries to avoid too many calls
                try:
                    output = self.client.run(
                        "adirik/grounding-dino:0e3b5b56379f48fab7cc7518c1ae622cae8e0831aaae8c9d75fc5e7140abb67c",
                        input={
                            "image": temp_buffer,
                            "query": query,
                            "box_threshold": 0.3,
                            "text_threshold": 0.25
                        }
                    )
                    
                    if output and 'boxes' in output:
                        for box in output['boxes']:
                            detection = RingDetection(
                                bbox=tuple(map(int, box)),
                                confidence=output.get('scores', [0.5])[0]
                            )
                            detections.append(detection)
                            
                except Exception as e:
                    logger.warning(f"Grounding DINO query '{query}' failed: {e}")
                    
            # Remove duplicate detections
            detections = self._remove_duplicates(detections)
            
            # If no detections with Replicate, fall back to local detection
            if not detections:
                logger.info("No rings detected with Replicate, using fallback detection")
                detections = self._fallback_detection(image)
                
            return detections
            
        except Exception as e:
            logger.error(f"Ring detection failed: {e}")
            return self._fallback_detection(image)
    
    def _remove_duplicates(self, detections: List[RingDetection]) -> List[RingDetection]:
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections
            
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        kept = []
        for detection in detections:
            # Check if this overlaps with any kept detection
            is_duplicate = False
            for kept_detection in kept:
                iou = self._calculate_iou(detection.bbox, kept_detection.bbox)
                if iou > 0.5:  # More than 50% overlap
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                kept.append(detection)
                
        return kept
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _fallback_detection(self, image: Image.Image) -> List[RingDetection]:
        """Fallback detection using traditional CV methods"""
        img_array = np.array(image)
        
        # Simple approach: assume ring is in center
        h, w = img_array.shape[:2]
        
        # Create a detection for center region
        center_bbox = (
            w // 4,  # x1
            h // 4,  # y1
            3 * w // 4,  # x2
            3 * h // 4   # y2
        )
        
        return [RingDetection(bbox=center_bbox, confidence=0.8)]

class QualityAssessment:
    """Assess ring image quality and determine needed enhancements"""
    
    @staticmethod
    def assess_ring_quality(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[float, bool]:
        """
        Returns quality score (0-1) and whether restoration is needed
        """
        # Crop ring area
        ring_crop = image.crop(bbox)
        ring_array = np.array(ring_crop)
        
        # Calculate quality metrics
        scores = []
        
        # 1. Sharpness score
        gray = cv2.cvtColor(ring_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000, 1.0)
        scores.append(sharpness_score)
        
        # 2. Contrast score
        contrast_score = gray.std() / 128.0
        scores.append(contrast_score)
        
        # 3. Brightness score
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Best at 0.5
        scores.append(brightness_score)
        
        # Overall quality
        quality_score = np.mean(scores)
        
        # Determine if restoration needed
        needs_restoration = (
            quality_score < 0.6 or
            sharpness_score < 0.4
        )
        
        return quality_score, needs_restoration

class HybridWeddingRingEnhancer:
    """Hybrid approach combining Replicate detection with v136 enhancement"""
    
    def __init__(self):
        self.ring_detector = ReplicateRingDetector()
        
        # Initialize Replicate client if available
        self.replicate_enabled = False
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        if REPLICATE_AVAILABLE and api_token:
            try:
                self.replicate_client = replicate.Client(api_token=api_token)
                self.replicate_enabled = True
                logger.info("Replicate client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Replicate client: {e}")
        else:
            if not REPLICATE_AVAILABLE:
                logger.warning("Replicate package not installed")
            else:
                logger.warning("REPLICATE_API_TOKEN not found")
            logger.info("Replicate enhancement features disabled")
        
    def enhance_image(self, image: Image.Image, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main enhancement pipeline"""
        try:
            start_time = time.time()
            
            # Step 1: Detect all rings
            logger.info("Detecting rings in image...")
            ring_detections = self.ring_detector.detect_rings(image)
            
            if not ring_detections:
                logger.warning("No rings detected in image")
                return self._create_response(image, [], "No rings detected")
            
            logger.info(f"Detected {len(ring_detections)} rings")
            
            # Step 2: Process each detected ring
            enhanced_rings = []
            final_image = image.copy()
            
            for idx, detection in enumerate(ring_detections):
                logger.info(f"Processing ring {idx+1}/{len(ring_detections)}")
                
                # Assess quality
                quality_score, needs_restoration = QualityAssessment.assess_ring_quality(
                    image, detection.bbox
                )
                detection.quality_score = quality_score
                detection.needs_restoration = needs_restoration
                
                # Detect metal type
                detection.metal_type = self._detect_metal_type(image, detection.bbox)
                logger.info(f"Ring {idx+1}: Metal type={detection.metal_type.value}, Quality={quality_score:.2f}")
                
                # Process ring
                enhanced_ring = self._process_single_ring(
                    final_image, 
                    detection,
                    use_replicate_restoration=params.get('use_replicate_restoration', True)
                )
                
                # Merge back to final image
                final_image = self._merge_enhanced_ring(final_image, enhanced_ring, detection.bbox)
                enhanced_rings.append(detection)
            
            # Step 3: Final polish on complete image
            final_image = self._final_polish(final_image)
            
            # Step 4: Create response
            processing_time = time.time() - start_time
            return self._create_response(final_image, enhanced_rings, "Success", processing_time)
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            logger.error(traceback.format_exc())
            return self._create_response(image, [], str(e))
    
    def _process_single_ring(self, image: Image.Image, detection: RingDetection, 
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
    
    def _create_response(self, image: Image.Image, detections: List[RingDetection], 
                        status: str, processing_time: float = 0) -> Dict[str, Any]:
        """Create standardized response"""
        # Convert image to base64
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG', quality=95)
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        output_base64 = output_base64.rstrip('=')
        
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
        
        return {
            "enhanced_image": f"data:image/png;base64,{output_base64}",
            "status": status,
            "rings_detected": len(detections),
            "detection_info": detection_info,
            "processing_time": f"{processing_time:.2f}s",
            "enhancement_applied": True,
            "version": "v138-make-fixed"
        }

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
                    "version": "v138-make-fixed",
                    "event_keys": list(event.keys()),
                    "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict"
                }
            }
        
        # Try to find image data from multiple possible keys
        image_data = None
        
        # Method 1: Direct common keys
        possible_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img', 'data', 'imageData', 'image_data']
        
        for key in possible_keys:
            if key in input_data:
                value = input_data[key]
                if isinstance(value, str) and len(value) > 100:
                    image_data = value
                    logger.info(f"[v138] Found image data in key: {key}")
                    break
        
        # Method 2: Check nested data structure
        if not image_data and 'data' in input_data:
            data = input_data['data']
            if isinstance(data, dict):
                for key in possible_keys:
                    if key in data:
                        value = data[key]
                        if isinstance(value, str) and len(value) > 100:
                            image_data = value
                            logger.info(f"[v138] Found image data in nested data.{key}")
                            break
        
        # Method 3: If input_data is a string itself (sometimes happens)
        if not image_data and isinstance(input_data, str) and len(input_data) > 100:
            image_data = input_data
            logger.info("[v138] Input data is string itself, using as image data")
        
        # Method 4: Find any large string that might be base64
        if not image_data:
            for key, value in input_data.items():
                if isinstance(value, str) and len(value) > 1000:
                    # Check if it looks like base64
                    if any(c in value for c in ['/', '+', '=']) or value.startswith('data:'):
                        image_data = value
                        logger.info(f"[v138] Found potential base64 data in key: {key}")
                        break
        
        if not image_data:
            logger.error("[v138] No image data found in any expected location")
            logger.error(f"[v138] Available keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
            
            # Log sample of each key's value for debugging
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
