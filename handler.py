import runpod
import os
import base64
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import io
import cv2
import json
import torch
from torchvision import transforms
from scipy.ndimage import binary_dilation, binary_erosion
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import traceback
from dataclasses import dataclass
from enum import Enum
import colorsys
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import replicate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ReplicateRingDetector:
    """Use Replicate's Grounding DINO for accurate ring detection"""
    
    def __init__(self):
        self.client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
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
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect circular objects
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = circle
                bbox = (x-r, y-r, x+r, y+r)
                detections.append(RingDetection(bbox=bbox, confidence=0.5))
                
        return detections

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
        
        # 4. Noise estimation
        noise_score = 1.0 - estimate_noise(gray) / 50.0
        scores.append(max(0, noise_score))
        
        # Overall quality
        quality_score = np.mean(scores)
        
        # Determine if restoration needed
        needs_restoration = (
            quality_score < 0.6 or
            sharpness_score < 0.4 or
            noise_score < 0.7
        )
        
        return quality_score, needs_restoration

def estimate_noise(image: np.ndarray) -> float:
    """Estimate noise level in image"""
    h, w = image.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w-2) * (h-2))
    
    return sigma

class HybridWeddingRingEnhancer:
    """Hybrid approach combining Replicate detection with v136 enhancement"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ring_detector = ReplicateRingDetector()
        self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
        self.setup_models()
        
    def setup_models(self):
        """Initialize enhancement models"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        
        # Step 1: Optional Replicate restoration
        if use_replicate_restoration and detection.needs_restoration:
            ring_crop = self._replicate_restore(ring_crop)
        
        # Step 2: Metal-specific color correction
        ring_crop = self._correct_metal_color(ring_crop, detection.metal_type)
        
        # Step 3: Detail enhancement
        ring_crop = self._enhance_details(ring_crop, detection.metal_type)
        
        # Step 4: Professional finish
        ring_crop = self._apply_professional_finish(ring_crop, detection.metal_type)
        
        # Step 5: Optional upscaling
        if detection.quality_score < 0.5 and use_replicate_restoration:
            ring_crop = self._replicate_upscale(ring_crop)
        
        return ring_crop
    
    def _replicate_restore(self, image: Image.Image) -> Image.Image:
        """Use Replicate's GFPGAN for restoration"""
        try:
            # Save to buffer
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Run GFPGAN
            output = self.replicate_client.run(
                "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
                input={
                    "img": buffer,
                    "scale": 2,
                    "version": "v1.4"
                }
            )
            
            # Download result
            response = requests.get(output)
            restored = Image.open(io.BytesIO(response.content))
            
            # Resize back to original size
            restored = restored.resize(image.size, Image.Resampling.LANCZOS)
            
            return restored
            
        except Exception as e:
            logger.warning(f"Replicate restoration failed: {e}")
            return image
    
    def _replicate_upscale(self, image: Image.Image) -> Image.Image:
        """Use Replicate's Real-ESRGAN for upscaling"""
        try:
            # Save to buffer
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Run Real-ESRGAN
            output = self.replicate_client.run(
                "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                input={
                    "image": buffer,
                    "scale": 4,
                    "face_enhance": False
                }
            )
            
            # Download result
            response = requests.get(output)
            upscaled = Image.open(io.BytesIO(response.content))
            
            # Resize to 2x original (not 4x to avoid over-processing)
            target_size = (image.width * 2, image.height * 2)
            upscaled = upscaled.resize(target_size, Image.Resampling.LANCZOS)
            
            return upscaled
            
        except Exception as e:
            logger.warning(f"Replicate upscaling failed: {e}")
            return image
    
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
        
        # Create mask for metallic areas
        metal_mask = self._create_metal_mask(img_array)
        
        if np.any(metal_mask):
            # Get current metal color
            metal_pixels = img_array[metal_mask > 0]
            current_avg = np.mean(metal_pixels, axis=0)
            
            # Calculate color matrix for transformation
            target_color = np.array(profile.base_rgb)
            
            # Create smooth color transformation
            transform_matrix = self._calculate_color_transform(current_avg, target_color)
            
            # Apply transformation to metal areas
            for i in range(3):
                channel = img_array[:,:,i]
                channel_transformed = np.dot(channel.reshape(-1, 1), transform_matrix[i].reshape(1, -1)).reshape(channel.shape)
                img_array[:,:,i] = np.where(metal_mask > 0, channel_transformed, channel)
            
            # Adjust saturation and brightness
            img_hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
            
            # Saturation adjustment
            target_sat = np.mean(profile.saturation_range)
            current_sat = np.mean(img_hsv[metal_mask > 0, 1]) / 255.0
            
            if current_sat > 0:
                sat_scale = target_sat / current_sat
                img_hsv[:,:,1] = np.where(
                    metal_mask > 0,
                    np.clip(img_hsv[:,:,1] * sat_scale, 0, 255),
                    img_hsv[:,:,1]
                )
            
            # Brightness adjustment with preservation of highlights
            target_val = np.mean(profile.brightness_range) * 255
            current_val = np.mean(img_hsv[metal_mask > 0, 2])
            
            if current_val > 0:
                val_scale = target_val / current_val
                # Preserve highlights by using non-linear scaling
                img_hsv[:,:,2] = np.where(
                    metal_mask > 0,
                    np.clip(img_hsv[:,:,2] ** 0.8 * val_scale ** 0.8, 0, 255),
                    img_hsv[:,:,2]
                )
            
            img_array = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _create_metal_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Create mask for metallic regions"""
        # Convert to LAB for better metal detection
        lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Detect high brightness areas (metallic reflections)
        l_channel = lab[:,:,0]
        bright_mask = l_channel > np.percentile(l_channel, 70)
        
        # Detect edges
        edges = cv2.Canny(img_array.astype(np.uint8), 50, 150)
        edge_mask = edges > 0
        
        # Combine masks
        combined = bright_mask | edge_mask
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes
        combined = binary_fill_holes(combined)
        
        return combined.astype(np.uint8)
    
    def _calculate_color_transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate color transformation matrix"""
        # Simple linear transformation with some non-linearity
        scale = target / (source + 1e-5)
        
        # Add some cross-channel influence for more natural color
        transform = np.eye(3)
        for i in range(3):
            transform[i, i] = scale[i]
            # Add slight influence from other channels
            for j in range(3):
                if i != j:
                    transform[i, j] = 0.05 * (scale[j] - 1.0)
        
        return transform
    
    def _enhance_details(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Enhance ring details while preserving metal characteristics"""
        img_array = np.array(image)
        profile = METAL_PROFILES[metal_type]
        
        # Create detail enhancement kernel based on metal type
        if metal_type in [MetalType.WHITE_GOLD, MetalType.PLATINUM]:
            # Sharper enhancement for white metals
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 1.0
        else:
            # Softer enhancement for gold metals
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]]) / 1.0
        
        # Apply sharpening
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend based on metal shine factor
        result = cv2.addWeighted(
            img_array, 
            1 - profile.shine_factor * 0.3, 
            sharpened, 
            profile.shine_factor * 0.3, 
            0
        )
        
        # Enhance edges specifically
        edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
        edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Apply edge enhancement
        for i in range(3):
            result[:,:,i] = np.where(
                edge_mask > 0,
                np.clip(result[:,:,i] * 1.1, 0, 255),
                result[:,:,i]
            )
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _apply_professional_finish(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Apply professional jewelry photography finish"""
        img_array = np.array(image).astype(float)
        profile = METAL_PROFILES[metal_type]
        
        # Create metallic shine effect
        metal_mask = self._create_metal_mask(img_array)
        
        if np.any(metal_mask):
            # Add realistic highlights
            highlight_map = self._create_highlight_map(img_array, metal_mask, profile)
            
            # Apply highlights
            highlight_color = np.array(profile.highlight_rgb)
            for i in range(3):
                img_array[:,:,i] = np.where(
                    highlight_map > 0,
                    img_array[:,:,i] * (1 - highlight_map) + highlight_color[i] * highlight_map,
                    img_array[:,:,i]
                )
            
            # Add subtle shadows for depth
            shadow_map = self._create_shadow_map(img_array, metal_mask)
            shadow_color = np.array(profile.shadow_rgb)
            
            for i in range(3):
                img_array[:,:,i] = np.where(
                    shadow_map > 0,
                    img_array[:,:,i] * (1 - shadow_map * 0.3) + shadow_color[i] * shadow_map * 0.3,
                    img_array[:,:,i]
                )
            
            # Add metallic reflection
            reflection = self._create_metallic_reflection(img_array, metal_mask, profile)
            img_array = cv2.addWeighted(
                img_array, 
                1 - profile.reflection_intensity * 0.2,
                reflection,
                profile.reflection_intensity * 0.2,
                0
            )
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _create_highlight_map(self, img: np.ndarray, mask: np.ndarray, profile: ColorProfile) -> np.ndarray:
        """Create realistic highlight map"""
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Find brightest areas
        _, bright = cv2.threshold(gray, int(255 * profile.brightness_range[1] * 0.9), 255, cv2.THRESH_BINARY)
        bright = bright & mask
        
        # Gaussian blur for soft highlights
        highlight_map = cv2.GaussianBlur(bright.astype(float) / 255, (15, 15), 5.0)
        
        # Adjust intensity based on metal type
        highlight_map *= profile.shine_factor
        
        return highlight_map
    
    def _create_shadow_map(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create shadow map for depth"""
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Find darker areas
        _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark = dark & mask
        
        # Soft shadows
        shadow_map = cv2.GaussianBlur(dark.astype(float) / 255, (11, 11), 3.0)
        
        return shadow_map
    
    def _create_metallic_reflection(self, img: np.ndarray, mask: np.ndarray, profile: ColorProfile) -> np.ndarray:
        """Create metallic reflection effect"""
        h, w = img.shape[:2]
        
        # Create gradient for reflection
        gradient = np.linspace(0, 1, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w))
        
        # Apply to metal areas only
        reflection = img.copy()
        
        for i in range(3):
            reflection[:,:,i] = np.where(
                mask > 0,
                reflection[:,:,i] + gradient * 30 * profile.reflection_intensity,
                reflection[:,:,i]
            )
        
        return reflection
    
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
        
        # 4. Optional subtle vignette
        image = self._add_subtle_vignette(image)
        
        return image
    
    def _add_subtle_vignette(self, image: Image.Image) -> Image.Image:
        """Add subtle vignette effect"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create radial gradient
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (dist / max_dist) * 0.2  # Very subtle
        vignette = np.clip(vignette, 0.8, 1.0)
        
        # Apply to image
        for i in range(3):
            img_array[:,:,i] = img_array[:,:,i] * vignette
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _create_response(self, image: Image.Image, detections: List[RingDetection], 
                        status: str, processing_time: float = 0) -> Dict[str, Any]:
        """Create standardized response"""
        # Convert image to base64
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG', quality=95)
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Remove padding
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
            "version": "v137-hybrid"
        }

def binary_fill_holes(binary_mask):
    """Fill holes in binary mask"""
    from scipy import ndimage
    return ndimage.binary_fill_holes(binary_mask)

def handler(event):
    """RunPod handler function"""
    try:
        input_data = event.get('input', {})
        
        # Extract data
        image_data = input_data.get('image', '')
        params = {
            'use_replicate_restoration': input_data.get('use_replicate_restoration', True),
            'enhancement_level': input_data.get('enhancement_level', 'high'),
            'detect_all_rings': input_data.get('detect_all_rings', True)
        }
        
        if not image_data:
            return {"output": {"error": "No image data provided"}}
        
        # Decode image
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # Add padding if needed
        image_data = image_data.replace(' ', '+')
        
        try:
            image_bytes = base64.b64decode(image_data + '==')
        except:
            try:
                image_bytes = base64.b64decode(image_data)
            except:
                return {"output": {"error": "Invalid base64 image data"}}
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize enhancer
        enhancer = HybridWeddingRingEnhancer()
        
        # Process image
        result = enhancer.enhance_image(image, params)
        
        # Return in RunPod format
        return {"output": result}
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
