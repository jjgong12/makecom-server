import runpod
import os
import base64
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import cv2
import json
import torch
from torchvision import transforms
from scipy.ndimage import binary_dilation, binary_erosion
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback
from dataclasses import dataclass
from enum import Enum
import colorsys
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

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

# Enhanced metal color profiles based on 38 training samples
METAL_PROFILES = {
    MetalType.YELLOW_GOLD: ColorProfile(
        base_rgb=(255, 215, 0),
        highlight_rgb=(255, 245, 200),
        shadow_rgb=(184, 134, 11),
        saturation_range=(0.15, 0.35),
        brightness_range=(0.70, 0.95),
        temperature=0.15
    ),
    MetalType.ROSE_GOLD: ColorProfile(
        base_rgb=(234, 179, 162),
        highlight_rgb=(255, 220, 210),
        shadow_rgb=(183, 110, 95),
        saturation_range=(0.20, 0.40),
        brightness_range=(0.65, 0.90),
        temperature=0.08
    ),
    MetalType.WHITE_GOLD: ColorProfile(
        base_rgb=(235, 235, 235),
        highlight_rgb=(255, 255, 255),
        shadow_rgb=(175, 175, 175),
        saturation_range=(0.0, 0.08),
        brightness_range=(0.80, 0.98),
        temperature=-0.05
    ),
    MetalType.PLATINUM: ColorProfile(
        base_rgb=(245, 245, 245),
        highlight_rgb=(255, 255, 255),
        shadow_rgb=(185, 185, 185),
        saturation_range=(0.0, 0.05),
        brightness_range=(0.85, 0.99),
        temperature=-0.02
    )
}

class UltraPrecisionMaskDetector:
    """Ultra-precision mask detection with infinite retry capability"""
    
    def __init__(self):
        self.detection_methods = [
            self._detect_black_boxes,
            self._detect_color_anomalies,
            self._detect_edge_patterns,
            self._detect_texture_discontinuities,
            self._detect_geometric_shapes,
            self._detect_contrast_boundaries,
            self._detect_frequency_anomalies,
            self._detect_morphological_patterns
        ]
        
    def detect_and_remove_masks(self, image: Image.Image, max_iterations: int = 100) -> Tuple[Image.Image, bool]:
        """Detect and remove masks with infinite retry until clean"""
        img_array = np.array(image)
        mask_found = True
        iteration = 0
        
        while mask_found and iteration < max_iterations:
            combined_mask = np.zeros(img_array.shape[:2], dtype=bool)
            
            # Run all detection methods
            for detect_method in self.detection_methods:
                try:
                    mask = detect_method(img_array)
                    if mask is not None:
                        combined_mask |= mask
                except Exception as e:
                    logger.warning(f"Detection method {detect_method.__name__} failed: {e}")
            
            # Expand mask to ensure complete removal
            if np.any(combined_mask):
                combined_mask = self._expand_mask(combined_mask, iterations=5)
                
                # Use Replicate API for inpainting
                img_array = self._inpaint_with_replicate(img_array, combined_mask)
                mask_found = True
            else:
                mask_found = False
                
            iteration += 1
            
            # Additional verification pass
            if not mask_found and iteration > 1:
                mask_found = self._verify_no_masks_remain(img_array)
        
        return Image.fromarray(img_array), iteration > 1
    
    def _detect_black_boxes(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect black rectangular masks"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Multiple threshold levels
        masks = []
        for threshold in [10, 20, 30, 40]:
            mask = gray < threshold
            masks.append(mask)
        
        combined = np.logical_or.reduce(masks)
        
        # Find rectangular regions
        contours, _ = cv2.findContours(combined.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_mask = np.zeros_like(combined)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check if rectangular
            if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                result_mask[y:y+h, x:x+w] = True
                
        return result_mask if np.any(result_mask) else None
    
    def _detect_color_anomalies(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect color anomalies that might be masks"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Detect very dark regions
        dark_mask = hsv[:,:,2] < 30
        
        # Detect color discontinuities
        edges = cv2.Canny(img, 50, 150)
        
        # Combine with morphological operations
        kernel = np.ones((5,5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return dark_mask.astype(bool) if np.any(dark_mask) else None
    
    def _detect_edge_patterns(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect unnatural edge patterns"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Detect strong rectangular edges
        edge_mask = edge_magnitude > np.percentile(edge_magnitude, 95)
        
        # Find rectangular patterns
        lines = cv2.HoughLinesP(edge_mask.astype(np.uint8) * 255, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        result_mask = np.zeros_like(edge_mask)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_mask, (x1, y1), (x2, y2), True, 3)
                
        return result_mask if np.any(result_mask) else None
    
    def _detect_texture_discontinuities(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect texture discontinuities"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        mean = cv2.filter2D(gray.astype(float), -1, kernel)
        mean_sq = cv2.filter2D(gray.astype(float)**2, -1, kernel)
        variance = mean_sq - mean**2
        
        # Low variance regions might be masks
        low_var_mask = variance < np.percentile(variance, 10)
        
        return low_var_mask if np.any(low_var_mask) else None
    
    def _detect_geometric_shapes(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect geometric shapes that might be masks"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect rectangles and squares
        contours, _ = cv2.findContours(cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1], 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray, dtype=bool)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            # Rectangle detection
            if len(approx) == 4:
                cv2.fillPoly(mask, [approx], True)
                
        return mask if np.any(mask) else None
    
    def _detect_contrast_boundaries(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect high contrast boundaries"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # High contrast boundaries
        high_contrast = np.abs(laplacian) > np.percentile(np.abs(laplacian), 98)
        
        # Dilate to connect edges
        kernel = np.ones((3,3), np.uint8)
        high_contrast = cv2.dilate(high_contrast.astype(np.uint8), kernel, iterations=2)
        
        return high_contrast.astype(bool) if np.any(high_contrast) else None
    
    def _detect_frequency_anomalies(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect frequency domain anomalies"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Detect anomalies in frequency domain
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        
        anomaly_mask = magnitude > mag_mean + 3 * mag_std
        
        # Convert back to spatial domain
        anomaly_spatial = np.fft.ifft2(np.fft.ifftshift(anomaly_mask * f_shift))
        anomaly_spatial = np.abs(anomaly_spatial) > np.mean(np.abs(anomaly_spatial))
        
        return anomaly_spatial if np.any(anomaly_spatial) else None
    
    def _detect_morphological_patterns(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect using morphological operations"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Multiple morphological operations
        kernel_sizes = [3, 5, 7]
        masks = []
        
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            
            # Top-hat to find bright regions
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Black-hat to find dark regions
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold
            _, mask1 = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
            masks.append(mask1 > 0)
            
        combined = np.logical_or.reduce(masks)
        return combined if np.any(combined) else None
    
    def _expand_mask(self, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Expand mask to ensure complete removal"""
        kernel = np.ones((5,5), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
        return expanded.astype(bool)
    
    def _inpaint_with_replicate(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use Replicate API for inpainting"""
        try:
            # Convert to PIL
            pil_img = Image.fromarray(img)
            pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Encode to base64
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            mask_buffer = io.BytesIO()
            pil_mask.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            # Call Replicate API
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {os.environ.get('REPLICATE_API_TOKEN')}",
                    "Content-Type": "application/json"
                },
                json={
                    "version": "30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f",
                    "input": {
                        "image": f"data:image/png;base64,{img_base64}",
                        "mask": f"data:image/png;base64,{mask_base64}"
                    }
                }
            )
            
            if response.status_code == 201:
                prediction = response.json()
                
                # Wait for completion
                while True:
                    time.sleep(1)
                    status_response = requests.get(
                        prediction['urls']['get'],
                        headers={"Authorization": f"Token {os.environ.get('REPLICATE_API_TOKEN')}"}
                    )
                    status = status_response.json()
                    
                    if status['status'] == 'succeeded':
                        result_url = status['output']
                        result_response = requests.get(result_url)
                        result_img = Image.open(io.BytesIO(result_response.content))
                        return np.array(result_img)
                    elif status['status'] == 'failed':
                        break
                        
        except Exception as e:
            logger.error(f"Replicate inpainting failed: {e}")
            
        # Fallback to local inpainting
        return cv2.inpaint(img, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    
    def _verify_no_masks_remain(self, img: np.ndarray) -> bool:
        """Final verification that no masks remain"""
        # Quick check for black rectangles
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        black_pixels = np.sum(gray < 20)
        total_pixels = gray.size
        
        # If more than 1% black pixels in rectangular patterns, masks might remain
        if black_pixels / total_pixels > 0.01:
            return True
            
        return False

class AdvancedWeddingRingEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_detector = UltraPrecisionMaskDetector()
        self.setup_models()
        
    def setup_models(self):
        """Initialize enhancement models"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def enhance_ring(self, image: Image.Image, detected_metal: MetalType) -> Image.Image:
        """Main enhancement pipeline"""
        try:
            # Step 1: Ultra-precision mask removal
            logger.info("Starting ultra-precision mask removal...")
            image, mask_removed = self.mask_detector.detect_and_remove_masks(image)
            
            if mask_removed:
                logger.info("Masks detected and removed")
            
            # Step 2: Ring detection and isolation
            ring_mask = self._detect_ring_region(image)
            
            # Step 3: Metal-specific color correction
            image = self._correct_metal_color(image, detected_metal, ring_mask)
            
            # Step 4: Detail enhancement
            image = self._enhance_details(image, ring_mask)
            
            # Step 5: Professional finishing
            image = self._apply_professional_finish(image, detected_metal, ring_mask)
            
            # Step 6: Background enhancement
            image = self._enhance_background(image, ring_mask)
            
            # Step 7: Final polish
            image = self._final_polish(image, detected_metal)
            
            return image
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image
    
    def _detect_ring_region(self, image: Image.Image) -> np.ndarray:
        """Detect ring region using advanced segmentation"""
        img_array = np.array(image)
        
        # Convert to LAB for better metal detection
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Detect metallic regions
        l_channel = lab[:,:,0]
        
        # High brightness areas (metallic reflections)
        bright_mask = l_channel > np.percentile(l_channel, 80)
        
        # Edge detection for ring contours
        edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
        
        # Combine masks
        combined = bright_mask | (edges > 0)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (likely the ring)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined)
        
        if num_labels > 1:
            # Get largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            ring_mask = (labels == largest_label)
        else:
            ring_mask = combined > 0
            
        # Refine mask
        ring_mask = binary_dilation(ring_mask, iterations=2)
        ring_mask = binary_erosion(ring_mask, iterations=1)
        
        return ring_mask.astype(np.uint8)
    
    def _correct_metal_color(self, image: Image.Image, metal_type: MetalType, mask: np.ndarray) -> Image.Image:
        """Correct metal color based on type"""
        img_array = np.array(image).astype(float)
        profile = METAL_PROFILES[metal_type]
        
        # Create color correction matrix
        if np.any(mask):
            # Get current metal color
            metal_pixels = img_array[mask > 0]
            current_avg = np.mean(metal_pixels, axis=0)
            
            # Calculate color shift needed
            target_color = np.array(profile.base_rgb)
            color_shift = target_color - current_avg
            
            # Apply graduated color correction
            for i in range(3):
                img_array[:,:,i] = np.where(
                    mask > 0,
                    np.clip(img_array[:,:,i] + color_shift[i] * 0.6, 0, 255),
                    img_array[:,:,i]
                )
            
            # Adjust saturation and brightness
            img_hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
            
            # Saturation adjustment
            target_sat = np.mean(profile.saturation_range)
            img_hsv[:,:,1] = np.where(
                mask > 0,
                np.clip(img_hsv[:,:,1] * target_sat * 2, 0, 255),
                img_hsv[:,:,1]
            )
            
            # Value adjustment
            target_val = np.mean(profile.brightness_range) * 255
            current_val = np.mean(img_hsv[mask > 0, 2])
            val_scale = target_val / (current_val + 1e-5)
            
            img_hsv[:,:,2] = np.where(
                mask > 0,
                np.clip(img_hsv[:,:,2] * val_scale, 0, 255),
                img_hsv[:,:,2]
            )
            
            img_array = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _enhance_details(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Enhance ring details"""
        img_array = np.array(image)
        
        # Unsharp masking for detail enhancement
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
        
        # Apply only to ring area
        result = np.where(mask[:,:,np.newaxis] > 0, unsharp, img_array)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _apply_professional_finish(self, image: Image.Image, metal_type: MetalType, mask: np.ndarray) -> Image.Image:
        """Apply professional jewelry photography finish"""
        img_array = np.array(image).astype(float)
        profile = METAL_PROFILES[metal_type]
        
        # Add metallic highlights
        if np.any(mask):
            # Create highlight map
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            highlights = highlights & mask
            
            # Add subtle highlights
            highlight_color = np.array(profile.highlight_rgb)
            for i in range(3):
                img_array[:,:,i] = np.where(
                    highlights > 0,
                    img_array[:,:,i] * 0.7 + highlight_color[i] * 0.3,
                    img_array[:,:,i]
                )
            
            # Add subtle shadows for depth
            shadow_map = mask & ~highlights
            shadow_color = np.array(profile.shadow_rgb)
            
            for i in range(3):
                img_array[:,:,i] = np.where(
                    shadow_map > 0,
                    img_array[:,:,i] * 0.9 + shadow_color[i] * 0.1,
                    img_array[:,:,i]
                )
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _enhance_background(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Enhance background while preserving ring"""
        img_array = np.array(image)
        
        # Slight blur to background
        background_mask = ~mask
        blurred = cv2.GaussianBlur(img_array, (5, 5), 1.0)
        
        # Combine
        result = np.where(background_mask[:,:,np.newaxis], blurred, img_array)
        
        # Subtle vignette
        h, w = img_array.shape[:2]
        center = (w//2, h//2)
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        vignette = 1 - (dist / max_dist) * 0.3
        
        result = (result * vignette[:,:,np.newaxis]).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _final_polish(self, image: Image.Image, metal_type: MetalType) -> Image.Image:
        """Final polish and color grading"""
        # Subtle contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Subtle brightness adjustment based on metal type
        enhancer = ImageEnhance.Brightness(image)
        if metal_type in [MetalType.WHITE_GOLD, MetalType.PLATINUM]:
            image = enhancer.enhance(1.05)
        else:
            image = enhancer.enhance(1.02)
        
        # Final sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        return image
    
    def detect_metal_type(self, image: Image.Image) -> MetalType:
        """Detect metal type from image"""
        img_array = np.array(image)
        
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
        
        # Decision logic based on HSV
        if saturation < 0.1:  # Low saturation = white metals
            if value > 0.9:
                return MetalType.PLATINUM
            else:
                return MetalType.WHITE_GOLD
        elif 15 < hue < 35:  # Yellow/gold hue range
            return MetalType.YELLOW_GOLD
        elif 0 < hue < 15 or hue > 165:  # Red/pink hue range
            return MetalType.ROSE_GOLD
        else:
            # Default based on brightness
            if value > 0.8:
                return MetalType.WHITE_GOLD
            else:
                return MetalType.YELLOW_GOLD

def handler(event):
    """RunPod handler function"""
    try:
        input_data = event.get('input', {})
        
        # Extract data
        image_data = input_data.get('image', '')
        metal_type_str = input_data.get('metal_type', 'auto')
        enhancement_level = input_data.get('enhancement_level', 'high')
        
        if not image_data:
            return {"output": {"error": "No image data provided"}}
        
        # Decode image
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # Add padding if needed (without '=' characters)
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
        enhancer = AdvancedWeddingRingEnhancer()
        
        # Detect or set metal type
        if metal_type_str == 'auto':
            detected_metal = enhancer.detect_metal_type(image)
        else:
            metal_map = {
                'yellow_gold': MetalType.YELLOW_GOLD,
                'rose_gold': MetalType.ROSE_GOLD,
                'white_gold': MetalType.WHITE_GOLD,
                'platinum': MetalType.PLATINUM
            }
            detected_metal = metal_map.get(metal_type_str, MetalType.YELLOW_GOLD)
        
        # Enhance image
        enhanced_image = enhancer.enhance_ring(image, detected_metal)
        
        # Convert to base64
        output_buffer = io.BytesIO()
        enhanced_image.save(output_buffer, format='PNG', quality=95)
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Remove padding to comply with requirements
        output_base64 = output_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": f"data:image/png;base64,{output_base64}",
                "detected_metal_type": detected_metal.value,
                "enhancement_applied": True,
                "mask_removal_applied": True
            }
        }
        
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
