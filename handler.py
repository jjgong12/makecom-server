import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
import requests
from typing import Dict, Any, Optional, Tuple
import json
import traceback
import re

class WeddingRingProcessor:
    def __init__(self):
        """Initialize the Wedding Ring Processor with v96 Ultimate settings"""
        self.setup_complete = False
        self.reference_colors = self._load_reference_colors()
        
        # v96 Core Parameters
        self.brightness_factor = 1.25
        self.contrast_factor = 1.2
        self.sharpness_factor = 1.15
        self.thumbnail_size = (1000, 1300)
        self.detail_enhancement = 1.3
        
    def _load_reference_colors(self) -> Dict[str, Any]:
        """Load reference colors from training data"""
        return {
            'white_gold': {
                'rgb_ranges': [(220, 250), (220, 250), (225, 255)],
                'brightness_min': 225,
                'blue_tint_max': 10
            },
            'yellow_gold': {
                'rgb_ranges': [(200, 255), (180, 235), (120, 200)],
                'warmth_min': 20,
                'rg_diff_min': 10
            },
            'rose_gold': {
                'rgb_ranges': [(220, 255), (180, 220), (150, 200)],
                'pink_tint_min': 15,
                'r_g_ratio_min': 1.08
            },
            'unplated_white': {
                'rgb_ranges': [(210, 240), (210, 240), (210, 240)],
                'brightness_min': 210,
                'saturation_max': 20
            }
        }
    
    def detect_wedding_ring_area(self, img_array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect and protect wedding ring area using multiple methods"""
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Hough Circle Detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=30, maxRadius=min(h, w) // 3
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]
            margin = int(r * 0.7)  # Increased margin
            return (
                max(0, x - r - margin),
                max(0, y - r - margin),
                min(w, x + r + margin),
                min(h, y + r + margin)
            )
        
        # Method 2: Bright region detection
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cont_w, cont_h = cv2.boundingRect(largest_contour)
            margin = 80  # Increased margin
            return (
                max(0, x - margin),
                max(0, y - margin),
                min(w, x + cont_w + margin),
                min(h, y + cont_h + margin)
            )
        
        # Method 3: Default center protection (60% of image)
        margin_x = w // 5
        margin_y = h // 5
        return (margin_x, margin_y, w - margin_x, h - margin_y)
    
    def remove_black_borders_ultimate(self, img_array: np.ndarray) -> np.ndarray:
        """Ultimate black border removal using aggressive multi-stage approach"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        h, w = img_array.shape[:2]
        result = img_array.copy()
        
        # Phase 1: Aggressive Inpainting
        result = self._phase1_aggressive_inpainting(result)
        
        # Phase 2: Smart Crop
        result = self._phase2_smart_crop(result)
        
        # Phase 3: Final Cleanup
        result = self._phase3_final_cleanup(result)
        
        return result
    
    def _phase1_aggressive_inpainting(self, img_array: np.ndarray) -> np.ndarray:
        """Phase 1: Multiple passes of aggressive inpainting"""
        h, w = img_array.shape[:2]
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect ring area for protection
        ring_area = self.detect_wedding_ring_area(img_array)
        
        # Pass 1: Very low threshold for pure black
        mask1 = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Scan from edges with very low threshold
        for threshold in [10, 20, 30, 40, 50]:
            # Top edge
            for y in range(min(h // 2, 300)):
                if ring_area and y >= ring_area[1]:
                    break
                if np.mean(gray[y, :]) < threshold:
                    mask1[y, :] = 255
                else:
                    break
            
            # Bottom edge
            for y in range(h - 1, max(h // 2, h - 300), -1):
                if ring_area and y < ring_area[3]:
                    break
                if np.mean(gray[y, :]) < threshold:
                    mask1[y, :] = 255
                else:
                    break
            
            # Left edge
            for x in range(min(w // 2, 300)):
                if ring_area and x >= ring_area[0]:
                    break
                if np.mean(gray[:, x]) < threshold:
                    mask1[:, x] = 255
                else:
                    break
            
            # Right edge
            for x in range(w - 1, max(w // 2, w - 300), -1):
                if ring_area and x < ring_area[2]:
                    break
                if np.mean(gray[:, x]) < threshold:
                    mask1[:, x] = 255
                else:
                    break
        
        # Apply morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask1 = cv2.dilate(mask1, kernel, iterations=2)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        
        # First inpainting
        if np.any(mask1):
            img_bgr = cv2.inpaint(img_bgr, mask1, 7, cv2.INPAINT_TELEA)
        
        # Pass 2: Edge detection based
        edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 30, 100)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        # Check edges for black borders
        edge_depth = 100
        if np.mean(edges[:edge_depth, :]) > 50:  # Top edge has many edges
            for y in range(edge_depth):
                if np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[y, :]) < 60:
                    mask2[y, :] = 255
        
        if np.mean(edges[-edge_depth:, :]) > 50:  # Bottom edge
            for y in range(h - edge_depth, h):
                if np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[y, :]) < 60:
                    mask2[y, :] = 255
        
        if np.mean(edges[:, :edge_depth]) > 50:  # Left edge
            for x in range(edge_depth):
                if np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[:, x]) < 60:
                    mask2[:, x] = 255
        
        if np.mean(edges[:, -edge_depth:]) > 50:  # Right edge
            for x in range(w - edge_depth, w):
                if np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[:, x]) < 60:
                    mask2[:, x] = 255
        
        # Second inpainting
        if np.any(mask2):
            img_bgr = cv2.inpaint(img_bgr, mask2, 5, cv2.INPAINT_NS)
        
        # Pass 3: Color-based detection
        mask3 = np.zeros((h, w), dtype=np.uint8)
        
        # Check for dark pixels in edges
        for y in range(50):
            for x in range(w):
                if np.all(img_bgr[y, x] < 40):
                    mask3[y, x] = 255
        
        for y in range(h - 50, h):
            for x in range(w):
                if np.all(img_bgr[y, x] < 40):
                    mask3[y, x] = 255
        
        for y in range(h):
            for x in range(50):
                if np.all(img_bgr[y, x] < 40):
                    mask3[y, x] = 255
        
        for y in range(h):
            for x in range(w - 50, w):
                if np.all(img_bgr[y, x] < 40):
                    mask3[y, x] = 255
        
        # Third inpainting
        if np.any(mask3):
            kernel_small = np.ones((3, 3), np.uint8)
            mask3 = cv2.dilate(mask3, kernel_small, iterations=1)
            img_bgr = cv2.inpaint(img_bgr, mask3, 3, cv2.INPAINT_TELEA)
        
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    def _phase2_smart_crop(self, img_array: np.ndarray) -> np.ndarray:
        """Phase 2: Smart crop to remove any remaining borders"""
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find the actual content boundaries
        top, bottom, left, right = 0, h, 0, w
        
        # Top
        for y in range(h // 3):
            if np.mean(gray[y, w//4:3*w//4]) > 100 and np.std(gray[y, w//4:3*w//4]) > 10:
                top = max(0, y - 5)
                break
        
        # Bottom
        for y in range(h - 1, 2 * h // 3, -1):
            if np.mean(gray[y, w//4:3*w//4]) > 100 and np.std(gray[y, w//4:3*w//4]) > 10:
                bottom = min(h, y + 5)
                break
        
        # Left
        for x in range(w // 3):
            if np.mean(gray[h//4:3*h//4, x]) > 100 and np.std(gray[h//4:3*h//4, x]) > 10:
                left = max(0, x - 5)
                break
        
        # Right
        for x in range(w - 1, 2 * w // 3, -1):
            if np.mean(gray[h//4:3*h//4, x]) > 100 and np.std(gray[h//4:3*h//4, x]) > 10:
                right = min(w, x + 5)
                break
        
        # Additional safety crop (remove 10 pixels from each edge)
        top += 10
        bottom -= 10
        left += 10
        right -= 10
        
        # Ensure valid crop
        if bottom > top and right > left:
            return img_array[top:bottom, left:right]
        
        return img_array
    
    def _phase3_final_cleanup(self, img_array: np.ndarray) -> np.ndarray:
        """Phase 3: Final cleanup and edge smoothing"""
        h, w = img_array.shape[:2]
        
        # Create a soft vignette mask for edges
        mask = np.ones((h, w), dtype=np.float32)
        
        # Fade edges
        fade_width = 20
        for i in range(fade_width):
            alpha = i / fade_width
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
        
        # Apply bilateral filter for edge smoothing
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Blend with white background at edges
        white_bg = np.ones_like(img_array) * 245
        mask_3d = np.stack([mask] * 3, axis=2)
        
        result = (img_array * mask_3d + white_bg * (1 - mask_3d)).astype(np.uint8)
        
        return result
    
    def detect_metal_type(self, img_array: np.ndarray) -> str:
        """Detect metal type - only detect unplated_white, white_gold, rose_gold; default to yellow_gold"""
        h, w = img_array.shape[:2]
        
        # Focus on center region where ring is likely to be
        center_y, center_x = h // 2, w // 2
        crop_size = min(h, w) // 3
        
        y1 = max(0, center_y - crop_size)
        y2 = min(h, center_y + crop_size)
        x1 = max(0, center_x - crop_size)
        x2 = min(w, center_x + crop_size)
        
        ring_region = img_array[y1:y2, x1:x2]
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(ring_region, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        mean_rgb = np.mean(ring_region.reshape(-1, 3), axis=0)
        r, g, b = mean_rgb
        
        # Calculate additional metrics
        brightness = np.mean(ring_region)
        saturation = np.mean(hsv[:, :, 1])
        
        # 1. Check for Unplated White (pure white, low saturation)
        if (brightness > 210 and saturation < 20 and 
            abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15):
            return 'unplated_white'
        
        # 2. Check for White Gold (bright, slight blue tint)
        if (brightness > 225 and b > g and b > r and 
            abs(b - g) < 10 and saturation < 30):
            return 'white_gold'
        
        # 3. Check for Rose Gold (pink tint)
        if (r > g and r > b and r / g > 1.08 and 
            r - g > 15 and brightness > 180):
            return 'rose_gold'
        
        # Default to Yellow Gold for everything else
        return 'yellow_gold'
    
    def enhance_wedding_ring_details(self, img_array: np.ndarray, metal_type: str) -> np.ndarray:
        """Enhance wedding ring details based on 38 training samples"""
        pil_img = Image.fromarray(img_array)
        
        # Base enhancement
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(self.brightness_factor)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(self.contrast_factor)
        
        # Metal-specific adjustments
        if metal_type == 'yellow_gold':
            # Enhance warm tones
            r, g, b = pil_img.split()
            r = r.point(lambda x: min(255, int(x * 1.05)))
            g = g.point(lambda x: min(255, int(x * 1.02)))
            pil_img = Image.merge('RGB', (r, g, b))
            
        elif metal_type == 'white_gold':
            # Enhance cool tones and brightness
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.05)
            
        elif metal_type == 'rose_gold':
            # Enhance pink tones
            r, g, b = pil_img.split()
            r = r.point(lambda x: min(255, int(x * 1.08)))
            pil_img = Image.merge('RGB', (r, g, b))
            
        elif metal_type == 'unplated_white':
            # Pure white enhancement
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.08)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(0.95)
        
        # Sharpening
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(self.sharpness_factor)
        
        # Final unsharp mask
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return np.array(pil_img)
    
    def create_thumbnail(self, img_array: np.ndarray) -> np.ndarray:
        """Create 98% zoomed thumbnail with perfect centering"""
        pil_img = Image.fromarray(img_array)
        
        # Calculate aspect ratios
        img_aspect = pil_img.width / pil_img.height
        target_aspect = self.thumbnail_size[0] / self.thumbnail_size[1]
        
        if img_aspect > target_aspect:
            # Image is wider - fit by height
            new_height = self.thumbnail_size[1]
            new_width = int(new_height * img_aspect)
        else:
            # Image is taller - fit by width
            new_width = self.thumbnail_size[0]
            new_height = int(new_width / img_aspect)
        
        # Resize to fit
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create thumbnail canvas with light background
        thumbnail = Image.new('RGB', self.thumbnail_size, (248, 248, 248))
        
        # Calculate position for centering
        x = (self.thumbnail_size[0] - new_width) // 2
        y = (self.thumbnail_size[1] - new_height) // 2
        
        # Paste the resized image
        thumbnail.paste(pil_img, (x, y))
        
        # Apply 98% zoom to remove any edge artifacts
        zoom_factor = 0.98
        cropped_size = (
            int(self.thumbnail_size[0] * zoom_factor),
            int(self.thumbnail_size[1] * zoom_factor)
        )
        
        left = (self.thumbnail_size[0] - cropped_size[0]) // 2
        top = (self.thumbnail_size[1] - cropped_size[1]) // 2
        right = left + cropped_size[0]
        bottom = top + cropped_size[1]
        
        thumbnail = thumbnail.crop((left, top, right, bottom))
        thumbnail = thumbnail.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
        
        return np.array(thumbnail)
    
    def process_image(self, image_base64: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Step 1: Remove black borders using ULTIMATE method
            img_array = self.remove_black_borders_ultimate(img_array)
            
            # Step 2: Detect metal type
            metal_type = self.detect_metal_type(img_array)
            
            # Step 3: Enhance wedding ring details
            enhanced_array = self.enhance_wedding_ring_details(img_array, metal_type)
            
            # Step 4: Create thumbnail
            thumbnail_array = self.create_thumbnail(enhanced_array)
            
            # Convert to base64
            enhanced_pil = Image.fromarray(enhanced_array)
            enhanced_buffer = BytesIO()
            enhanced_pil.save(enhanced_buffer, format='PNG', optimize=True)
            enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
            
            thumbnail_pil = Image.fromarray(thumbnail_array)
            thumbnail_buffer = BytesIO()
            thumbnail_pil.save(thumbnail_buffer, format='PNG', optimize=True)
            thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')
            
            # Remove base64 padding for Make.com
            enhanced_base64 = enhanced_base64.rstrip('=')
            thumbnail_base64 = thumbnail_base64.rstrip('=')
            
            return {
                "output": {
                    "enhanced_image": enhanced_base64,
                    "thumbnail": thumbnail_base64,
                    "metal_type": metal_type,
                    "processing_version": "v96_ultimate_black_removal",
                    "status": "success"
                }
            }
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            traceback.print_exc()
            return {
                "output": {
                    "error": str(e),
                    "status": "error",
                    "processing_version": "v96_ultimate_black_removal"
                }
            }

# RunPod Handler
processor = WeddingRingProcessor()

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        image_input = event.get("input", {})
        
        # Try multiple possible input keys
        image_base64 = (
            image_input.get("image") or 
            image_input.get("image_base64") or 
            image_input.get("base64_image")
        )
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error"
                }
            }
        
        # Process image
        return processor.process_image(image_base64)
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        traceback.print_exc()
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }

# RunPod serverless worker
runpod.serverless.start({"handler": handler})
