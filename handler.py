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
        """Initialize the Wedding Ring Processor with v98 Perfect Black Removal"""
        self.setup_complete = False
        self.reference_colors = self._load_reference_colors()
        
        # v98 Core Parameters
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
            margin = int(r * 0.8)  # Increased protection margin
            return (
                max(0, x - r - margin),
                max(0, y - r - margin),
                min(w, x + r + margin),
                min(h, y + r + margin)
            )
        
        # Method 2: Bright region detection
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cont_w, cont_h = cv2.boundingRect(largest_contour)
            margin = 100  # Large margin for safety
            return (
                max(0, x - margin),
                max(0, y - margin),
                min(w, x + cont_w + margin),
                min(h, y + cont_h + margin)
            )
        
        # Method 3: Default center protection (70% of image)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        return (margin_x, margin_y, w - margin_x, h - margin_y)
    
    def remove_black_borders_perfect(self, img_array: np.ndarray) -> np.ndarray:
        """Perfect black border removal with maximum aggression"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect ring area for protection
        ring_area = self.detect_wedding_ring_area(img_array)
        print(f"Ring area detected: {ring_area}")
        
        # Initialize with aggressive defaults
        top, bottom, left, right = 50, h - 50, 50, w - 50  # Start with 50px crop
        
        # PHASE 1: Ultra aggressive threshold scanning (5-120)
        print("PHASE 1: Ultra aggressive scanning")
        for threshold in range(5, 121, 5):
            # Top edge
            for y in range(min(h // 2, 400)):
                if ring_area and y >= ring_area[1] - 20:
                    break
                if np.mean(gray[y, :]) < threshold:
                    top = max(top, y + 1)
            
            # Bottom edge
            for y in range(h - 1, max(h // 2, h - 400), -1):
                if ring_area and y <= ring_area[3] + 20:
                    break
                if np.mean(gray[y, :]) < threshold:
                    bottom = min(bottom, y)
            
            # Left edge
            for x in range(min(w // 2, 400)):
                if ring_area and x >= ring_area[0] - 20:
                    break
                if np.mean(gray[:, x]) < threshold:
                    left = max(left, x + 1)
            
            # Right edge
            for x in range(w - 1, max(w // 2, w - 400), -1):
                if ring_area and x <= ring_area[2] + 20:
                    break
                if np.mean(gray[:, x]) < threshold:
                    right = min(right, x)
        
        # PHASE 2: Gradient-based detection
        print("PHASE 2: Gradient detection")
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Top edge gradient check
        for y in range(top, min(top + 100, h // 2)):
            if np.mean(grad_mag[y, :]) < 10 and np.mean(gray[y, :]) < 100:
                top = y + 1
        
        # Bottom edge gradient check
        for y in range(bottom - 1, max(bottom - 100, h // 2), -1):
            if np.mean(grad_mag[y, :]) < 10 and np.mean(gray[y, :]) < 100:
                bottom = y
        
        # Left edge gradient check
        for x in range(left, min(left + 100, w // 2)):
            if np.mean(grad_mag[:, x]) < 10 and np.mean(gray[:, x]) < 100:
                left = x + 1
        
        # Right edge gradient check
        for x in range(right - 1, max(right - 100, w // 2), -1):
            if np.mean(grad_mag[:, x]) < 10 and np.mean(gray[:, x]) < 100:
                right = x
        
        # PHASE 3: Percentile-based detection
        print("PHASE 3: Percentile detection")
        
        # Check if edge pixels are consistently dark
        for y in range(top, min(top + 50, h // 2)):
            if np.percentile(gray[y, :], 90) < 80:  # 90% of pixels are dark
                top = y + 1
        
        for y in range(bottom - 1, max(bottom - 50, h // 2), -1):
            if np.percentile(gray[y, :], 90) < 80:
                bottom = y
        
        for x in range(left, min(left + 50, w // 2)):
            if np.percentile(gray[:, x], 90) < 80:
                left = x + 1
        
        for x in range(right - 1, max(right - 50, w // 2), -1):
            if np.percentile(gray[:, x], 90) < 80:
                right = x
        
        # PHASE 4: Color variance detection
        print("PHASE 4: Color variance detection")
        
        # Low variance = likely black border
        for y in range(top, min(top + 30, h // 2)):
            row_variance = np.var(img_array[y, :].reshape(-1))
            if row_variance < 100:
                top = y + 1
        
        for y in range(bottom - 1, max(bottom - 30, h // 2), -1):
            row_variance = np.var(img_array[y, :].reshape(-1))
            if row_variance < 100:
                bottom = y
        
        for x in range(left, min(left + 30, w // 2)):
            col_variance = np.var(img_array[:, x].reshape(-1))
            if col_variance < 100:
                left = x + 1
        
        for x in range(right - 1, max(right - 30, w // 2), -1):
            col_variance = np.var(img_array[:, x].reshape(-1))
            if col_variance < 100:
                right = x
        
        # PHASE 5: Fixed aggressive crop
        print("PHASE 5: Fixed aggressive crop")
        
        # Always remove at least this many pixels
        min_crop = 30
        top = max(top, min_crop)
        bottom = min(bottom, h - min_crop)
        left = max(left, min_crop)
        right = min(right, w - min_crop)
        
        # PHASE 6: Corner double-check
        print("PHASE 6: Corner analysis")
        
        corner_size = 80
        corners = [
            (gray[:corner_size, :corner_size], 'top-left'),
            (gray[:corner_size, -corner_size:], 'top-right'),
            (gray[-corner_size:, :corner_size], 'bottom-left'),
            (gray[-corner_size:, -corner_size:], 'bottom-right')
        ]
        
        dark_corners = 0
        for corner, name in corners:
            if np.mean(corner) < 60:
                dark_corners += 1
                print(f"Dark corner detected: {name}")
        
        if dark_corners >= 2:
            # Extra aggressive crop
            top += 25
            bottom -= 25
            left += 25
            right -= 25
        
        # PHASE 7: Final safety adjustments
        print("PHASE 7: Final adjustments")
        
        # Ensure we don't crop into the ring area
        if ring_area:
            top = min(top, ring_area[1] - 30)
            bottom = max(bottom, ring_area[3] + 30)
            left = min(left, ring_area[0] - 30)
            right = max(right, ring_area[2] + 30)
        
        # Ensure valid crop
        if bottom <= top + 100 or right <= left + 100:
            print("Invalid crop detected, using fallback")
            top = 60
            bottom = h - 60
            left = 60
            right = w - 60
        
        # Apply crop
        cropped = img_array[top:bottom, left:right]
        
        print(f"Final crop: top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"Original: {h}x{w}, Cropped: {cropped.shape[0]}x{cropped.shape[1]}")
        print(f"Removed: top={top}px, bottom={h-bottom}px, left={left}px, right={w-right}px")
        
        return cropped
    
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
            
            # Step 1: Remove black borders using perfect detection
            img_array = self.remove_black_borders_perfect(img_array)
            
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
                    "processing_version": "v98_perfect_black_removal",
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
                    "processing_version": "v98_perfect_black_removal"
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
