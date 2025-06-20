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
        """Initialize the Wedding Ring Processor with v97 Ultra Fine Detection"""
        self.setup_complete = False
        self.reference_colors = self._load_reference_colors()
        
        # v97 Core Parameters
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
            margin = int(r * 0.7)
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
            margin = 80
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
    
    def remove_black_borders_ultra_fine(self, img_array: np.ndarray) -> np.ndarray:
        """Ultra fine black border detection and removal using multi-stage scanning"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect ring area for protection
        ring_area = self.detect_wedding_ring_area(img_array)
        
        # Initialize crop boundaries
        top, bottom, left, right = 0, h, 0, w
        
        # STAGE 1: Pixel-by-pixel ultra fine scanning (1 pixel at a time)
        print("STAGE 1: Pixel-by-pixel scanning")
        
        # Top edge - pixel by pixel
        for y in range(min(h // 2, 300)):
            if ring_area and y >= ring_area[1]:
                break
            # Check multiple conditions
            row_mean = np.mean(gray[y, :])
            row_max = np.max(gray[y, :])
            row_std = np.std(gray[y, :])
            
            if row_mean < 40 or (row_max < 60 and row_std < 10):
                top = y + 1
            else:
                break
        
        # Bottom edge - pixel by pixel
        for y in range(h - 1, max(h // 2, h - 300), -1):
            if ring_area and y < ring_area[3]:
                break
            row_mean = np.mean(gray[y, :])
            row_max = np.max(gray[y, :])
            row_std = np.std(gray[y, :])
            
            if row_mean < 40 or (row_max < 60 and row_std < 10):
                bottom = y
            else:
                break
        
        # Left edge - pixel by pixel
        for x in range(min(w // 2, 300)):
            if ring_area and x >= ring_area[0]:
                break
            col_mean = np.mean(gray[:, x])
            col_max = np.max(gray[:, x])
            col_std = np.std(gray[:, x])
            
            if col_mean < 40 or (col_max < 60 and col_std < 10):
                left = x + 1
            else:
                break
        
        # Right edge - pixel by pixel
        for x in range(w - 1, max(w // 2, w - 300), -1):
            if ring_area and x < ring_area[2]:
                break
            col_mean = np.mean(gray[:, x])
            col_max = np.max(gray[:, x])
            col_std = np.std(gray[:, x])
            
            if col_mean < 40 or (col_max < 60 and col_std < 10):
                right = x
            else:
                break
        
        # STAGE 2: 2-pixel step scanning for grey borders
        print("STAGE 2: 2-pixel step scanning")
        
        # Check for grey borders (threshold 60)
        for y in range(top, min(top + 100, h // 2), 2):
            if np.mean(gray[y, :]) < 60:
                top = y + 2
            else:
                break
        
        for y in range(bottom - 1, max(bottom - 100, h // 2), -2):
            if np.mean(gray[y, :]) < 60:
                bottom = y - 1
            else:
                break
        
        for x in range(left, min(left + 100, w // 2), 2):
            if np.mean(gray[:, x]) < 60:
                left = x + 2
            else:
                break
        
        for x in range(right - 1, max(right - 100, w // 2), -2):
            if np.mean(gray[:, x]) < 60:
                right = x - 1
            else:
                break
        
        # STAGE 3: 5-pixel step scanning for light grey
        print("STAGE 3: 5-pixel step scanning")
        
        for y in range(top, min(top + 50, h // 2), 5):
            if np.mean(gray[y, :]) < 80:
                top = y + 5
            else:
                break
        
        for y in range(bottom - 1, max(bottom - 50, h // 2), -5):
            if np.mean(gray[y, :]) < 80:
                bottom = y - 4
            else:
                break
        
        for x in range(left, min(left + 50, w // 2), 5):
            if np.mean(gray[:, x]) < 80:
                left = x + 5
            else:
                break
        
        for x in range(right - 1, max(right - 50, w // 2), -5):
            if np.mean(gray[:, x]) < 80:
                right = x - 4
            else:
                break
        
        # STAGE 4: Adaptive threshold scanning
        print("STAGE 4: Adaptive threshold scanning")
        
        for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            # Top
            for y in range(top, min(top + 20, h // 2)):
                if np.mean(gray[y, w//4:3*w//4]) < threshold:
                    top = y + 1
                else:
                    break
            
            # Bottom
            for y in range(bottom - 1, max(bottom - 20, h // 2), -1):
                if np.mean(gray[y, w//4:3*w//4]) < threshold:
                    bottom = y
                else:
                    break
            
            # Left
            for x in range(left, min(left + 20, w // 2)):
                if np.mean(gray[h//4:3*h//4, x]) < threshold:
                    left = x + 1
                else:
                    break
            
            # Right
            for x in range(right - 1, max(right - 20, w // 2), -1):
                if np.mean(gray[h//4:3*h//4, x]) < threshold:
                    right = x
                else:
                    break
        
        # STAGE 5: Edge detection based scanning
        print("STAGE 5: Edge detection scanning")
        
        edges = cv2.Canny(gray, 30, 100)
        
        # Check if edges are mostly black borders
        for y in range(top, min(top + 30, h // 2)):
            edge_ratio = np.sum(edges[y, :] > 0) / w
            if edge_ratio < 0.1 and np.mean(gray[y, :]) < 100:
                top = y + 1
        
        for y in range(bottom - 1, max(bottom - 30, h // 2), -1):
            edge_ratio = np.sum(edges[y, :] > 0) / w
            if edge_ratio < 0.1 and np.mean(gray[y, :]) < 100:
                bottom = y
        
        for x in range(left, min(left + 30, w // 2)):
            edge_ratio = np.sum(edges[:, x] > 0) / h
            if edge_ratio < 0.1 and np.mean(gray[:, x]) < 100:
                left = x + 1
        
        for x in range(right - 1, max(right - 30, w // 2), -1):
            edge_ratio = np.sum(edges[:, x] > 0) / h
            if edge_ratio < 0.1 and np.mean(gray[:, x]) < 100:
                right = x
        
        # STAGE 6: Corner analysis
        print("STAGE 6: Corner analysis")
        
        corner_size = 50
        corners = [
            gray[:corner_size, :corner_size],  # Top-left
            gray[:corner_size, -corner_size:],  # Top-right
            gray[-corner_size:, :corner_size],  # Bottom-left
            gray[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        # If all corners are dark, add extra crop
        if all(np.mean(corner) < 50 for corner in corners):
            top += 20
            bottom -= 20
            left += 20
            right -= 20
        
        # STAGE 7: Final safety crop
        print("STAGE 7: Final safety crop")
        
        # Always remove at least 10 pixels from each edge as safety
        top += 10
        bottom -= 10
        left += 10
        right -= 10
        
        # STAGE 8: RGB channel analysis
        print("STAGE 8: RGB channel analysis")
        
        # Check each color channel separately
        for c in range(3):
            channel = img_array[:, :, c]
            
            # Top
            for y in range(top, min(top + 20, h // 2)):
                if np.mean(channel[y, :]) < 40:
                    top = max(top, y + 1)
            
            # Bottom
            for y in range(bottom - 1, max(bottom - 20, h // 2), -1):
                if np.mean(channel[y, :]) < 40:
                    bottom = min(bottom, y)
            
            # Left
            for x in range(left, min(left + 20, w // 2)):
                if np.mean(channel[:, x]) < 40:
                    left = max(left, x + 1)
            
            # Right
            for x in range(right - 1, max(right - 20, w // 2), -1):
                if np.mean(channel[:, x]) < 40:
                    right = min(right, x)
        
        # Ensure valid crop region
        if bottom <= top or right <= left:
            print("Invalid crop region detected, using fallback")
            top, bottom = h // 10, 9 * h // 10
            left, right = w // 10, 9 * w // 10
        
        # Apply crop
        cropped = img_array[top:bottom, left:right]
        
        print(f"Cropped: top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"Original size: {h}x{w}, Cropped size: {cropped.shape[0]}x{cropped.shape[1]}")
        
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
            
            # Step 1: Remove black borders using ultra fine detection
            img_array = self.remove_black_borders_ultra_fine(img_array)
            
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
                    "processing_version": "v97_ultra_fine_detection",
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
                    "processing_version": "v97_ultra_fine_detection"
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
