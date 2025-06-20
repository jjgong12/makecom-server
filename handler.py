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
        """Initialize the Wedding Ring Processor with v95 Inpainting Perfect settings"""
        self.setup_complete = False
        self.reference_colors = self._load_reference_colors()
        
        # v95 Core Parameters
        self.brightness_factor = 1.25
        self.contrast_factor = 1.2
        self.sharpness_factor = 1.15
        self.thumbnail_size = (1000, 1300)
        self.detail_enhancement = 1.3
        
        # Inpainting parameters
        self.inpaint_radius = 5
        self.black_threshold = 30
        self.edge_scan_depth = 150
        
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
            margin = int(r * 0.5)
            return (
                max(0, x - r - margin),
                max(0, y - r - margin),
                min(w, x + r + margin),
                min(h, y + r + margin)
            )
        
        # Method 2: Bright region detection
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cont_w, cont_h = cv2.boundingRect(largest_contour)
            margin = 50
            return (
                max(0, x - margin),
                max(0, y - margin),
                min(w, x + cont_w + margin),
                min(h, y + cont_h + margin)
            )
        
        # Method 3: Default center protection (50% of image)
        return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    
    def remove_black_borders_inpainting(self, img_array: np.ndarray) -> np.ndarray:
        """Remove black borders using advanced inpainting technique"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        h, w = img_array.shape[:2]
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect wedding ring area for protection
        ring_area = self.detect_wedding_ring_area(img_array)
        
        # Create mask for black borders
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Multi-stage black border detection
        for threshold in [30, 40, 50, 60]:
            # Top edge
            for y in range(min(self.edge_scan_depth, h // 3)):
                # Skip if in ring area
                if ring_area and y >= ring_area[1] and y < ring_area[3]:
                    continue
                    
                row_mean = np.mean(img_bgr[y, :])
                if row_mean < threshold:
                    mask[y, :] = 255
                else:
                    break
            
            # Bottom edge
            for y in range(h - 1, max(h - self.edge_scan_depth, 2 * h // 3), -1):
                if ring_area and y >= ring_area[1] and y < ring_area[3]:
                    continue
                    
                row_mean = np.mean(img_bgr[y, :])
                if row_mean < threshold:
                    mask[y, :] = 255
                else:
                    break
            
            # Left edge
            for x in range(min(self.edge_scan_depth, w // 3)):
                if ring_area and x >= ring_area[0] and x < ring_area[2]:
                    continue
                    
                col_mean = np.mean(img_bgr[:, x])
                if col_mean < threshold:
                    mask[:, x] = 255
                else:
                    break
            
            # Right edge
            for x in range(w - 1, max(w - self.edge_scan_depth, 2 * w // 3), -1):
                if ring_area and x >= ring_area[0] and x < ring_area[2]:
                    continue
                    
                col_mean = np.mean(img_bgr[:, x])
                if col_mean < threshold:
                    mask[:, x] = 255
                else:
                    break
        
        # Detect corners for additional black removal
        corners_to_check = [
            (slice(0, 100), slice(0, 100)),      # Top-left
            (slice(0, 100), slice(-100, None)),  # Top-right
            (slice(-100, None), slice(0, 100)),  # Bottom-left
            (slice(-100, None), slice(-100, None)) # Bottom-right
        ]
        
        for y_slice, x_slice in corners_to_check:
            corner_region = img_bgr[y_slice, x_slice]
            if np.mean(corner_region) < 50:
                mask[y_slice, x_slice] = 255
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Inpaint the black regions
        if np.any(mask):
            result = cv2.inpaint(img_bgr, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
            
            # Second pass with different method for better results
            remaining_mask = np.zeros_like(mask)
            gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            # Check edges again after first inpainting
            edge_width = 20
            edges_to_check = [
                (gray_result[:edge_width, :], remaining_mask[:edge_width, :]),
                (gray_result[-edge_width:, :], remaining_mask[-edge_width:, :]),
                (gray_result[:, :edge_width], remaining_mask[:, :edge_width]),
                (gray_result[:, -edge_width:], remaining_mask[:, -edge_width:])
            ]
            
            for edge_region, mask_region in edges_to_check:
                if np.mean(edge_region) < 40:
                    mask_region[:] = 255
            
            if np.any(remaining_mask):
                result = cv2.inpaint(result, remaining_mask, 3, cv2.INPAINT_NS)
            
            # Convert back to RGB
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return img_array
    
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
        
        # Create thumbnail canvas
        thumbnail = Image.new('RGB', self.thumbnail_size, (250, 250, 250))
        
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
            
            # Step 1: Remove black borders using inpainting
            img_array = self.remove_black_borders_inpainting(img_array)
            
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
                    "processing_version": "v95_inpainting_perfect",
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
                    "processing_version": "v95_inpainting_perfect"
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
