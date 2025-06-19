import os
import io
import cv2
import base64
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import runpod
from PIL import Image, ImageEnhance, ImageFilter
import json
import traceback

class RingImageEnhancer:
    def __init__(self):
        self.debug_mode = True
        self.output_dir = "/workspace/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def multi_stage_black_detection(self, img: np.ndarray) -> np.ndarray:
        """Multi-stage black border detection and removal"""
        result = img.copy()
        h, w = result.shape[:2]
        
        # Stage 1: Scan in 5-pixel increments
        print("Stage 1: 5-pixel increment scan...")
        for edge in ['top', 'bottom', 'left', 'right']:
            for depth in range(5, min(150, h//4 if edge in ['top','bottom'] else w//4), 5):
                if self._check_and_remove_black_strip(result, edge, depth, threshold=30):
                    print(f"  Removed {depth}px from {edge}")
                    break
        
        # Stage 2: Scan in 10-pixel increments with different threshold
        print("Stage 2: 10-pixel increment scan...")
        for edge in ['top', 'bottom', 'left', 'right']:
            for depth in range(10, min(100, h//4 if edge in ['top','bottom'] else w//4), 10):
                if self._check_and_remove_black_strip(result, edge, depth, threshold=40):
                    print(f"  Removed {depth}px from {edge}")
                    break
        
        # Stage 3: Pixel-by-pixel scan for remaining black edges
        print("Stage 3: Pixel-by-pixel scan...")
        for _ in range(3):  # Multiple passes
            old_shape = result.shape
            result = self._remove_black_pixels_aggressive(result)
            if result.shape == old_shape:
                break
            print(f"  Trimmed to {result.shape}")
        
        # Stage 4: Corner analysis
        print("Stage 4: Corner analysis...")
        result = self._remove_black_corners(result)
        
        # Stage 5: Final cleanup with multiple thresholds
        print("Stage 5: Final cleanup...")
        for threshold in [20, 30, 40, 50]:
            result = self._final_black_cleanup(result, threshold)
        
        return result
    
    def _check_and_remove_black_strip(self, img: np.ndarray, edge: str, depth: int, threshold: int) -> bool:
        """Check and remove black strip from specific edge"""
        h, w = img.shape[:2]
        
        if edge == 'top':
            strip = img[:depth, :]
            if np.mean(strip) < threshold:
                # Inpaint the region
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[:depth, :] = 0
                img[:] = cv2.inpaint(img, 255 - mask, 3, cv2.INPAINT_TELEA)
                return True
                
        elif edge == 'bottom':
            strip = img[-depth:, :]
            if np.mean(strip) < threshold:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[-depth:, :] = 0
                img[:] = cv2.inpaint(img, 255 - mask, 3, cv2.INPAINT_TELEA)
                return True
                
        elif edge == 'left':
            strip = img[:, :depth]
            if np.mean(strip) < threshold:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[:, :depth] = 0
                img[:] = cv2.inpaint(img, 255 - mask, 3, cv2.INPAINT_TELEA)
                return True
                
        elif edge == 'right':
            strip = img[:, -depth:]
            if np.mean(strip) < threshold:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[:, -depth:] = 0
                img[:] = cv2.inpaint(img, 255 - mask, 3, cv2.INPAINT_TELEA)
                return True
        
        return False
    
    def _remove_black_pixels_aggressive(self, img: np.ndarray) -> np.ndarray:
        """Aggressively remove black pixels from edges"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Find first non-black pixel from each edge
        top = 0
        for y in range(h):
            if np.max(gray[y, :]) > 50 or np.mean(gray[y, :]) > 30:
                top = y
                break
        
        bottom = h
        for y in range(h-1, -1, -1):
            if np.max(gray[y, :]) > 50 or np.mean(gray[y, :]) > 30:
                bottom = y + 1
                break
        
        left = 0
        for x in range(w):
            if np.max(gray[:, x]) > 50 or np.mean(gray[:, x]) > 30:
                left = x
                break
        
        right = w
        for x in range(w-1, -1, -1):
            if np.max(gray[:, x]) > 50 or np.mean(gray[:, x]) > 30:
                right = x + 1
                break
        
        return img[top:bottom, left:right]
    
    def _remove_black_corners(self, img: np.ndarray) -> np.ndarray:
        """Remove black corners specifically"""
        h, w = img.shape[:2]
        corner_size = min(50, h//10, w//10)
        
        # Check each corner
        corners = [
            (0, 0, corner_size, corner_size),  # Top-left
            (w-corner_size, 0, w, corner_size),  # Top-right
            (0, h-corner_size, corner_size, h),  # Bottom-left
            (w-corner_size, h-corner_size, w, h)  # Bottom-right
        ]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for x1, y1, x2, y2 in corners:
            corner_region = img[y1:y2, x1:x2]
            if np.mean(corner_region) < 40:
                mask[y1:y2, x1:x2] = 255
        
        if np.any(mask):
            img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        
        return img
    
    def _final_black_cleanup(self, img: np.ndarray, threshold: int) -> np.ndarray:
        """Final cleanup pass with specific threshold"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create mask for black pixels
        mask = (gray < threshold).astype(np.uint8) * 255
        
        # Only process edge regions
        h, w = mask.shape
        edge_mask = np.zeros_like(mask)
        edge_width = 20
        edge_mask[:edge_width, :] = mask[:edge_width, :]  # Top
        edge_mask[-edge_width:, :] = mask[-edge_width:, :]  # Bottom
        edge_mask[:, :edge_width] = mask[:, :edge_width]  # Left
        edge_mask[:, -edge_width:] = mask[:, -edge_width:]  # Right
        
        # Inpaint edge black pixels
        if np.any(edge_mask):
            img = cv2.inpaint(img, edge_mask, 3, cv2.INPAINT_TELEA)
        
        return img
    
    def enhance_brightness_contrast_extreme(self, img: np.ndarray) -> np.ndarray:
        """Extreme brightness and contrast enhancement"""
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Extreme L channel processing
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE with high clip limit
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Boost brightness
        l = cv2.add(l, 70)
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gamma correction for extra brightness
        gamma = 0.6
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def super_sharpening(self, img: np.ndarray) -> np.ndarray:
        """Super aggressive sharpening"""
        # Unsharp mask with high amount
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)
        
        # Additional kernel sharpening
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 9, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 9.0
        sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 30, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(sharpened, 1.0, edges_colored, 0.3, 0)
        
        return sharpened
    
    def process_ring_image(self, img_array: np.ndarray) -> np.ndarray:
        """Complete ring image processing pipeline"""
        print(f"Input shape: {img_array.shape}")
        
        # Step 1: Multi-stage black border removal
        print("Step 1: Multi-stage black border removal...")
        img_no_border = self.multi_stage_black_detection(img_array)
        print(f"After border removal: {img_no_border.shape}")
        
        # Step 2: Denoising
        print("Step 2: Denoising...")
        denoised = cv2.bilateralFilter(img_no_border, 9, 75, 75)
        denoised = cv2.edgePreservingFilter(denoised, flags=2, sigma_s=60, sigma_r=0.4)
        
        # Step 3: Find ring and crop tightly
        print("Step 3: Finding ring boundaries...")
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # Use multiple methods to find ring
        edges = cv2.Canny(gray, 20, 80)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Minimal padding
            padding = 1
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(denoised.shape[1] - x, w + 2 * padding)
            h = min(denoised.shape[0] - y, h + 2 * padding)
            
            img_cropped = denoised[y:y+h, x:x+w]
        else:
            img_cropped = denoised
        
        print(f"After crop: {img_cropped.shape}")
        
        # Step 4: Extreme enhancement
        print("Step 4: Extreme enhancement...")
        img_bright = self.enhance_brightness_contrast_extreme(img_cropped)
        img_sharp = self.super_sharpening(img_bright)
        
        # Step 5: PIL final touches
        print("Step 5: Final enhancements...")
        pil_img = Image.fromarray(cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB))
        
        # Maximum enhancements
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(2.0)
        
        # Convert back
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return result
    
    def create_thumbnail(self, img: np.ndarray, size: Tuple[int, int] = (1000, 1300)) -> np.ndarray:
        """Create perfect thumbnail"""
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Scale to overfill frame
        scale = max(target_w / w, target_h / h) * 1.1  # 10% extra
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # High quality resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        
        thumbnail = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        # Final sharpening
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        thumbnail = cv2.filter2D(thumbnail, -1, kernel)
        
        return thumbnail
    
    def process_image(self, image_data: str) -> Dict[str, Any]:
        """Process base64 image and return enhanced version"""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Process the image
            enhanced = self.process_ring_image(img_array)
            
            # Create thumbnail
            thumbnail = self.create_thumbnail(enhanced)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
            
            _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
            thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
            
            # Save debug images if enabled
            if self.debug_mode:
                debug_path = os.path.join(self.output_dir, "debug_enhanced.png")
                cv2.imwrite(debug_path, enhanced)
                thumb_path = os.path.join(self.output_dir, "debug_thumbnail.jpg")
                cv2.imwrite(thumb_path, thumbnail)
                print(f"Debug images saved to {self.output_dir}")
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "original_size": img_array.shape[:2],
                "enhanced_size": enhanced.shape[:2],
                "thumbnail_size": thumbnail.shape[:2]
            }
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            traceback.print_exc()
            raise

def handler(job):
    """RunPod handler function"""
    job_input = job["input"]
    
    if "image" not in job_input:
        return {"error": "No image provided in input"}
    
    try:
        enhancer = RingImageEnhancer()
        result = enhancer.process_image(job_input["image"])
        
        # Return with proper structure for Make.com
        return {
            "output": result
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
