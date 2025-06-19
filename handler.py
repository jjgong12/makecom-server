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
    
    def detect_black_borders(self, img: np.ndarray, threshold: int = 40) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular black border regions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        borders = []
        
        # Detect continuous black regions from each edge
        # Top edge
        for y in range(min(150, h//4)):
            if np.mean(gray[y, :]) < threshold:
                # Check if entire row is black
                black_pixels = np.sum(gray[y, :] < threshold)
                if black_pixels > w * 0.9:  # 90% of row is black
                    continue
                else:
                    if y > 10:  # Minimum 10 pixels
                        borders.append(('top', 0, 0, w, y))
                    break
        
        # Bottom edge
        for y in range(min(150, h//4)):
            if np.mean(gray[h-1-y, :]) < threshold:
                black_pixels = np.sum(gray[h-1-y, :] < threshold)
                if black_pixels > w * 0.9:
                    continue
                else:
                    if y > 10:
                        borders.append(('bottom', 0, h-y, w, h))
                    break
        
        # Left edge
        for x in range(min(150, w//4)):
            if np.mean(gray[:, x]) < threshold:
                black_pixels = np.sum(gray[:, x] < threshold)
                if black_pixels > h * 0.9:
                    continue
                else:
                    if x > 10:
                        borders.append(('left', 0, 0, x, h))
                    break
        
        # Right edge
        for x in range(min(150, w//4)):
            if np.mean(gray[:, w-1-x]) < threshold:
                black_pixels = np.sum(gray[:, w-1-x] < threshold)
                if black_pixels > h * 0.9:
                    continue
                else:
                    if x > 10:
                        borders.append(('right', w-x, 0, w, h))
                    break
        
        return borders
    
    def remove_black_borders_smart(self, img: np.ndarray) -> np.ndarray:
        """Smart black border removal with inpainting"""
        borders = self.detect_black_borders(img, threshold=50)
        
        if not borders:
            print("No black borders detected")
            return img
        
        result = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Create mask for all border regions
        for border_type, x1, y1, x2, y2 in borders:
            print(f"Detected {border_type} border: {x2-x1}x{y2-y1} pixels")
            mask[y1:y2, x1:x2] = 255
        
        # Inpaint the masked regions
        if np.any(mask):
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        
        # Final crop to remove any remaining borders
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray > 30)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add small margin
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(result.shape[1] - x, w + 2 * margin)
            h = min(result.shape[0] - y, h + 2 * margin)
            result = result[y:y+h, x:x+w]
        
        return result
    
    def denoise_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced denoising for both background and product"""
        # Convert to LAB for better noise reduction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Denoise each channel
        l = cv2.bilateralFilter(l, 9, 50, 50)
        a = cv2.bilateralFilter(a, 9, 75, 75)
        b = cv2.bilateralFilter(b, 9, 75, 75)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional edge-preserving smoothing
        result = cv2.edgePreservingFilter(result, flags=2, sigma_s=50, sigma_r=0.4)
        
        return result
    
    def enhance_brightness_extreme(self, img: np.ndarray) -> np.ndarray:
        """Extreme brightness enhancement"""
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Extreme L channel enhancement
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        l = cv2.add(l, 60)  # Add brightness
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def extreme_sharpening(self, img: np.ndarray) -> np.ndarray:
        """Extreme sharpening for jewelry details"""
        # Multiple sharpening passes
        sharpened = img.copy()
        
        # First pass - unsharp mask
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.8, gaussian, -0.8, 0)
        
        # Second pass - kernel sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(sharpened, 1.0, edges_colored, 0.3, 0)
        
        return sharpened
    
    def process_ring_image(self, img_array: np.ndarray) -> np.ndarray:
        """Complete ring image processing pipeline"""
        print(f"Input shape: {img_array.shape}")
        
        # Step 1: Smart black border removal
        print("Step 1: Removing black borders...")
        img_no_border = self.remove_black_borders_smart(img_array)
        print(f"After border removal: {img_no_border.shape}")
        
        # Step 2: Advanced denoising
        print("Step 2: Advanced denoising...")
        img_denoised = self.denoise_advanced(img_no_border)
        
        # Step 3: Detect ring and create tight crop
        print("Step 3: Finding ring boundaries...")
        gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find ring
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (likely the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Tight crop with minimal padding
            padding = 2  # Minimal padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_denoised.shape[1] - x, w + 2 * padding)
            h = min(img_denoised.shape[0] - y, h + 2 * padding)
            
            img_cropped = img_denoised[y:y+h, x:x+w]
        else:
            img_cropped = img_denoised
        
        print(f"After crop: {img_cropped.shape}")
        
        # Step 4: Extreme enhancement
        print("Step 4: Extreme enhancement...")
        img_bright = self.enhance_brightness_extreme(img_cropped)
        img_sharp = self.extreme_sharpening(img_bright)
        
        # Step 5: PIL enhancements
        print("Step 5: Final PIL enhancements...")
        pil_img = Image.fromarray(cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB))
        
        # Strong enhancements
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.4)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.4)
        
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.6)
        
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Convert back
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return result
    
    def create_thumbnail(self, img: np.ndarray, size: Tuple[int, int] = (1000, 1300)) -> np.ndarray:
        """Create perfect thumbnail with ring filling the frame"""
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Scale to fill the entire frame (no black borders)
        scale = max(target_w / w, target_h / h) * 1.05  # 5% extra to ensure full coverage
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop to exact size
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
            
            # Save debug images
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
