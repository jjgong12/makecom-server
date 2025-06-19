import cv2
import numpy as np
from rembg import remove
import runpod
import base64
import io
from PIL import Image
import json
import traceback
import sys

class WeddingRingEnhancer:
    def __init__(self):
        """Initialize the wedding ring enhancement system"""
        self.min_area = 5000
        self.v13_3_params = {
            'brightness': 1.24,
            'contrast': 1.15,
            'saturation': 1.05,
            'sharpness': 1.2,
            'denoise': 3,
            'detail_enhance': 1.15,
            'edge_preserve': 0.8,
            'bilateral_d': 9,
            'bilateral_sigma_color': 50,
            'bilateral_sigma_space': 50,
            'unsharp_radius': 2,
            'unsharp_amount': 0.8,
            'final_blend': 0.65
        }
        
    def find_black_lines_precise(self, image):
        """Find black lines with precise coordinate detection"""
        print("Finding black lines with precise coordinates...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Create mask for very dark pixels (black lines)
        black_mask = gray < 20  # Very strict threshold for black
        
        # Find horizontal black lines (top and bottom)
        horizontal_lines = []
        
        # Check top area
        for y in range(min(100, h//4)):
            row = gray[y, :]
            if np.mean(row) < 30:  # Dark row
                # Check if it's a continuous black line
                black_pixels = np.sum(row < 20)
                if black_pixels > w * 0.8:  # 80% of width is black
                    horizontal_lines.append(('top', y))
        
        # Check bottom area
        for y in range(max(0, h - h//4), h):
            row = gray[y, :]
            if np.mean(row) < 30:
                black_pixels = np.sum(row < 20)
                if black_pixels > w * 0.8:
                    horizontal_lines.append(('bottom', y))
        
        # Find vertical black lines (left and right)
        vertical_lines = []
        
        # Check left area
        for x in range(min(100, w//4)):
            col = gray[:, x]
            if np.mean(col) < 30:
                black_pixels = np.sum(col < 20)
                if black_pixels > h * 0.8:
                    vertical_lines.append(('left', x))
        
        # Check right area
        for x in range(max(0, w - w//4), w):
            col = gray[:, x]
            if np.mean(col) < 30:
                black_pixels = np.sum(col < 20)
                if black_pixels > h * 0.8:
                    vertical_lines.append(('right', x))
        
        # Create removal mask
        removal_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Mark horizontal lines
        if horizontal_lines:
            # Find continuous top black region
            top_end = 0
            for pos, y in horizontal_lines:
                if pos == 'top' and y < h//4:
                    top_end = max(top_end, y + 1)
            if top_end > 0:
                removal_mask[:top_end, :] = 255
                print(f"Found top black border: 0 to {top_end}")
            
            # Find continuous bottom black region
            bottom_start = h
            for pos, y in horizontal_lines:
                if pos == 'bottom' and y > h * 3//4:
                    bottom_start = min(bottom_start, y)
            if bottom_start < h:
                removal_mask[bottom_start:, :] = 255
                print(f"Found bottom black border: {bottom_start} to {h}")
        
        # Mark vertical lines
        if vertical_lines:
            # Find continuous left black region
            left_end = 0
            for pos, x in vertical_lines:
                if pos == 'left' and x < w//4:
                    left_end = max(left_end, x + 1)
            if left_end > 0:
                removal_mask[:, :left_end] = 255
                print(f"Found left black border: 0 to {left_end}")
            
            # Find continuous right black region
            right_start = w
            for pos, x in vertical_lines:
                if pos == 'right' and x > w * 3//4:
                    right_start = min(right_start, x)
            if right_start < w:
                removal_mask[:, right_start:] = 255
                print(f"Found right black border: {right_start} to {w}")
        
        return removal_mask
    
    def detect_wedding_ring_bbox(self, image):
        """Detect wedding ring bounding box with protection margin"""
        print("Detecting wedding ring location...")
        
        # Remove background
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output = remove(pil_image)
        result = np.array(output)
        
        # Get alpha channel
        if result.shape[2] == 4:
            alpha = result[:, :, 3]
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (wedding ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add 30 pixel protection margin
        margin = 30
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        print(f"Wedding ring bbox with margin: x={x}, y={y}, w={w}, h={h}")
        return (x, y, w, h)
    
    def sample_background_color(self, image, black_mask, ring_bbox=None):
        """Sample background color from clean areas"""
        h, w = image.shape[:2]
        
        # Define sampling regions (avoiding black areas and ring)
        sample_regions = []
        
        # Top region
        if ring_bbox:
            rx, ry, rw, rh = ring_bbox
            # Sample above the ring if possible
            if ry > h//4:
                sample_regions.append(image[h//8:ry-20, w//4:w*3//4])
        else:
            sample_regions.append(image[:h//4, w//4:w*3//4])
        
        # Left and right regions
        if ring_bbox:
            if rx > w//4:
                sample_regions.append(image[h//4:h*3//4, :rx-20])
            if rx + rw < w*3//4:
                sample_regions.append(image[h//4:h*3//4, rx+rw+20:])
        else:
            sample_regions.append(image[h//4:h*3//4, :w//4])
            sample_regions.append(image[h//4:h*3//4, w*3//4:])
        
        # Collect valid background pixels
        bg_pixels = []
        for region in sample_regions:
            if region.size > 0:
                # Flatten and filter out black pixels
                pixels = region.reshape(-1, 3)
                # Only use pixels that are not too dark
                valid_pixels = pixels[np.mean(pixels, axis=1) > 50]
                if len(valid_pixels) > 0:
                    bg_pixels.extend(valid_pixels)
        
        if not bg_pixels:
            # Fallback: use overall image average excluding very dark pixels
            pixels = image.reshape(-1, 3)
            valid_pixels = pixels[np.mean(pixels, axis=1) > 50]
            if len(valid_pixels) > 0:
                bg_color = np.mean(valid_pixels, axis=0)
            else:
                bg_color = np.array([245, 245, 245])  # Default light background
        else:
            bg_color = np.mean(bg_pixels, axis=0)
        
        print(f"Sampled background color: {bg_color}")
        return bg_color.astype(np.uint8)
    
    def remove_black_lines_natural(self, image, black_mask, ring_bbox):
        """Remove black lines and blend naturally with background"""
        result = image.copy()
        
        # Sample background color from clean areas
        bg_color = self.sample_background_color(image, black_mask, ring_bbox)
        
        # Direct replacement of black pixels
        black_indices = np.where(black_mask > 0)
        result[black_indices] = bg_color
        
        # Create smooth transition at edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_mask = cv2.dilate(black_mask, kernel, iterations=2) - black_mask
        edge_indices = np.where(edge_mask > 0)
        
        if len(edge_indices[0]) > 0:
            # Blend edge pixels
            alpha = 0.7
            result[edge_indices] = (image[edge_indices] * alpha + bg_color * (1 - alpha)).astype(np.uint8)
        
        # Apply light Gaussian blur to blend boundaries
        if np.any(black_mask > 0):
            # Create blur mask for affected areas only
            blur_mask = cv2.dilate(black_mask, kernel, iterations=3)
            blurred = cv2.GaussianBlur(result, (5, 5), 1)
            
            # Blend only in transition areas
            blur_indices = np.where(blur_mask > 0)
            alpha_map = blur_mask[blur_indices].astype(float) / 255.0
            result[blur_indices] = (
                result[blur_indices] * (1 - alpha_map[:, np.newaxis]) +
                blurred[blur_indices] * alpha_map[:, np.newaxis]
            ).astype(np.uint8)
        
        return result
    
    def apply_v13_3_enhancement(self, image):
        """Apply the proven v13.3 enhancement parameters"""
        print("Applying v13.3 enhancement...")
        
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # 1. Brightness adjustment
        brightness = self.v13_3_params['brightness']
        img_float = img_float * brightness
        
        # 2. Contrast adjustment
        contrast = self.v13_3_params['contrast']
        img_float = (img_float - 0.5) * contrast + 0.5
        
        # 3. Saturation adjustment
        saturation = self.v13_3_params['saturation']
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        img_float = gray_3ch * (1 - saturation) + img_float * saturation
        
        # Clip values
        img_float = np.clip(img_float, 0, 1)
        enhanced = (img_float * 255).astype(np.uint8)
        
        # 4. Denoise with bilateral filter
        denoised = cv2.bilateralFilter(
            enhanced,
            d=self.v13_3_params['bilateral_d'],
            sigmaColor=self.v13_3_params['bilateral_sigma_color'],
            sigmaSpace=self.v13_3_params['bilateral_sigma_space']
        )
        
        # 5. Sharpening with unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), self.v13_3_params['unsharp_radius'])
        sharpened = cv2.addWeighted(
            denoised, 
            1 + self.v13_3_params['unsharp_amount'], 
            gaussian, 
            -self.v13_3_params['unsharp_amount'], 
            0
        )
        
        # 6. Detail enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Blend enhanced L with original
        detail_factor = self.v13_3_params['detail_enhance']
        l_final = cv2.addWeighted(l, 1 - detail_factor + 1, l_enhanced, detail_factor - 1, 0)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_final, a, b])
        enhanced_detail = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 7. Final blend with original
        final_blend = self.v13_3_params['final_blend']
        result = cv2.addWeighted(image, 1 - final_blend, enhanced_detail, final_blend, 0)
        
        return result
    
    def create_thumbnail(self, image):
        """Create thumbnail with wedding ring filling the frame"""
        print("Creating thumbnail...")
        
        # Remove background to find ring
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output = remove(pil_image)
        result = np.array(output)
        
        # Get alpha channel
        if result.shape[2] == 4:
            alpha = result[:, :, 3]
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find ring contour
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: resize to target
            return cv2.resize(image, (1000, 1300))
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate crop with aspect ratio 1000:1300
        target_ratio = 1000 / 1300  # 0.769
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Wider than target - adjust height
            new_h = int(w / target_ratio)
            pad_h = new_h - h
            y = max(0, y - pad_h // 2)
            h = new_h
        else:
            # Taller than target - adjust width
            new_w = int(h * target_ratio)
            pad_w = new_w - w
            x = max(0, x - pad_w // 2)
            w = new_w
        
        # Ensure within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Crop and resize
        cropped = image[y:y+h, x:x+w]
        thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        return thumbnail
    
    def process_image(self, image):
        """Main processing pipeline"""
        try:
            print("\n=== Starting Wedding Ring Enhancement v23.3 ===")
            
            # Step 1: Detect wedding ring location with protection margin
            ring_bbox = self.detect_wedding_ring_bbox(image)
            if not ring_bbox:
                print("Warning: Could not detect wedding ring")
                ring_bbox = None
            
            # Step 2: Find black lines with precise coordinates
            black_mask = self.find_black_lines_precise(image)
            
            # Step 3: Remove black lines if found
            if np.any(black_mask > 0):
                print("Removing black lines and blending with background...")
                # Protect wedding ring area
                if ring_bbox:
                    x, y, w, h = ring_bbox
                    black_mask[y:y+h, x:x+w] = 0  # Don't touch ring area
                
                image = self.remove_black_lines_natural(image, black_mask, ring_bbox)
            else:
                print("No black lines detected")
            
            # Step 4: Apply v13.3 enhancement
            enhanced = self.apply_v13_3_enhancement(image)
            
            # Step 5: Create thumbnail from enhanced image
            thumbnail = self.create_thumbnail(enhanced)
            
            # Step 6: Upscale 2x
            height, width = enhanced.shape[:2]
            upscaled = cv2.resize(enhanced, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            print("=== Enhancement Complete ===\n")
            
            return upscaled, thumbnail
            
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            traceback.print_exc()
            
            # Return original at 2x scale as fallback
            height, width = image.shape[:2]
            upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            thumbnail = self.create_thumbnail(image)
            
            return upscaled, thumbnail


def handler(job):
    """RunPod handler function"""
    try:
        print("Starting RunPod handler...")
        job_input = job["input"]
        
        if "image" not in job_input:
            return {"error": "No image provided in input"}
        
        # Decode base64 image
        image_data = base64.b64decode(job_input["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image"}
        
        print(f"Received image: {image.shape}")
        
        # Process image
        enhancer = WeddingRingEnhancer()
        enhanced_image, thumbnail = enhancer.process_image(image)
        
        # Encode results
        _, buffer = cv2.imencode('.png', enhanced_image)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, thumb_buffer = cv2.imencode('.png', thumbnail)
        thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        # Return with proper structure for Make.com
        # RunPod wraps this in {"output": ...} automatically
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "status": "success",
                "original_size": f"{image.shape[1]}x{image.shape[0]}",
                "enhanced_size": f"{enhanced_image.shape[1]}x{enhanced_image.shape[0]}",
                "thumbnail_size": "1000x1300"
            }
        }
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "status": "error"
            }
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
