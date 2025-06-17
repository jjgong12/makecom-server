import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional, List
import base64
from io import BytesIO
from PIL import Image

class WeddingRingEnhancer:
    def __init__(self):
        """Wedding Ring AI Enhancement System v15.0"""
        self.version = "15.0"
        
        # v13.3 Enhancement parameters - COMPLETE SET
        self.enhancement_params = {
            'gold': {
                'studio': {
                    'brightness': 1.15, 'contrast': 1.25, 'saturation': 1.30,
                    'highlights': 0.92, 'shadows': 1.08, 'clarity': 0.75,
                    'vibrance': 0.85, 'warmth': 1.10, 'exposure': 0.15,
                    'sharpness': 0.70, 'noise_reduction': 0.30,
                    'metal_enhancement': {'shine': 0.80, 'reflection': 0.75, 'polish': 0.85}
                },
                'natural': {
                    'brightness': 1.08, 'contrast': 1.15, 'saturation': 1.20,
                    'highlights': 0.95, 'shadows': 1.12, 'clarity': 0.65,
                    'vibrance': 0.80, 'warmth': 1.05, 'exposure': 0.10,
                    'sharpness': 0.60, 'noise_reduction': 0.25,
                    'metal_enhancement': {'shine': 0.70, 'reflection': 0.65, 'polish': 0.75}
                },
                'warm': {
                    'brightness': 1.20, 'contrast': 1.30, 'saturation': 1.35,
                    'highlights': 0.88, 'shadows': 1.05, 'clarity': 0.80,
                    'vibrance': 0.90, 'warmth': 1.20, 'exposure': 0.20,
                    'sharpness': 0.75, 'noise_reduction': 0.35,
                    'metal_enhancement': {'shine': 0.85, 'reflection': 0.80, 'polish': 0.90}
                }
            },
            'silver': {
                'studio': {
                    'brightness': 1.10, 'contrast': 1.20, 'saturation': 0.95,
                    'highlights': 0.90, 'shadows': 1.10, 'clarity': 0.80,
                    'vibrance': 0.70, 'warmth': 0.95, 'exposure': 0.12,
                    'sharpness': 0.75, 'noise_reduction': 0.30,
                    'metal_enhancement': {'shine': 0.85, 'reflection': 0.82, 'polish': 0.88}
                },
                'natural': {
                    'brightness': 1.05, 'contrast': 1.12, 'saturation': 0.90,
                    'highlights': 0.93, 'shadows': 1.15, 'clarity': 0.70,
                    'vibrance': 0.65, 'warmth': 0.90, 'exposure': 0.08,
                    'sharpness': 0.65, 'noise_reduction': 0.25,
                    'metal_enhancement': {'shine': 0.75, 'reflection': 0.72, 'polish': 0.78}
                },
                'warm': {
                    'brightness': 1.12, 'contrast': 1.25, 'saturation': 1.00,
                    'highlights': 0.87, 'shadows': 1.08, 'clarity': 0.85,
                    'vibrance': 0.75, 'warmth': 1.05, 'exposure': 0.15,
                    'sharpness': 0.80, 'noise_reduction': 0.35,
                    'metal_enhancement': {'shine': 0.90, 'reflection': 0.85, 'polish': 0.92}
                }
            },
            'rose_gold': {
                'studio': {
                    'brightness': 1.12, 'contrast': 1.22, 'saturation': 1.25,
                    'highlights': 0.91, 'shadows': 1.09, 'clarity': 0.72,
                    'vibrance': 0.82, 'warmth': 1.15, 'exposure': 0.13,
                    'sharpness': 0.68, 'noise_reduction': 0.28,
                    'metal_enhancement': {'shine': 0.78, 'reflection': 0.73, 'polish': 0.83}
                },
                'natural': {
                    'brightness': 1.06, 'contrast': 1.13, 'saturation': 1.15,
                    'highlights': 0.94, 'shadows': 1.13, 'clarity': 0.62,
                    'vibrance': 0.78, 'warmth': 1.08, 'exposure': 0.09,
                    'sharpness': 0.58, 'noise_reduction': 0.23,
                    'metal_enhancement': {'shine': 0.68, 'reflection': 0.63, 'polish': 0.73}
                },
                'warm': {
                    'brightness': 1.18, 'contrast': 1.28, 'saturation': 1.32,
                    'highlights': 0.86, 'shadows': 1.06, 'clarity': 0.78,
                    'vibrance': 0.88, 'warmth': 1.25, 'exposure': 0.18,
                    'sharpness': 0.73, 'noise_reduction': 0.33,
                    'metal_enhancement': {'shine': 0.83, 'reflection': 0.78, 'polish': 0.88}
                }
            },
            'platinum': {
                'studio': {
                    'brightness': 1.08, 'contrast': 1.18, 'saturation': 0.92,
                    'highlights': 0.89, 'shadows': 1.11, 'clarity': 0.82,
                    'vibrance': 0.68, 'warmth': 0.92, 'exposure': 0.11,
                    'sharpness': 0.77, 'noise_reduction': 0.32,
                    'metal_enhancement': {'shine': 0.87, 'reflection': 0.85, 'polish': 0.90}
                },
                'natural': {
                    'brightness': 1.03, 'contrast': 1.10, 'saturation': 0.88,
                    'highlights': 0.92, 'shadows': 1.16, 'clarity': 0.72,
                    'vibrance': 0.63, 'warmth': 0.88, 'exposure': 0.07,
                    'sharpness': 0.67, 'noise_reduction': 0.27,
                    'metal_enhancement': {'shine': 0.77, 'reflection': 0.75, 'polish': 0.80}
                },
                'warm': {
                    'brightness': 1.10, 'contrast': 1.23, 'saturation': 0.98,
                    'highlights': 0.85, 'shadows': 1.09, 'clarity': 0.87,
                    'vibrance': 0.73, 'warmth': 1.02, 'exposure': 0.14,
                    'sharpness': 0.82, 'noise_reduction': 0.37,
                    'metal_enhancement': {'shine': 0.92, 'reflection': 0.88, 'polish': 0.95}
                }
            }
        }
        
        # After file background colors for each metal and lighting
        self.after_bg_colors = {
            'gold': {'studio': (245, 242, 238), 'natural': (248, 245, 241), 'warm': (250, 247, 243)},
            'silver': {'studio': (242, 243, 244), 'natural': (245, 246, 247), 'warm': (247, 245, 243)},
            'rose_gold': {'studio': (246, 243, 240), 'natural': (249, 246, 243), 'warm': (251, 248, 245)},
            'platinum': {'studio': (241, 242, 243), 'natural': (244, 245, 246), 'warm': (246, 244, 242)}
        }

    def detect_black_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect black rectangular frame with improved accuracy"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Multiple threshold values for better detection
        for threshold in [30, 40, 50, 60, 70]:
            # Create binary mask
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, rect_w, rect_h = cv2.boundingRect(largest_contour)
            
            # Check if it's a proper frame (close to image borders)
            if (x < 50 and y < 50 and 
                (x + rect_w) > w - 50 and (y + rect_h) > h - 50 and
                rect_w > w * 0.7 and rect_h > h * 0.7):
                
                # Find the inner edge of the black frame
                # Scan inward to find where black ends
                inner_x1, inner_y1 = x, y
                inner_x2, inner_y2 = x + rect_w, y + rect_h
                
                # Scan from edges to find exact black frame boundaries
                # Top edge
                for scan_y in range(y, y + 200):
                    if scan_y >= h:
                        break
                    row = gray[scan_y, x:x+rect_w]
                    if np.mean(row) > threshold + 20:
                        inner_y1 = scan_y
                        break
                
                # Bottom edge
                for scan_y in range(y + rect_h - 1, y + rect_h - 200, -1):
                    if scan_y < 0:
                        break
                    row = gray[scan_y, x:x+rect_w]
                    if np.mean(row) > threshold + 20:
                        inner_y2 = scan_y + 1
                        break
                
                # Left edge
                for scan_x in range(x, x + 200):
                    if scan_x >= w:
                        break
                    col = gray[y:y+rect_h, scan_x]
                    if np.mean(col) > threshold + 20:
                        inner_x1 = scan_x
                        break
                
                # Right edge
                for scan_x in range(x + rect_w - 1, x + rect_w - 200, -1):
                    if scan_x < 0:
                        break
                    col = gray[y:y+rect_h, scan_x]
                    if np.mean(col) > threshold + 20:
                        inner_x2 = scan_x + 1
                        break
                
                return (inner_x1, inner_y1, inner_x2, inner_y2)
        
        return None

    def remove_black_frame(self, image: np.ndarray, frame_coords: Tuple[int, int, int, int], 
                          metal_type: str, lighting: str) -> np.ndarray:
        """Remove black frame completely and fill with appropriate background"""
        x1, y1, x2, y2 = frame_coords
        result = image.copy()
        h, w = image.shape[:2]
        
        # Get background color
        bg_color = self.after_bg_colors.get(metal_type, {}).get(lighting, (245, 243, 240))
        
        # Create mask for black frame area
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill outer area (black frame area)
        mask[:y1, :] = 255  # Top
        mask[y2:, :] = 255  # Bottom
        mask[:, :x1] = 255  # Left
        mask[:, x2:] = 255  # Right
        
        # Also detect any remaining black pixels in frame area
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 80
        
        # Combine masks - only in frame area
        frame_mask = np.zeros_like(mask)
        frame_mask[:y1, :] = black_mask[:y1, :]
        frame_mask[y2:, :] = black_mask[y2:, :]
        frame_mask[:, :x1] = black_mask[:, :x1]
        frame_mask[:, x2:] = black_mask[:, x2:]
        
        # Apply background color
        result[mask > 0] = bg_color
        result[frame_mask > 0] = bg_color
        
        # Smooth edges
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        
        # Create smooth transition
        dist_transform = cv2.distanceTransform(255 - mask_dilated, cv2.DIST_L2, 5)
        dist_transform = np.clip(dist_transform / 5.0, 0, 1)
        
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - dist_transform) + bg_color[c] * dist_transform
        
        return result

    def detect_ring(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """Detect wedding ring in image or specific ROI"""
        if roi:
            x1, y1, x2, y2 = roi
            roi_img = image[y1:y2, x1:x2]
            search_img = roi_img
            offset_x, offset_y = x1, y1
        else:
            search_img = image
            offset_x, offset_y = 0, 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Detect edges
        edges = cv2.Canny(enhanced, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours for ring-like shapes
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.3:  # Reasonably circular
                valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        # Get bounding box of all valid contours
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(search_img.shape[1] - x, w + 2 * padding)
        h = min(search_img.shape[0] - y, h + 2 * padding)
        
        return (x + offset_x, y + offset_y, x + offset_x + w, y + offset_y + h)

    def apply_enhancement(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply v13.3 enhancement parameters"""
        result = image.copy().astype(np.float32)
        
        # Apply brightness
        result = result * params.get('brightness', 1.0)
        
        # Apply contrast
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = mean + params.get('contrast', 1.0) * (result - mean)
        
        # Apply saturation
        gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        result = gray + params.get('saturation', 1.0) * (result - gray)
        
        # Apply highlights and shadows
        highlights_mask = (gray[:,:,0] > 180).astype(np.float32)
        shadows_mask = (gray[:,:,0] < 75).astype(np.float32)
        
        result = result * (1 + (params.get('highlights', 1.0) - 1) * highlights_mask[:,:,np.newaxis])
        result = result * (1 + (params.get('shadows', 1.0) - 1) * shadows_mask[:,:,np.newaxis])
        
        # Apply warmth
        warmth = params.get('warmth', 1.0)
        if warmth != 1.0:
            result[:,:,2] *= warmth  # Red channel
            result[:,:,0] *= (2 - warmth)  # Blue channel
        
        # Apply exposure
        exposure = params.get('exposure', 0.0)
        if exposure != 0:
            result = result * (2 ** exposure)
        
        # Apply vibrance
        vibrance = params.get('vibrance', 0.0)
        if vibrance != 0:
            sat = np.sqrt(np.sum((result - gray) ** 2, axis=2))
            sat_mask = 1 - sat / (np.max(sat) + 1e-6)
            result = result + vibrance * sat_mask[:,:,np.newaxis] * (result - gray) * 0.5
        
        # Apply clarity (local contrast)
        clarity = params.get('clarity', 0.0)
        if clarity != 0:
            blurred = cv2.GaussianBlur(result, (15, 15), 0)
            detail = result - blurred
            result = result + clarity * detail * 0.5
        
        # Apply sharpness
        sharpness = params.get('sharpness', 0.0)
        if sharpness > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpness * 0.1
            kernel[1,1] = 1 + kernel[1,1]
            sharpened = cv2.filter2D(result, -1, kernel)
            result = result * (1 - sharpness) + sharpened * sharpness
        
        # Apply metal enhancement
        metal_params = params.get('metal_enhancement', {})
        if metal_params:
            # Enhance metallic shine
            shine = metal_params.get('shine', 0.0)
            if shine > 0:
                highlights = np.maximum(result - 200, 0) * shine * 2
                result = result + highlights
            
            # Enhance reflections
            reflection = metal_params.get('reflection', 0.0)
            if reflection > 0:
                lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:,:,0] = lab[:,:,0] * (1 + reflection * 0.1)
                result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)

    def create_thumbnail(self, image: np.ndarray, frame_coords: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Create 1000x1300 thumbnail from inner area"""
        if frame_coords:
            x1, y1, x2, y2 = frame_coords
            # Crop to inner area (without black frame)
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
        
        # Calculate aspect ratios
        target_ratio = 1000 / 1300  # 0.769
        h, w = cropped.shape[:2]
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Image is wider, crop width
            new_w = int(h * target_ratio)
            x_offset = (w - new_w) // 2
            final_crop = cropped[:, x_offset:x_offset + new_w]
        else:
            # Image is taller, crop height
            new_h = int(w / target_ratio)
            y_offset = (h - new_h) // 2
            final_crop = cropped[y_offset:y_offset + new_h, :]
        
        # Resize to exact 1000x1300
        thumbnail = cv2.resize(final_crop, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        return thumbnail

    def detect_metal_type(self, image: np.ndarray) -> str:
        """Detect metal type from ring color"""
        # Get center region
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        
        # Calculate average color
        avg_color = np.mean(center_region, axis=(0, 1))
        b, g, r = avg_color
        
        # Calculate color ratios
        if r > b and r > g:
            if r - b > 20:
                return 'rose_gold'
            else:
                return 'gold'
        elif b > r and g > r:
            return 'silver'
        elif abs(r - g) < 10 and abs(g - b) < 10:
            return 'platinum'
        else:
            return 'gold'  # Default

    def detect_lighting(self, image: np.ndarray) -> str:
        """Detect lighting condition"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate statistics
        mean_l = np.mean(l_channel)
        std_l = np.std(l_channel)
        
        # Determine lighting
        if mean_l > 200 and std_l < 30:
            return 'studio'
        elif mean_l > 150 and std_l > 40:
            return 'natural'
        else:
            return 'warm'

    def process_image(self, image_path: str) -> Dict[str, str]:
        """Main processing function"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect black frame
        frame_coords = self.detect_black_frame(image)
        
        if frame_coords:
            # Detect ring within frame
            ring_bbox = self.detect_ring(image, frame_coords)
            
            if ring_bbox:
                # Extract ring region
                rx1, ry1, rx2, ry2 = ring_bbox
                ring_region = image[ry1:ry2, rx1:rx2].copy()
                
                # Detect metal and lighting
                metal_type = self.detect_metal_type(ring_region)
                lighting = self.detect_lighting(ring_region)
                
                # Get enhancement parameters
                params = self.enhancement_params.get(metal_type, {}).get(lighting, {})
                
                # Apply enhancement to ring
                enhanced_ring = self.apply_enhancement(ring_region, params)
                
                # Put enhanced ring back
                image[ry1:ry2, rx1:rx2] = enhanced_ring
            
            # Remove black frame
            image = self.remove_black_frame(image, frame_coords, 
                                          metal_type if ring_bbox else 'gold', 
                                          lighting if ring_bbox else 'studio')
        else:
            # No frame detected, process entire image
            ring_bbox = self.detect_ring(image)
            
            if ring_bbox:
                rx1, ry1, rx2, ry2 = ring_bbox
                ring_region = image[ry1:ry2, rx1:rx2].copy()
                
                metal_type = self.detect_metal_type(ring_region)
                lighting = self.detect_lighting(ring_region)
                
                params = self.enhancement_params.get(metal_type, {}).get(lighting, {})
                enhanced_ring = self.apply_enhancement(ring_region, params)
                
                image[ry1:ry2, rx1:rx2] = enhanced_ring
        
        # Create thumbnail
        thumbnail = self.create_thumbnail(image, frame_coords)
        
        # Save outputs
        output_path = image_path.replace('.', '_enhanced.')
        thumbnail_path = image_path.replace('.', '_thumbnail.')
        
        cv2.imwrite(output_path, image)
        cv2.imwrite(thumbnail_path, thumbnail)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'enhanced_image': enhanced_base64,
            'thumbnail': thumbnail_base64,
            'metal_type': metal_type if ring_bbox else 'unknown',
            'lighting': lighting if ring_bbox else 'unknown',
            'version': self.version
        }

# RunPod handler
def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event.get('input', {})
        
        # Get base64 image
        image_base64 = input_data.get('image')
        if not image_base64:
            return {'error': 'No image provided'}
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temporary file
        temp_path = '/tmp/input_image.jpg'
        cv2.imwrite(temp_path, image)
        
        # Process image
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# Test mode
if __name__ == "__main__":
    print("Wedding Ring AI Enhancement v15.0")
    print("Testing mode...")
    
    # Test with a sample image if available
    test_image = "test_wedding_ring.jpg"
    if os.path.exists(test_image):
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(test_image)
        print(f"Processing complete!")
        print(f"Metal type: {result.get('metal_type')}")
        print(f"Lighting: {result.get('lighting')}")
    else:
        print("No test image found. Please provide 'test_wedding_ring.jpg'")
