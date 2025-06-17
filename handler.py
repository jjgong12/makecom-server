import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional, List
import base64
from io import BytesIO
from PIL import Image

class WeddingRingEnhancer:
    def __init__(self):
        """Wedding Ring AI Enhancement System v16.0 - Perfect Frame Removal"""
        self.version = "16.0"
        
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
        
        # Bright and clean background colors like image 5
        self.after_bg_colors = {
            'gold': {'studio': (250, 248, 245), 'natural': (252, 250, 248), 'warm': (254, 252, 250)},
            'silver': {'studio': (248, 249, 250), 'natural': (250, 251, 252), 'warm': (252, 250, 248)},
            'rose_gold': {'studio': (251, 249, 247), 'natural': (253, 251, 249), 'warm': (255, 253, 251)},
            'platinum': {'studio': (247, 248, 249), 'natural': (249, 250, 251), 'warm': (251, 249, 247)}
        }

    def detect_black_frame_advanced(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Advanced black frame detection with multiple methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Multiple threshold detection
        frame_coords = None
        for threshold in [20, 30, 40, 50, 60, 70, 80]:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find largest contour
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest = max(contours, key=cv2.contourArea)
            x, y, rect_w, rect_h = cv2.boundingRect(largest)
            
            # Check if it's a frame
            if (x < 100 and y < 100 and 
                (x + rect_w) > w - 100 and (y + rect_h) > h - 100 and
                rect_w > w * 0.6 and rect_h > h * 0.6):
                frame_coords = self._find_exact_black_edges(gray, (x, y, x + rect_w, y + rect_h), threshold)
                if frame_coords:
                    break
        
        # Method 2: Edge-based detection if method 1 fails
        if not frame_coords:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # Find rectangular frame from lines
                frame_coords = self._find_frame_from_lines(lines, w, h)
        
        # Method 3: Gradient-based detection
        if not frame_coords:
            # Find strong gradients indicating frame edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # Find frame from gradient
            frame_coords = self._find_frame_from_gradient(gradient, w, h)
        
        return frame_coords

    def _find_exact_black_edges(self, gray: np.ndarray, rough_coords: Tuple[int, int, int, int], 
                               threshold: int) -> Optional[Tuple[int, int, int, int]]:
        """Find exact edges of black frame"""
        x1, y1, x2, y2 = rough_coords
        h, w = gray.shape
        
        # Scan inward to find exact edges
        # Top edge
        for y in range(y1, min(y1 + 200, h)):
            if np.mean(gray[y, x1:x2]) > threshold + 30:
                y1 = y
                break
        
        # Bottom edge
        for y in range(y2 - 1, max(y2 - 200, 0), -1):
            if np.mean(gray[y, x1:x2]) > threshold + 30:
                y2 = y + 1
                break
        
        # Left edge
        for x in range(x1, min(x1 + 200, w)):
            if np.mean(gray[y1:y2, x]) > threshold + 30:
                x1 = x
                break
        
        # Right edge
        for x in range(x2 - 1, max(x2 - 200, 0), -1):
            if np.mean(gray[y1:y2, x]) > threshold + 30:
                x2 = x + 1
                break
        
        # Validate frame
        if x2 - x1 > w * 0.5 and y2 - y1 > h * 0.5:
            return (x1, y1, x2, y2)
        return None

    def _find_frame_from_lines(self, lines: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Find frame from detected lines"""
        # Group lines by orientation and position
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1) * 0.1:  # Horizontal
                h_lines.append((y1 + y2) // 2)
            elif abs(x2 - x1) < abs(y2 - y1) * 0.1:  # Vertical
                v_lines.append((x1 + x2) // 2)
        
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            h_lines.sort()
            v_lines.sort()
            
            # Find frame edges
            top = min([y for y in h_lines if y < h * 0.3])
            bottom = max([y for y in h_lines if y > h * 0.7])
            left = min([x for x in v_lines if x < w * 0.3])
            right = max([x for x in v_lines if x > w * 0.7])
            
            return (left, top, right, bottom)
        return None

    def _find_frame_from_gradient(self, gradient: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Find frame from gradient magnitude"""
        # Find strong gradient lines
        _, binary = cv2.threshold(gradient.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # Project to find edges
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        
        # Find peaks in projections
        h_peaks = np.where(h_proj > w * 0.3)[0]
        v_peaks = np.where(v_proj > h * 0.3)[0]
        
        if len(h_peaks) >= 2 and len(v_peaks) >= 2:
            return (v_peaks[0], h_peaks[0], v_peaks[-1], h_peaks[-1])
        return None

    def remove_black_frame_completely(self, image: np.ndarray, frame_coords: Tuple[int, int, int, int], 
                                    metal_type: str, lighting: str) -> np.ndarray:
        """Remove black frame completely with bright background"""
        x1, y1, x2, y2 = frame_coords
        result = image.copy()
        h, w = image.shape[:2]
        
        # Get bright background color
        bg_color = self.after_bg_colors.get(metal_type, {}).get(lighting, (252, 250, 248))
        
        # Create comprehensive mask for all black pixels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple masks for thorough detection
        masks = []
        for threshold in [30, 40, 50, 60, 70, 80]:
            mask = gray < threshold
            masks.append(mask)
        
        # Combine all masks
        combined_mask = np.logical_or.reduce(masks)
        
        # Only apply to frame area (not inside)
        frame_mask = np.zeros_like(combined_mask)
        frame_mask[:y1, :] = combined_mask[:y1, :]  # Top
        frame_mask[y2:, :] = combined_mask[y2:, :]  # Bottom  
        frame_mask[:, :x1] = combined_mask[:, :x1]  # Left
        frame_mask[:, x2:] = combined_mask[:, x2:]  # Right
        
        # Add border detection for remaining black pixels
        # Top border
        for y in range(max(0, y1-10), min(y1+10, h)):
            for x in range(w):
                if gray[y, x] < 80:
                    frame_mask[y, x] = True
        
        # Bottom border
        for y in range(max(0, y2-10), min(y2+10, h)):
            for x in range(w):
                if gray[y, x] < 80:
                    frame_mask[y, x] = True
        
        # Left border
        for x in range(max(0, x1-10), min(x1+10, w)):
            for y in range(h):
                if gray[y, x] < 80:
                    frame_mask[y, x] = True
        
        # Right border
        for x in range(max(0, x2-10), min(x2+10, w)):
            for y in range(h):
                if gray[y, x] < 80:
                    frame_mask[y, x] = True
        
        # Apply bright background color
        result[frame_mask] = bg_color
        
        # Smooth transition
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame_mask_dilated = cv2.dilate(frame_mask.astype(np.uint8), kernel, iterations=1)
        
        # Gaussian blur for smooth edges
        transition_area = frame_mask_dilated - frame_mask.astype(np.uint8)
        if np.any(transition_area):
            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                channel[transition_area > 0] = (
                    channel[transition_area > 0] * 0.3 + 
                    bg_color[c] * 0.7
                )
                result[:, :, c] = channel.astype(np.uint8)
        
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
        
        # Multiple edge detection methods
        edges1 = cv2.Canny(enhanced, 30, 100)
        edges2 = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
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
                
            # Check shape characteristics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Fit ellipse if possible
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, (width, height), angle) = ellipse
                    
                    # Check if ellipse-like
                    aspect_ratio = min(width, height) / max(width, height)
                    if aspect_ratio > 0.3:  # Not too elongated
                        valid_contours.append(contour)
                except:
                    pass
        
        if not valid_contours:
            # Fallback: use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            valid_contours = [largest_contour]
        
        # Get bounding box of all valid contours
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(search_img.shape[1] - x, w + 2 * padding)
        h = min(search_img.shape[0] - y, h + 2 * padding)
        
        return (x + offset_x, y + offset_y, x + offset_x + w, y + offset_y + h)

    def apply_enhancement(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply v13.3 enhancement parameters with brightness adjustment"""
        result = image.copy().astype(np.float32)
        
        # Apply brightness (increased for brighter result)
        result = result * params.get('brightness', 1.0) * 1.1  # Extra 10% brightness
        
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

    def create_perfect_thumbnail(self, image: np.ndarray, frame_coords: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Create perfect 1000x1300 thumbnail with ring centered"""
        if frame_coords:
            x1, y1, x2, y2 = frame_coords
            # Use inner area only
            inner_image = image[y1:y2, x1:x2]
        else:
            inner_image = image
        
        # Detect ring in inner area for better centering
        ring_bbox = self.detect_ring(inner_image)
        
        if ring_bbox:
            rx1, ry1, rx2, ry2 = ring_bbox
            ring_center_x = (rx1 + rx2) // 2
            ring_center_y = (ry1 + ry2) // 2
            ring_width = rx2 - rx1
            ring_height = ry2 - ry1
            
            # Calculate crop area to center the ring
            # Make crop area larger than ring to include some background
            crop_size = max(ring_width, ring_height) * 1.8
            
            # Adjust for 1000:1300 aspect ratio
            crop_width = int(crop_size)
            crop_height = int(crop_size * 1.3)
            
            # Calculate crop coordinates
            crop_x1 = max(0, int(ring_center_x - crop_width // 2))
            crop_y1 = max(0, int(ring_center_y - crop_height // 2))
            crop_x2 = min(inner_image.shape[1], crop_x1 + crop_width)
            crop_y2 = min(inner_image.shape[0], crop_y1 + crop_height)
            
            # Adjust if crop goes out of bounds
            if crop_x2 > inner_image.shape[1]:
                crop_x1 = inner_image.shape[1] - crop_width
                crop_x2 = inner_image.shape[1]
            if crop_y2 > inner_image.shape[0]:
                crop_y1 = inner_image.shape[0] - crop_height
                crop_y2 = inner_image.shape[0]
            
            cropped = inner_image[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            # Fallback: center crop
            h, w = inner_image.shape[:2]
            target_ratio = 1000 / 1300
            current_ratio = w / h
            
            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                x_offset = (w - new_w) // 2
                cropped = inner_image[:, x_offset:x_offset + new_w]
            else:
                new_h = int(w / target_ratio)
                y_offset = (h - new_h) // 2
                cropped = inner_image[y_offset:y_offset + new_h, :]
        
        # Resize to exact 1000x1300 with high quality
        thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply slight brightness boost to match reference
        thumbnail = cv2.convertScaleAbs(thumbnail, alpha=1.05, beta=5)
        
        return thumbnail

    def detect_metal_type(self, image: np.ndarray) -> str:
        """Detect metal type from ring color"""
        # Get center region
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        
        # Calculate average color in LAB space for better color detection
        lab = cv2.cvtColor(center_region, cv2.COLOR_BGR2LAB)
        avg_lab = np.mean(lab, axis=(0, 1))
        l, a, b = avg_lab
        
        # Determine metal type based on LAB values
        if a > 5 and b > 15:  # Warm tones
            if a > 10:
                return 'rose_gold'
            else:
                return 'gold'
        elif abs(a) < 3 and abs(b) < 10:  # Neutral tones
            if l > 180:
                return 'platinum'
            else:
                return 'silver'
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
        
        # Check color temperature
        avg_b = np.mean(lab[:, :, 2])
        avg_a = np.mean(lab[:, :, 1])
        
        # Determine lighting
        if mean_l > 200 and std_l < 30:
            return 'studio'
        elif avg_b > 135:  # Warm tone
            return 'warm'
        else:
            return 'natural'

    def process_image(self, image_path: str) -> Dict[str, str]:
        """Main processing function with perfect black frame removal"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect black frame with advanced method
        frame_coords = self.detect_black_frame_advanced(image)
        
        # Default metal and lighting
        metal_type = 'gold'
        lighting = 'natural'
        
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
                params = self.enhancement_params.get(metal_type, {}).get(lighting, 
                                                    self.enhancement_params['gold']['natural'])
                
                # Apply enhancement to ring
                enhanced_ring = self.apply_enhancement(ring_region, params)
                
                # Put enhanced ring back
                image[ry1:ry2, rx1:rx2] = enhanced_ring
            
            # Remove black frame completely with bright background
            image = self.remove_black_frame_completely(image, frame_coords, metal_type, lighting)
        else:
            # No frame detected, process entire image
            ring_bbox = self.detect_ring(image)
            
            if ring_bbox:
                rx1, ry1, rx2, ry2 = ring_bbox
                ring_region = image[ry1:ry2, rx1:rx2].copy()
                
                metal_type = self.detect_metal_type(ring_region)
                lighting = self.detect_lighting(ring_region)
                
                params = self.enhancement_params.get(metal_type, {}).get(lighting,
                                                    self.enhancement_params['gold']['natural'])
                enhanced_ring = self.apply_enhancement(ring_region, params)
                
                image[ry1:ry2, rx1:rx2] = enhanced_ring
        
        # Create perfect thumbnail
        thumbnail = self.create_perfect_thumbnail(image, frame_coords)
        
        # Save outputs
        output_path = image_path.replace('.', '_enhanced.')
        thumbnail_path = image_path.replace('.', '_thumbnail.')
        
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'enhanced_image': enhanced_base64,
            'thumbnail': thumbnail_base64,
            'metal_type': metal_type,
            'lighting': lighting,
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
    print("Wedding Ring AI Enhancement v16.0")
    print("Perfect Black Frame Removal System")
    print("Testing mode...")
    
    # Test with a sample image if available
    test_image = "test_wedding_ring.jpg"
    if os.path.exists(test_image):
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(test_image)
        print(f"Processing complete!")
        print(f"Metal type: {result.get('metal_type')}")
        print(f"Lighting: {result.get('lighting')}")
        print(f"Version: {result.get('version')}")
    else:
        print("No test image found. Please provide 'test_wedding_ring.jpg'")
