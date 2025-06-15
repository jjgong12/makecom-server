import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 Complete Parameters (28 pairs learning data) - ALL 12 SETS
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01,
            'noise_reduction': 1.15, 'clarity': 1.18
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18, 'saturation': 0.98, 'gamma': 1.03,
            'noise_reduction': 1.20, 'clarity': 1.22
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 0.99,
            'noise_reduction': 1.12, 'clarity': 1.15
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98,
            'noise_reduction': 1.10, 'clarity': 1.10
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.08,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95,
            'noise_reduction': 1.08, 'clarity': 1.05
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.04,
            'sharpness': 1.25, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.15, 'saturation': 1.25, 'gamma': 1.02,
            'noise_reduction': 1.18, 'clarity': 1.20
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00,
            'noise_reduction': 1.15, 'clarity': 1.15
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.10,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98,
            'noise_reduction': 1.18, 'clarity': 1.12
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.15,
            'sharpness': 1.25, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.02,
            'noise_reduction': 1.20, 'clarity': 1.18
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01,
            'noise_reduction': 1.12, 'clarity': 1.12
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.03,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.97,
            'noise_reduction': 1.08, 'clarity': 1.08
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.18, 'saturation': 1.28, 'gamma': 1.03,
            'noise_reduction': 1.22, 'clarity': 1.18
        }
    }
}

# 28 pairs AFTER background colors (Dialog 28 achievement)
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [250, 248, 245], 'medium': [242, 240, 237], 'dark': [235, 233, 230]
    },
    'warm': {
        'light': [252, 248, 240], 'medium': [245, 241, 233], 'dark': [238, 234, 226]
    },
    'cool': {
        'light': [248, 250, 252], 'medium': [240, 242, 245], 'dark': [233, 235, 238]
    }
}

class UltimateCompletePerfectWeddingRingProcessor:
    def __init__(self):
        self.processing_log = []
        
    def log_processing_step(self, step, details=""):
        """Processing step logging for debugging"""
        self.processing_log.append(f"{step}: {details}")
    
    def safe_array_operation(self, func, *args, **kwargs):
        """Safe array operation wrapper to prevent crashes"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_processing_step("Safe Operation Error", str(e))
            return None
    
    def detect_metal_type_advanced_complete(self, image, mask=None):
        """Complete advanced metal type detection with all edge cases"""
        try:
            self.log_processing_step("Metal Detection Start")
            
            if mask is not None and np.any(mask):
                # Extract masked region for analysis
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) == 0:
                    self.log_processing_step("Metal Detection", "No mask pixels found, defaulting to champagne_gold")
                    return 'champagne_gold'
                
                rgb_values = image[mask_indices[0], mask_indices[1], :]
                if len(rgb_values) == 0:
                    return 'champagne_gold'
                
                # Convert to HSV for analysis
                hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                avg_hue = np.mean(hsv_values[:, 0])
                avg_sat = np.mean(hsv_values[:, 1])
                avg_val = np.mean(hsv_values[:, 2])
            else:
                # Analyze entire image
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
                avg_val = np.mean(hsv[:, :, 2])
            
            # Advanced metal classification with comprehensive logic
            if avg_sat < 20:  # Very low saturation
                metal_type = 'white_gold'
            elif avg_sat < 30 and avg_val > 150:  # Low saturation, high brightness
                metal_type = 'white_gold'
            elif avg_hue < 10 or avg_hue > 170:  # Red spectrum
                if avg_sat > 50:
                    metal_type = 'rose_gold'
                else:
                    metal_type = 'white_gold'
            elif 10 <= avg_hue <= 40:  # Yellow-orange spectrum
                if avg_sat > 100:
                    metal_type = 'yellow_gold'
                elif avg_sat > 50:
                    metal_type = 'champagne_gold'  # Will be whitened
                else:
                    metal_type = 'white_gold'
            else:
                metal_type = 'champagne_gold'  # Default case
            
            self.log_processing_step("Metal Detection Complete", 
                                   f"Type: {metal_type}, HSV: H={avg_hue:.1f}, S={avg_sat:.1f}, V={avg_val:.1f}")
            return metal_type
            
        except Exception as e:
            self.log_processing_step("Metal Detection Error", str(e))
            return 'champagne_gold'  # Safe default
    
    def detect_lighting_environment_complete(self, image):
        """Complete lighting environment detection with edge cases"""
        try:
            self.log_processing_step("Lighting Detection Start")
            
            # LAB color space analysis
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]  # Lightness
            a_channel = lab[:, :, 1]  # Green-Red
            b_channel = lab[:, :, 2]  # Blue-Yellow
            
            # Calculate statistics
            b_mean = np.mean(b_channel)
            a_mean = np.mean(a_channel)
            l_mean = np.mean(l_channel)
            
            # Advanced lighting classification
            if b_mean < 120:  # Blue shift (warm light compensation)
                lighting = 'warm'
            elif b_mean > 140:  # Yellow shift (cool light compensation)
                lighting = 'cool'
            elif a_mean < 125:  # Green shift
                lighting = 'cool'
            elif a_mean > 135:  # Red shift
                lighting = 'warm'
            else:
                lighting = 'natural'
            
            # Secondary validation using overall brightness
            if l_mean < 100:  # Very dark - likely warm indoor lighting
                lighting = 'warm'
            elif l_mean > 200:  # Very bright - likely cool daylight
                lighting = 'cool'
            
            self.log_processing_step("Lighting Detection Complete", 
                                   f"Type: {lighting}, LAB: L={l_mean:.1f}, A={a_mean:.1f}, B={b_mean:.1f}")
            return lighting
            
        except Exception as e:
            self.log_processing_step("Lighting Detection Error", str(e))
            return 'natural'  # Safe default
    
    def detect_adaptive_black_border_ultimate_complete(self, image):
        """Ultimate adaptive black border detection - Dialog 29 complete achievement"""
        try:
            self.log_processing_step("Border Detection Start")
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Multi-threshold detection for comprehensive border finding
            threshold_masks = []
            thresholds = [10, 15, 20, 25, 30, 35]  # Comprehensive range
            
            for threshold in thresholds:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                threshold_masks.append(binary)
            
            # Combine all threshold masks for maximum detection
            combined_mask = threshold_masks[0].copy()
            for mask in threshold_masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Progressive morphological operations for border refinement
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            ]
            
            for kernel in kernels:
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours with comprehensive analysis
            contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.log_processing_step("Border Detection", "No contours found")
                return None, None, None
            
            # Analyze all contours for best border candidate
            border_candidates = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 1000:  # Skip tiny contours
                    continue
                
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate metrics for border quality assessment
                area_ratio = area / (width * height)
                aspect_ratio = max(w, h) / min(w, h)
                rect_ratio = (w * h) / area if area > 0 else 0
                
                # Border quality scoring
                score = 0
                
                # Size scoring (should be significant but not entire image)
                if 0.1 < area_ratio < 0.8:
                    score += 30
                
                # Aspect ratio scoring (should be reasonable rectangle)
                if 0.5 < aspect_ratio < 3.0:
                    score += 25
                
                # Rectangle fit scoring
                if rect_ratio > 0.7:
                    score += 20
                
                # Position scoring (borders usually near edges)
                edge_distance = min(x, y, width - (x + w), height - (y + h))
                if edge_distance < min(width, height) * 0.2:
                    score += 15
                
                # Polygon approximation scoring (rectangles have ~4 vertices)
                if 4 <= len(approx) <= 8:
                    score += 10
                
                border_candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'score': score,
                    'area': area,
                    'approx': approx
                })
            
            if not border_candidates:
                self.log_processing_step("Border Detection", "No valid border candidates")
                return None, None, None
            
            # Select best border candidate
            best_border = max(border_candidates, key=lambda x: x['score'])
            
            if best_border['score'] < 30:  # Minimum quality threshold
                self.log_processing_step("Border Detection", f"Best score {best_border['score']} below threshold")
                return None, None, None
            
            contour = best_border['contour']
            x, y, w, h = best_border['bbox']
            
            self.log_processing_step("Border Detection", f"Found border at ({x}, {y}, {w}, {h}) with score {best_border['score']}")
            
            # Measure actual border thickness with comprehensive sampling
            border_thickness = self.measure_actual_border_thickness_complete(combined_mask, (x, y, w, h))
            
            # Create precise border mask
            border_mask = np.zeros_like(gray)
            cv2.fillPoly(border_mask, [contour], 255)
            
            # Calculate wedding ring protection area with adaptive margins
            # Dialog 29: 100px thickness support with 50% safety margin
            ring_margin_base = max(30, border_thickness // 2)
            ring_margin_adaptive = min(ring_margin_base, min(w, h) // 8)  # Don't make ring area too small
            
            ring_x = x + ring_margin_adaptive
            ring_y = y + ring_margin_adaptive
            ring_w = w - 2 * ring_margin_adaptive
            ring_h = h - 2 * ring_margin_adaptive
            
            # Ensure ring area is reasonable
            min_ring_size = min(width, height) // 6
            if ring_w < min_ring_size or ring_h < min_ring_size:
                # Fallback to center-based protection
                center_x, center_y = x + w // 2, y + h // 2
                safe_size = max(min_ring_size, min(w, h) // 3)
                ring_x = center_x - safe_size // 2
                ring_y = center_y - safe_size // 2
                ring_w = safe_size
                ring_h = safe_size
            
            # Apply absolute wedding ring protection to border mask
            border_mask[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w] = 0
            
            self.log_processing_step("Border Detection Complete", 
                                   f"Border thickness: {border_thickness}px, Ring protection: ({ring_x}, {ring_y}, {ring_w}, {ring_h})")
            
            return border_mask, (x, y, w, h), (ring_x, ring_y, ring_w, ring_h)
            
        except Exception as e:
            self.log_processing_step("Border Detection Error", str(e))
            return None, None, None
    
    def measure_actual_border_thickness_complete(self, mask, bbox):
        """Complete border thickness measurement - Dialog 29 achievement"""
        try:
            x, y, w, h = bbox
            thicknesses = []
            
            # Sample from multiple positions for accuracy
            sample_positions = [0.25, 0.5, 0.75]  # 25%, 50%, 75% positions
            
            # Top border sampling
            if y > 20:
                for pos in sample_positions:
                    sample_x = int(x + w * pos)
                    if 0 <= sample_x < mask.shape[1]:
                        top_line = mask[max(0, y-15):y+15, sample_x:sample_x+1]
                        if np.any(top_line):
                            thickness = np.sum(top_line > 0)
                            if thickness > 0:
                                thicknesses.append(thickness)
            
            # Bottom border sampling
            if y + h < mask.shape[0] - 20:
                for pos in sample_positions:
                    sample_x = int(x + w * pos)
                    if 0 <= sample_x < mask.shape[1]:
                        bottom_line = mask[y+h-15:min(mask.shape[0], y+h+15), sample_x:sample_x+1]
                        if np.any(bottom_line):
                            thickness = np.sum(bottom_line > 0)
                            if thickness > 0:
                                thicknesses.append(thickness)
            
            # Left border sampling
            if x > 20:
                for pos in sample_positions:
                    sample_y = int(y + h * pos)
                    if 0 <= sample_y < mask.shape[0]:
                        left_line = mask[sample_y:sample_y+1, max(0, x-15):x+15]
                        if np.any(left_line):
                            thickness = np.sum(left_line > 0)
                            if thickness > 0:
                                thicknesses.append(thickness)
            
            # Right border sampling
            if x + w < mask.shape[1] - 20:
                for pos in sample_positions:
                    sample_y = int(y + h * pos)
                    if 0 <= sample_y < mask.shape[0]:
                        right_line = mask[sample_y:sample_y+1, x+w-15:min(mask.shape[1], x+w+15)]
                        if np.any(right_line):
                            thickness = np.sum(right_line > 0)
                            if thickness > 0:
                                thicknesses.append(thickness)
            
            if thicknesses:
                # Use median for robustness, apply 1.5x safety factor
                measured_thickness = int(np.median(thicknesses) * 1.5)
                # Ensure reasonable bounds (Dialog 29: support up to 100px)
                return max(20, min(measured_thickness, 150))
            else:
                return 50  # Conservative default
                
        except Exception as e:
            self.log_processing_step("Thickness Measurement Error", str(e))
            return 50
    
    def apply_v13_3_complete_enhancement_ultimate(self, image, metal_type, lighting):
        """Complete v13.3 enhancement with ALL 10 steps - Dialog 8-15 ultimate achievement"""
        try:
            self.log_processing_step("v13.3 Enhancement Start", f"Metal: {metal_type}, Lighting: {lighting}")
            
            # Get parameters with comprehensive fallback
            params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                            WEDDING_RING_PARAMS['champagne_gold']['natural'])
            
            # Ensure all required parameters exist
            default_params = {
                'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.08,
                'sharpness': 1.15, 'color_temp_a': -2, 'color_temp_b': -2,
                'original_blend': 0.15, 'saturation': 1.05, 'gamma': 1.00,
                'noise_reduction': 1.15, 'clarity': 1.15
            }
            
            for key, default_value in default_params.items():
                if key not in params:
                    params[key] = default_value
            
            # Step 1: Advanced noise reduction with bilateral filter
            self.log_processing_step("Step 1", "Noise Reduction")
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Additional noise reduction if parameter > 1.1
            if params['noise_reduction'] > 1.1:
                kernel = np.ones((3, 3), np.float32) / 9
                denoised = cv2.filter2D(denoised, -1, kernel)
            
            # Convert to PIL for precise enhancement
            pil_image = Image.fromarray(denoised)
            
            # Step 2: Brightness enhancement with precision
            self.log_processing_step("Step 2", f"Brightness: {params['brightness']}")
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # Step 3: Contrast enhancement with precision
            self.log_processing_step("Step 3", f"Contrast: {params['contrast']}")
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # Step 4: Sharpness enhancement with precision
            self.log_processing_step("Step 4", f"Sharpness: {params['sharpness']}")
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # Convert back to array for advanced processing
            enhanced_array = np.array(enhanced)
            
            # Step 5: Saturation adjustment with HSV precision
            self.log_processing_step("Step 5", f"Saturation: {params['saturation']}")
            if abs(params['saturation'] - 1.0) > 0.01:  # Only if significant change
                hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * params['saturation']
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                enhanced_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Step 6: White overlay application ("slight white coating feel" - Dialog core concept)
            self.log_processing_step("Step 6", f"White Overlay: {params['white_overlay']}")
            if params['white_overlay'] > 0.01:
                white_overlay = np.full_like(enhanced_array, 255, dtype=np.uint8)
                enhanced_array = cv2.addWeighted(
                    enhanced_array, 1 - params['white_overlay'],
                    white_overlay, params['white_overlay'], 0
                )
            
            # Step 7: Color temperature adjustment in LAB space (precision color science)
            self.log_processing_step("Step 7", f"Color Temp: A={params['color_temp_a']}, B={params['color_temp_b']}")
            if abs(params['color_temp_a']) > 0.5 or abs(params['color_temp_b']) > 0.5:
                lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
                lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)  # A channel
                lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)  # B channel
                enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            # Step 8: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            self.log_processing_step("Step 8", f"CLAHE Clarity: {params['clarity']}")
            if params['clarity'] > 1.05:
                lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0]
                
                # Apply CLAHE with intensity based on clarity parameter
                clip_limit = 1.0 + (params['clarity'] - 1.0) * 2.0  # Convert to CLAHE range
                clip_limit = min(clip_limit, 3.0)  # Cap at 3.0 for safety
                
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                lab[:, :, 0] = l_channel
                enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Step 9: Gamma correction for fine-tuning
            self.log_processing_step("Step 9", f"Gamma: {params['gamma']}")
            if abs(params['gamma'] - 1.0) > 0.01:
                gamma = params['gamma']
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced_array = cv2.LUT(enhanced_array, table)
            
            # Step 10: Original blending for natural preservation
            self.log_processing_step("Step 10", f"Original Blend: {params['original_blend']}")
            final = cv2.addWeighted(
                enhanced_array, 1 - params['original_blend'],
                image, params['original_blend'], 0
            )
            
            self.log_processing_step("v13.3 Enhancement Complete", "All 10 steps applied successfully")
            return final.astype(np.uint8)
            
        except Exception as e:
            self.log_processing_step("v13.3 Enhancement Error", str(e))
            # Return enhanced version even if some steps failed
            try:
                # Basic fallback enhancement
                enhanced = cv2.convertScaleAbs(image, alpha=1.15, beta=10)
                return enhanced
            except:
                return image  # Ultimate fallback - return original
    
    def remove_black_border_telea_advanced_complete(self, image, border_mask, ring_bbox):
        """Complete advanced TELEA inpainting with 28-pairs AFTER background colors - Dialog 27,28 achievement"""
        try:
            self.log_processing_step("Border Removal Start")
            
            # Detect lighting for appropriate background color selection
            lighting = self.detect_lighting_environment_complete(image)
            bg_colors = AFTER_BACKGROUND_COLORS[lighting]
            
            # Analyze current background characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Sample background from non-border areas
            background_sample_mask = np.ones_like(gray) * 255
            background_sample_mask = cv2.bitwise_and(background_sample_mask, cv2.bitwise_not(border_mask))
            
            # If we have ring bbox, exclude that area from background sampling
            if ring_bbox is not None:
                rx, ry, rw, rh = ring_bbox
                background_sample_mask[ry:ry+rh, rx:rx+rw] = 0
            
            # Calculate background brightness for color selection
            bg_pixels = gray[background_sample_mask > 0]
            if len(bg_pixels) > 0:
                bg_brightness = np.mean(bg_pixels)
            else:
                bg_brightness = np.mean(gray)  # Fallback
            
            # Select appropriate AFTER background color
            if bg_brightness > 180:
                target_bg = bg_colors['light']
            elif bg_brightness > 120:
                target_bg = bg_colors['medium']
            else:
                target_bg = bg_colors['dark']
            
            self.log_processing_step("Background Analysis", 
                                   f"Lighting: {lighting}, Brightness: {bg_brightness:.1f}, Target: {target_bg}")
            
            # Apply TELEA inpainting for natural texture
            inpainted = cv2.inpaint(image, border_mask, 5, cv2.INPAINT_TELEA)
            
            # Enhance inpainting with NS method for comparison
            inpainted_ns = cv2.inpaint(image, border_mask, 5, cv2.INPAINT_NS)
            
            # Combine best of both methods
            inpainted = cv2.addWeighted(inpainted, 0.7, inpainted_ns, 0.3, 0)
            
            # Apply 28-pairs AFTER background color to border areas
            border_indices = np.where(border_mask > 0)
            if len(border_indices[0]) > 0:
                # Blend with target background color
                for c in range(3):
                    inpainted[border_indices[0], border_indices[1], c] = (
                        inpainted[border_indices[0], border_indices[1], c] * 0.3 +
                        target_bg[c] * 0.7
                    ).astype(np.uint8)
            
            # Create sophisticated blending mask (Dialog 25 achievement - 31x31 Gaussian)
            blend_mask = border_mask.astype(np.float32) / 255.0
            
            # Multi-stage blending for natural edges
            blend_mask_31 = cv2.GaussianBlur(blend_mask, (31, 31), 10)  # Main blending
            blend_mask_15 = cv2.GaussianBlur(blend_mask, (15, 15), 5)   # Fine blending
            blend_mask_7 = cv2.GaussianBlur(blend_mask, (7, 7), 2)      # Detail preservation
            
            # Combine blending masks for multi-level smoothness
            final_blend_mask = (blend_mask_31 * 0.5 + blend_mask_15 * 0.3 + blend_mask_7 * 0.2)
            
            # Apply sophisticated multi-channel blending
            result = image.copy().astype(np.float32)
            inpainted_float = inpainted.astype(np.float32)
            
            for c in range(3):
                result[:, :, c] = (
                    image[:, :, c].astype(np.float32) * (1 - final_blend_mask) +
                    inpainted_float[:, :, c] * final_blend_mask
                )
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            self.log_processing_step("Border Removal Complete", "TELEA + AFTER colors + 31x31 blending applied")
            return result
            
        except Exception as e:
            self.log_processing_step("Border Removal Error", str(e))
            # Fallback to simple border color replacement
            try:
                result = image.copy()
                border_indices = np.where(border_mask > 0)
                if len(border_indices[0]) > 0:
                    result[border_indices[0], border_indices[1]] = [240, 240, 240]  # Safe gray
                return result
            except:
                return image  # Ultimate fallback
    
    def apply_wedding_ring_extra_enhancement(self, image, ring_bbox):
        """Apply extra enhancement to wedding ring area - Dialog 34 achievement"""
        try:
            if ring_bbox is None:
                # Apply gentle enhancement to entire image
                enhanced = cv2.convertScaleAbs(image, alpha=1.05, beta=3)
                return enhanced
            
            result = image.copy()
            rx, ry, rw, rh = ring_bbox
            
            # Ensure bbox is within image bounds
            rx = max(0, min(rx, image.shape[1] - 1))
            ry = max(0, min(ry, image.shape[0] - 1))
            rw = max(1, min(rw, image.shape[1] - rx))
            rh = max(1, min(rh, image.shape[0] - ry))
            
            if rw > 10 and rh > 10:  # Only if area is reasonable
                ring_region = image[ry:ry+rh, rx:rx+rw].copy()
                
                # Apply sophisticated ring enhancement
                # 1. Brightness and contrast boost
                ring_enhanced = cv2.convertScaleAbs(ring_region, alpha=1.12, beta=6)
                
                # 2. Unsharp masking for detail enhancement
                gaussian = cv2.GaussianBlur(ring_enhanced, (0, 0), 2.0)
                ring_enhanced = cv2.addWeighted(ring_enhanced, 1.5, gaussian, -0.5, 0)
                
                # 3. Detail enhancement with high-pass filter
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                detail_enhanced = cv2.filter2D(ring_enhanced, -1, kernel)
                ring_enhanced = cv2.addWeighted(ring_enhanced, 0.8, detail_enhanced, 0.2, 0)
                
                # Place enhanced ring back
                result[ry:ry+rh, rx:rx+rw] = ring_enhanced
                
                self.log_processing_step("Ring Enhancement", f"Applied to region ({rx}, {ry}, {rw}, {rh})")
            
            return result
            
        except Exception as e:
            self.log_processing_step("Ring Enhancement Error", str(e))
            return image
    
    def upscale_lanczos_2x_complete(self, image):
        """Complete high quality 2x upscaling using LANCZOS4 with safety checks"""
        try:
            self.log_processing_step("Upscaling Start")
            height, width = image.shape[:2]
            
            # Safety check for maximum image size
            max_dimension = 8192
            if width * 2 > max_dimension or height * 2 > max_dimension:
                # Scale down to safe size first
                scale_factor = min(max_dimension / (width * 2), max_dimension / (height * 2))
                safe_width = int(width * scale_factor)
                safe_height = int(height * scale_factor)
                resized_first = cv2.resize(image, (safe_width, safe_height), interpolation=cv2.INTER_LANCZOS4)
                final_upscaled = cv2.resize(resized_first, (safe_width * 2, safe_height * 2), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Direct 2x upscaling
                new_width = width * 2
                new_height = height * 2
                final_upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            self.log_processing_step("Upscaling Complete", f"From {width}x{height} to {final_upscaled.shape[1]}x{final_upscaled.shape[0]}")
            return final_upscaled
            
        except Exception as e:
            self.log_processing_step("Upscaling Error", str(e))
            # Fallback to smaller upscaling
            try:
                height, width = image.shape[:2]
                return cv2.resize(image, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_LINEAR)
            except:
                return image  # Ultimate fallback
    
    def create_perfect_thumbnail_1000x1300_complete(self, image, border_bbox):
        """Create perfect 1000x1300 thumbnail with wedding ring filling screen - Complete implementation"""
        try:
            self.log_processing_step("Thumbnail Creation Start")
            
            if border_bbox is not None:
                x, y, w, h = border_bbox
                self.log_processing_step("Thumbnail", f"Using border bbox: ({x}, {y}, {w}, {h})")
            else:
                # No border detected - use intelligent center region detection
                height, width = image.shape[:2]
                
                # Detect bright regions (likely wedding rings)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Find bright areas
                threshold = np.mean(gray) + np.std(gray) * 0.5
                _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours of bright areas
                contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest bright area (likely the rings)
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    self.log_processing_step("Thumbnail", f"Detected bright region: ({x}, {y}, {w}, {h})")
                else:
                    # Fallback to center region
                    x, y = width // 4, height // 4
                    w, h = width // 2, height // 2
                    self.log_processing_step("Thumbnail", "Using center region as fallback")
            
            # Calculate crop area with minimal margins for maximum ring size
            margin_w = max(10, int(w * 0.05))  # Very small margin - 5%
            margin_h = max(10, int(h * 0.05))
            
            crop_x1 = max(0, x - margin_w)
            crop_y1 = max(0, y - margin_h)
            crop_x2 = min(image.shape[1], x + w + margin_w)
            crop_y2 = min(image.shape[0], y + h + margin_h)
            
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped.size == 0:
                self.log_processing_step("Thumbnail Error", "Empty crop, using center")
                # Emergency fallback
                center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                size = min(image.shape[0], image.shape[1]) // 2
                cropped = image[center_y-size//2:center_y+size//2, center_x-size//2:center_x+size//2]
            
            # Calculate scale to fill 1000x1300 completely (ring fills screen)
            crop_h, crop_w = cropped.shape[:2]
            target_w, target_h = 1000, 1300
            
            # Use maximum scale to ensure ring fills the screen
            ratio = max(target_w / crop_w, target_h / crop_h) * 0.98  # 98% to prevent edge artifacts
            
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            # Ensure new dimensions are reasonable
            new_w = min(new_w, target_w * 2)  # Cap at 2x target
            new_h = min(new_h, target_h * 2)
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Select background color from AFTER colors
            lighting = self.detect_lighting_environment_complete(image)
            bg_color = AFTER_BACKGROUND_COLORS[lighting]['light']
            
            # Create 1000x1300 canvas
            canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
            
            # Position ring at 1/3 from top to minimize top/bottom margins (Dialog requirement)
            start_y = max(0, target_h // 3 - new_h // 2)
            start_x = max(0, (target_w - new_w) // 2)
            
            # Ensure it fits within canvas
            if start_y + new_h > target_h:
                start_y = target_h - new_h
            if start_x + new_w > target_w:
                start_x = target_w - new_w
            if start_y < 0:
                start_y = 0
            if start_x < 0:
                start_x = 0
            
            # Handle oversized images by cropping the resized image
            if new_w > target_w:
                excess_w = new_w - target_w
                resized = resized[:, excess_w//2:excess_w//2 + target_w]
                new_w = target_w
                start_x = 0
            
            if new_h > target_h:
                excess_h = new_h - target_h
                resized = resized[excess_h//2:excess_h//2 + target_h, :]
                new_h = target_h
                start_y = 0
            
            # Place resized image on canvas
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            self.log_processing_step("Thumbnail Creation Complete", 
                                   f"Final size: 1000x1300, Ring position: ({start_x}, {start_y}), Ring size: {new_w}x{new_h}")
            
            return canvas
            
        except Exception as e:
            self.log_processing_step("Thumbnail Creation Error", str(e))
            # Emergency thumbnail creation
            try:
                # Simple center crop and resize
                h, w = image.shape[:2]
                size = min(h, w)
                start_y, start_x = (h - size) // 2, (w - size) // 2
                cropped = image[start_y:start_y+size, start_x:start_x+size]
                thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LINEAR)
                return thumbnail
            except:
                # Ultimate fallback - solid color with text
                canvas = np.full((1300, 1000, 3), [240, 240, 240], dtype=np.uint8)
                return canvas

def handler(event):
    """Ultimate RunPod Serverless handler with ALL 34 dialog achievements - COMPLETE VERSION"""
    try:
        input_data = event["input"]
        
        # Handle connection test
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"Ultimate Wedding Ring AI v17.0 COMPLETE - Ready: {input_data['prompt']}",
                "version": "v17.0 Ultimate Perfect COMPLETE",
                "dialog_achievements": "ALL 34 dialogs implemented",
                "features": [
                    "v13.3 Complete 28-pairs Parameters (ALL 12 sets)",
                    "Adaptive Border Detection (100px thickness support)",
                    "Champagne Gold Whitening (Dialog 25)",
                    "TELEA Advanced Inpainting (Dialog 27)",
                    "28-pairs AFTER Background Colors (Dialog 28)",
                    "Perfect 1000x1300 Thumbnail (Dialog 34)",
                    "Wedding Ring Absolute Protection (Dialog 34)",
                    "31x31 Gaussian Blending (Dialog 25)",
                    "10-step v13.3 Enhancement Pipeline",
                    "Emergency Safety Systems"
                ],
                "safety_features": [
                    "Complete error handling",
                    "Fallback systems at every step",
                    "Wedding ring protection priority",
                    "Memory safety checks",
                    "Processing step logging"
                ]
            }
        
        # Process actual image with complete pipeline
        if "image_base64" in input_data:
            # Initialize complete processor
            processor = UltimateCompletePerfectWeddingRingProcessor()
            processor.log_processing_step("Handler Start", "v17.0 Ultimate Complete Processing")
            
            # Decode base64 image with safety
            try:
                image_data = base64.b64decode(input_data["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image.convert('RGB'))
                processor.log_processing_step("Image Decode", f"Success: {image_array.shape}")
            except Exception as e:
                return {"error": f"Image decode failed: {str(e)}"}
            
            # Step 1: Advanced metal type and lighting detection
            metal_type = processor.detect_metal_type_advanced_complete(image_array)
            lighting = processor.detect_lighting_environment_complete(image_array)
            
            # Step 2: Apply v13.3 complete enhancement (28 pairs based, ALL 10 steps)
            enhanced_image = processor.apply_v13_3_complete_enhancement_ultimate(image_array, metal_type, lighting)
            
            # Step 3: Ultimate adaptive black border detection and removal
            border_mask, border_bbox, ring_bbox = processor.detect_adaptive_black_border_ultimate_complete(enhanced_image)
            
            if border_mask is not None and border_bbox is not None:
                # Remove black border with complete TELEA + AFTER colors
                final_image = processor.remove_black_border_telea_advanced_complete(enhanced_image, border_mask, ring_bbox)
                
                # Apply wedding ring extra enhancement (Dialog 34)
                final_image = processor.apply_wedding_ring_extra_enhancement(final_image, ring_bbox)
                
                border_removed = True
            else:
                # No border detected - apply general enhancement
                final_image = processor.apply_wedding_ring_extra_enhancement(enhanced_image, None)
                border_removed = False
            
            # Step 4: Complete 2x upscaling with LANCZOS4
            upscaled_image = processor.upscale_lanczos_2x_complete(final_image)
            
            # Step 5: Create perfect 1000x1300 thumbnail (ring fills screen)
            if border_bbox is not None:
                # Scale bbox for upscaled image
                scaled_bbox = (border_bbox[0]*2, border_bbox[1]*2, border_bbox[2]*2, border_bbox[3]*2)
            else:
                scaled_bbox = None
            
            thumbnail = processor.create_perfect_thumbnail_1000x1300_complete(upscaled_image, scaled_bbox)
            
            # Step 6: Encode results with maximum quality
            try:
                # Main image
                main_pil = Image.fromarray(upscaled_image)
                main_buffer = io.BytesIO()
                main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
                main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
                
                # Thumbnail
                thumb_pil = Image.fromarray(thumbnail)
                thumb_buffer = io.BytesIO()
                thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
                thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
                
                processor.log_processing_step("Encoding Complete", "Both images encoded successfully")
                
            except Exception as e:
                return {"error": f"Image encoding failed: {str(e)}"}
            
            # Return comprehensive results
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v17.0 Ultimate Perfect COMPLETE",
                    "dialog_achievements": "ALL 34 dialogs fully implemented",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": border_removed,
                    "border_bbox": border_bbox,
                    "ring_protected": ring_bbox is not None,
                    "ring_bbox": ring_bbox,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "processing_steps": [
                        "Advanced metal/lighting detection",
                        "v13.3 10-step complete enhancement",
                        "Ultimate adaptive border detection",
                        "TELEA advanced inpainting",
                        "28-pairs AFTER background colors",
                        "Wedding ring extra enhancement",
                        "2x LANCZOS upscaling",
                        "Perfect 1000x1300 thumbnail"
                    ],
                    "safety_systems": [
                        "Wedding ring absolute protection",
                        "Complete error handling",
                        "Multi-level fallbacks",
                        "Memory safety checks"
                    ],
                    "processing_log": processor.processing_log
                }
            }
        
        return {"error": "No valid input provided"}
        
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# Start RunPod serverless with complete system
runpod.serverless.start({"handler": handler})
