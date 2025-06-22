import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
import io
import os
import json
import re

# Import Replicate only when available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("[v150] Replicate not available")

class WeddingRingEnhancer:
    """v150 Wedding Ring Enhancement System - Complete Version"""
    
    def __init__(self):
        print("[v150] Initializing Wedding Ring Enhancer - Full Version")
        self.replicate_client = None
        
        # 38 pairs learning data parameters (28 + 10 additional)
        self.enhancement_params = {
            'yellow_gold': {
                'brightness': 1.15,
                'contrast': 1.08,
                'sharpness': 1.10,
                'saturation': 1.06,
                'white_overlay': 0.03,
                'temperature': 1.02,
                'clahe_limit': 2.5,
                'gamma': 0.98,
                'blend_original': 0.15,
                'h_shift': -2,
                's_mult': 0.95,
                'v_mult': 1.05,
                'warmth': 1.03,
                'target_rgb': (255, 235, 190)
            },
            'rose_gold': {
                'brightness': 1.12,
                'contrast': 1.10,
                'sharpness': 1.12,
                'saturation': 1.05,
                'white_overlay': 0.02,
                'temperature': 1.00,
                'clahe_limit': 2.8,
                'gamma': 0.97,
                'blend_original': 0.12,
                'h_shift': 0,
                's_mult': 0.92,
                'v_mult': 1.06,
                'warmth': 1.01,
                'target_rgb': (245, 220, 200)
            },
            'white_gold': {
                'brightness': 1.18,
                'contrast': 1.06,
                'sharpness': 1.08,
                'saturation': 0.98,
                'white_overlay': 0.08,
                'temperature': 0.98,
                'clahe_limit': 2.2,
                'gamma': 0.96,
                'blend_original': 0.10,
                'h_shift': 0,
                's_mult': 0.85,
                'v_mult': 1.08,
                'warmth': 0.99,
                'target_rgb': (250, 250, 245)
            },
            'plain_white': {
                'brightness': 1.30,
                'contrast': 1.04,
                'sharpness': 1.05,
                'saturation': 0.95,
                'white_overlay': 0.15,
                'temperature': 0.97,
                'clahe_limit': 2.0,
                'gamma': 0.94,
                'blend_original': 0.08,
                'h_shift': 2,
                's_mult': 0.80,
                'v_mult': 1.10,
                'warmth': 1.02,
                'target_rgb': (252, 247, 232)
            }
        }
        
        # 28 pairs AFTER background colors
        self.after_bg_colors = {
            'yellow_gold': np.array([248, 243, 238]),
            'rose_gold': np.array([245, 240, 235]),
            'white_gold': np.array([250, 248, 245]),
            'plain_white': np.array([252, 250, 248])
        }

    def _init_replicate_client(self):
        """Initialize Replicate client only when needed"""
        if self.replicate_client is None and REPLICATE_AVAILABLE:
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if api_token:
                try:
                    print("[v150] Initializing Replicate client...")
                    self.replicate_client = replicate.Client(api_token=api_token)
                    print("[v150] Replicate client initialized successfully")
                except Exception as e:
                    print(f"[v150] Failed to initialize Replicate client: {e}")
                    self.replicate_client = None
            else:
                print("[v150] No REPLICATE_API_TOKEN found")

    def decode_base64_image(self, base64_string):
        """Decode base64 image with comprehensive error handling"""
        print(f"[v150] Decoding base64 image, length: {len(base64_string)}")
        
        try:
            # Clean the string
            base64_string = base64_string.strip()
            
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                print("[v150] Removed data URL prefix")
            
            # Try multiple decoding approaches
            image_data = None
            
            # Method 1: Direct decode
            try:
                image_data = base64.b64decode(base64_string, validate=True)
                print("[v150] Direct decode successful")
            except Exception as e1:
                print(f"[v150] Direct decode failed: {e1}")
                
                # Method 2: Add padding
                try:
                    missing_padding = len(base64_string) % 4
                    if missing_padding:
                        base64_string += '=' * (4 - missing_padding)
                        print(f"[v150] Added {4 - missing_padding} padding characters")
                    image_data = base64.b64decode(base64_string)
                    print("[v150] Decode with padding successful")
                except Exception as e2:
                    print(f"[v150] Decode with padding failed: {e2}")
                    
                    # Method 3: Force decode
                    try:
                        image_data = base64.b64decode(base64_string + '==')
                        print("[v150] Force decode successful")
                    except Exception as e3:
                        print(f"[v150] Force decode failed: {e3}")
                        
                        # Method 4: Clean and retry
                        base64_string = re.sub(r'[^A-Za-z0-9+/]', '', base64_string)
                        image_data = base64.b64decode(base64_string + '==')
                        print("[v150] Clean and decode successful")
            
            if image_data:
                image = Image.open(io.BytesIO(image_data))
                print(f"[v150] Image decoded successfully: {image.size}")
                return image
            else:
                raise ValueError("All decoding methods failed")
                
        except Exception as e:
            print(f"[v150] Error decoding base64: {e}")
            raise

    def detect_and_remove_black_borders(self, image_np):
        """Detect and remove black borders with adaptive thickness up to 200 pixels"""
        print("[v150] Starting adaptive black border detection (up to 200px)")
        
        h, w = image_np.shape[:2]
        
        # Multiple threshold values for different black levels
        thresholds = [30, 40, 50, 60, 70]
        max_border = 200  # Maximum border thickness to check
        
        for threshold in thresholds:
            # Create mask for dark pixels
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            mask = gray < threshold
            
            # Check each edge
            for edge in ['top', 'bottom', 'left', 'right']:
                if edge == 'top':
                    for i in range(min(max_border, h//3)):
                        if np.mean(mask[i, :]) > 0.8:  # 80% black pixels
                            image_np[:i+1, :] = self.after_bg_colors.get('white_gold', [250, 248, 245])
                        else:
                            break
                
                elif edge == 'bottom':
                    for i in range(min(max_border, h//3)):
                        if np.mean(mask[h-i-1, :]) > 0.8:
                            image_np[h-i-1:, :] = self.after_bg_colors.get('white_gold', [250, 248, 245])
                        else:
                            break
                
                elif edge == 'left':
                    for i in range(min(max_border, w//3)):
                        if np.mean(mask[:, i]) > 0.8:
                            image_np[:, :i+1] = self.after_bg_colors.get('white_gold', [250, 248, 245])
                        else:
                            break
                
                elif edge == 'right':
                    for i in range(min(max_border, w//3)):
                        if np.mean(mask[:, w-i-1]) > 0.8:
                            image_np[:, w-i-1:] = self.after_bg_colors.get('white_gold', [250, 248, 245])
                        else:
                            break
        
        print("[v150] Black border removal completed")
        return image_np

    def detect_masking_ultra_advanced(self, image_np):
        """Ultra-advanced masking detection with multiple methods"""
        h, w = image_np.shape[:2]
        print(f"[v150] Starting ultra-advanced masking detection on {w}x{h} image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Central region gray box detection (30-70% of image)
        center_x_start = int(w * 0.3)
        center_x_end = int(w * 0.7)
        center_y_start = int(h * 0.3)
        center_y_end = int(h * 0.7)
        
        center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        
        # Check for uniform gray areas
        gray_mask = ((center_region > 100) & (center_region < 170)).astype(np.uint8) * 255
        
        # Method 2: Standard deviation check for uniform areas
        local_std = cv2.GaussianBlur(gray, (31, 31), 0)
        std_map = np.abs(gray.astype(float) - local_std.astype(float))
        uniform_mask = std_map < 10  # Very uniform areas
        
        # Method 3: Color uniformity check
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        color_std = np.std(lab, axis=2)
        color_uniform = color_std < 15
        
        # Combine all methods
        combined_mask = np.zeros_like(gray, dtype=bool)
        
        # Add gray detection
        gray_full = ((gray > 100) & (gray < 170))
        combined_mask = combined_mask | gray_full
        
        # Add uniform areas
        combined_mask = combined_mask | (uniform_mask & color_uniform)
        
        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Get bounding box
            x = stats[largest_component, cv2.CC_STAT_LEFT]
            y = stats[largest_component, cv2.CC_STAT_TOP]
            w_box = stats[largest_component, cv2.CC_STAT_WIDTH]
            h_box = stats[largest_component, cv2.CC_STAT_HEIGHT]
            area = stats[largest_component, cv2.CC_STAT_AREA]
            
            # Check if significant (>5% of image)
            if area > (w * h * 0.05):
                print(f"[v150] Advanced masking detected at ({x}, {y}), size: {w_box}x{h_box}")
                
                return {
                    'has_masking': True,
                    'type': 'central_box',
                    'bounds': {
                        'x': x,
                        'y': y,
                        'width': w_box,
                        'height': h_box
                    }
                }
        
        # Method 4: Specific gray value detection
        gray_values = [128, 140, 150, 160]
        for gray_val in gray_values:
            mask = np.abs(gray.astype(float) - gray_val) < 15
            if np.sum(mask) > (w * h * 0.1):
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = np.min(coords[0]), np.max(coords[0])
                    x_min, x_max = np.min(coords[1]), np.max(coords[1])
                    
                    print(f"[v150] Gray value {gray_val} masking detected")
                    return {
                        'has_masking': True,
                        'type': 'central_box',
                        'bounds': {
                            'x': x_min,
                            'y': y_min,
                            'width': x_max - x_min,
                            'height': y_max - y_min
                        }
                    }
        
        print("[v150] No masking detected")
        return {'has_masking': False, 'type': None}

    def remove_masking_with_replicate(self, image, masking_info):
        """Remove masking using Replicate API"""
        print("[v150] Starting masking removal with Replicate")
        
        try:
            # Initialize Replicate client if needed
            self._init_replicate_client()
            
            if not self.replicate_client:
                print("[v150] Replicate client not available, returning original")
                return image
            
            # Convert to base64 for Replicate
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"
            
            # Create mask based on masking bounds
            mask = Image.new('L', image.size, 0)
            bounds = masking_info['bounds']
            
            # Draw white rectangle on mask where masking is
            draw = ImageDraw.Draw(mask)
            draw.rectangle([
                bounds['x'], 
                bounds['y'], 
                bounds['x'] + bounds['width'],
                bounds['y'] + bounds['height']
            ], fill=255)
            
            # Add some dilation to mask for better results
            mask_np = np.array(mask)
            kernel = np.ones((5,5), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=2)
            mask = Image.fromarray(mask_np)
            
            # Convert mask to base64
            mask_buffered = io.BytesIO()
            mask.save(mask_buffered, format="PNG")
            mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode('utf-8')
            mask_data_url = f"data:image/png;base64,{mask_base64}"
            
            print("[v150] Running Replicate background removal...")
            
            # Run Replicate
            output = self.replicate_client.run(
                "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
                input={
                    "image": img_data_url,
                    "mask": mask_data_url,
                    "alpha_matting": True,
                    "alpha_matting_foreground_threshold": 240,
                    "alpha_matting_background_threshold": 50,
                    "alpha_matting_erode_size": 10
                }
            )
            
            if output:
                print("[v150] Replicate processing successful")
                # Decode result
                result_base64 = output.split(',')[1] if ',' in output else output
                result_data = base64.b64decode(result_base64)
                return Image.open(io.BytesIO(result_data))
            
        except Exception as e:
            print(f"[v150] Replicate processing failed: {e}")
        
        print("[v150] Returning original image")
        return image

    def detect_metal_type(self, image_np):
        """Detect wedding ring metal type with improved accuracy"""
        print("[v150] Detecting metal type...")
        
        # Sample center region
        h, w = image_np.shape[:2]
        center_y, center_x = h // 2, w // 2
        sample_size = min(w, h) // 4
        
        roi = image_np[
            max(0, center_y - sample_size):min(h, center_y + sample_size),
            max(0, center_x - sample_size):min(w, center_x + sample_size)
        ]
        
        # Calculate color metrics
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        # HSV analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        h, s, v = avg_hsv
        
        # LAB analysis for better color distinction
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        avg_lab = np.mean(lab_roi.reshape(-1, 3), axis=0)
        l, a, b_lab = avg_lab
        
        # Gold detection
        gold_score = (r - b) / (r + b + 1e-5)
        warm_score = (r + g) / (2 * b + 1e-5)
        
        print(f"[v150] Color analysis - RGB: {avg_color}, HSV: {avg_hsv}, LAB: {avg_lab}")
        print(f"[v150] Gold score: {gold_score:.3f}, Warm score: {warm_score:.3f}")
        
        # Determine metal type with improved logic
        if gold_score > 0.15 and warm_score > 1.8 and a > 5:
            metal_type = 'yellow_gold'
        elif gold_score > 0.05 and s > 20 and 10 < h < 25 and a > 10:
            metal_type = 'rose_gold'
        elif s < 30 and v > 180 and abs(r - g) < 10 and abs(g - b) < 10 and l > 200:
            metal_type = 'plain_white'
        else:
            metal_type = 'white_gold'
        
        print(f"[v150] Detected metal type: {metal_type}")
        return metal_type

    def enhance_wedding_ring(self, image, metal_type):
        """Apply 10-step enhancement based on 38 pairs learning data"""
        print(f"[v150] Enhancing {metal_type} wedding ring with 10-step process")
        
        params = self.enhancement_params[metal_type]
        enhanced = image.copy()
        
        # Step 1: Brightness adjustment
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(params['brightness'])
        
        # Step 2: Contrast adjustment
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # Step 3: Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # Step 4: Saturation adjustment
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(params['saturation'])
        
        # Step 5: White overlay
        if params['white_overlay'] > 0:
            white = Image.new('RGB', enhanced.size, (255, 255, 255))
            enhanced = Image.blend(enhanced, white, params['white_overlay'])
        
        # Step 6: Temperature adjustment in LAB
        enhanced_np = np.array(enhanced)
        lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB).astype(float)
        lab[:,:,1] = lab[:,:,1] * params['temperature']  # a channel
        lab[:,:,2] = lab[:,:,2] * params['warmth']       # b channel
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        enhanced_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Step 7: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=params['clahe_limit'], tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Step 8: Gamma correction
        gamma_corrected = np.power(enhanced_np / 255.0, params['gamma']) * 255
        enhanced_np = gamma_corrected.astype(np.uint8)
        
        # Step 9: HSV fine-tuning
        hsv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2HSV).astype(float)
        hsv[:,:,0] = (hsv[:,:,0] + params['h_shift']) % 180
        hsv[:,:,1] = hsv[:,:,1] * params['s_mult']
        hsv[:,:,2] = hsv[:,:,2] * params['v_mult']
        hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
        enhanced_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Step 10: Blend with original
        enhanced = Image.fromarray(enhanced_np)
        final = Image.blend(enhanced, image, params['blend_original'])
        
        print("[v150] 10-step enhancement completed")
        return final

    def apply_natural_blending(self, image_np, metal_type):
        """Apply natural blending with 31x31 Gaussian blur"""
        print("[v150] Applying natural blending with 31x31 Gaussian")
        
        h, w = image_np.shape[:2]
        
        # Create smooth transition mask
        mask = np.ones((h, w), dtype=np.float32)
        
        # Reduce mask at edges
        edge_width = 50
        for i in range(edge_width):
            alpha = i / edge_width
            mask[i, :] *= alpha
            mask[h-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, w-i-1] *= alpha
        
        # Apply 31x31 Gaussian blur for natural transition
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        # Get background color
        bg_color = self.after_bg_colors[metal_type]
        background = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        # Blend
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = image_np * mask_3d + background * (1 - mask_3d)
        
        print("[v150] Natural blending completed")
        return result.astype(np.uint8)

    def create_perfect_thumbnail(self, image, masking_info=None):
        """Create 1000x1300 thumbnail with ring perfectly centered and filling the frame"""
        print("[v150] Creating perfect 1000x1300 thumbnail")
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # If masking detected, use masking bounds as reference
        if masking_info and masking_info.get('has_masking'):
            bounds = masking_info['bounds']
            ring_x = bounds['x']
            ring_y = bounds['y']
            ring_w = bounds['width']
            ring_h = bounds['height']
            
            print(f"[v150] Using masking bounds for thumbnail: {ring_w}x{ring_h}")
        else:
            # Try to detect ring using multiple methods
            # Method 1: Edge detection
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate edges to connect components
            kernel = np.ones((5,5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find contour closest to center
                center_x, center_y = w // 2, h // 2
                best_contour = None
                min_dist = float('inf')
                
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        
                        if dist < min_dist and cv2.contourArea(contour) > 1000:
                            min_dist = dist
                            best_contour = contour
                
                if best_contour is not None:
                    ring_x, ring_y, ring_w, ring_h = cv2.boundingRect(best_contour)
                    print(f"[v150] Detected ring bounds: {ring_w}x{ring_h}")
                else:
                    # Use center 50% as fallback
                    ring_x = int(w * 0.25)
                    ring_y = int(h * 0.25)
                    ring_w = int(w * 0.5)
                    ring_h = int(h * 0.5)
                    print("[v150] Using center region as fallback")
            else:
                # Use center 50% as fallback
                ring_x = int(w * 0.25)
                ring_y = int(h * 0.25)
                ring_w = int(w * 0.5)
                ring_h = int(h * 0.5)
                print("[v150] No contours found, using center region")
        
        # Calculate center of ring
        ring_center_x = ring_x + ring_w // 2
        ring_center_y = ring_y + ring_h // 2
        
        # Target size: 1000x1300
        target_w = 1000
        target_h = 1300
        target_ratio = target_w / target_h  # 0.769
        
        # Ring should fill 90% of the frame (maximum fill)
        ring_scale = 0.90
        
        # Calculate required crop size to achieve target ratio
        # and ensure ring fills the desired portion
        required_w = int(ring_w / ring_scale)
        required_h = int(ring_h / ring_scale)
        
        # Adjust to match target aspect ratio
        current_ratio = required_w / required_h
        if current_ratio > target_ratio:
            # Too wide, increase height
            required_h = int(required_w / target_ratio)
        else:
            # Too tall, increase width
            required_w = int(required_h * target_ratio)
        
        # Ensure minimum size
        required_w = max(required_w, target_w)
        required_h = max(required_h, target_h)
        
        # Calculate crop region centered on ring
        crop_x1 = ring_center_x - required_w // 2
        crop_y1 = ring_center_y - required_h // 2
        crop_x2 = crop_x1 + required_w
        crop_y2 = crop_y1 + required_h
        
        # Adjust if crop goes out of bounds
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > w:
            crop_x1 -= (crop_x2 - w)
            crop_x2 = w
        if crop_y2 > h:
            crop_y1 -= (crop_y2 - h)
            crop_y2 = h
        
        # Final safety check
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)
        
        # Crop the image
        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize to exact 1000x1300
        thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"[v150] Created thumbnail: crop region ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
        print(f"[v150] Ring fills approximately {int(ring_scale * 100)}% of frame")
        
        return Image.fromarray(thumbnail)

    def process_image(self, base64_image):
        """Main processing pipeline"""
        try:
            print("[v150] Starting image processing - Complete Version")
            
            # Decode image
            image = self.decode_base64_image(base64_image)
            original_image = image.copy()
            
            # Convert to numpy
            image_np = np.array(image)
            
            # Remove black borders first (adaptive up to 200px)
            image_np = self.detect_and_remove_black_borders(image_np)
            
            # Ultra-advanced masking detection
            masking_info = self.detect_masking_ultra_advanced(image_np)
            
            # Remove masking if detected
            if masking_info['has_masking'] and masking_info['type'] == 'central_box':
                print("[v150] Central box masking detected - removing with Replicate")
                image = self.remove_masking_with_replicate(Image.fromarray(image_np), masking_info)
                image_np = np.array(image)
            
            # Detect metal type
            metal_type = self.detect_metal_type(image_np)
            
            # Apply AFTER background color
            if metal_type in self.after_bg_colors:
                bg_color = self.after_bg_colors[metal_type]
                print(f"[v150] Applying AFTER background color for {metal_type}: {bg_color}")
                
                # Apply natural blending with 31x31 Gaussian
                image_np = self.apply_natural_blending(image_np, metal_type)
                image = Image.fromarray(image_np)
            
            # Enhance wedding ring (10-step process)
            enhanced = self.enhance_wedding_ring(image, metal_type)
            
            # Create perfect thumbnail (masking_info helps with positioning)
            thumbnail = self.create_perfect_thumbnail(enhanced, masking_info)
            
            # Convert to base64 (without padding for Make.com)
            # Enhanced image
            buffer = io.BytesIO()
            enhanced.save(buffer, format='PNG', optimize=True)
            enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            enhanced_base64 = enhanced_base64.rstrip('=')  # Remove padding
            
            # Thumbnail
            buffer_thumb = io.BytesIO()
            thumbnail.save(buffer_thumb, format='PNG', optimize=True)
            thumbnail_base64 = base64.b64encode(buffer_thumb.getvalue()).decode('utf-8')
            thumbnail_base64 = thumbnail_base64.rstrip('=')  # Remove padding
            
            print("[v150] Processing completed successfully")
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": "v150",
                    "metal_type": metal_type,
                    "masking_detected": masking_info['has_masking'],
                    "masking_type": masking_info.get('type'),
                    "enhancement_applied": True,
                    "thumbnail_size": "1000x1300",
                    "after_bg_applied": True,
                    "black_border_removed": True,
                    "natural_blending": True,
                    "adaptive_thickness": "200px"
                }
            }
            
        except Exception as e:
            print(f"[v150] Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return original on error
            try:
                buffer = io.BytesIO()
                original_image.save(buffer, format='PNG')
                original_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
                
                # Simple center crop for thumbnail
                w, h = original_image.size
                aspect = 1000 / 1300
                if w / h > aspect:
                    new_w = int(h * aspect)
                    left = (w - new_w) // 2
                    thumb = original_image.crop((left, 0, left + new_w, h))
                else:
                    new_h = int(w / aspect)
                    top = (h - new_h) // 2
                    thumb = original_image.crop((0, top, w, top + new_h))
                
                thumb = thumb.resize((1000, 1300), Image.Resampling.LANCZOS)
                
                buffer_thumb = io.BytesIO()
                thumb.save(buffer_thumb, format='PNG')
                thumb_base64 = base64.b64encode(buffer_thumb.getvalue()).decode('utf-8').rstrip('=')
                
                return {
                    "enhanced_image": original_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "version": "v150",
                        "error": str(e),
                        "fallback": True
                    }
                }
            except:
                return {
                    "enhanced_image": base64_image.rstrip('='),
                    "thumbnail": base64_image.rstrip('='),
                    "processing_info": {
                        "version": "v150",
                        "error": "Critical error",
                        "fallback": True
                    }
                }

def handler(event):
    """RunPod handler function"""
    try:
        print("[v150] Handler started - Complete Version")
        print(f"[v150] Event type: {type(event)}")
        print(f"[v150] Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
        
        # Try multiple paths to find the image
        base64_image = None
        
        # Path 1: event["input"]["image"]
        if isinstance(event, dict) and "input" in event:
            job_input = event.get("input", {})
            print(f"[v150] Job input type: {type(job_input)}")
            print(f"[v150] Job input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
            
            # Debug: print first 100 chars of each value
            if isinstance(job_input, dict):
                for key, value in job_input.items():
                    if isinstance(value, str):
                        print(f"[v150] {key}: {value[:100]}...")
                    else:
                        print(f"[v150] {key}: {type(value)}")
            
            base64_image = job_input.get("image", "") if isinstance(job_input, dict) else ""
        
        # Path 2: event["image"]
        if not base64_image and isinstance(event, dict):
            base64_image = event.get("image", "")
            if base64_image:
                print("[v150] Found image in event['image']")
        
        # Path 3: Direct string
        if not base64_image and isinstance(event, str):
            base64_image = event
            print("[v150] Event is direct base64 string")
        
        # Path 4: Nested in data
        if not base64_image and isinstance(event, dict) and "data" in event:
            data = event.get("data", {})
            if isinstance(data, dict) and "image" in data:
                base64_image = data.get("image", "")
                print("[v150] Found image in event['data']['image']")
        
        if not base64_image:
            print("[v150] No image found in any expected location")
            print(f"[v150] Full event structure: {json.dumps(event, indent=2) if isinstance(event, dict) else str(event)[:500]}")
            return {
                "output": {
                    "error": "No image provided",
                    "processing_info": {"version": "v150", "error": "No input image"}
                }
            }
        
        print(f"[v150] Received image, length: {len(base64_image)}")
        
        # Process image
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(base64_image)
        
        # Return with proper structure for Make.com
        return {
            "output": result
        }
        
    except Exception as e:
        print(f"[v150] Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "processing_info": {"version": "v150", "error": "Handler exception"}
            }
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print("[v150] Starting RunPod serverless handler - Complete Version")
    runpod.serverless.start({"handler": handler})
