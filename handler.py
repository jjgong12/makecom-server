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
    print("[v151] Replicate not available")

class WeddingRingEnhancer:
    """v151 Wedding Ring Enhancement System - Advanced Detection & Natural Processing"""
    
    def __init__(self):
        print("[v151] Initializing Wedding Ring Enhancer - Advanced Version")
        self.replicate_client = None
        
        # 38 pairs learning data parameters (28 + 10 additional) - ENHANCED for v151
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
            'plain_white': {  # Enhanced for v151 - more white
                'brightness': 1.35,  # Increased from 1.30
                'contrast': 1.02,    # Reduced for softer look
                'sharpness': 1.05,
                'saturation': 0.90,  # More desaturated
                'white_overlay': 0.20,  # Increased from 0.15
                'temperature': 0.95,    # Cooler
                'clahe_limit': 2.0,
                'gamma': 0.92,
                'blend_original': 0.05,  # Less original
                'h_shift': 2,
                's_mult': 0.75,  # More desaturated
                'v_mult': 1.15,  # Brighter
                'warmth': 0.98,
                'target_rgb': (255, 253, 250)  # More white
            }
        }
        
        # 28 pairs AFTER background colors - more natural for v151
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
                    print("[v151] Initializing Replicate client...")
                    self.replicate_client = replicate.Client(api_token=api_token)
                    print("[v151] Replicate client initialized successfully")
                except Exception as e:
                    print(f"[v151] Failed to initialize Replicate client: {e}")
                    self.replicate_client = None
            else:
                print("[v151] No REPLICATE_API_TOKEN found")

    def decode_base64_image(self, base64_string):
        """Decode base64 image with comprehensive error handling for Make.com"""
        print(f"[v151] Decoding base64 image, length: {len(base64_string)}")
        
        try:
            # Clean the string
            base64_string = base64_string.strip()
            
            # Remove data URL prefix if present
            if 'base64,' in base64_string:
                base64_string = base64_string.split('base64,')[1]
                print("[v151] Removed data URL prefix")
            elif base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]
                print("[v151] Removed data: prefix")
            
            # Remove any whitespace, newlines
            base64_string = base64_string.replace('\n', '').replace('\r', '').replace(' ', '')
            
            # Try multiple decoding approaches
            image_data = None
            
            # Method 1: Direct decode (no padding manipulation for Make.com data)
            try:
                image_data = base64.b64decode(base64_string, validate=True)
                print("[v151] Direct decode successful")
            except Exception as e1:
                print(f"[v151] Direct decode failed: {e1}")
                
                # Method 2: Add padding only if needed
                try:
                    missing_padding = len(base64_string) % 4
                    if missing_padding:
                        base64_string += '=' * (4 - missing_padding)
                        print(f"[v151] Added {4 - missing_padding} padding characters")
                    image_data = base64.b64decode(base64_string)
                    print("[v151] Decode with padding successful")
                except Exception as e2:
                    print(f"[v151] Decode with padding failed: {e2}")
                    
                    # Method 3: Clean non-base64 characters and retry
                    try:
                        base64_string = re.sub(r'[^A-Za-z0-9+/]', '', base64_string)
                        missing_padding = len(base64_string) % 4
                        if missing_padding:
                            base64_string += '=' * (4 - missing_padding)
                        image_data = base64.b64decode(base64_string)
                        print("[v151] Clean and decode successful")
                    except Exception as e3:
                        print(f"[v151] All decode methods failed: {e3}")
                        raise ValueError("Unable to decode base64 image")
            
            if image_data:
                image = Image.open(io.BytesIO(image_data))
                print(f"[v151] Image decoded successfully: {image.size}, mode: {image.mode}")
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    print(f"[v151] Converted to RGB from {image.mode}")
                
                return image
            else:
                raise ValueError("No image data after decoding")
                
        except Exception as e:
            print(f"[v151] Error decoding base64: {e}")
            print(f"[v151] Base64 preview: {base64_string[:100]}...")
            raise

    def detect_and_remove_black_borders_natural(self, image_np, metal_type):
        """Detect and remove black borders with natural gradient transition"""
        print("[v151] Starting natural black border detection and removal")
        
        h, w = image_np.shape[:2]
        result = image_np.copy()
        
        # Get appropriate background color for detected metal type
        bg_color = self.after_bg_colors.get(metal_type, np.array([250, 248, 245]))
        
        # Multiple threshold values for different black levels
        thresholds = [30, 40, 50, 60, 70]
        max_border = 200  # Maximum border thickness to check
        
        # Track border sizes
        borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        for threshold in thresholds:
            # Create mask for dark pixels
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            mask = gray < threshold
            
            # Check each edge and find actual border size
            # Top edge
            for i in range(min(max_border, h//3)):
                if np.mean(mask[i, :]) > 0.8:  # 80% black pixels
                    borders['top'] = max(borders['top'], i + 1)
                else:
                    break
            
            # Bottom edge
            for i in range(min(max_border, h//3)):
                if np.mean(mask[h-i-1, :]) > 0.8:
                    borders['bottom'] = max(borders['bottom'], i + 1)
                else:
                    break
            
            # Left edge
            for i in range(min(max_border, w//3)):
                if np.mean(mask[:, i]) > 0.8:
                    borders['left'] = max(borders['left'], i + 1)
                else:
                    break
            
            # Right edge
            for i in range(min(max_border, w//3)):
                if np.mean(mask[:, w-i-1]) > 0.8:
                    borders['right'] = max(borders['right'], i + 1)
                else:
                    break
        
        # Apply natural gradient transition instead of hard replacement
        if any(borders.values()):
            print(f"[v151] Detected borders: {borders}")
            
            # Create gradient masks for each edge
            for edge, size in borders.items():
                if size > 0:
                    if edge == 'top':
                        # Create gradient from background to image
                        for i in range(size):
                            alpha = i / size  # 0 to 1
                            result[i, :] = result[i, :] * alpha + bg_color * (1 - alpha)
                    
                    elif edge == 'bottom':
                        for i in range(size):
                            alpha = i / size
                            row = h - size + i
                            result[row, :] = result[row, :] * alpha + bg_color * (1 - alpha)
                    
                    elif edge == 'left':
                        for i in range(size):
                            alpha = i / size
                            result[:, i] = result[:, i] * alpha + bg_color.reshape(1, 3) * (1 - alpha)
                    
                    elif edge == 'right':
                        for i in range(size):
                            alpha = i / size
                            col = w - size + i
                            result[:, col] = result[:, col] * alpha + bg_color.reshape(1, 3) * (1 - alpha)
        
        print("[v151] Natural black border removal completed")
        return result

    def detect_masking_ultra_advanced(self, image_np):
        """Ultra-advanced masking detection with multiple methods - v151 enhanced"""
        h, w = image_np.shape[:2]
        print(f"[v151] Starting ultra-advanced masking detection on {w}x{h} image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Central region gray box detection (25-75% of image for better coverage)
        center_x_start = int(w * 0.25)
        center_x_end = int(w * 0.75)
        center_y_start = int(h * 0.25)
        center_y_end = int(h * 0.75)
        
        center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        
        # Enhanced gray detection with wider range
        gray_ranges = [
            (90, 180),   # Wide gray range
            (120, 160),  # Common masking gray
            (140, 145),  # Specific gray value
        ]
        
        for min_val, max_val in gray_ranges:
            gray_mask = ((center_region > min_val) & (center_region < max_val)).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                center_area = (center_x_end - center_x_start) * (center_y_end - center_y_start)
                
                # Lower threshold for detection (15% instead of 20%)
                if area > center_area * 0.15:
                    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                    
                    # Convert to full image coordinates
                    mask_x = center_x_start + x
                    mask_y = center_y_start + y
                    
                    print(f"[v151] Central box masking detected at ({mask_x}, {mask_y}), size: {w_box}x{h_box}")
                    
                    return {
                        'has_masking': True,
                        'type': 'central_box',
                        'bounds': {
                            'x': mask_x,
                            'y': mask_y,
                            'width': w_box,
                            'height': h_box
                        }
                    }
        
        # Method 2: Standard deviation check for uniform areas
        local_std = cv2.GaussianBlur(gray, (31, 31), 0)
        std_map = np.abs(gray.astype(float) - local_std.astype(float))
        uniform_mask = std_map < 15  # Increased threshold
        
        # Method 3: Color uniformity check
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        color_std = np.std(lab, axis=2)
        color_uniform = color_std < 20  # Increased threshold
        
        # Combine methods
        combined_mask = uniform_mask & color_uniform
        
        # Find largest uniform area
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
            
            # Check if significant (>3% of image for better detection)
            if area > (w * h * 0.03):
                print(f"[v151] Uniform area masking detected at ({x}, {y}), size: {w_box}x{h_box}")
                
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
        
        print("[v151] No masking detected")
        return {'has_masking': False, 'type': None}

    def remove_masking_with_replicate(self, image, masking_info):
        """Remove masking using Replicate API"""
        print("[v151] Starting masking removal with Replicate")
        
        try:
            # Initialize Replicate client if needed
            self._init_replicate_client()
            
            if not self.replicate_client:
                print("[v151] Replicate client not available, returning original")
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
            kernel = np.ones((7,7), np.uint8)  # Larger kernel
            mask_np = cv2.dilate(mask_np, kernel, iterations=3)
            mask = Image.fromarray(mask_np)
            
            # Convert mask to base64
            mask_buffered = io.BytesIO()
            mask.save(mask_buffered, format="PNG")
            mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode('utf-8')
            mask_data_url = f"data:image/png;base64,{mask_base64}"
            
            print("[v151] Running Replicate background removal...")
            
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
                print("[v151] Replicate processing successful")
                # Decode result
                result_base64 = output.split(',')[1] if ',' in output else output
                result_data = base64.b64decode(result_base64)
                return Image.open(io.BytesIO(result_data))
            
        except Exception as e:
            print(f"[v151] Replicate processing failed: {e}")
        
        print("[v151] Returning original image")
        return image

    def detect_metal_type_enhanced(self, image_np):
        """Enhanced metal type detection for v151"""
        print("[v151] Detecting metal type with enhanced algorithm...")
        
        # Sample multiple regions for better accuracy
        h, w = image_np.shape[:2]
        
        # Sample regions: center, corners, and mid-points
        sample_regions = []
        sample_size = min(w, h) // 6
        
        # Center
        center_y, center_x = h // 2, w // 2
        sample_regions.append(image_np[
            max(0, center_y - sample_size):min(h, center_y + sample_size),
            max(0, center_x - sample_size):min(w, center_x + sample_size)
        ])
        
        # Additional sampling points
        for y_ratio in [0.3, 0.7]:
            for x_ratio in [0.3, 0.7]:
                y = int(h * y_ratio)
                x = int(w * x_ratio)
                region = image_np[
                    max(0, y - sample_size//2):min(h, y + sample_size//2),
                    max(0, x - sample_size//2):min(w, x + sample_size//2)
                ]
                if region.size > 0:
                    sample_regions.append(region)
        
        # Analyze all regions
        all_metrics = []
        for region in sample_regions:
            if region.size == 0:
                continue
                
            # Calculate color metrics
            avg_color = np.mean(region.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            # HSV analysis
            hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            avg_hsv = np.mean(hsv_region.reshape(-1, 3), axis=0)
            h_val, s, v = avg_hsv
            
            # LAB analysis
            lab_region = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
            avg_lab = np.mean(lab_region.reshape(-1, 3), axis=0)
            l, a, b_lab = avg_lab
            
            # Calculate scores
            gold_score = (r - b) / (r + b + 1e-5)
            warm_score = (r + g) / (2 * b + 1e-5)
            white_score = min(r, g, b) / (max(r, g, b) + 1e-5)  # How close to gray/white
            
            all_metrics.append({
                'rgb': (r, g, b),
                'hsv': (h_val, s, v),
                'lab': (l, a, b_lab),
                'gold_score': gold_score,
                'warm_score': warm_score,
                'white_score': white_score
            })
        
        # Average all metrics
        if all_metrics:
            avg_metrics = {
                'gold_score': np.mean([m['gold_score'] for m in all_metrics]),
                'warm_score': np.mean([m['warm_score'] for m in all_metrics]),
                'white_score': np.mean([m['white_score'] for m in all_metrics]),
                'avg_s': np.mean([m['hsv'][1] for m in all_metrics]),
                'avg_v': np.mean([m['hsv'][2] for m in all_metrics]),
                'avg_a': np.mean([m['lab'][1] for m in all_metrics]),
                'avg_l': np.mean([m['lab'][0] for m in all_metrics])
            }
            
            print(f"[v151] Averaged metrics: {avg_metrics}")
            
            # Enhanced detection logic
            if avg_metrics['gold_score'] > 0.15 and avg_metrics['warm_score'] > 1.8 and avg_metrics['avg_a'] > 5:
                metal_type = 'yellow_gold'
            elif avg_metrics['gold_score'] > 0.05 and avg_metrics['avg_s'] > 20 and avg_metrics['avg_a'] > 8:
                metal_type = 'rose_gold'
            elif avg_metrics['white_score'] > 0.85 and avg_metrics['avg_s'] < 15 and avg_metrics['avg_l'] > 210:
                # Enhanced plain white detection
                metal_type = 'plain_white'
            elif avg_metrics['avg_s'] < 25 and avg_metrics['avg_v'] > 180:
                metal_type = 'white_gold'
            else:
                # Default based on brightness and saturation
                if avg_metrics['avg_l'] > 200 and avg_metrics['avg_s'] < 20:
                    metal_type = 'plain_white'
                else:
                    metal_type = 'white_gold'
        else:
            metal_type = 'white_gold'  # Default
        
        print(f"[v151] Detected metal type: {metal_type}")
        return metal_type

    def enhance_wedding_ring(self, image, metal_type):
        """Apply 10-step enhancement based on 38 pairs learning data"""
        print(f"[v151] Enhancing {metal_type} wedding ring with 10-step process")
        
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
        
        print("[v151] 10-step enhancement completed")
        return final

    def apply_natural_blending(self, image_np, metal_type):
        """Apply natural blending with 31x31 Gaussian blur"""
        print("[v151] Applying natural blending with 31x31 Gaussian")
        
        h, w = image_np.shape[:2]
        
        # Create smooth transition mask
        mask = np.ones((h, w), dtype=np.float32)
        
        # Reduce mask at edges with smoother transition
        edge_width = 60  # Wider edge for more natural blend
        for i in range(edge_width):
            alpha = (i / edge_width) ** 2  # Quadratic for smoother transition
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
        
        print("[v151] Natural blending completed")
        return result.astype(np.uint8)

    def create_perfect_thumbnail(self, image, masking_info=None):
        """Create 1000x1300 thumbnail with ring perfectly centered and filling the frame"""
        print("[v151] Creating perfect 1000x1300 thumbnail")
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # If masking detected, use masking bounds as reference
        if masking_info and masking_info.get('has_masking'):
            bounds = masking_info['bounds']
            ring_x = bounds['x']
            ring_y = bounds['y']
            ring_w = bounds['width']
            ring_h = bounds['height']
            
            print(f"[v151] Using masking bounds for thumbnail: {ring_w}x{ring_h}")
        else:
            # Enhanced ring detection for v151
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Multiple edge detection methods
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Dilate edges to connect components
            kernel = np.ones((7,7), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area and position
                center_x, center_y = w // 2, h // 2
                valid_contours = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area threshold
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # Check if contour is reasonably centered
                            if abs(cx - center_x) < w * 0.3 and abs(cy - center_y) < h * 0.3:
                                valid_contours.append(contour)
                
                if valid_contours:
                    # Combine all valid contours
                    all_points = np.vstack(valid_contours)
                    ring_x, ring_y, ring_w, ring_h = cv2.boundingRect(all_points)
                    print(f"[v151] Detected combined ring bounds: {ring_w}x{ring_h}")
                else:
                    # Use center 45% as fallback
                    ring_x = int(w * 0.275)
                    ring_y = int(h * 0.275)
                    ring_w = int(w * 0.45)
                    ring_h = int(h * 0.45)
                    print("[v151] Using center region as fallback")
            else:
                # Use center 45% as fallback
                ring_x = int(w * 0.275)
                ring_y = int(h * 0.275)
                ring_w = int(w * 0.45)
                ring_h = int(h * 0.45)
                print("[v151] No contours found, using center region")
        
        # Calculate center of ring
        ring_center_x = ring_x + ring_w // 2
        ring_center_y = ring_y + ring_h // 2
        
        # Target size: 1000x1300
        target_w = 1000
        target_h = 1300
        target_ratio = target_w / target_h  # 0.769
        
        # Ring should fill 85-90% of the frame
        ring_scale = 0.875  # Between 85-90%
        
        # Calculate required crop size
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
        
        # Resize to exact 1000x1300 with high quality
        thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"[v151] Created thumbnail: crop region ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
        print(f"[v151] Ring fills approximately {int(ring_scale * 100)}% of frame")
        
        return Image.fromarray(thumbnail)

    def process_image(self, base64_image):
        """Main processing pipeline"""
        try:
            print("[v151] Starting image processing - Advanced Version")
            
            # Decode image
            image = self.decode_base64_image(base64_image)
            original_image = image.copy()
            
            # Convert to numpy
            image_np = np.array(image)
            
            # Detect metal type first (before border removal for accuracy)
            metal_type = self.detect_metal_type_enhanced(image_np)
            
            # Remove black borders with natural gradient (using detected metal type)
            image_np = self.detect_and_remove_black_borders_natural(image_np, metal_type)
            
            # Ultra-advanced masking detection
            masking_info = self.detect_masking_ultra_advanced(image_np)
            
            # Remove masking if detected
            if masking_info['has_masking'] and masking_info['type'] == 'central_box':
                print("[v151] Central box masking detected - removing with Replicate")
                image = self.remove_masking_with_replicate(Image.fromarray(image_np), masking_info)
                image_np = np.array(image)
            
            # Apply AFTER background color with natural blending
            if metal_type in self.after_bg_colors:
                print(f"[v151] Applying natural background blending for {metal_type}")
                image_np = self.apply_natural_blending(image_np, metal_type)
                image = Image.fromarray(image_np)
            
            # Enhance wedding ring (10-step process)
            enhanced = self.enhance_wedding_ring(image, metal_type)
            
            # Create perfect thumbnail
            thumbnail = self.create_perfect_thumbnail(enhanced, masking_info)
            
            # Convert to base64 (without padding for Make.com)
            # Enhanced image
            buffer = io.BytesIO()
            enhanced.save(buffer, format='PNG', optimize=True, quality=95)
            enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            enhanced_base64 = enhanced_base64.rstrip('=')  # Remove padding
            
            # Thumbnail
            buffer_thumb = io.BytesIO()
            thumbnail.save(buffer_thumb, format='PNG', optimize=True, quality=95)
            thumbnail_base64 = base64.b64encode(buffer_thumb.getvalue()).decode('utf-8')
            thumbnail_base64 = thumbnail_base64.rstrip('=')  # Remove padding
            
            print("[v151] Processing completed successfully")
            
            return {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": "v151",
                    "metal_type": metal_type,
                    "masking_detected": masking_info['has_masking'],
                    "masking_type": masking_info.get('type'),
                    "enhancement_applied": True,
                    "thumbnail_size": "1000x1300",
                    "after_bg_applied": True,
                    "natural_border_removal": True,
                    "natural_blending": True,
                    "adaptive_thickness": "200px"
                }
            }
            
        except Exception as e:
            print(f"[v151] Error in processing: {str(e)}")
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
                        "version": "v151",
                        "error": str(e),
                        "fallback": True
                    }
                }
            except:
                return {
                    "enhanced_image": base64_image.rstrip('='),
                    "thumbnail": base64_image.rstrip('='),
                    "processing_info": {
                        "version": "v151",
                        "error": "Critical error",
                        "fallback": True
                    }
                }

def handler(event):
    """RunPod handler function"""
    try:
        print("[v151] Handler started - Advanced Version")
        print(f"[v151] Event type: {type(event)}")
        print(f"[v151] Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
        
        # Get input from event (validated structure from previous versions)
        image_input = event.get("input", {})
        print(f"[v151] Input type: {type(image_input)}")
        print(f"[v151] Input keys: {list(image_input.keys()) if isinstance(image_input, dict) else 'Not a dict'}")
        
        # Debug: print sample of values
        if isinstance(image_input, dict):
            for key, value in image_input.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"[v151] {key}: {value[:100]}... (length: {len(value)})")
                else:
                    print(f"[v151] {key}: {value if isinstance(value, str) else type(value)}")
        
        # Try to get image from multiple possible keys - Enhanced for v151
        base64_image = None
        
        # Check input dictionary first
        if isinstance(image_input, dict):
            for key in ['image', 'image_base64', 'base64', 'data', 'imageData', 'img', 'photo', 'picture']:
                if key in image_input and image_input[key]:
                    base64_image = image_input[key]
                    print(f"[v151] Found image in input['{key}']")
                    break
        
        # If still no image, check direct event keys
        if not base64_image and isinstance(event, dict):
            for key in ['image', 'image_base64', 'base64', 'data', 'imageData', 'img']:
                if key in event and event[key]:
                    base64_image = event[key]
                    print(f"[v151] Found image in event['{key}']")
                    break
        
        # Check if event itself is the base64 string
        if not base64_image and isinstance(event, str) and len(event) > 100:
            base64_image = event
            print("[v151] Event is direct base64 string")
        
        # Check if input is string
        if not base64_image and isinstance(image_input, str) and len(image_input) > 100:
            base64_image = image_input
            print("[v151] Input is direct base64 string")
        
        # Last resort: check for nested structures
        if not base64_image:
            # Check event.data
            if isinstance(event, dict) and 'data' in event and isinstance(event['data'], dict):
                for key in ['image', 'image_base64', 'base64']:
                    if key in event['data'] and event['data'][key]:
                        base64_image = event['data'][key]
                        print(f"[v151] Found image in event['data']['{key}']")
                        break
        
        # Remove data URL prefix if present (common from Make.com)
        if base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
            print("[v151] Removed data URL prefix")
        
        if not base64_image:
            print("[v151] No image found in any expected location")
            print(f"[v151] Full event structure: {json.dumps(event, indent=2)[:1000] if isinstance(event, dict) else str(event)[:1000]}")
            print("[v151] Available event keys:", list(event.keys()) if isinstance(event, dict) else "Not a dict")
            print("[v151] Available input keys:", list(image_input.keys()) if isinstance(image_input, dict) else "Not a dict")
            
            # Return detailed error for debugging
            return {
                "output": {
                    "error": "No image provided",
                    "processing_info": {
                        "version": "v151", 
                        "error": "No input image found",
                        "event_type": str(type(event)),
                        "input_type": str(type(image_input)),
                        "event_keys": list(event.keys()) if isinstance(event, dict) else [],
                        "input_keys": list(image_input.keys()) if isinstance(image_input, dict) else [],
                        "checked_locations": [
                            "input.image", "input.image_base64", "input.base64", "input.data",
                            "event.image", "event.image_base64", "event.data.image",
                            "direct string"
                        ]
                    }
                }
            }
        
        print(f"[v151] Received image, length: {len(base64_image)}")
        
        # Process image
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(base64_image)
        
        # Return with proper structure for Make.com
        # Make.com expects: {{4.data.output.output.enhanced_image}}
        return {
            "output": result
        }
        
    except Exception as e:
        print(f"[v151] Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "processing_info": {
                    "version": "v151", 
                    "error": "Handler exception",
                    "traceback": traceback.format_exc()
                }
            }
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print("[v151] Starting RunPod serverless handler - Advanced Version")
    runpod.serverless.start({"handler": handler})
