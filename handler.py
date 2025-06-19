import runpod
import cv2
import numpy as np
from PIL import Image
import base64
import io

def handler(event):
    """RunPod handler for wedding ring enhancement v83"""
    try:
        # Get input
        image_input = event.get("input", {})
        image_base64 = image_input.get("image") or image_input.get("image_base64")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error"
                }
            }
        
        # Decode base64
        try:
            # Handle padding
            image_base64 = image_base64.strip()
            missing_padding = len(image_base64) % 4
            if missing_padding:
                image_base64 += '=' * (4 - missing_padding)
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "error"
                }
            }
        
        # Metal type detection - ONLY 4 TYPES (NO SILVER)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        center_y, center_x = img_array.shape[0] // 2, img_array.shape[1] // 2
        sample_size = min(img_array.shape[0], img_array.shape[1]) // 4
        center_region = img_rgb[center_y-sample_size:center_y+sample_size,
                                center_x-sample_size:center_x+sample_size]
        
        r_mean = np.mean(center_region[:, :, 0])
        g_mean = np.mean(center_region[:, :, 1])
        b_mean = np.mean(center_region[:, :, 2])
        
        # Calculate color characteristics
        rg_ratio = r_mean / (g_mean + 1)
        rb_ratio = r_mean / (b_mean + 1)
        gb_ratio = g_mean / (b_mean + 1)
        brightness = np.mean(center_region)
        
        # Determine metal type
        if rg_ratio > 1.12 and rb_ratio > 1.15:  # Pink/rose tint
            metal_type = "rose_gold"
        elif rg_ratio > 1.05 and gb_ratio > 1.05:  # Yellow tint
            metal_type = "yellow_gold"
        elif abs(r_mean - g_mean) < 10 and abs(g_mean - b_mean) < 10 and brightness > 160:
            # Very neutral and bright = champagne gold
            metal_type = "champagne_gold"
        else:
            metal_type = "white_gold"
        
        # Lighting detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        center_brightness = np.mean(gray[center_y-sample_size:center_y+sample_size,
                                         center_x-sample_size:center_x+sample_size])
        
        if center_brightness > 180:
            lighting = "bright"
        elif center_brightness > 120:
            lighting = "normal"
        else:
            lighting = "dark"
        
        # Enhancement parameters - 4 metal types only
        params = {
            # White Gold
            ("white_gold", "bright"): {
                "denoise_h": 3, "enhance_strength": 1.15, "contrast": 1.02,
                "brightness": 3, "shadows": 0.5, "highlights": 0.7,
                "temperature": -3, "saturation": 0.95, "whites": 0.3,
                "sharpen": 1.5, "small_details": 0.7, "medium_details": 0.8
            },
            ("white_gold", "normal"): {
                "denoise_h": 5, "enhance_strength": 1.25, "contrast": 1.08,
                "brightness": 8, "shadows": 0.6, "highlights": 0.8,
                "temperature": -5, "saturation": 0.9, "whites": 0.5,
                "sharpen": 1.8, "small_details": 0.8, "medium_details": 0.9
            },
            ("white_gold", "dark"): {
                "denoise_h": 8, "enhance_strength": 1.35, "contrast": 1.12,
                "brightness": 15, "shadows": 0.7, "highlights": 0.9,
                "temperature": -8, "saturation": 0.85, "whites": 0.8,
                "sharpen": 2.2, "small_details": 0.9, "medium_details": 1.0
            },
            # Yellow Gold
            ("yellow_gold", "bright"): {
                "denoise_h": 4, "enhance_strength": 1.18, "contrast": 1.05,
                "brightness": 5, "shadows": 0.5, "highlights": 0.7,
                "temperature": -10, "saturation": 0.8, "whites": 0.6,
                "sharpen": 1.8, "small_details": 0.8, "medium_details": 0.9
            },
            ("yellow_gold", "normal"): {
                "denoise_h": 6, "enhance_strength": 1.28, "contrast": 1.1,
                "brightness": 10, "shadows": 0.6, "highlights": 0.8,
                "temperature": -15, "saturation": 0.75, "whites": 0.8,
                "sharpen": 2.0, "small_details": 0.9, "medium_details": 1.0
            },
            ("yellow_gold", "dark"): {
                "denoise_h": 10, "enhance_strength": 1.4, "contrast": 1.15,
                "brightness": 18, "shadows": 0.8, "highlights": 1.0,
                "temperature": -20, "saturation": 0.7, "whites": 1.0,
                "sharpen": 2.5, "small_details": 1.0, "medium_details": 1.1
            },
            # Rose Gold
            ("rose_gold", "bright"): {
                "denoise_h": 4, "enhance_strength": 1.2, "contrast": 1.05,
                "brightness": 5, "shadows": 0.5, "highlights": 0.7,
                "temperature": -8, "saturation": 0.85, "whites": 0.5,
                "sharpen": 1.7, "small_details": 0.8, "medium_details": 0.9
            },
            ("rose_gold", "normal"): {
                "denoise_h": 6, "enhance_strength": 1.28, "contrast": 1.1,
                "brightness": 10, "shadows": 0.6, "highlights": 0.8,
                "temperature": -12, "saturation": 0.8, "whites": 0.7,
                "sharpen": 2.0, "small_details": 0.9, "medium_details": 1.0
            },
            ("rose_gold", "dark"): {
                "denoise_h": 9, "enhance_strength": 1.38, "contrast": 1.15,
                "brightness": 18, "shadows": 0.7, "highlights": 0.9,
                "temperature": -18, "saturation": 0.75, "whites": 0.9,
                "sharpen": 2.3, "small_details": 1.0, "medium_details": 1.1
            },
            # Champagne Gold (무도금화이트)
            ("champagne_gold", "bright"): {
                "denoise_h": 3, "enhance_strength": 1.2, "contrast": 1.03,
                "brightness": 5, "shadows": 0.5, "highlights": 0.7,
                "temperature": -15, "saturation": 0.85, "whites": 0.7,
                "sharpen": 1.8, "small_details": 0.8, "medium_details": 0.9
            },
            ("champagne_gold", "normal"): {
                "denoise_h": 5, "enhance_strength": 1.3, "contrast": 1.1,
                "brightness": 12, "shadows": 0.6, "highlights": 0.8,
                "temperature": -20, "saturation": 0.8, "whites": 0.9,
                "sharpen": 2.0, "small_details": 0.9, "medium_details": 1.0
            },
            ("champagne_gold", "dark"): {
                "denoise_h": 8, "enhance_strength": 1.4, "contrast": 1.15,
                "brightness": 20, "shadows": 0.7, "highlights": 0.9,
                "temperature": -25, "saturation": 0.75, "whites": 1.0,
                "sharpen": 2.3, "small_details": 1.0, "medium_details": 1.1
            }
        }
        
        p = params.get((metal_type, lighting), params[("white_gold", "normal")])
        
        # Apply enhancement
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, p["denoise_h"], p["denoise_h"], 7, 21)
        enhanced = cv2.convertScaleAbs(denoised, alpha=p["enhance_strength"], beta=p["brightness"])
        
        # Additional enhancements
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Detail enhancement
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # AGGRESSIVE BLACK EDGE DETECTION - PIXEL BY PIXEL
        h, w = img_array.shape[:2]
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Multiple thresholds for black detection
        thresholds = [20, 30, 40, 50, 60]
        
        # Scan every single pixel from edges
        for threshold in thresholds:
            # Top edge - scan line by line
            for y in range(min(200, h)):  # Up to 200 pixels from top
                for x in range(w):
                    if gray[y, x] < threshold:
                        mask[y, x] = 255
                        # If black pixel found, check if it continues
                        if y < h - 1 and gray[y + 1, x] >= threshold:
                            break  # Stop at this depth for this column
            
            # Bottom edge
            for y in range(max(0, h - 200), h):
                for x in range(w):
                    if gray[y, x] < threshold:
                        mask[y, x] = 255
                        if y > 0 and gray[y - 1, x] >= threshold:
                            break
            
            # Left edge
            for x in range(min(200, w)):
                for y in range(h):
                    if gray[y, x] < threshold:
                        mask[y, x] = 255
                        if x < w - 1 and gray[y, x + 1] >= threshold:
                            break
            
            # Right edge
            for x in range(max(0, w - 200), w):
                for y in range(h):
                    if gray[y, x] < threshold:
                        mask[y, x] = 255
                        if x > 0 and gray[y, x - 1] >= threshold:
                            break
        
        # Also check for continuous black lines
        # Horizontal lines
        for y in range(h):
            black_count = 0
            for x in range(w):
                if gray[y, x] < 50:
                    black_count += 1
                else:
                    black_count = 0
                
                # If we find 50+ continuous black pixels, it's likely a border
                if black_count > 50:
                    for xx in range(max(0, x - black_count), x + 1):
                        mask[y, xx] = 255
        
        # Vertical lines
        for x in range(w):
            black_count = 0
            for y in range(h):
                if gray[y, x] < 50:
                    black_count += 1
                else:
                    black_count = 0
                
                if black_count > 50:
                    for yy in range(max(0, y - black_count), y + 1):
                        mask[yy, x] = 255
        
        # Protect wedding ring area (center 50%)
        center_protection = 0.25  # 25% margin on each side
        protected_x1 = int(w * center_protection)
        protected_x2 = int(w * (1 - center_protection))
        protected_y1 = int(h * center_protection)
        protected_y2 = int(h * (1 - center_protection))
        mask[protected_y1:protected_y2, protected_x1:protected_x2] = 0
        
        # Apply mask
        if np.any(mask):
            # Dilate slightly to ensure complete removal
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Apply background replacement
            mask_float = mask.astype(float) / 255
            mask_blurred = cv2.GaussianBlur(mask_float, (31, 31), 10)
            
            target_color = np.array([245, 243, 240], dtype=np.uint8)
            for c in range(3):
                enhanced[:, :, c] = (enhanced[:, :, c] * (1 - mask_blurred) + 
                                     target_color[c] * mask_blurred).astype(np.uint8)
        
        # IMPROVED THUMBNAIL - FIND RING AND MAKE IT HUGE
        pil_enhanced = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Find wedding ring using multiple methods
        # Method 1: Bright region detection
        center_h = h // 2
        center_w = w // 2
        search_region = gray_enhanced[center_h - h//3:center_h + h//3, 
                                      center_w - w//3:center_w + w//3]
        
        # Adaptive threshold based on center region
        thresh_value = np.mean(search_region) + np.std(search_region) * 0.3
        _, binary = cv2.threshold(gray_enhanced, min(thresh_value, 220), 255, cv2.THRESH_BINARY)
        
        # Method 2: Edge detection for ring shape
        edges = cv2.Canny(gray_enhanced, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        
        # Combine both methods
        combined = cv2.bitwise_or(binary, edges_dilated)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the best contour (ring)
        best_contour = None
        max_score = 0
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            cx = x + cw // 2
            cy = y + ch // 2
            
            # Score based on:
            # 1. Distance from center (closer is better)
            # 2. Size (bigger is better, but not too big)
            # 3. Aspect ratio (rings are usually somewhat square)
            
            dist_from_center = np.sqrt((cx - center_w)**2 + (cy - center_h)**2)
            center_score = 1.0 - (dist_from_center / (w + h))
            
            size_score = min(cw * ch / (w * h * 0.3), 1.0)  # Max 30% of image
            
            aspect_ratio = min(cw, ch) / max(cw, ch)
            aspect_score = aspect_ratio  # Closer to 1 is better
            
            # Combined score
            score = center_score * 0.5 + size_score * 0.3 + aspect_score * 0.2
            
            if score > max_score and cw > 50 and ch > 50:  # Minimum size
                max_score = score
                best_contour = contour
        
        # Crop based on best contour
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            
            # Expand to square aspect ratio
            max_dim = max(cw, ch)
            cx = x + cw // 2
            cy = y + ch // 2
            
            # Expand by 10% for minimal padding
            expand = 1.1
            half_size = int(max_dim * expand / 2)
            
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)
            
            crop = pil_enhanced.crop((x1, y1, x2, y2))
        else:
            # Fallback: aggressive center crop
            size = min(w, h) * 0.6
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            crop = pil_enhanced.crop((x1, y1, x1 + size, y1 + size))
        
        # Resize to fill 98% of thumbnail (almost no padding)
        aspect = crop.width / crop.height
        if aspect > 1000/1300:  # Wider than target
            new_width = 980  # 98% of 1000
            new_height = int(new_width / aspect)
        else:  # Taller than target
            new_height = 1274  # 98% of 1300
            new_width = int(new_height * aspect)
        
        resized = crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create thumbnail with minimal padding
        thumbnail = Image.new('RGB', (1000, 1300), (245, 243, 240))
        x_offset = (1000 - new_width) // 2
        y_offset = (1300 - new_height) // 2
        thumbnail.paste(resized, (x_offset, y_offset))
        
        # Encode results
        enhanced_buffer = io.BytesIO()
        pil_enhanced.save(enhanced_buffer, format='PNG', quality=95)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "status": "success",
                    "version": "v83"
                }
            }
        }
    
    except Exception as e:
        return {
            "output": {
                "error": f"Processing failed: {str(e)}",
                "status": "error"
            }
        }

runpod.serverless.start({"handler": handler})
