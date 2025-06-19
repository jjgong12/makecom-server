import runpod
import cv2
import numpy as np
from PIL import Image
import base64
import io

def handler(event):
    """RunPod handler for wedding ring enhancement v82"""
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
        
        # Metal type detection
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        center_y, center_x = img_array.shape[0] // 2, img_array.shape[1] // 2
        sample_size = min(img_array.shape[0], img_array.shape[1]) // 4
        center_region = img_rgb[center_y-sample_size:center_y+sample_size,
                                center_x-sample_size:center_x+sample_size]
        
        b_mean = np.mean(center_region[:, :, 2])
        rg_diff = np.mean(center_region[:, :, 0]) - np.mean(center_region[:, :, 1])
        
        if b_mean > 180 and abs(rg_diff) < 20:
            metal_type = "white_gold"
        elif rg_diff > 10:
            metal_type = "rose_gold"
        elif rg_diff > 0:
            metal_type = "yellow_gold"
        else:
            metal_type = "silver"
        
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
        
        # Enhancement parameters
        params = {
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
            ("silver", "bright"): {
                "denoise_h": 3, "enhance_strength": 1.15, "contrast": 1.02,
                "brightness": 3, "shadows": 0.5, "highlights": 0.7,
                "temperature": -5, "saturation": 0.9, "whites": 0.4,
                "sharpen": 1.6, "small_details": 0.7, "medium_details": 0.8
            },
            ("silver", "normal"): {
                "denoise_h": 5, "enhance_strength": 1.25, "contrast": 1.08,
                "brightness": 8, "shadows": 0.6, "highlights": 0.8,
                "temperature": -8, "saturation": 0.85, "whites": 0.6,
                "sharpen": 1.9, "small_details": 0.8, "medium_details": 0.9
            },
            ("silver", "dark"): {
                "denoise_h": 8, "enhance_strength": 1.35, "contrast": 1.12,
                "brightness": 15, "shadows": 0.7, "highlights": 0.9,
                "temperature": -12, "saturation": 0.8, "whites": 0.9,
                "sharpen": 2.3, "small_details": 0.9, "medium_details": 1.0
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
        
        # NEW: Progressive black edge detection
        h, w = img_array.shape[:2]
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Detect black edges progressively from 10px to 150px
        edge_coords = []
        for edge_width in range(10, 151, 10):  # 10, 20, 30, ... 150
            # Top edge
            for y in range(min(edge_width, h)):
                for x in range(w):
                    if gray[y, x] < 40:  # Black threshold
                        edge_coords.append((y, x))
            
            # Bottom edge
            for y in range(max(0, h - edge_width), h):
                for x in range(w):
                    if gray[y, x] < 40:
                        edge_coords.append((y, x))
            
            # Left edge
            for x in range(min(edge_width, w)):
                for y in range(h):
                    if gray[y, x] < 40:
                        edge_coords.append((y, x))
            
            # Right edge
            for x in range(max(0, w - edge_width), w):
                for y in range(h):
                    if gray[y, x] < 40:
                        edge_coords.append((y, x))
        
        # Apply mask only to detected black pixels
        for y, x in edge_coords:
            mask[y, x] = 255
        
        # Dilate to ensure complete removal
        if np.any(mask):
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Protect wedding ring area (center 60%)
            center_protection = 0.3  # 30% margin on each side
            protected_x1 = int(w * center_protection)
            protected_x2 = int(w * (1 - center_protection))
            protected_y1 = int(h * center_protection)
            protected_y2 = int(h * (1 - center_protection))
            mask[protected_y1:protected_y2, protected_x1:protected_x2] = 0
            
            # Apply background replacement
            mask_float = mask.astype(float) / 255
            mask_blurred = cv2.GaussianBlur(mask_float, (31, 31), 10)
            
            target_color = np.array([245, 243, 240], dtype=np.uint8)
            for c in range(3):
                enhanced[:, :, c] = (enhanced[:, :, c] * (1 - mask_blurred) + 
                                     target_color[c] * mask_blurred).astype(np.uint8)
        
        # Create thumbnail with improved ring detection
        pil_enhanced = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Find wedding ring for thumbnail
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Find bright regions (wedding rings are usually bright)
        center_region = gray_enhanced[h//4:3*h//4, w//4:3*w//4]
        threshold = np.mean(center_region) + np.std(center_region) * 0.5
        _, binary = cv2.threshold(gray_enhanced, min(threshold, 200), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour in center region (likely the ring)
        best_contour = None
        max_area = 0
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            center_x = x + cw // 2
            center_y = y + ch // 2
            # Check if contour center is in middle region
            if (w * 0.2 < center_x < w * 0.8 and h * 0.2 < center_y < h * 0.8):
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    best_contour = contour
        
        # Crop based on ring detection
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            
            # Expand bounding box by 20% for context
            expand = 0.2
            x = max(0, int(x - cw * expand))
            y = max(0, int(y - ch * expand))
            cw = min(w - x, int(cw * (1 + 2 * expand)))
            ch = min(h - y, int(ch * (1 + 2 * expand)))
            
            # Ensure minimum size
            if cw > 100 and ch > 100:
                crop = pil_enhanced.crop((x, y, x + cw, y + ch))
            else:
                crop = pil_enhanced
        else:
            # Fallback: use center crop
            crop_size = min(w, h) * 0.8
            x = (w - crop_size) // 2
            y = (h - crop_size) // 2
            crop = pil_enhanced.crop((x, y, x + crop_size, y + crop_size))
        
        # Resize to fill 95% of thumbnail
        aspect = crop.width / crop.height
        if aspect > 1000/1300:
            new_width = 950  # 95% of 1000
            new_height = int(new_width / aspect)
        else:
            new_height = 1235  # 95% of 1300
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
        enhanced_base64 = enhanced_base64.rstrip('=')  # Remove padding for Make.com
        
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding for Make.com
        
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "status": "success",
                    "version": "v82"
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
