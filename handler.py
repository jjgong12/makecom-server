import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import time
import runpod

# V13.3 Complete Parameters (28 Pairs Study)
COMPLETE_PARAMETERS = {
    'white_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.14, 'exposure': 0.05,
            'highlights': -0.08, 'shadows': 0.15, 'vibrance': 1.22,
            'saturation': 1.10, 'clarity': 12, 'color_temp': 0,
            'white_overlay': 0.03
        },
        'bright': {
            'brightness': 1.12, 'contrast': 1.10, 'exposure': 0.02,
            'highlights': -0.15, 'shadows': 0.08, 'vibrance': 1.18,
            'saturation': 1.05, 'clarity': 8, 'color_temp': -2,
            'white_overlay': 0.05
        },
        'shadow': {
            'brightness': 1.20, 'contrast': 1.18, 'exposure': 0.08,
            'highlights': -0.05, 'shadows': 0.25, 'vibrance': 1.25,
            'saturation': 1.15, 'clarity': 15, 'color_temp': 2,
            'white_overlay': 0.04
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'exposure': 0.04,
            'highlights': -0.10, 'shadows': 0.12, 'vibrance': 1.20,
            'saturation': 0.95, 'clarity': 10, 'color_temp': -4,
            'white_overlay': 0.08
        },
        'bright': {
            'brightness': 1.14, 'contrast': 1.08, 'exposure': 0.00,
            'highlights': -0.20, 'shadows': 0.05, 'vibrance': 1.15,
            'saturation': 0.92, 'clarity': 6, 'color_temp': -5,
            'white_overlay': 0.10
        },
        'shadow': {
            'brightness': 1.24, 'contrast': 1.16, 'exposure': 0.06,
            'highlights': -0.03, 'shadows': 0.20, 'vibrance': 1.28,
            'saturation': 0.98, 'clarity': 14, 'color_temp': -3,
            'white_overlay': 0.07
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.14, 'contrast': 1.12, 'exposure': 0.03,
            'highlights': -0.09, 'shadows': 0.14, 'vibrance': 1.25,
            'saturation': 1.12, 'clarity': 13, 'color_temp': 3,
            'white_overlay': 0.02
        },
        'bright': {
            'brightness': 1.10, 'contrast': 1.10, 'exposure': 0.01,
            'highlights': -0.10, 'shadows': 0.10, 'vibrance': 1.20,
            'saturation': 1.08, 'clarity': 11, 'color_temp': 2,
            'white_overlay': 0.04
        },
        'shadow': {
            'brightness': 1.22, 'contrast': 1.15, 'exposure': 0.07,
            'highlights': -0.04, 'shadows': 0.22, 'vibrance': 1.30,
            'saturation': 1.12, 'clarity': 16, 'color_temp': 4,
            'white_overlay': 0.06
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.30, 'contrast': 1.08, 'exposure': 0.06,
            'highlights': -0.12, 'shadows': 0.18, 'vibrance': 1.15,
            'saturation': 0.90, 'clarity': 14, 'color_temp': -6,
            'white_overlay': 0.15
        },
        'bright': {
            'brightness': 1.28, 'contrast': 1.06, 'exposure': 0.03,
            'highlights': -0.18, 'shadows': 0.12, 'vibrance': 1.10,
            'saturation': 0.88, 'clarity': 11, 'color_temp': -7,
            'white_overlay': 0.18
        },
        'shadow': {
            'brightness': 1.35, 'contrast': 1.10, 'exposure': 0.10,
            'highlights': -0.06, 'shadows': 0.28, 'vibrance': 1.20,
            'saturation': 0.85, 'clarity': 17, 'color_temp': -8,
            'white_overlay': 0.20
        }
    }
}

def remove_black_borders(image):
    """v62: 100픽셀 두께 검은색 테두리도 완전 제거"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # v60 → v62: threshold 120 → 150 (더 밝은 회색도 테두리로 인식)
    threshold = 150  # ← 이것만 변경!
    
    # v60: 최대 200픽셀까지 스캔 (100픽셀 테두리 + 여유)
    max_scan = min(200, h // 3, w // 3)
    
    # Find borders - 전체 라인의 평균값으로 판단
    top = 0
    for y in range(max_scan):
        if np.mean(gray[y, :]) < threshold:
            top = y + 1
        else:
            break
    
    bottom = 0
    for y in range(h-1, max(h-max_scan-1, h//2), -1):
        if np.mean(gray[y, :]) < threshold:
            bottom = h - y
        else:
            break
    
    left = 0
    for x in range(max_scan):
        if np.mean(gray[:, x]) < threshold:
            left = x + 1
        else:
            break
    
    right = 0
    for x in range(w-1, max(w-max_scan-1, w//2), -1):
        if np.mean(gray[:, x]) < threshold:
            right = w - x
        else:
            break
    
    # v62: 추가 안전 마진 (35픽셀)
    safety_margin = 35
    top = min(top + safety_margin, h // 4)
    bottom = min(bottom + safety_margin, h // 4)
    left = min(left + safety_margin, w // 4)
    right = min(right + safety_margin, w // 4)
    
    # Crop the image
    if top + bottom < h and left + right < w:
        cropped = image[top:h-bottom, left:w-right]
        border_removed = (top > 35 or bottom > 35 or left > 35 or right > 35)
        return cropped, border_removed
    else:
        return image, False

def detect_metal_type(image):
    """Detect metal type from image colors"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    center_region = img_rgb[h//4:3*h//4, w//4:3*w//4]
    
    avg_color = np.mean(center_region, axis=(0, 1))
    r, g, b = avg_color
    
    # Metal detection logic based on color ratios
    yellow_ratio = (r + g) / (2 * b) if b > 0 else 2.0
    rose_ratio = r / g if g > 0 else 1.0
    
    if yellow_ratio > 1.4 and r > 140 and g > 120:
        return 'yellow_gold'
    elif rose_ratio > 1.15 and r > 150:
        return 'rose_gold'
    elif r > 190 and g > 170 and yellow_ratio > 1.3:
        return 'champagne_gold'
    else:
        return 'white_gold'

def detect_lighting(image):
    """Detect lighting condition from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness > 180:
        return 'bright'
    elif brightness < 120:
        return 'shadow'
    else:
        return 'natural'

def enhance_image(image, params):
    """Apply v13.3 enhancement parameters"""
    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Basic adjustments
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(params['brightness'])
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(params['contrast'])
    
    # Convert back to numpy for advanced processing
    img_array = np.array(pil_img)
    
    # Apply exposure adjustment
    exposure_factor = 1 + params['exposure']
    img_array = np.clip(img_array * exposure_factor, 0, 255).astype(np.uint8)
    
    # Vibrance and Saturation
    enhancer = ImageEnhance.Color(Image.fromarray(img_array))
    img_array = np.array(enhancer.enhance(params['saturation']))
    
    # Color temperature adjustment
    temp = params['color_temp']
    if temp != 0:
        img_array[:,:,0] = np.clip(img_array[:,:,0] * (1 + temp/100), 0, 255)
        img_array[:,:,2] = np.clip(img_array[:,:,2] * (1 - temp/100), 0, 255)
    
    # White overlay
    white_overlay = np.full_like(img_array, 255)
    alpha = params['white_overlay']
    img_array = cv2.addWeighted(img_array, 1-alpha, white_overlay, alpha, 0)
    
    # Convert back to BGR for OpenCV
    final = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Clarity (unsharp mask)
    clarity_strength = params['clarity'] / 100
    if clarity_strength > 0:
        blurred = cv2.GaussianBlur(final, (0, 0), 3)
        final = cv2.addWeighted(final, 1 + clarity_strength, blurred, -clarity_strength, 0)
    
    # 2x upscaling
    h, w = final.shape[:2]
    final = cv2.resize(final, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    
    return final

def create_thumbnail(image, target_w=1000, target_h=1300):
    """Create thumbnail with ring maximized in frame"""
    h, w = image.shape[:2]
    
    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to find the ring
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find the bounding box of edges
    points = cv2.findNonZero(edges)
    
    if points is not None:
        x, y, edge_w, edge_h = cv2.boundingRect(points)
        
        # v60: 더 타이트한 크롭 (padding 5%)
        pad = int(max(edge_w, edge_h) * 0.05)
        x = max(0, x - pad)
        y = max(0, y - pad)
        edge_w = min(w - x, edge_w + 2 * pad)
        edge_h = min(h - y, edge_h + 2 * pad)
        
        cropped = image[y:y+edge_h, x:x+edge_w]
    else:
        # Fallback: try thresholding
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, ring_w, ring_h = cv2.boundingRect(largest_contour)
            
            # v60: 최소한의 padding (10%)
            pad = int(max(ring_w, ring_h) * 0.1)
            x = max(0, x - pad)
            y = max(0, y - pad)
            ring_w = min(w - x, ring_w + 2 * pad)
            ring_h = min(h - y, ring_h + 2 * pad)
            
            # Crop to ring area
            cropped = image[y:y+ring_h, x:x+ring_w]
        else:
            # Fallback: 이미지 전체 사용 (이미 검은 테두리 제거됨)
            cropped = image
    
    # v60: ring이 썸네일의 99%를 차지하도록
    crop_h, crop_w = cropped.shape[:2]
    scale = min(target_w * 0.99 / crop_w, target_h * 0.99 / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize with high quality
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    thumb = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
    
    # Center the ring
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    thumb[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumb

def handler(job):
    """Wedding Ring AI v62 Handler - 100픽셀 검은 테두리 완전 제거"""
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get("image") or job_input.get("image_base64")
        
        if not image_data:
            return {"output": {"error": "No image provided", "status": "error", "version": "v62"}}
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            return {"output": {"error": f"Image decode error: {str(e)}", "status": "error", "version": "v62"}}
        
        # Step 1: Remove black borders (v62 - 100픽셀 대응)
        image, border_removed = remove_black_borders(image)
        
        # Step 2: Detect metal and lighting
        metal_type = job_input.get("metal_type", "auto")
        lighting = job_input.get("lighting", "auto")
        
        if metal_type == "auto":
            metal_type = detect_metal_type(image)
        
        if lighting == "auto":
            lighting = detect_lighting(image)
        
        # Step 3: Get parameters and enhance
        params = COMPLETE_PARAMETERS.get(metal_type, {}).get(lighting, COMPLETE_PARAMETERS['white_gold']['natural'])
        enhanced = enhance_image(image, params)
        
        # Step 4: Create thumbnail (v62 - 최대화)
        thumbnail = create_thumbnail(enhanced)
        
        # Step 5: Convert to base64
        # Main image
        _, main_buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        
        # Thumbnail
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        # CRITICAL: Return with proper output structure
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_removed": border_removed,
                    "processing_time": time.time() - start_time,
                    "version": "v62",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "error",
                "version": "v62"
            }
        }

# RunPod serverless start
runpod.serverless.start({"handler": handler})
