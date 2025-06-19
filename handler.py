import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import time

# v58 EXTREME WHITE Parameters - Champagne Gold becomes almost White Gold
COMPLETE_PARAMETERS = {
    'white_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.10, 'exposure': 0.05,
            'highlights': -0.10, 'shadows': 0.15, 'vibrance': 1.20,
            'saturation': 1.08, 'clarity': 15, 'color_temp': -5,
            'white_overlay': 0.06
        },
        'bright': {
            'brightness': 1.10, 'contrast': 1.08, 'exposure': 0.02,
            'highlights': -0.15, 'shadows': 0.10, 'vibrance': 1.15,
            'saturation': 1.06, 'clarity': 12, 'color_temp': -4,
            'white_overlay': 0.05
        },
        'shadow': {
            'brightness': 1.25, 'contrast': 1.12, 'exposure': 0.08,
            'highlights': -0.05, 'shadows': 0.25, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 18, 'color_temp': -6,
            'white_overlay': 0.08
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.12, 'contrast': 1.15, 'exposure': 0.03,
            'highlights': -0.08, 'shadows': 0.12, 'vibrance': 1.30,
            'saturation': 1.12, 'clarity': 12, 'color_temp': 8,
            'white_overlay': 0.03
        },
        'bright': {
            'brightness': 1.08, 'contrast': 1.12, 'exposure': 0.00,
            'highlights': -0.12, 'shadows': 0.08, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 10, 'color_temp': 6,
            'white_overlay': 0.02
        },
        'shadow': {
            'brightness': 1.20, 'contrast': 1.18, 'exposure': 0.06,
            'highlights': -0.03, 'shadows': 0.20, 'vibrance': 1.35,
            'saturation': 1.15, 'clarity': 15, 'color_temp': 10,
            'white_overlay': 0.04
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.14, 'contrast': 1.12, 'exposure': 0.04,
            'highlights': -0.07, 'shadows': 0.14, 'vibrance': 1.25,
            'saturation': 1.10, 'clarity': 13, 'color_temp': 3,
            'white_overlay': 0.05
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
    """v60: 100픽셀 두께 검은색 테두리도 완전 제거"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # v60: threshold 조정 (밝은 회색도 테두리로 인식)
    threshold = 120
    
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
    
    # v60: safety margin 추가 (검은 테두리 잔여물 제거)
    safety_margin = 20
    
    # Apply borders with safety margin
    top = min(top + safety_margin, h // 3)
    bottom = min(bottom + safety_margin, h // 3)
    left = min(left + safety_margin, w // 3)
    right = min(right + safety_margin, w // 3)
    
    # Crop
    cropped = image[top:h-bottom, left:w-right]
    
    # Check if we removed too much
    if cropped.shape[0] < 100 or cropped.shape[1] < 100:
        # 너무 많이 잘렸으면 원본 반환
        return image, False
    
    border_removed = (top > 10 or bottom > 10 or left > 10 or right > 10)
    return cropped, border_removed

def detect_metal_type(image):
    """Detect metal type from image"""
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get center region (more reliable)
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        roi_size = min(h, w) // 4
        roi = hsv[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        # Calculate average hue and saturation
        avg_hue = np.mean(roi[:, :, 0])
        avg_sat = np.mean(roi[:, :, 1])
        avg_val = np.mean(roi[:, :, 2])
        
        # Metal detection logic
        if avg_sat < 30:  # Low saturation = white gold
            return 'white_gold'
        elif 15 <= avg_hue <= 25 and avg_sat > 50:  # Orange hue = yellow gold
            return 'yellow_gold'
        elif 5 <= avg_hue <= 15 and avg_sat > 30:  # Red-orange = rose gold
            return 'rose_gold'
        elif avg_hue < 20 and avg_val > 180:  # Bright low hue = champagne
            return 'champagne_gold'
        else:
            return 'white_gold'
    except:
        return 'white_gold'

def detect_lighting(image):
    """Detect lighting condition from image brightness"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 180:
            return 'bright'
        elif avg_brightness < 120:
            return 'shadow'
        else:
            return 'natural'
    except:
        return 'natural'

def enhance_image(image, params):
    """Apply enhancement parameters to image"""
    # Convert to PIL for easier manipulation
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply brightness
    if params.get('brightness', 1.0) != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(params['brightness'])
    
    # Apply contrast
    if params.get('contrast', 1.0) != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(params['contrast'])
    
    # Convert back to numpy for advanced operations
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Apply white overlay
    if params.get('white_overlay', 0) > 0:
        white = np.ones_like(enhanced) * 255
        enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                                 white, params['white_overlay'], 0)
    
    # Apply color temperature adjustment
    if params.get('color_temp', 0) != 0:
        temp = params['color_temp']
        if temp > 0:  # Warmer
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 - temp/100), 0, 255)  # Less blue
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 + temp/100), 0, 255)  # More red
        else:  # Cooler
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 - temp/100), 0, 255)  # More blue
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 + temp/100), 0, 255)  # Less red
    
    return enhanced

def create_thumbnail(image):
    """v60: 강력한 썸네일 생성 - 검은 테두리 제거된 이미지에서 ring 최대화"""
    h, w = image.shape[:2]
    target_w, target_h = 1000, 1300
    
    # Find the ring area using stronger detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # v60: ring 감지를 위한 이진화
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    
    # Clean up the binary image
    kernel = np.ones((7, 7), np.uint8)
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
    """Wedding Ring AI v60 Handler - 100픽셀 검은 테두리 완전 제거"""
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get("image") or job_input.get("image_base64")
        
        if not image_data:
            return {"output": {"error": "No image provided", "status": "error", "version": "v60"}}
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            return {"output": {"error": f"Image decode error: {str(e)}", "status": "error", "version": "v60"}}
        
        # Step 1: Remove black borders (v60 - 100픽셀 대응)
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
        
        # Step 4: Create thumbnail (v60 - 최대화)
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
                    "version": "v60",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "error",
                "version": "v60"
            }
        }

# RunPod serverless start
runpod.serverless.start({"handler": handler})
