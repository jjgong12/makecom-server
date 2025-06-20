#!/usr/bin/env python3
"""
Wedding Ring AI v108 - Perfect Crop & Metal Detection
- Perfect 1000x1300 crop
- Auto metal type detection
- Special processing for white metal
"""

import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import base64
from io import BytesIO
import traceback
import os
import replicate
from typing import Dict, Tuple, Optional

# Wedding ring enhancement parameters (28 pairs of training data)
WEDDING_RING_PARAMS = {
    "white_gold": {
        "brightness": 1.18,
        "contrast": 1.12,
        "sharpness": 1.85,
        "saturation": 0.82,
        "white_overlay": 0.12,
        "gamma": 0.93,
        "highlights": 1.15,
        "shadows": 0.85,
        "temperature": -5,
        "vibrance": 1.1
    },
    "rose_gold": {
        "brightness": 1.2,
        "contrast": 1.15,
        "sharpness": 1.88,
        "saturation": 0.9,
        "white_overlay": 0.08,
        "gamma": 0.92,
        "highlights": 1.12,
        "shadows": 0.88,
        "temperature": -3,
        "vibrance": 1.15
    },
    "yellow_gold": {
        "brightness": 1.25,
        "contrast": 1.2,
        "sharpness": 1.92,
        "saturation": 0.95,
        "white_overlay": 0.05,
        "gamma": 0.88,
        "highlights": 1.18,
        "shadows": 0.82,
        "temperature": -2,
        "vibrance": 1.2
    },
    "white": {  # 무도금화이트
        "brightness": 1.22,
        "contrast": 1.25,
        "sharpness": 1.95,
        "saturation": 0.65,  # 채도 낮춤
        "white_overlay": 0.18,  # 화이트 오버레이 증가
        "gamma": 0.98,
        "highlights": 1.25,
        "shadows": 0.95,
        "temperature": -8,  # 차가운 톤
        "vibrance": 0.9
    }
}

def detect_metal_type(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """
    자동으로 웨딩링의 금속 타입을 감지
    """
    x, y, w, h = bbox
    
    # 웨딩링 영역만 추출
    ring_area = image[y:y+h, x:x+w]
    
    # HSV로 변환
    hsv = cv2.cvtColor(ring_area, cv2.COLOR_BGR2HSV)
    
    # 평균 색상값 계산
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])
    
    # RGB 평균값도 계산
    avg_b = np.mean(ring_area[:,:,0])
    avg_g = np.mean(ring_area[:,:,1])
    avg_r = np.mean(ring_area[:,:,2])
    
    print(f"Metal detection - H:{avg_hue:.1f}, S:{avg_sat:.1f}, V:{avg_val:.1f}, RGB:({avg_r:.1f},{avg_g:.1f},{avg_b:.1f})")
    
    # 무도금화이트 감지 (매우 낮은 채도)
    if avg_sat < 30 and avg_val > 150:
        return "white"
    
    # 로즈골드 감지 (붉은 색조)
    if avg_r > avg_g + 10 and avg_r > avg_b + 15:
        return "rose_gold"
    
    # 화이트골드 감지 (차가운 톤)
    if avg_b > avg_r and avg_sat < 50:
        return "white_gold"
    
    # 옐로우골드 감지 (따뜻한 황금색, 높은 채도)
    if avg_hue > 20 and avg_hue < 40 and avg_sat > 50:
        return "yellow_gold"
    
    # 기본값: 화이트골드
    return "white_gold"

def detect_wedding_ring_replicate(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect wedding ring using Replicate AI
    """
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save temporary file
        temp_path = "/tmp/temp_ring.jpg"
        pil_image.save(temp_path, quality=95)
        
        # Use object detection model
        output = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={
                "image": open(temp_path, "rb"),
                "task": "image_captioning"
            }
        )
        
        # Simple detection based on caption
        if output and "ring" in output.lower():
            # Use fallback to get actual bbox
            os.remove(temp_path)
            return detect_wedding_ring_opencv(image)
        
        os.remove(temp_path)
        
    except Exception as e:
        print(f"Replicate detection failed: {str(e)}")
    
    return None

def detect_wedding_ring_opencv(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    OpenCV를 사용한 정확한 웨딩링 감지
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 엣지 감지
    edges = cv2.Canny(gray, 50, 150)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 면적이 큰 상위 5개 컨투어 검토
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        best_bbox = None
        best_score = 0
        
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 너무 작은 컨투어 제외
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # 중앙에서의 거리 계산
            center_x = x + w // 2
            center_y = y + h // 2
            dist_from_center = np.sqrt((center_x - width//2)**2 + (center_y - height//2)**2)
            
            # 종횡비 계산 (1에 가까울수록 정사각형)
            aspect_ratio = float(w) / h
            aspect_score = 1 - abs(1 - aspect_ratio)
            
            # 점수 계산 (중앙에 가깝고, 정사각형에 가까울수록 높은 점수)
            score = aspect_score * 1000 / (dist_from_center + 1)
            
            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)
        
        if best_bbox:
            x, y, w, h = best_bbox
            
            # 패딩 추가 (10% - 너무 크지 않게)
            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            return x, y, w, h
    
    # 웨딩링을 찾지 못한 경우 중앙 영역 반환
    size = int(min(width, height) * 0.5)
    x = (width - size) // 2
    y = (height - size) // 2
    
    return x, y, size, size

def crop_to_target_size(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       target_size: Tuple[int, int] = (1000, 1300), 
                       ring_percentage: float = 0.45) -> np.ndarray:
    """
    웨딩링을 중심으로 크롭 (ring_percentage로 크기 조절 가능)
    """
    x, y, w, h = bbox
    height, width = image.shape[:2]
    
    # 웨딩링 중심점
    ring_center_x = x + w // 2
    ring_center_y = y + h // 2
    
    # 타겟 비율 (1000:1300 = 0.77)
    target_ratio = target_size[0] / target_size[1]
    
    # 웨딩링이 화면의 ring_percentage를 차지하도록 크기 계산
    ring_size = max(w, h)
    crop_size = int(ring_size / ring_percentage)  # 웨딩링이 지정된 비율 차지
    
    # 세로가 더 긴 비율로 크롭 영역 계산
    crop_w = int(crop_size * target_ratio)
    crop_h = crop_size
    
    # 크롭 시작점 계산 (웨딩링이 중앙에 오도록)
    crop_x = ring_center_x - crop_w // 2
    crop_y = ring_center_y - crop_h // 2
    
    # 경계 체크 및 조정
    if crop_x < 0:
        crop_x = 0
    elif crop_x + crop_w > width:
        crop_x = width - crop_w
        
    if crop_y < 0:
        crop_y = 0
    elif crop_y + crop_h > height:
        crop_y = height - crop_h
    
    # 크롭 영역이 이미지를 벗어나는 경우 패딩 추가
    if crop_x < 0 or crop_y < 0 or crop_x + crop_w > width or crop_y + crop_h > height:
        # 패딩 추가
        pad_top = max(0, -crop_y)
        pad_bottom = max(0, crop_y + crop_h - height)
        pad_left = max(0, -crop_x)
        pad_right = max(0, crop_x + crop_w - width)
        
        # 배경색 (이미지 가장자리 색상 평균)
        bg_color = np.mean(image[0:10, 0:10], axis=(0,1))
        
        # 패딩 적용
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=bg_color)
        
        # 크롭 좌표 조정
        crop_x += pad_left
        crop_y += pad_top
    
    # 크롭
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # PIL로 변환하여 고품질 리사이즈
    pil_cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    resized = pil_cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

def enhance_wedding_ring_advanced(image: Image.Image, metal_type: str = "white_gold") -> Image.Image:
    """
    Advanced wedding ring enhancement with metal-specific adjustments
    """
    params = WEDDING_RING_PARAMS.get(metal_type, WEDDING_RING_PARAMS["white_gold"])
    
    # Keep original for reference
    original = image.copy()
    
    # Convert to numpy for advanced processing
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 무도금화이트 특별 처리
    if metal_type == "white":
        # 채도를 크게 낮춤
        hsv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * 0.5  # 채도 50% 감소
        img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # 밝은 영역 강조
        luminance = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        bright_mask = np.where(luminance > 0.7, 1.0, 0.0)
        img_array = img_array + bright_mask[:,:,np.newaxis] * 0.1
    
    # 1. Adjust highlights and shadows
    luminance = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    highlights_mask = np.where(luminance > 0.6, 1.0, 0.0)
    shadows_mask = np.where(luminance < 0.3, 1.0, 0.0)
    
    img_array = img_array * (1 + highlights_mask[:,:,np.newaxis] * (params['highlights'] - 1))
    img_array = img_array * (1 + shadows_mask[:,:,np.newaxis] * (params['shadows'] - 1))
    
    # 2. Color temperature adjustment
    temp_shift = params['temperature'] / 100.0
    img_array[:,:,0] = np.clip(img_array[:,:,0] + temp_shift, 0, 1)  # Red
    img_array[:,:,2] = np.clip(img_array[:,:,2] - temp_shift, 0, 1)  # Blue
    
    # 3. Vibrance
    hsv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * params['vibrance']
    img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    
    # 4. Apply gamma correction
    gamma = params['gamma']
    img_array = np.power(img_array, gamma)
    
    # 5. White overlay for metallic shine
    white_overlay = np.ones_like(img_array)
    img_array = img_array * (1 - params['white_overlay']) + white_overlay * params['white_overlay']
    
    # Convert back to PIL
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    enhanced = Image.fromarray(img_array)
    
    # 6. Apply PIL enhancements
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(params['brightness'])
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(params['contrast'])
    
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(params['sharpness'])
    
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(params['saturation'])
    
    # 7. 무도금화이트 추가 처리
    if metal_type == "white":
        # 추가 화이트 오버레이
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, 0.1)
        
        # 추가 샤프닝
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=3))
    
    # 8. Final sharpening
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    
    return enhanced

def process_wedding_ring_v108(image_base64: str, metal_type: str = "auto") -> Dict:
    """
    Main processing function v108
    """
    try:
        # Decode base64
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Add padding if needed
        padding = 4 - len(image_base64) % 4
        if padding != 4:
            image_base64 += '=' * padding
        
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_bytes))
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        print(f"Processing v108: size={img_array.shape}")
        
        # Step 1: Detect wedding ring
        bbox = detect_wedding_ring_replicate(img_array)
        
        if bbox is None:
            print("Using OpenCV detection")
            bbox = detect_wedding_ring_opencv(img_array)
        
        print(f"Ring detected at: {bbox}")
        
        # Step 2: Auto detect metal type if needed
        if metal_type == "auto":
            detected_metal = detect_metal_type(img_array, bbox)
            print(f"Auto-detected metal type: {detected_metal}")
        else:
            detected_metal = metal_type
            
        # Step 3: Apply enhancement to original image (no crop, just color enhancement)
        pil_original = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        enhanced = enhance_wedding_ring_advanced(pil_original, detected_metal)
        
        # Step 4: Create thumbnail (1000x1300) - 웨딩링이 큰 크롭
        thumbnail_cropped = crop_to_target_size(img_array, bbox, (1000, 1300), ring_percentage=0.85)
        pil_thumb = Image.fromarray(cv2.cvtColor(thumbnail_cropped, cv2.COLOR_BGR2RGB))
        thumbnail = enhance_wedding_ring_advanced(pil_thumb, detected_metal)
        
        # Convert to base64
        # Main image (original size)
        main_buffer = BytesIO()
        enhanced.save(main_buffer, format='PNG', quality=95, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail (1000x1300)
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Log sizes
        print(f"Main image (original size): color enhanced only - {len(main_base64)} bytes")
        print(f"Thumbnail (1000x1300): ring at 85% - {len(thumb_base64)} bytes")
        print(f"Metal type used: {detected_metal}")
        
        # Return with nested output structure
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "metal_type": detected_metal,
                "processing_version": "v108_perfect_crop",
                "dimensions": {
                    "main": f"{enhanced.width}x{enhanced.height}",
                    "thumbnail": "1000x1300"
                },
                "ring_detection": {
                    "bbox": bbox,
                    "auto_metal": metal_type == "auto"
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v108: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v108_perfect_crop"
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        input_data = event.get("input", {})
        
        # Test mode check
        if input_data.get("test") == True:
            return {
                "status": "test_success",
                "message": "Wedding Ring Processor v108 - Perfect Crop & Metal Detection",
                "version": "v108_perfect_crop",
                "features": [
                    "Main image: Original size with color enhancement only",
                    "Thumbnail (1000x1300): ring at 85% size for close-up",
                    "Auto metal type detection (white/rose_gold/white_gold/yellow_gold)",
                    "Special white metal processing",
                    "Improved ring detection accuracy"
                ]
            }
        
        # Get image
        image_base64 = input_data.get("image") or input_data.get("image_base64")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error"
                }
            }
        
        # Get metal type (default: auto)
        metal_type = input_data.get("metal_type", "auto")
        
        # Process image
        return process_wedding_ring_v108(image_base64, metal_type)
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
