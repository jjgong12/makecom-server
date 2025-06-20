#!/usr/bin/env python3
"""
Wedding Ring AI v109 - Final Perfect Version
- Original: No change
- Enhanced: Original size with color enhancement only
- Thumbnail: 1000x1300 crop with ring at 85%
- Fixed OpenCV float type error
"""

import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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
        "saturation": 0.65,
        "white_overlay": 0.18,
        "gamma": 0.98,
        "highlights": 1.25,
        "shadows": 0.95,
        "temperature": -8,
        "vibrance": 0.9
    }
}

def detect_metal_type(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """자동으로 웨딩링의 금속 타입을 감지"""
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

def detect_wedding_ring_opencv(image: np.ndarray) -> Tuple[int, int, int, int]:
    """OpenCV를 사용한 정확한 웨딩링 감지"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이진화로 밝은 영역 찾기
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 면적이 큰 상위 컨투어 중 가장 중앙에 있는 것 선택
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 면적
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                dist_from_center = np.sqrt((center_x - width//2)**2 + (center_y - height//2)**2)
                valid_contours.append((contour, area, dist_from_center))
        
        if valid_contours:
            # 중앙에 가장 가까운 큰 컨투어 선택
            valid_contours.sort(key=lambda x: x[2])  # 거리순 정렬
            best_contour = valid_contours[0][0]
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # 패딩 추가 (10%)
            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            return x, y, w, h
    
    # 웨딩링을 찾지 못한 경우 중앙 영역
    size = int(min(width, height) * 0.5)
    x = (width - size) // 2
    y = (height - size) // 2
    
    return x, y, size, size

def enhance_wedding_ring_simple(image: Image.Image, metal_type: str = "white_gold") -> Image.Image:
    """간단한 웨딩링 보정 - PIL만 사용"""
    params = WEDDING_RING_PARAMS.get(metal_type, WEDDING_RING_PARAMS["white_gold"])
    
    # PIL 기본 보정만 적용
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(params["brightness"])
    
    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(params["contrast"])
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(params["sharpness"])
    
    # Saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(params["saturation"])
    
    # 무도금화이트 추가 처리
    if metal_type == "white":
        # 화이트 오버레이
        white_layer = Image.new('RGB', image.size, (255, 255, 255))
        image = Image.blend(image, white_layer, params["white_overlay"])
        
        # 추가 샤프닝
        image = image.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=3))
    else:
        # 일반 샤프닝
        image = image.filter(ImageFilter.SHARPEN)
    
    return image

def crop_for_thumbnail(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      target_size: Tuple[int, int] = (1000, 1300)) -> np.ndarray:
    """썸네일용 크롭 - 웨딩링이 85% 차지"""
    x, y, w, h = bbox
    height, width = image.shape[:2]
    
    # 웨딩링 중심점
    ring_center_x = x + w // 2
    ring_center_y = y + h // 2
    
    # 타겟 비율 (1000:1300 = 0.77)
    target_ratio = target_size[0] / target_size[1]
    
    # 웨딩링이 85% 차지하도록
    ring_size = max(w, h)
    crop_size = int(ring_size / 0.85)
    
    # 세로형 비율로 크롭 영역 계산
    crop_w = int(crop_size * target_ratio)
    crop_h = crop_size
    
    # 크롭 시작점 (웨딩링 중앙)
    crop_x = max(0, ring_center_x - crop_w // 2)
    crop_y = max(0, ring_center_y - crop_h // 2)
    
    # 경계 체크
    if crop_x + crop_w > width:
        crop_x = width - crop_w
    if crop_y + crop_h > height:
        crop_y = height - crop_h
    
    # 크롭 영역이 이미지를 벗어나는 경우 조정
    if crop_x < 0 or crop_y < 0 or crop_w > width or crop_h > height:
        # 이미지가 작은 경우 최대한 크롭
        aspect = target_ratio
        if width / height > aspect:
            # 높이 기준
            crop_h = height
            crop_w = int(height * aspect)
            crop_x = (width - crop_w) // 2
            crop_y = 0
        else:
            # 너비 기준
            crop_w = width
            crop_h = int(width / aspect)
            crop_x = 0
            crop_y = (height - crop_h) // 2
    
    # 크롭
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # PIL로 변환하여 리사이즈
    pil_cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    resized = pil_cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

def process_wedding_ring_v109(image_base64: str, metal_type: str = "auto") -> Dict:
    """Main processing function v109"""
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
        
        print(f"Processing v109: size={img_array.shape}")
        
        # Step 1: Detect wedding ring
        bbox = detect_wedding_ring_opencv(img_array)
        print(f"Ring detected at: {bbox}")
        
        # Step 2: Auto detect metal type if needed
        if metal_type == "auto":
            detected_metal = detect_metal_type(img_array, bbox)
            print(f"Auto-detected metal type: {detected_metal}")
        else:
            detected_metal = metal_type
        
        # Step 3: Create enhanced version (원본 크기 유지, 색감만 보정)
        pil_original = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        enhanced = enhance_wedding_ring_simple(pil_original, detected_metal)
        
        # Step 4: Create thumbnail (1000x1300 크롭)
        thumbnail_array = crop_for_thumbnail(img_array, bbox, (1000, 1300))
        pil_thumbnail = Image.fromarray(cv2.cvtColor(thumbnail_array, cv2.COLOR_BGR2RGB))
        thumbnail = enhance_wedding_ring_simple(pil_thumbnail, detected_metal)
        
        # Convert to base64
        # Enhanced image (원본 크기)
        enhanced_buffer = BytesIO()
        enhanced.save(enhanced_buffer, format='PNG', quality=95, optimize=True)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail (1000x1300)
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Log info
        print(f"Enhanced: {enhanced.size} - color enhanced only")
        print(f"Thumbnail: 1000x1300 - ring at 85%")
        print(f"Metal type: {detected_metal}")
        
        # Return with nested output structure
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "metal_type": detected_metal,
                "processing_version": "v109_final",
                "dimensions": {
                    "enhanced": f"{enhanced.width}x{enhanced.height}",
                    "thumbnail": "1000x1300"
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v109: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v109_final"
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        input_data = event.get("input", {})
        
        # Test mode
        if input_data.get("test") == True:
            return {
                "status": "test_success",
                "message": "Wedding Ring Processor v109 - Final Perfect Version",
                "version": "v109_final",
                "features": [
                    "Enhanced image: Original size with color enhancement only",
                    "Thumbnail: 1000x1300 crop with ring at 85%",
                    "Auto metal type detection",
                    "Fixed OpenCV float type error",
                    "Simple and reliable processing"
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
        
        # Get metal type
        metal_type = input_data.get("metal_type", "auto")
        
        # Process image
        return process_wedding_ring_v109(image_base64, metal_type)
        
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
