import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
from io import BytesIO
import runpod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_ring_area(img_array):
    """웨딩링 영역 감지 - 원형과 금속 특성으로 판단"""
    height, width = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 중앙 60% 영역에서 웨딩링 찾기
    center_y, center_x = height // 2, width // 2
    roi_h, roi_w = int(height * 0.6), int(width * 0.6)
    roi_y, roi_x = center_y - roi_h // 2, center_x - roi_w // 2
    
    roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # 밝은 영역 감지 (웨딩링은 일반적으로 밝음)
    threshold = np.mean(roi) + np.std(roi) * 0.5
    _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
    
    # 원형 감지
    circles = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=min(roi_h, roi_w)//2
    )
    
    if circles is not None:
        # 가장 큰 원을 웨딩링으로 간주
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda c: c[2])
        x, y, r = largest_circle
        
        # 전체 이미지 좌표로 변환
        ring_x = roi_x + x
        ring_y = roi_y + y
        ring_r = int(r * 1.3)  # 여유 있게 30% 확대
        
        return {
            'x': max(0, ring_x - ring_r),
            'y': max(0, ring_y - ring_r),
            'w': min(width - (ring_x - ring_r), ring_r * 2),
            'h': min(height - (ring_y - ring_r), ring_r * 2)
        }
    
    # 원형 감지 실패시 밝은 영역 기준
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 전체 이미지 좌표로 변환하고 여유 추가
        padding = int(max(w, h) * 0.2)
        return {
            'x': max(0, roi_x + x - padding),
            'y': max(0, roi_y + y - padding),
            'w': min(width - (roi_x + x - padding), w + padding * 2),
            'h': min(height - (roi_y + y - padding), h + padding * 2)
        }
    
    # 감지 실패시 중앙 50% 영역 보호
    return {
        'x': int(width * 0.25),
        'y': int(height * 0.25),
        'w': int(width * 0.5),
        'h': int(height * 0.5)
    }

def remove_black_borders_with_protection(img_array):
    """검은색 테두리를 크롭으로 제거 - 웨딩링 보호"""
    height, width = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    logger.info(f"Original size: {width}x{height}")
    
    # 웨딩링 영역 감지
    ring_area = detect_ring_area(img_array)
    ring_x, ring_y = ring_area['x'], ring_area['y']
    ring_w, ring_h = ring_area['w'], ring_area['h']
    
    logger.info(f"Ring area detected: x={ring_x}, y={ring_y}, w={ring_w}, h={ring_h}")
    
    # 가장자리에서만 검은색 찾기
    edges = {'top': 0, 'bottom': height, 'left': 0, 'right': width}
    
    # 상단 - 웨딩링 영역 전까지만 검사
    max_top_scan = min(ring_y, height // 3)
    for y in range(max_top_scan):
        row_mean = np.mean(gray[y, :])
        if row_mean > 50:  # 검은색 끝
            edges['top'] = y
            break
    
    # 하단 - 웨딩링 영역 이후부터만 검사
    min_bottom_scan = max(ring_y + ring_h, 2 * height // 3)
    for y in range(height - 1, min_bottom_scan, -1):
        row_mean = np.mean(gray[y, :])
        if row_mean > 50:
            edges['bottom'] = y + 1
            break
    
    # 좌측 - 웨딩링 영역 전까지만 검사
    max_left_scan = min(ring_x, width // 3)
    for x in range(max_left_scan):
        col_mean = np.mean(gray[:, x])
        if col_mean > 50:
            edges['left'] = x
            break
    
    # 우측 - 웨딩링 영역 이후부터만 검사
    min_right_scan = max(ring_x + ring_w, 2 * width // 3)
    for x in range(width - 1, min_right_scan, -1):
        col_mean = np.mean(gray[:, x])
        if col_mean > 50:
            edges['right'] = x + 1
            break
    
    logger.info(f"Black borders detected: top={edges['top']}, bottom={edges['bottom']}, left={edges['left']}, right={edges['right']}")
    
    # 크롭 실행
    cropped = img_array[edges['top']:edges['bottom'], edges['left']:edges['right']]
    logger.info(f"Cropped to: {cropped.shape[1]}x{cropped.shape[0]}")
    
    return cropped

def enhance_wedding_ring(img_array, metal_type="white_gold"):
    """웨딩링 디테일 보정 - 색상 변환 없이 원본 유지"""
    
    # 1. 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    
    # 2. 샤프닝
    pil_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    sharpener = ImageEnhance.Sharpness(pil_img)
    sharpened = sharpener.enhance(1.5)
    
    # 3. 밝기/대비 - PIL로만 처리 (색상 변환 없이)
    brightness = ImageEnhance.Brightness(sharpened)
    brightened = brightness.enhance(1.15)
    
    contrast = ImageEnhance.Contrast(brightened)
    contrasted = contrast.enhance(1.1)
    
    # 4. 다시 OpenCV 형식으로
    enhanced = cv2.cvtColor(np.array(contrasted), cv2.COLOR_RGB2BGR)
    
    # 5. 추가 디테일 향상
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    detailed = cv2.filter2D(enhanced, -1, kernel)
    
    # 6. 블렌딩
    final = cv2.addWeighted(enhanced, 0.7, detailed, 0.3, 0)
    
    return final

def create_centered_thumbnail(img_array):
    """중앙 정렬된 1000x1300 썸네일 생성"""
    height, width = img_array.shape[:2]
    target_w, target_h = 1000, 1300
    
    # 웨딩링 영역 감지
    ring_area = detect_ring_area(img_array)
    ring_center_x = ring_area['x'] + ring_area['w'] // 2
    ring_center_y = ring_area['y'] + ring_area['h'] // 2
    
    logger.info(f"Ring center for thumbnail: ({ring_center_x}, {ring_center_y})")
    
    # 웨딩링 중심으로 크롭 영역 계산
    # 비율 유지하면서 최대한 크게
    scale = max(target_w / width, target_h / height) * 1.2  # 20% 여유
    
    crop_w = int(target_w / scale)
    crop_h = int(target_h / scale)
    
    # 웨딩링이 중앙에 오도록
    crop_x = max(0, min(width - crop_w, ring_center_x - crop_w // 2))
    crop_y = max(0, min(height - crop_h, ring_center_y - crop_h // 2))
    
    # 크롭
    cropped = img_array[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # 리사이즈
    thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 약간 밝게
    thumbnail = cv2.convertScaleAbs(thumbnail, alpha=1.05, beta=5)
    
    logger.info(f"Thumbnail created: {target_w}x{target_h}, centered on ring")
    
    return thumbnail

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job['input']
        
        # Base64 이미지 디코딩
        image_data = job_input.get('image', '')
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # padding 추가 (Make.com 호환성)
        padding = 4 - len(image_data) % 4
        if padding != 4:
            image_data += '=' * padding
            
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 메탈 타입
        metal_type = job_input.get('metal_type', 'white_gold').lower()
        
        logger.info(f"Processing: metal={metal_type}, original_size={img_array.shape}")
        
        # 1. 검은색 테두리 제거 (웨딩링 보호)
        no_border = remove_black_borders_with_protection(img_array)
        
        # 2. 웨딩링 보정 (색상 변환 없이)
        enhanced = enhance_wedding_ring(no_border, metal_type)
        
        # 3. 중앙 정렬 썸네일 생성
        thumbnail = create_centered_thumbnail(enhanced)
        
        # Base64 인코딩 (padding 제거)
        _, buffer = cv2.imencode('.png', enhanced)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8').rstrip('=')
        
        _, thumb_buffer = cv2.imencode('.png', thumbnail)
        thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8').rstrip('=')
        
        logger.info("Processing completed successfully")
        
        # RunPod가 자동으로 data.output으로 감싸므로 우리는 output만 반환
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "original_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
                    "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                    "thumbnail_size": f"{thumbnail.shape[1]}x{thumbnail.shape[0]}",
                    "status": "success",
                    "version": "v91-crop-ring-protection"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v91-crop-ring-protection"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
