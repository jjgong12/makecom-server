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

def remove_black_borders_pixel_perfect(img_array):
    """픽셀 단위 정밀 검은색 제거"""
    height, width = img_array.shape[:2]
    
    # RGB로 작업 (정확한 검은색 감지)
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array.copy()
    
    logger.info(f"Original size: {width}x{height}")
    
    # 순수 검은색 픽셀 찾기 - RGB(0,0,0) ~ RGB(10,10,10)
    BLACK_THRESHOLD = 10
    
    # 위쪽에서 아래로 스캔
    top_crop = 0
    for y in range(height // 3):  # 상위 1/3만 스캔
        row = img_rgb[y, :]
        # 행의 90% 이상이 검은색이면 제거 대상
        black_pixels = np.sum(np.all(row <= BLACK_THRESHOLD, axis=1))
        if black_pixels > width * 0.9:
            top_crop = y + 1
        else:
            break
    
    # 아래에서 위로 스캔
    bottom_crop = height
    for y in range(height - 1, 2 * height // 3, -1):  # 하위 1/3만 스캔
        row = img_rgb[y, :]
        black_pixels = np.sum(np.all(row <= BLACK_THRESHOLD, axis=1))
        if black_pixels > width * 0.9:
            bottom_crop = y
        else:
            break
    
    # 왼쪽에서 오른쪽으로 스캔
    left_crop = 0
    for x in range(width // 3):
        col = img_rgb[:, x]
        black_pixels = np.sum(np.all(col <= BLACK_THRESHOLD, axis=1))
        if black_pixels > height * 0.9:
            left_crop = x + 1
        else:
            break
    
    # 오른쪽에서 왼쪽으로 스캔
    right_crop = width
    for x in range(width - 1, 2 * width // 3, -1):
        col = img_rgb[:, x]
        black_pixels = np.sum(np.all(col <= BLACK_THRESHOLD, axis=1))
        if black_pixels > height * 0.9:
            right_crop = x
        else:
            break
    
    logger.info(f"Black borders detected - Top: {top_crop}, Bottom: {height-bottom_crop}, Left: {left_crop}, Right: {width-right_crop}")
    
    # 크롭 실행
    if top_crop > 0 or bottom_crop < height or left_crop > 0 or right_crop < width:
        cropped = img_array[top_crop:bottom_crop, left_crop:right_crop]
        logger.info(f"Cropped to: {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped
    
    return img_array

def process_wedding_ring(img_array, metal_type="gold", lighting="studio"):
    """웨딩링 전문 보정 - 28쌍 학습데이터 + 10쌍 보정전/후 데이터 기반"""
    
    # 1. 먼저 검은색 테두리 제거
    img_array = remove_black_borders_pixel_perfect(img_array)
    
    # 2. 메탈 타입별 보정
    def enhance_metal(img, metal):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if metal == "gold":
            # 골드 향상
            l = cv2.add(l, 15)
            b = cv2.add(b, 10)
        elif metal == "silver":
            # 실버 향상
            l = cv2.add(l, 20)
            a = cv2.subtract(a, 5)
        elif metal == "rose_gold":
            # 로즈골드 향상
            l = cv2.add(l, 12)
            a = cv2.add(a, 8)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. 디테일 향상
    def enhance_details(img):
        # 샤프닝
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 블렌딩
        return cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
    
    # 4. 반사광 처리
    def enhance_reflections(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        # 하이라이트 부드럽게
        bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
        
        # 적용
        highlight = cv2.addWeighted(img, 1, cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR), 0.2, 0)
        return highlight
    
    # 5. 처리 실행
    processed = enhance_metal(img_array, metal_type)
    processed = enhance_details(processed)
    processed = enhance_reflections(processed)
    
    # 6. 최종 조정
    pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    # 명도/대비
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    # 채도
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.15)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_thumbnail_ultra(img_array, target_size=(800, 800)):
    """웨딩링 자동 감지 + 꽉찬 썸네일"""
    height, width = img_array.shape[:2]
    
    # 이미지를 grayscale로 변환하여 링 감지
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 이진화로 링 영역 찾기 (밝은 부분이 링)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어 찾기 (주로 링)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 15% 패딩 추가 (98% 차지하도록)
        padding = int(max(w, h) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        # 정사각형으로 만들기
        size = max(w, h)
        center_x = x + w // 2
        center_y = y + h // 2
        
        x = max(0, center_x - size // 2)
        y = max(0, center_y - size // 2)
        x2 = min(width, x + size)
        y2 = min(height, y + size)
        
        # 크롭
        cropped = img_array[y:y2, x:x2]
        
        logger.info(f"Ring detected and cropped: {cropped.shape}")
    else:
        # 링을 못 찾으면 중앙 크롭
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        cropped = img_array[y:y+size, x:x+size]
    
    # 리사이즈
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 추가 샤프닝
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def handler(event):
    """RunPod 핸들러"""
    try:
        logger.info("Handler started - v72 Pixel Perfect")
        
        # 입력 데이터 추출
        input_data = event.get("input", {})
        image_data = input_data.get("image") or input_data.get("image_base64")
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Base64 디코드 (padding 자동 처리)
        if not image_data.startswith("data:"):
            # padding 추가 (있어도 상관없음)
            padding = 4 - len(image_data) % 4
            if padding != 4:
                image_data += '=' * padding
        else:
            image_data = image_data.split(",")[1]
        
        # 이미지 디코드
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 파라미터
        metal_type = input_data.get("metal_type", "gold")
        lighting = input_data.get("lighting", "studio")
        
        logger.info(f"Processing: metal={metal_type}, lighting={lighting}, original_size={img_array.shape}")
        
        # 웨딩링 처리
        enhanced = process_wedding_ring(img_array, metal_type, lighting)
        
        # 썸네일 생성 (꽉찬 버전)
        thumbnail = create_thumbnail_ultra(enhanced)
        
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
                    "lighting": lighting,
                    "original_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
                    "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                    "thumbnail_size": f"{thumbnail.shape[1]}x{thumbnail.shape[0]}",
                    "status": "success",
                    "version": "v72-pixel-perfect"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v72-pixel-perfect"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
