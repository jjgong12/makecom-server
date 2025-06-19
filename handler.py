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

def remove_black_borders_aggressive(img_array):
    """공격적인 검은색 테두리 제거 - 확실하게"""
    height, width = img_array.shape[:2]
    
    # 매우 넓은 범위의 검은색/회색 감지
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    # 위에서 아래로 - threshold 80까지 확대
    top_crop = 0
    for y in range(height // 2):
        if np.mean(gray[y, :]) < 80:  # 매우 관대한 threshold
            top_crop = y + 1
        else:
            break
    
    # 아래에서 위로
    bottom_crop = height
    for y in range(height - 1, height // 2, -1):
        if np.mean(gray[y, :]) < 80:
            bottom_crop = y
        else:
            break
    
    # 왼쪽에서 오른쪽으로
    left_crop = 0
    for x in range(width // 2):
        if np.mean(gray[:, x]) < 80:
            left_crop = x + 1
        else:
            break
    
    # 오른쪽에서 왼쪽으로
    right_crop = width
    for x in range(width - 1, width // 2, -1):
        if np.mean(gray[:, x]) < 80:
            right_crop = x
        else:
            break
    
    logger.info(f"Aggressive crop - T:{top_crop}, B:{height-bottom_crop}, L:{left_crop}, R:{width-right_crop}")
    
    # 크롭 실행
    cropped = img_array[top_crop:bottom_crop, left_crop:right_crop]
    
    # 추가 안전 마진 (10픽셀 더 제거)
    if cropped.shape[0] > 20 and cropped.shape[1] > 20:
        cropped = cropped[10:-10, 10:-10]
    
    return cropped

def process_wedding_ring(img_array, metal_type="gold", lighting="studio"):
    """웨딩링 전문 보정 - 배경 깨끗하게"""
    
    # 1. 검은색 테두리 제거
    img_array = remove_black_borders_aggressive(img_array)
    
    # 2. 배경을 깨끗한 화이트/라이트그레이로 변경
    def clean_background(img):
        # HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 링 마스크 생성 (밝은 부분이 링)
        _, _, v = cv2.split(hsv)
        _, ring_mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)
        
        # 모폴로지로 마스크 정리
        kernel = np.ones((5,5), np.uint8)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
        
        # 배경을 깨끗한 라이트그레이로
        background = np.full_like(img, [245, 245, 245])  # 밝은 회색
        
        # 링 부분만 원본 유지
        result = cv2.bitwise_and(img, img, mask=ring_mask)
        background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(ring_mask))
        
        return cv2.add(result, background_masked)
    
    # 3. 메탈 타입별 보정
    def enhance_metal(img, metal):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if metal == "gold" or metal == "yellow_gold":
            # 골드 - 따뜻한 톤
            l = cv2.add(l, 10)
            b = cv2.add(b, 15)
        elif metal == "white" or metal == "white_gold":
            # 화이트/무도금화이트 - 차가운 실버톤
            l = cv2.add(l, 25)  # 더 밝게
            a = cv2.subtract(a, 8)  # 약간 블루톤
            b = cv2.subtract(b, 10)  # 옐로우 제거
        elif metal == "rose_gold":
            # 로즈골드
            l = cv2.add(l, 12)
            a = cv2.add(a, 10)
            b = cv2.add(b, 5)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. 처리 실행
    processed = clean_background(img_array)
    processed = enhance_metal(processed, metal_type)
    
    # 5. 디테일 향상
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(processed, -1, kernel)
    processed = cv2.addWeighted(processed, 0.8, sharpened, 0.2, 0)
    
    # 6. 최종 조정
    pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    # 밝기/대비
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    # 채도 (화이트는 낮게)
    if metal_type in ["white", "white_gold"]:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(0.9)  # 채도 낮춤
    else:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.2)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_thumbnail_perfect(img_array, target_size=(800, 800)):
    """완벽한 썸네일 - 링만 꽉차게"""
    height, width = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 적응형 threshold로 링 감지
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 반전 (링이 흰색이 되도록)
    binary = cv2.bitwise_not(binary)
    
    # 노이즈 제거
    kernel = np.ones((7,7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 면적 기준 상위 2개 컨투어 (두 개의 링)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        # 전체 바운딩 박스
        x_min, y_min = width, height
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 5% 패딩
        w = x_max - x_min
        h = y_max - y_min
        padding = int(max(w, h) * 0.05)
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # 정사각형으로
        size = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(width, x1 + size)
        y2 = min(height, y1 + size)
        
        cropped = img_array[y1:y2, x1:x2]
    else:
        # 중앙 크롭
        size = int(min(width, height) * 0.8)
        x = (width - size) // 2
        y = (height - size) // 2
        cropped = img_array[y:y+size, x:x+size]
    
    # 리사이즈
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 샤프닝
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def handler(event):
    """RunPod 핸들러 v90"""
    try:
        logger.info("Handler started - v90 Final")
        
        # 입력 데이터 추출
        input_data = event.get("input", {})
        image_data = input_data.get("image") or input_data.get("image_base64")
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Base64 디코드
        if not image_data.startswith("data:"):
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
        
        # 무도금화이트는 white로 처리
        if metal_type == "무도금화이트":
            metal_type = "white"
        
        logger.info(f"Processing: metal={metal_type}, lighting={lighting}, size={img_array.shape}")
        
        # 웨딩링 처리
        enhanced = process_wedding_ring(img_array, metal_type, lighting)
        
        # 썸네일 생성
        thumbnail = create_thumbnail_perfect(enhanced)
        
        # Base64 인코딩 (padding 제거)
        _, buffer = cv2.imencode('.png', enhanced)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8').rstrip('=')
        
        _, thumb_buffer = cv2.imencode('.png', thumbnail)
        thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8').rstrip('=')
        
        logger.info("Processing completed successfully")
        
        # RunPod return 구조
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
                    "version": "v90-final"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v90-final"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
