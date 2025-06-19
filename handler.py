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

def remove_black_borders_adaptive_canny(img_array):
    """단계적 threshold + Canny Edge Detection 결합"""
    height, width = img_array.shape[:2]
    
    # RGB로 작업
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        gray = img_array
    else:
        img_rgb = img_array.copy()
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    logger.info(f"Original size: {width}x{height}")
    
    # 방법 1: Canny Edge Detection
    def get_canny_borders():
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # 엣지 찾기
        canny_edges = {'top': 0, 'bottom': height, 'left': 0, 'right': width}
        
        # 위에서 아래로
        for y in range(height // 2):
            if np.sum(edges[y, :]) > width * 0.1:  # 10% 이상 엣지 있으면
                canny_edges['top'] = max(0, y - 5)  # 5픽셀 여유
                break
        
        # 아래에서 위로
        for y in range(height - 1, height // 2, -1):
            if np.sum(edges[y, :]) > width * 0.1:
                canny_edges['bottom'] = min(height, y + 5)
                break
        
        # 왼쪽에서 오른쪽으로
        for x in range(width // 2):
            if np.sum(edges[:, x]) > height * 0.1:
                canny_edges['left'] = max(0, x - 5)
                break
        
        # 오른쪽에서 왼쪽으로
        for x in range(width - 1, width // 2, -1):
            if np.sum(edges[:, x]) > height * 0.1:
                canny_edges['right'] = min(width, x + 5)
                break
        
        return canny_edges
    
    # 방법 2: 단계적 Threshold
    def get_adaptive_borders():
        threshold_edges = None
        detected_threshold = None
        
        # 10부터 50까지 단계적 시도
        for threshold in [10, 20, 30, 40, 50]:
            edges = {'top': 0, 'bottom': height, 'left': 0, 'right': width}
            found = False
            
            # 위쪽 스캔
            for y in range(height // 2):
                row = img_rgb[y, :]
                # 모든 픽셀이 threshold 이하인지 체크
                if np.all(np.max(row, axis=1) <= threshold):
                    edges['top'] = y + 1
                    found = True
                else:
                    break
            
            # 검은색 감지되면 이 threshold로 나머지 방향도 스캔
            if found:
                # 아래쪽
                for y in range(height - 1, height // 2, -1):
                    row = img_rgb[y, :]
                    if np.all(np.max(row, axis=1) <= threshold):
                        edges['bottom'] = y
                    else:
                        break
                
                # 왼쪽
                for x in range(width // 2):
                    col = img_rgb[:, x]
                    if np.all(np.max(col, axis=1) <= threshold):
                        edges['left'] = x + 1
                    else:
                        break
                
                # 오른쪽
                for x in range(width - 1, width // 2, -1):
                    col = img_rgb[:, x]
                    if np.all(np.max(col, axis=1) <= threshold):
                        edges['right'] = x
                    else:
                        break
                
                threshold_edges = edges
                detected_threshold = threshold
                logger.info(f"Black borders detected with threshold {threshold}")
                break
        
        return threshold_edges, detected_threshold
    
    # 두 방법 실행
    canny_borders = get_canny_borders()
    threshold_borders, used_threshold = get_adaptive_borders()
    
    # 두 방법 중 더 보수적인 값 선택 (더 많이 자르는 쪽)
    if threshold_borders:
        final_edges = {
            'top': max(canny_borders['top'], threshold_borders['top']),
            'bottom': min(canny_borders['bottom'], threshold_borders['bottom']),
            'left': max(canny_borders['left'], threshold_borders['left']),
            'right': min(canny_borders['right'], threshold_borders['right'])
        }
        logger.info(f"Combined borders - Threshold({used_threshold}): {threshold_borders}, Canny: {canny_borders}")
    else:
        final_edges = canny_borders
        logger.info(f"Using only Canny borders: {canny_borders}")
    
    # 크롭 실행
    top, bottom = final_edges['top'], final_edges['bottom']
    left, right = final_edges['left'], final_edges['right']
    
    if top > 0 or bottom < height or left > 0 or right < width:
        cropped = img_array[top:bottom, left:right]
        logger.info(f"Cropped from {width}x{height} to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped
    
    return img_array

def process_wedding_ring(img_array, metal_type="gold", lighting="studio"):
    """웨딩링 전문 보정 - 28쌍 학습데이터 + 10쌍 보정전/후 데이터 기반"""
    
    # 1. 먼저 검은색 테두리 제거 (Adaptive + Canny)
    img_array = remove_black_borders_adaptive_canny(img_array)
    
    # 2. 메탈 타입별 보정
    def enhance_metal(img, metal):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if metal == "gold" or metal == "yellow_gold":
            # 골드/옐로우골드 향상
            l = cv2.add(l, 15)
            b = cv2.add(b, 10)
        elif metal == "silver" or metal == "white_gold" or metal == "white":
            # 실버/화이트골드 향상
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
    """웨딩링 자동 감지 + 꽉찬 썸네일 (threshold 60)"""
    height, width = img_array.shape[:2]
    
    # 이미지를 grayscale로 변환하여 링 감지
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # threshold 60으로 링 영역 찾기 (어두운 배경 제외)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어 찾기 (주로 링)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 10% 패딩 추가 (더 꽉 차게)
        padding = int(max(w, h) * 0.1)
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
        
        logger.info(f"Ring detected and cropped: {cropped.shape} (threshold=60)")
    else:
        # 링을 못 찾으면 중앙 크롭
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        cropped = img_array[y:y+size, x:x+size]
        logger.info("Ring not detected, using center crop")
    
    # 리사이즈
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 추가 샤프닝
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def handler(event):
    """RunPod 핸들러 v90"""
    try:
        logger.info("Handler started - v90 Adaptive+Canny")
        
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
        
        # 썸네일 생성 (threshold 60)
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
                    "version": "v90-adaptive-canny"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v90-adaptive-canny"
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
