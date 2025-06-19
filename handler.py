import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import time
import traceback

def handler(event):
    """
    Wedding Ring AI v61 - Ultimate Black Border Removal
    대화 59번 최종 버전: 검은색 테두리 완전 제거
    """
    start_time = time.time()
    
    try:
        # 디버깅: 전체 이벤트 출력
        print(f"=== FULL EVENT DEBUG ===")
        print(f"Event: {event}")
        print(f"Event keys: {list(event.keys())}")
        
        if "input" in event:
            print(f"Input: {event['input']}")
            if isinstance(event['input'], dict):
                print(f"Input keys: {list(event['input'].keys())}")
                for key, value in event['input'].items():
                    print(f"  {key}: {str(value)[:100]}...")
        
        # Base64 이미지 추출
        base64_image = event.get("input", {}).get("image") or event.get("input", {}).get("image_base64")
        if not base64_image:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "failed",
                    "debug": event  # 전체 event 반환
                }
            }
        
        # 이미지 디코딩
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Padding 문제 해결
        # base64 문자열 길이가 4의 배수가 되도록 패딩 추가
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += '=' * (4 - missing_padding)
        
        try:
            image_data = base64.b64decode(base64_image)
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            print(f"Base64 preview: {base64_image[:100]}...")
            return {
                "output": {
                    "error": f"Failed to decode base64: {str(e)}",
                    "status": "failed"
                }
            }
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "output": {
                    "error": "Failed to decode image",
                    "status": "failed"
                }
            }
        
        print(f"Original image size: {img.shape}")
        
        # ========= PHASE 1: ULTRA BLACK BORDER REMOVAL =========
        # 대화 58번 + 59번 최강 버전
        img_removed = remove_black_border_ultra(img.copy())
        print(f"After border removal: {img_removed.shape}")
        
        # ========= PHASE 2: METAL & LIGHTING DETECTION =========
        metal_type, lighting = detect_metal_and_lighting(img_removed)
        print(f"Detected: {metal_type} under {lighting} lighting")
        
        # ========= PHASE 3: V13.3 ENHANCEMENT =========
        enhanced = apply_v13_3_enhancement(img_removed, metal_type, lighting)
        
        # ========= PHASE 4: ENSURE PURE WHITE BACKGROUND =========
        final = ensure_white_background(enhanced)
        
        # ========= PHASE 5: CREATE PERFECT THUMBNAIL =========
        # 검은색 제거된 이미지로 썸네일 생성
        thumbnail = create_perfect_thumbnail(final)
        
        # Convert to base64
        _, main_buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        return {
            "output": {
                "enhanced_image": f"data:image/jpeg;base64,{main_base64}",
                "thumbnail": f"data:image/jpeg;base64,{thumb_base64}",
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_removed": True,
                    "processing_time": time.time() - start_time,
                    "version": "v61_ultimate"
                }
            }
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

def remove_black_border_ultra(img):
    """
    대화 59번 최강 검은색 테두리 제거
    - 더 높은 threshold (150)
    - 더 넓은 스캔 범위 (50%)
    - 다단계 제거 프로세스
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1차: 초강력 검은색 감지 (threshold 150)
    borders = find_borders_ultra(gray, threshold=150, max_scan_ratio=0.5)
    
    # 2차: 추가 안전 마진 (50픽셀)
    borders = {
        'top': borders['top'] + 50,
        'bottom': borders['bottom'] + 50,
        'left': borders['left'] + 50,
        'right': borders['right'] + 50
    }
    
    # 크롭
    y1 = borders['top']
    y2 = h - borders['bottom']
    x1 = borders['left']
    x2 = w - borders['right']
    
    if y2 > y1 and x2 > x1:
        cropped = img[y1:y2, x1:x2]
        
        # 3차: 남은 회색 테두리 추가 제거
        h2, w2 = cropped.shape[:2]
        gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # 가장자리 20픽셀 체크하여 회색이면 추가 제거
        edge_check = 20
        if np.mean(gray2[:edge_check, :]) < 200:  # 상단
            cropped = cropped[edge_check:, :]
        if np.mean(gray2[-edge_check:, :]) < 200:  # 하단
            cropped = cropped[:-edge_check, :]
        if np.mean(gray2[:, :edge_check]) < 200:  # 좌측
            cropped = cropped[:, edge_check:]
        if np.mean(gray2[:, -edge_check:]) < 200:  # 우측
            cropped = cropped[:, :-edge_check]
        
        return cropped
    
    return img

def find_borders_ultra(gray, threshold=150, max_scan_ratio=0.5):
    """
    초강력 검은색 테두리 감지
    - threshold 150: 밝은 회색도 테두리로 감지
    - max_scan_ratio 0.5: 이미지의 50%까지 스캔
    """
    h, w = gray.shape
    max_scan = int(min(h, w) * max_scan_ratio)
    
    borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # Top border
    for y in range(min(max_scan, h)):
        if np.mean(gray[y, :]) > threshold:
            borders['top'] = y
            break
    
    # Bottom border
    for y in range(min(max_scan, h)):
        if np.mean(gray[h-1-y, :]) > threshold:
            borders['bottom'] = y
            break
    
    # Left border
    for x in range(min(max_scan, w)):
        if np.mean(gray[:, x]) > threshold:
            borders['left'] = x
            break
    
    # Right border
    for x in range(min(max_scan, w)):
        if np.mean(gray[:, w-1-x]) > threshold:
            borders['right'] = x
            break
    
    print(f"Detected borders: {borders}")
    return borders

def ensure_white_background(img):
    """
    완전한 흰색 배경 보장
    """
    # HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 회색/베이지색 배경을 흰색으로
    # Saturation이 낮고 Value가 높은 픽셀을 찾아 흰색으로
    mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 180)
    
    # 흰색으로 변경
    img[mask] = [255, 255, 255]
    
    return img

def create_perfect_thumbnail(img):
    """
    완벽한 1000x1300 썸네일 생성
    - 여백 완전 제거
    - 웨딩링이 화면의 85% 차지
    """
    h, w = img.shape[:2]
    
    # 웨딩링 영역 찾기 (더 민감하게)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 배경이 아닌 영역 찾기 (threshold 낮춤)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 모폴로지 연산으로 웨딩링 영역 확장
    kernel = np.ones((10, 10), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 모든 컨투어를 포함하는 바운딩 박스
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)
        
        # 여백 최소화 (10픽셀만)
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # 크롭
        ring_area = img[y_min:y_max, x_min:x_max]
        
        # 1000x1300으로 리사이즈
        # 비율 유지하면서 최대한 크게
        rh, rw = ring_area.shape[:2]
        scale = min(1000/rw, 1300/rh) * 0.85  # 85% 크기
        
        new_w = int(rw * scale)
        new_h = int(rh * scale)
        
        # 고품질 리사이즈
        pil_ring = Image.fromarray(cv2.cvtColor(ring_area, cv2.COLOR_BGR2RGB))
        pil_ring = pil_ring.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 흰 배경에 중앙 배치
        thumbnail = Image.new('RGB', (1000, 1300), (255, 255, 255))
        x_offset = (1000 - new_w) // 2
        y_offset = (1300 - new_h) // 2
        thumbnail.paste(pil_ring, (x_offset, y_offset))
        
        return cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)
    
    # 실패 시 기본 리사이즈
    return cv2.resize(img, (1000, 1300))

def detect_metal_and_lighting(img):
    """금속 타입과 조명 감지"""
    # 중앙 영역 추출
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    size = min(h, w) // 4
    
    roi = img[center_y-size:center_y+size, center_x-size:center_x+size]
    
    # HSV 분석
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_roi[:,:,0])
    avg_sat = np.mean(hsv_roi[:,:,1])
    avg_val = np.mean(hsv_roi[:,:,2])
    
    # 금속 타입 결정
    b, g, r = cv2.mean(roi)[:3]
    
    if avg_sat < 15:  # 저채도
        if avg_val > 200:
            metal_type = "white_gold"
        else:
            metal_type = "white_gold"  # 기본값
    elif 10 <= avg_hue <= 25:  # 주황색 계열
        if r > g > b and (r - b) > 30:
            metal_type = "rose_gold"
        else:
            metal_type = "champagne_gold"
    elif 25 <= avg_hue <= 35:  # 노란색 계열
        metal_type = "yellow_gold"
    else:
        metal_type = "champagne_gold"
    
    # 조명 조건 결정
    brightness = avg_val
    
    if brightness > 200:
        lighting = "natural"
    elif brightness > 150:
        lighting = "warm"
    else:
        lighting = "cool"
    
    return metal_type, lighting

def get_enhancement_params(metal_type, lighting):
    """v13.3 파라미터 - 28쌍 데이터 기반"""
    params = {
        "white_gold": {
            "natural": {"brightness": 1.08, "contrast": 1.05, "saturation": 0.95, 
                       "highlights": 1.10, "shadows": 0.95, "clarity": 1.10,
                       "white_balance": 0, "tint": 0, "gamma": 1.05, 
                       "sharpness": 1.15, "white_overlay": 0.02, "color_temp": 0},
            "warm": {"brightness": 1.10, "contrast": 1.08, "saturation": 0.90,
                    "highlights": 1.15, "shadows": 0.90, "clarity": 1.15,
                    "white_balance": -2, "tint": 0, "gamma": 1.08,
                    "sharpness": 1.18, "white_overlay": 0.03, "color_temp": -2},
            "cool": {"brightness": 1.05, "contrast": 1.10, "saturation": 0.88,
                    "highlights": 1.12, "shadows": 0.88, "clarity": 1.12,
                    "white_balance": 2, "tint": 0, "gamma": 1.10,
                    "sharpness": 1.20, "white_overlay": 0.05, "color_temp": 2}
        },
        "rose_gold": {
            "natural": {"brightness": 1.10, "contrast": 1.08, "saturation": 1.05,
                       "highlights": 1.08, "shadows": 0.92, "clarity": 1.08,
                       "white_balance": 2, "tint": 2, "gamma": 1.02,
                       "sharpness": 1.10, "white_overlay": 0, "color_temp": 1},
            "warm": {"brightness": 1.12, "contrast": 1.10, "saturation": 1.10,
                    "highlights": 1.10, "shadows": 0.90, "clarity": 1.10,
                    "white_balance": 3, "tint": 3, "gamma": 1.00,
                    "sharpness": 1.12, "white_overlay": 0, "color_temp": 2},
            "cool": {"brightness": 1.08, "contrast": 1.12, "saturation": 1.00,
                    "highlights": 1.05, "shadows": 0.88, "clarity": 1.05,
                    "white_balance": 0, "tint": 1, "gamma": 1.05,
                    "sharpness": 1.15, "white_overlay": 0.02, "color_temp": -1}
        },
        "yellow_gold": {
            "natural": {"brightness": 1.12, "contrast": 1.10, "saturation": 1.08,
                       "highlights": 1.05, "shadows": 0.90, "clarity": 1.05,
                       "white_balance": 3, "tint": 0, "gamma": 0.98,
                       "sharpness": 1.08, "white_overlay": 0, "color_temp": 2},
            "warm": {"brightness": 1.15, "contrast": 1.12, "saturation": 1.12,
                    "highlights": 1.08, "shadows": 0.88, "clarity": 1.08,
                    "white_balance": 5, "tint": 0, "gamma": 0.95,
                    "sharpness": 1.10, "white_overlay": 0, "color_temp": 3},
            "cool": {"brightness": 1.10, "contrast": 1.15, "saturation": 1.05,
                    "highlights": 1.00, "shadows": 0.85, "clarity": 1.00,
                    "white_balance": 1, "tint": 0, "gamma": 1.02,
                    "sharpness": 1.12, "white_overlay": 0.02, "color_temp": 0}
        },
        "champagne_gold": {
            "natural": {"brightness": 1.25, "contrast": 1.15, "saturation": 0.85,
                       "highlights": 1.20, "shadows": 0.85, "clarity": 1.20,
                       "white_balance": -3, "tint": 0, "gamma": 1.10,
                       "sharpness": 1.25, "white_overlay": 0.12, "color_temp": -5},
            "warm": {"brightness": 1.28, "contrast": 1.18, "saturation": 0.80,
                    "highlights": 1.25, "shadows": 0.80, "clarity": 1.25,
                    "white_balance": -5, "tint": 0, "gamma": 1.12,
                    "sharpness": 1.28, "white_overlay": 0.15, "color_temp": -6},
            "cool": {"brightness": 1.22, "contrast": 1.20, "saturation": 0.82,
                    "highlights": 1.18, "shadows": 0.82, "clarity": 1.18,
                    "white_balance": -2, "tint": 0, "gamma": 1.15,
                    "sharpness": 1.30, "white_overlay": 0.18, "color_temp": -4}
        }
    }
    
    return params.get(metal_type, params["white_gold"]).get(lighting, params["white_gold"]["natural"])

def apply_v13_3_enhancement(img, metal_type, lighting):
    """v13.3 10단계 보정 프로세스"""
    params = get_enhancement_params(metal_type, lighting)
    
    # 1. 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    # PIL 변환
    pil_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    
    # 2. 밝기 조정
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(params["brightness"])
    
    # 3. 대비 조정  
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(params["contrast"])
    
    # 4. 선명도 조정
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(params["sharpness"])
    
    # OpenCV로 변환
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 5. 채도 조정
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * params["saturation"]
    hsv[:,:,1][hsv[:,:,1] > 255] = 255
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 6. 하얀색 오버레이 (샴페인골드 특별 처리)
    if params["white_overlay"] > 0:
        white_layer = np.ones_like(enhanced) * 255
        enhanced = cv2.addWeighted(enhanced, 1 - params["white_overlay"], 
                                 white_layer, params["white_overlay"], 0)
    
    # 7. 색온도 조정 (LAB)
    if params["color_temp"] != 0:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:,:,2] = lab[:,:,2] + params["color_temp"] * 2
        lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 8. CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 9. 감마 보정
    gamma = params["gamma"]
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    # 10. 원본과 블렌딩
    result = cv2.addWeighted(denoised, 0.2, enhanced, 0.8, 0)
    
    return result

if __name__ == "__main__":
    # 테스트용
    test_event = {
        "input": {
            "image": "base64_encoded_image_here"
        }
    }
    result = handler(test_event)
    print(result)
