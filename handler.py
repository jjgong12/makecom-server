import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

def handler(event):
    """
    RunPod handler - v21.0 Perfect
    검은색 선 100% 제거 + 완벽한 보정
    """
    try:
        # 입력 데이터 확인
        if "input" not in event:
            return {"error": "No input provided"}
            
        input_data = event["input"]
        
        # 간단한 연결 테스트
        if "test" in input_data:
            return {"status": "Wedding Ring AI v21.0 Ready", "message": "System operational"}
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "No image_base64 in input"}
            
        # Base64 디코딩
        try:
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return {"error": f"Image decode error: {str(e)}"}
        
        # ========== STEP 1: 금속 타입 감지 ==========
        def detect_metal_type(img):
            """금속 타입 감지 - 샴페인골드 우선 감지"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # 중앙 영역에서 샘플링
            h, w = img.shape[:2]
            center_region = hsv[h//3:2*h//3, w//3:2*w//3]
            
            avg_hue = np.mean(center_region[:, :, 0])
            avg_sat = np.mean(center_region[:, :, 1])
            avg_val = np.mean(center_region[:, :, 2])
            
            # 샴페인골드 우선 감지 (베이지톤)
            if 15 <= avg_hue <= 35 and avg_sat < 50 and avg_val > 150:
                return 'champagne_gold'
            elif avg_sat < 30:
                return 'white_gold' 
            elif 10 <= avg_hue <= 25:
                return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
            elif avg_hue < 10 or avg_hue > 170:
                return 'rose_gold'
            else:
                return 'white_gold'
        
        # ========== STEP 2: 조명 환경 감지 ==========
        def detect_lighting(img):
            """조명 환경 감지"""
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            b_channel = lab[:, :, 2]
            b_mean = np.mean(b_channel)
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
        
        # ========== STEP 3: v13.3 완전한 파라미터 ==========
        PARAMS = {
            'white_gold': {
                'natural': {'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
                           'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
                           'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01},
                'warm': {'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
                        'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
                        'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98},
                'cool': {'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
                        'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
                        'original_blend': 0.12, 'saturation': 1.03, 'gamma': 1.02}
            },
            'rose_gold': {
                'natural': {'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
                           'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
                           'original_blend': 0.20, 'saturation': 1.10, 'gamma': 0.98},
                'warm': {'brightness': 1.12, 'contrast': 1.06, 'white_overlay': 0.08,
                        'sharpness': 1.12, 'color_temp_a': 3, 'color_temp_b': 2,
                        'original_blend': 0.22, 'saturation': 1.08, 'gamma': 0.96},
                'cool': {'brightness': 1.18, 'contrast': 1.10, 'white_overlay': 0.05,
                        'sharpness': 1.18, 'color_temp_a': 0, 'color_temp_b': 0,
                        'original_blend': 0.18, 'saturation': 1.12, 'gamma': 1.00}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.30, 'contrast': 1.15, 'white_overlay': 0.15,
                           'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
                           'original_blend': 0.12, 'saturation': 0.90, 'gamma': 1.02},
                'warm': {'brightness': 1.28, 'contrast': 1.13, 'white_overlay': 0.18,
                        'sharpness': 1.18, 'color_temp_a': -8, 'color_temp_b': -8,
                        'original_blend': 0.15, 'saturation': 0.88, 'gamma': 1.00},
                'cool': {'brightness': 1.32, 'contrast': 1.17, 'white_overlay': 0.13,
                        'sharpness': 1.22, 'color_temp_a': -5, 'color_temp_b': -5,
                        'original_blend': 0.10, 'saturation': 0.92, 'gamma': 1.03}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
                           'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
                           'original_blend': 0.22, 'saturation': 1.15, 'gamma': 0.97},
                'warm': {'brightness': 1.14, 'contrast': 1.07, 'white_overlay': 0.07,
                        'sharpness': 1.12, 'color_temp_a': 4, 'color_temp_b': 3,
                        'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.95},
                'cool': {'brightness': 1.18, 'contrast': 1.11, 'white_overlay': 0.04,
                        'sharpness': 1.16, 'color_temp_a': 2, 'color_temp_b': 1,
                        'original_blend': 0.20, 'saturation': 1.18, 'gamma': 0.99}
            }
        }
        
        # ========== STEP 4: 10단계 보정 프로세스 ==========
        def enhance_ring_v13_3(img, params):
            """v13.3 완전한 10단계 보정"""
            result = img.copy()
            
            # 1. 노이즈 제거
            result = cv2.bilateralFilter(result, 9, 75, 75)
            
            # PIL 변환
            pil_img = Image.fromarray(result)
            
            # 2. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(params['brightness'])
            
            # 3. 대비 조정
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(params['contrast'])
            
            # 4. 선명도 조정
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(params['sharpness'])
            
            # 다시 numpy로
            result = np.array(pil_img)
            
            # 5. 채도 조정
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * params['saturation']
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # 6. 하얀색 오버레이
            white_layer = np.full_like(result, 255)
            result = cv2.addWeighted(result, 1 - params['white_overlay'], 
                                   white_layer, params['white_overlay'], 0)
            
            # 7. 색온도 조정 (LAB)
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            # 8. CLAHE (명료도)
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9. 감마 보정
            gamma = params['gamma']
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(result, table)
            
            # 10. 원본과 블렌딩
            result = cv2.addWeighted(result, 1 - params['original_blend'],
                                   img, params['original_blend'], 0)
            
            return result
        
        # ========== STEP 5: 검은색 테두리 감지 및 제거 ==========
        def detect_and_remove_border(img):
            """검은색 테두리 강력하게 감지하고 제거"""
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 검은색 테두리 감지 - 매우 낮은 threshold부터 시작
            border_mask = np.zeros_like(gray)
            
            # 여러 threshold로 시도 (5-40)
            for thresh in [5, 10, 15, 20, 25, 30, 35, 40]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                border_mask = cv2.bitwise_or(border_mask, binary)
            
            # 테두리만 찾기 (전체 이미지 스캔)
            # 상하좌우 각각 200픽셀까지 확인
            edge_thickness = 200
            
            # 상단
            top_region = border_mask[:edge_thickness, :]
            if np.sum(top_region) > w * edge_thickness * 0.5:  # 50% 이상이 검은색이면
                # 실제 두께 측정
                for y in range(edge_thickness):
                    if np.sum(border_mask[y, :]) < w * 0.5:
                        top_thickness = y
                        break
                else:
                    top_thickness = edge_thickness
            else:
                top_thickness = 0
            
            # 하단
            bottom_region = border_mask[-edge_thickness:, :]
            if np.sum(bottom_region) > w * edge_thickness * 0.5:
                for y in range(edge_thickness):
                    if np.sum(border_mask[h-1-y, :]) < w * 0.5:
                        bottom_thickness = y
                        break
                else:
                    bottom_thickness = edge_thickness
            else:
                bottom_thickness = 0
            
            # 좌측
            left_region = border_mask[:, :edge_thickness]
            if np.sum(left_region) > h * edge_thickness * 0.5:
                for x in range(edge_thickness):
                    if np.sum(border_mask[:, x]) < h * 0.5:
                        left_thickness = x
                        break
                else:
                    left_thickness = edge_thickness
            else:
                left_thickness = 0
            
            # 우측
            right_region = border_mask[:, -edge_thickness:]
            if np.sum(right_region) > h * edge_thickness * 0.5:
                for x in range(edge_thickness):
                    if np.sum(border_mask[:, w-1-x]) < h * 0.5:
                        right_thickness = x
                        break
                else:
                    right_thickness = edge_thickness
            else:
                right_thickness = 0
            
            # 테두리가 감지되면 제거
            if max(top_thickness, bottom_thickness, left_thickness, right_thickness) > 10:
                result = img.copy()
                
                # 배경색 설정 (깨끗한 밝은 회색)
                bg_color = np.array([250, 248, 245])
                
                # 각 테두리 제거 + 블렌딩
                if top_thickness > 0:
                    # 완전 제거
                    result[:top_thickness+20, :] = bg_color
                    # 부드러운 블렌딩
                    for i in range(20):
                        alpha = i / 20.0
                        y = top_thickness + 20 + i
                        if y < h:
                            result[y, :] = result[y, :] * alpha + bg_color * (1 - alpha)
                
                if bottom_thickness > 0:
                    result[h-bottom_thickness-20:, :] = bg_color
                    for i in range(20):
                        alpha = i / 20.0
                        y = h - bottom_thickness - 20 - i
                        if y >= 0:
                            result[y, :] = result[y, :] * alpha + bg_color * (1 - alpha)
                
                if left_thickness > 0:
                    result[:, :left_thickness+20] = bg_color
                    for i in range(20):
                        alpha = i / 20.0
                        x = left_thickness + 20 + i
                        if x < w:
                            result[:, x] = result[:, x] * alpha + bg_color * (1 - alpha)
                
                if right_thickness > 0:
                    result[:, w-right_thickness-20:] = bg_color
                    for i in range(20):
                        alpha = i / 20.0
                        x = w - right_thickness - 20 - i
                        if x >= 0:
                            result[:, x] = result[:, x] * alpha + bg_color * (1 - alpha)
                
                # 웨딩링 영역 추가 보정 (더 밝고 선명하게)
                inner_y = top_thickness + 50
                inner_x = left_thickness + 50
                inner_h = h - top_thickness - bottom_thickness - 100
                inner_w = w - left_thickness - right_thickness - 100
                
                if inner_h > 100 and inner_w > 100:
                    ring_region = result[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
                    # 추가 밝기와 선명도
                    ring_region = cv2.convertScaleAbs(ring_region, alpha=1.1, beta=10)
                    # 언샤프 마스크
                    gaussian = cv2.GaussianBlur(ring_region, (0, 0), 2.0)
                    ring_region = cv2.addWeighted(ring_region, 1.5, gaussian, -0.5, 0)
                    result[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = ring_region
                
                return result, (left_thickness, top_thickness, right_thickness, bottom_thickness)
            
            return img, (0, 0, 0, 0)
        
        # ========== STEP 6: 썸네일 생성 ==========
        def create_thumbnail(img, border_info):
            """1000x1300 썸네일 생성"""
            h, w = img.shape[:2]
            left, top, right, bottom = border_info
            
            # 웨딩링 영역 계산
            ring_x = left + 30
            ring_y = top + 30
            ring_w = w - left - right - 60
            ring_h = h - top - bottom - 60
            
            # 안전 체크
            if ring_w < 100 or ring_h < 100:
                # 중앙 60% 영역 사용
                ring_x = int(w * 0.2)
                ring_y = int(h * 0.2)
                ring_w = int(w * 0.6)
                ring_h = int(h * 0.6)
            
            # 크롭
            cropped = img[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w]
            
            # 1000x1300 비율 맞추기
            target_w, target_h = 1000, 1300
            crop_h, crop_w = cropped.shape[:2]
            
            # 스케일 계산 (여백 최소화)
            scale = max(target_w / crop_w, target_h / crop_h) * 0.95
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            # 리사이즈
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 캔버스에 배치
            canvas = np.full((target_h, target_w, 3), 250, dtype=np.uint8)
            start_y = max(0, (target_h - new_h) // 3)  # 위쪽에 배치
            start_x = (target_w - new_w) // 2
            
            # 범위 체크
            end_y = min(start_y + new_h, target_h)
            end_x = min(start_x + new_w, target_w)
            
            canvas[start_y:end_y, start_x:end_x] = resized[:end_y-start_y, :end_x-start_x]
            
            return canvas
        
        # ========== 메인 처리 프로세스 ==========
        # 1. 금속과 조명 감지
        metal_type = detect_metal_type(image_array)
        lighting = detect_lighting(image_array)
        
        # 2. v13.3 보정 적용
        params = PARAMS.get(metal_type, PARAMS['white_gold']).get(lighting, PARAMS['white_gold']['natural'])
        enhanced = enhance_ring_v13_3(image_array, params)
        
        # 3. 검은색 테두리 제거
        borderless, border_info = detect_and_remove_border(enhanced)
        
        # 4. 2x 업스케일링
        h, w = borderless.shape[:2]
        upscaled = cv2.resize(borderless, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
        
        # 5. 썸네일 생성
        thumbnail = create_thumbnail(borderless, border_info)
        
        # ========== 결과 인코딩 ==========
        # 메인 이미지
        main_pil = Image.fromarray(upscaled.astype(np.uint8))
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail.astype(np.uint8))
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": max(border_info) > 0,
                    "border_thickness": border_info,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "upscaled_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "version": "v21.0 Perfect"
                }
            }
        }
        
    except Exception as e:
        # 에러가 나도 무조건 보정된 이미지 반환
        try:
            # 최소한의 보정이라도 적용
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=20)
            h, w = enhanced.shape[:2]
            
            # 간단한 업스케일링
            upscaled = cv2.resize(enhanced, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
            
            # 간단한 썸네일
            thumb_size = min(h, w)
            start_y = (h - thumb_size) // 2
            start_x = (w - thumb_size) // 2
            cropped = enhanced[start_y:start_y+thumb_size, start_x:start_x+thumb_size]
            thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LINEAR)
            
            # 인코딩
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=90)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=90)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "output": {
                    "enhanced_image": main_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "error": str(e),
                        "fallback": True,
                        "version": "v21.0 Fallback"
                    }
                }
            }
        except:
            return {"error": f"Critical error: {str(e)}"}

# RunPod handler 시작
runpod.serverless.start({"handler": handler})
