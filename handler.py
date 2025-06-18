import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import json
import time

class WeddingRingEnhancer:
    def __init__(self):
        self.version = "v23.0"
        
        # v13.3 완전한 파라미터 세트 (28쌍 학습 데이터 기반)
        self.metal_params = {
            'white_gold': {
                'natural': {
                    'brightness': 1.15,
                    'contrast': 1.05,
                    'highlights': 1.20,
                    'shadows': 0.85,
                    'whites': 1.25,
                    'blacks': 0.10,
                    'vibrance': 1.05,
                    'saturation': 0.95,
                    'white_overlay': 0.04,
                    'gaussian_blur': 0.5
                },
                'warm': {
                    'brightness': 1.18,
                    'contrast': 1.08,
                    'highlights': 1.25,
                    'shadows': 0.82,
                    'whites': 1.30,
                    'blacks': 0.08,
                    'vibrance': 1.08,
                    'saturation': 0.93,
                    'white_overlay': 0.05,
                    'gaussian_blur': 0.3
                },
                'cool': {
                    'brightness': 1.12,
                    'contrast': 1.03,
                    'highlights': 1.18,
                    'shadows': 0.87,
                    'whites': 1.22,
                    'blacks': 0.12,
                    'vibrance': 1.02,
                    'saturation': 0.97,
                    'white_overlay': 0.03,
                    'gaussian_blur': 0.7
                }
            },
            'yellow_gold': {
                'natural': {
                    'brightness': 1.08,
                    'contrast': 1.10,
                    'highlights': 1.15,
                    'shadows': 0.88,
                    'whites': 1.12,
                    'blacks': 0.15,
                    'vibrance': 1.12,
                    'saturation': 0.88,
                    'white_overlay': 0.08,
                    'gaussian_blur': 0.4
                },
                'warm': {
                    'brightness': 1.10,
                    'contrast': 1.12,
                    'highlights': 1.18,
                    'shadows': 0.85,
                    'whites': 1.15,
                    'blacks': 0.12,
                    'vibrance': 1.15,
                    'saturation': 0.85,
                    'white_overlay': 0.10,
                    'gaussian_blur': 0.3
                },
                'cool': {
                    'brightness': 1.06,
                    'contrast': 1.08,
                    'highlights': 1.12,
                    'shadows': 0.90,
                    'whites': 1.10,
                    'blacks': 0.18,
                    'vibrance': 1.10,
                    'saturation': 0.90,
                    'white_overlay': 0.06,
                    'gaussian_blur': 0.5
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.10,
                    'contrast': 1.08,
                    'highlights': 1.18,
                    'shadows': 0.85,
                    'whites': 1.15,
                    'blacks': 0.12,
                    'vibrance': 1.10,
                    'saturation': 0.90,
                    'white_overlay': 0.06,
                    'gaussian_blur': 0.4
                },
                'warm': {
                    'brightness': 1.12,
                    'contrast': 1.10,
                    'highlights': 1.20,
                    'shadows': 0.83,
                    'whites': 1.18,
                    'blacks': 0.10,
                    'vibrance': 1.12,
                    'saturation': 0.88,
                    'white_overlay': 0.08,
                    'gaussian_blur': 0.3
                },
                'cool': {
                    'brightness': 1.08,
                    'contrast': 1.06,
                    'highlights': 1.15,
                    'shadows': 0.87,
                    'whites': 1.12,
                    'blacks': 0.14,
                    'vibrance': 1.08,
                    'saturation': 0.92,
                    'white_overlay': 0.04,
                    'gaussian_blur': 0.5
                }
            },
            'champagne_gold': {
                'natural': {
                    'brightness': 1.30,
                    'contrast': 1.08,
                    'highlights': 1.25,
                    'shadows': 0.80,
                    'whites': 1.35,
                    'blacks': 0.08,
                    'vibrance': 1.05,
                    'saturation': 0.90,
                    'white_overlay': 0.15,
                    'gaussian_blur': 0.3
                },
                'warm': {
                    'brightness': 1.32,
                    'contrast': 1.10,
                    'highlights': 1.28,
                    'shadows': 0.78,
                    'whites': 1.38,
                    'blacks': 0.06,
                    'vibrance': 1.08,
                    'saturation': 0.88,
                    'white_overlay': 0.17,
                    'gaussian_blur': 0.2
                },
                'cool': {
                    'brightness': 1.28,
                    'contrast': 1.06,
                    'highlights': 1.22,
                    'shadows': 0.82,
                    'whites': 1.32,
                    'blacks': 0.10,
                    'vibrance': 1.02,
                    'saturation': 0.92,
                    'white_overlay': 0.13,
                    'gaussian_blur': 0.4
                }
            }
        }

    def detect_and_remove_black_border(self, image):
        """검은 테두리 완전 제거 - 초강력 버전"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 각 방향에서 컨텐츠 시작점 찾기
        top, bottom, left, right = 0, h, 0, w
        
        # 더 넓은 범위 스캔 (최대 이미지의 40%)
        max_border = min(int(h * 0.4), int(w * 0.4), 400)
        
        # 상단 - 여러 threshold로 검사
        for threshold in [20, 30, 40, 50, 60]:
            for y in range(max_border):
                row_mean = np.mean(gray[y, w//4:3*w//4])  # 중앙 50% 영역만 체크
                if row_mean > threshold:
                    top = max(top, y)
                    break
        
        # 하단
        for threshold in [20, 30, 40, 50, 60]:
            for y in range(max_border):
                row_mean = np.mean(gray[h-1-y, w//4:3*w//4])
                if row_mean > threshold:
                    bottom = min(bottom, h-y)
                    break
        
        # 좌측
        for threshold in [20, 30, 40, 50, 60]:
            for x in range(max_border):
                col_mean = np.mean(gray[h//4:3*h//4, x])
                if col_mean > threshold:
                    left = max(left, x)
                    break
        
        # 우측
        for threshold in [20, 30, 40, 50, 60]:
            for x in range(max_border):
                col_mean = np.mean(gray[h//4:3*h//4, w-1-x])
                if col_mean > threshold:
                    right = min(right, w-x)
                    break
        
        # 안전 마진 추가
        margin = 20
        top += margin
        left += margin
        bottom = max(top + 100, bottom - margin)  # 최소 크기 보장
        right = max(left + 100, right - margin)
        
        # 크롭
        cropped = image[top:bottom, left:right]
        
        # 2차 정밀 크롭 - 엣지의 어두운 부분 추가 제거
        if cropped.size > 0:
            gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            h2, w2 = gray2.shape
            
            # 엣지 픽셀 밝기 체크 (더 민감하게)
            edge_cut = 5  # 기본 5픽셀 추가 제거
            
            # 상단 엣지가 어두우면 추가 크롭
            if np.mean(gray2[:10, :]) < 100:
                edge_cut = 15
            
            if h2 > edge_cut * 2 and w2 > edge_cut * 2:
                cropped = cropped[edge_cut:h2-edge_cut, edge_cut:w2-edge_cut]
        
        return cropped

    def detect_rings_simple(self, image):
        """간단한 반지 감지 - 중앙 영역 활용"""
        h, w = image.shape[:2]
        
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 이진화로 밝은 영역(반지) 찾기
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어들 찾기
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            # 모든 컨투어를 포함하는 바운딩 박스
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + cw)
                y_max = max(y_max, y + ch)
            
            # 여유 공간 추가
            margin = 50
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # 컨투어를 못 찾으면 중앙 영역 반환
        return (w//4, h//4, w//2, h//2)

    def detect_metal_and_lighting(self, image):
        """금속 타입과 조명 감지 - 간단한 버전"""
        # 중앙 영역에서 색상 분석
        h, w = image.shape[:2]
        center_roi = image[h//3:2*h//3, w//3:2*w//3]
        
        # BGR 평균값 계산
        b_mean = np.mean(center_roi[:,:,0])
        g_mean = np.mean(center_roi[:,:,1])
        r_mean = np.mean(center_roi[:,:,2])
        
        # 밝기 계산
        brightness = (b_mean + g_mean + r_mean) / 3
        
        # 금속 타입 결정 (간단한 규칙)
        if r_mean > g_mean and r_mean > b_mean:
            if r_mean - b_mean > 30:
                metal_type = 'yellow_gold'
            else:
                metal_type = 'rose_gold'
        elif brightness > 180:
            metal_type = 'white_gold'
        else:
            metal_type = 'champagne_gold'
        
        # 조명 타입 결정
        if brightness > 200:
            lighting = 'cool'
        elif brightness < 150:
            lighting = 'warm'
        else:
            lighting = 'natural'
        
        return metal_type, lighting

    def apply_v13_params_simple(self, image, metal_type, lighting):
        """v13.3 파라미터 적용 - 색상 왜곡 없는 버전"""
        params = self.metal_params[metal_type][lighting]
        
        # PIL로 변환
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. 밝기 조정
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = brightness_enhancer.enhance(params['brightness'])
        
        # 2. 대비 조정
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(params['contrast'])
        
        # 3. 채도 조정
        color_enhancer = ImageEnhance.Color(pil_image)
        pil_image = color_enhancer.enhance(params['saturation'])
        
        # OpenCV로 다시 변환
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 4. 화이트 오버레이 (약하게)
        if params['white_overlay'] > 0:
            white_layer = np.full_like(result, 255, dtype=np.float32)
            alpha = params['white_overlay']
            result = cv2.addWeighted(result.astype(np.float32), 1-alpha, white_layer, alpha, 0)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 5. 약간의 블러 (선택적)
        if params['gaussian_blur'] > 0:
            result = cv2.bilateralFilter(result, 5, 50, 50)
        
        return result

    def create_thumbnail_large(self, image, ring_bbox):
        """반지가 90% 차지하는 큰 썸네일"""
        x, y, w, h = ring_bbox
        
        # 반지 중심
        cx = x + w // 2
        cy = y + h // 2
        
        # 목표 크기
        target_w, target_h = 1000, 1300
        
        # 반지가 90% 차지하도록 (더 크게)
        scale = min(target_w * 0.9 / w, target_h * 0.9 / h)
        
        # 크롭 영역 계산
        crop_size = max(int(target_w / scale), int(target_h / scale))
        
        # 정사각형으로 크롭 (나중에 리사이즈)
        half_size = crop_size // 2
        
        # 크롭 영역
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(image.shape[1], cx + half_size)
        y2 = min(image.shape[0], cy + half_size)
        
        # 이미지 경계 조정
        if x2 - x1 < crop_size:
            if x1 == 0:
                x2 = min(image.shape[1], x1 + crop_size)
            else:
                x1 = max(0, x2 - crop_size)
        
        if y2 - y1 < crop_size:
            if y1 == 0:
                y2 = min(image.shape[0], y1 + crop_size)
            else:
                y1 = max(0, y2 - crop_size)
        
        # 크롭
        cropped = image[y1:y2, x1:x2]
        
        # 1000x1300으로 리사이즈
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            # 비율 유지하면서 리사이즈
            h_crop, w_crop = cropped.shape[:2]
            if w_crop / h_crop > target_w / target_h:
                # 너비 기준
                new_w = target_w
                new_h = int(h_crop * target_w / w_crop)
            else:
                # 높이 기준
                new_h = target_h
                new_w = int(w_crop * target_h / h_crop)
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 캔버스에 중앙 배치
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)  # 흰색 배경
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def process_image(self, image_base64):
        """메인 처리 함수"""
        start_time = time.time()
        
        # 이미지 디코드
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # 1. 검은 테두리 제거
        print("Removing black borders...")
        image_no_border = self.detect_and_remove_black_border(image)
        
        # 2. 금속 타입과 조명 감지
        metal_type, lighting = self.detect_metal_and_lighting(image_no_border)
        print(f"Detected: {metal_type} with {lighting} lighting")
        
        # 3. v13.3 파라미터 적용 (색상 왜곡 없는 버전)
        enhanced = self.apply_v13_params_simple(image_no_border, metal_type, lighting)
        
        # 4. 배경을 더 밝게 (흰색으로)
        # 배경 마스크 생성
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # 배경을 더 밝게
        background = np.full_like(enhanced, 248, dtype=np.uint8)  # 거의 흰색
        result = cv2.bitwise_and(enhanced, enhanced, mask=mask_inv)
        background_part = cv2.bitwise_and(background, background, mask=mask)
        enhanced = cv2.add(result, background_part)
        
        # 5. 반지 영역 감지
        ring_bbox = self.detect_rings_simple(enhanced)
        
        # 6. 2x 업스케일
        height, width = enhanced.shape[:2]
        upscaled = cv2.resize(enhanced, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # 7. 썸네일 생성 (반지 90% - 더 크게)
        ring_bbox_scaled = (ring_bbox[0] * 2, ring_bbox[1] * 2, ring_bbox[2] * 2, ring_bbox[3] * 2)
        thumbnail = self.create_thumbnail_large(upscaled, ring_bbox_scaled)
        
        # 이미지를 base64로 인코딩
        _, main_buffer = cv2.imencode('.jpg', upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 90])
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "processing_time": processing_time,
                    "version": self.version,
                    "border_removed": True,
                    "thumbnail_size": "1000x1300",
                    "ring_coverage": "90%"
                }
            }
        }

def handler(event):
    """RunPod 핸들러"""
    try:
        input_data = event["input"]
        
        # 테스트 모드 확인
        if input_data.get("test") == "ping":
            return {"output": {"status": "pong", "message": "v23.0 ready!"}}
        
        # 이미지 처리
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            raise ValueError("No image_base64 provided")
        
        # 처리
        enhancer = WeddingRingEnhancer()
        result = enhancer.process_image(image_base64)
        
        return result
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v23.0"
            }
        }

# RunPod 진입점
runpod.serverless.start({"handler": handler})
