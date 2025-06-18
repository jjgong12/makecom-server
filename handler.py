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
        self.version = "v22.0"
        
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
                    'color_temp_r': 0.98,
                    'color_temp_b': 1.03,
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
                    'color_temp_r': 0.97,
                    'color_temp_b': 1.02,
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
                    'color_temp_r': 0.99,
                    'color_temp_b': 1.04,
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
                    'color_temp_r': 1.05,
                    'color_temp_b': 0.92,
                    'color_temp_a': -8,
                    'color_temp_b_lab': -5,
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
                    'color_temp_r': 1.06,
                    'color_temp_b': 0.90,
                    'color_temp_a': -10,
                    'color_temp_b_lab': -6,
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
                    'color_temp_r': 1.04,
                    'color_temp_b': 0.94,
                    'color_temp_a': -6,
                    'color_temp_b_lab': -4,
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
                    'color_temp_r': 1.02,
                    'color_temp_b': 0.96,
                    'color_temp_a': -3,
                    'color_temp_b_lab': -2,
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
                    'color_temp_r': 1.03,
                    'color_temp_b': 0.94,
                    'color_temp_a': -4,
                    'color_temp_b_lab': -3,
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
                    'color_temp_r': 1.01,
                    'color_temp_b': 0.98,
                    'color_temp_a': -2,
                    'color_temp_b_lab': -1,
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
                    'color_temp_r': 0.96,
                    'color_temp_b': 1.02,
                    'color_temp_a': -6,
                    'color_temp_b_lab': -6,
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
                    'color_temp_r': 0.95,
                    'color_temp_b': 1.00,
                    'color_temp_a': -7,
                    'color_temp_b_lab': -7,
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
                    'color_temp_r': 0.97,
                    'color_temp_b': 1.04,
                    'color_temp_a': -5,
                    'color_temp_b_lab': -5,
                    'white_overlay': 0.13,
                    'gaussian_blur': 0.4
                }
            }
        }

    def detect_black_border(self, image):
        """검은 테두리 완벽 감지 - v22.0 초강력 버전"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 다양한 threshold로 테두리 감지
        borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # 각 방향별로 최대 300픽셀까지 스캔 (더 넓게)
        max_scan = min(300, h//3, w//3)
        
        for threshold in [10, 20, 30, 40, 50, 60, 70, 80]:
            # 상단
            for i in range(max_scan):
                row = gray[i, :]
                if np.mean(row) > threshold:
                    borders['top'] = max(borders['top'], i)
                    break
            
            # 하단
            for i in range(max_scan):
                row = gray[h-1-i, :]
                if np.mean(row) > threshold:
                    borders['bottom'] = max(borders['bottom'], i)
                    break
            
            # 왼쪽
            for i in range(max_scan):
                col = gray[:, i]
                if np.mean(col) > threshold:
                    borders['left'] = max(borders['left'], i)
                    break
            
            # 오른쪽
            for i in range(max_scan):
                col = gray[:, w-1-i]
                if np.mean(col) > threshold:
                    borders['right'] = max(borders['right'], i)
                    break
        
        # 안전 마진 추가 (발견된 값의 1.5배 + 30픽셀)
        margin = 30
        borders['top'] = min(int(borders['top'] * 1.5) + margin, h//3)
        borders['bottom'] = min(int(borders['bottom'] * 1.5) + margin, h//3)
        borders['left'] = min(int(borders['left'] * 1.5) + margin, w//3)
        borders['right'] = min(int(borders['right'] * 1.5) + margin, w//3)
        
        print(f"Detected borders: {borders}")
        return borders

    def remove_black_border_v22(self, image):
        """검은 테두리 완전 제거 - 2단계 크롭"""
        # 1차 크롭
        borders = self.detect_black_border(image)
        h, w = image.shape[:2]
        
        cropped = image[
            borders['top']:h-borders['bottom'],
            borders['left']:w-borders['right']
        ]
        
        # 2차 정밀 크롭 - 남은 검은색 제거
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        h2, w2 = gray.shape
        
        # 엣지에서 안쪽으로 스캔하면서 첫 번째 밝은 픽셀 찾기
        final_borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # 더 민감한 threshold (50)
        edge_threshold = 50
        
        # 상단
        for i in range(min(100, h2//4)):
            if np.max(gray[i, w2//4:3*w2//4]) > edge_threshold:
                final_borders['top'] = i
                break
        
        # 하단
        for i in range(min(100, h2//4)):
            if np.max(gray[h2-1-i, w2//4:3*w2//4]) > edge_threshold:
                final_borders['bottom'] = i
                break
        
        # 왼쪽
        for i in range(min(100, w2//4)):
            if np.max(gray[h2//4:3*h2//4, i]) > edge_threshold:
                final_borders['left'] = i
                break
        
        # 오른쪽
        for i in range(min(100, w2//4)):
            if np.max(gray[h2//4:3*h2//4, w2-1-i]) > edge_threshold:
                final_borders['right'] = i
                break
        
        # 최종 크롭
        final_cropped = cropped[
            final_borders['top']:h2-final_borders['bottom'],
            final_borders['left']:w2-final_borders['right']
        ]
        
        return final_cropped

    def detect_rings(self, image):
        """반지 영역 감지 - HSV 기반"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 금속 반지 감지를 위한 다중 마스크
        masks = []
        
        # 밝은 금속 (화이트골드, 실버)
        lower1 = np.array([0, 0, 100])
        upper1 = np.array([180, 30, 255])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # 금색 계열
        lower2 = np.array([10, 20, 100])
        upper2 = np.array([30, 255, 255])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # 로즈골드 계열
        lower3 = np.array([0, 20, 100])
        upper3 = np.array([20, 100, 255])
        masks.append(cv2.inRange(hsv, lower3, upper3))
        
        # 모든 마스크 합치기
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 반지를 못 찾으면 이미지 중앙 영역 반환
            h, w = image.shape[:2]
            return (w//4, h//4, w//2, h//2)
        
        # 가장 큰 컨투어 2개 찾기 (반지 2개)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        # 모든 컨투어를 포함하는 바운딩 박스
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def detect_metal_and_lighting(self, image):
        """금속 타입과 조명 감지"""
        # 반지 영역에서만 분석
        bbox = self.detect_rings(image)
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 'white_gold', 'natural'
        
        # HSV로 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 평균 색상 계산
        avg_hue = np.mean(hsv_roi[:,:,0])
        avg_sat = np.mean(hsv_roi[:,:,1])
        avg_val = np.mean(hsv_roi[:,:,2])
        
        # 금속 타입 결정
        if avg_sat < 30:  # 채도가 낮으면
            if avg_val > 200:
                metal_type = 'white_gold'
            else:
                metal_type = 'champagne_gold'
        elif avg_hue < 20:  # 붉은색 계열
            metal_type = 'rose_gold'
        elif avg_hue < 30:  # 노란색 계열
            metal_type = 'yellow_gold'
        else:
            metal_type = 'white_gold'
        
        # 조명 타입 결정
        brightness_variance = np.var(hsv_roi[:,:,2])
        if brightness_variance > 2000:
            lighting = 'natural'
        elif avg_val > 180:
            lighting = 'cool'
        else:
            lighting = 'warm'
        
        return metal_type, lighting

    def apply_v13_params(self, image, metal_type, lighting):
        """v13.3 파라미터 적용"""
        params = self.metal_params[metal_type][lighting]
        result = image.copy().astype(np.float32)
        
        # 1. 노이즈 제거
        if params['gaussian_blur'] > 0:
            result = cv2.bilateralFilter(result.astype(np.uint8), 9, 75, 75)
            result = result.astype(np.float32)
        
        # 2. 밝기 조정
        result = result * params['brightness']
        
        # 3. 대비 조정
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * params['contrast'] + mean
        
        # 4. 하이라이트/섀도우
        gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        highlight_mask = np.power(gray, 0.5)
        shadow_mask = 1 - np.power(1 - gray, 0.5)
        
        for c in range(3):
            result[:,:,c] = result[:,:,c] * (1 + (params['highlights'] - 1) * highlight_mask)
            result[:,:,c] = result[:,:,c] * (1 + (params['shadows'] - 1) * shadow_mask)
        
        # 5. 흰색/검은색 포인트
        result = np.where(result > 200, result * params['whites'], result)
        result = np.where(result < 50, result * (1 - params['blacks']), result)
        
        # 6. 색온도 조정
        if 'color_temp_r' in params:
            result[:,:,2] *= params['color_temp_r']  # R
            result[:,:,0] *= params['color_temp_b']  # B
        
        # 7. Lab 색공간 조정
        if 'color_temp_a' in params:
            lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:,:,1] += params['color_temp_a']
            lab[:,:,2] += params.get('color_temp_b_lab', 0)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # 8. 채도 조정
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= params['saturation']
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # 9. 화이트 오버레이
        white_overlay = np.full_like(result, 255, dtype=np.float32)
        alpha = params['white_overlay']
        result = result * (1 - alpha) + white_overlay * alpha
        
        # 10. 최종 클리핑
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)

    def create_perfect_thumbnail(self, image, ring_bbox):
        """반지가 80% 차지하는 완벽한 썸네일"""
        x, y, w, h = ring_bbox
        
        # 반지 중심점
        ring_center_x = x + w // 2
        ring_center_y = y + h // 2
        
        # 썸네일 크기
        thumb_w, thumb_h = 1000, 1300
        
        # 반지가 썸네일의 80% 차지하도록 스케일 계산
        scale = min(thumb_w * 0.8 / w, thumb_h * 0.8 / h)
        
        # 크롭 영역 계산 (원본 이미지에서)
        crop_w = int(thumb_w / scale)
        crop_h = int(thumb_h / scale)
        
        # 크롭 시작점 (반지 중심 기준)
        crop_x = max(0, ring_center_x - crop_w // 2)
        crop_y = max(0, ring_center_y - crop_h // 2)
        
        # 이미지 경계 체크
        img_h, img_w = image.shape[:2]
        if crop_x + crop_w > img_w:
            crop_x = img_w - crop_w
        if crop_y + crop_h > img_h:
            crop_y = img_h - crop_h
        
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        
        # 크롭
        cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # 리사이즈
        thumbnail = cv2.resize(cropped, (thumb_w, thumb_h), interpolation=cv2.INTER_LANCZOS4)
        
        return thumbnail

    def process_image(self, image_base64):
        """메인 처리 함수"""
        start_time = time.time()
        
        # 이미지 디코드
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # 1. 검은 테두리 완전 제거 (v22.0 강화)
        image_no_border = self.remove_black_border_v22(image)
        
        # 2. 금속 타입과 조명 감지
        metal_type, lighting = self.detect_metal_and_lighting(image_no_border)
        print(f"Detected: {metal_type} with {lighting} lighting")
        
        # 3. v13.3 파라미터 적용
        enhanced = self.apply_v13_params(image_no_border, metal_type, lighting)
        
        # 4. 반지 영역 감지 (보정된 이미지에서)
        ring_bbox = self.detect_rings(enhanced)
        
        # 5. 2x 업스케일
        height, width = enhanced.shape[:2]
        upscaled = cv2.resize(enhanced, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # 6. 썸네일 생성 (반지 80%)
        ring_bbox_scaled = (ring_bbox[0] * 2, ring_bbox[1] * 2, ring_bbox[2] * 2, ring_bbox[3] * 2)
        thumbnail = self.create_perfect_thumbnail(upscaled, ring_bbox_scaled)
        
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
                    "ring_coverage": "80%"
                }
            }
        }

def handler(event):
    """RunPod 핸들러"""
    try:
        input_data = event["input"]
        
        # 테스트 모드 확인
        if input_data.get("test") == "ping":
            return {"output": {"status": "pong", "message": "v22.0 ready!"}}
        
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
                "version": "v22.0"
            }
        }

# RunPod 진입점
runpod.serverless.start({"handler": handler})
