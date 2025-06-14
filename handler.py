import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 파라미터 (28쌍 학습 데이터 기반) - 완전한 12가지 세트
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15,
            'contrast': 1.08,
            'white_overlay': 0.06,
            'sharpness': 1.15,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.05,
            'white_overlay': 0.04,
            'sharpness': 1.10,
            'color_temp_a': 0,
            'color_temp_b': -1,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.25,
            'contrast': 1.15,
            'white_overlay': 0.08,
            'sharpness': 1.25,
            'color_temp_a': 4,
            'color_temp_b': 3,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17,
            'contrast': 1.11,
            'white_overlay': 0.08,
            'sharpness': 1.16,
            'color_temp_a': -1,
            'color_temp_b': -1,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.13,
            'contrast': 1.08,
            'white_overlay': 0.10,
            'sharpness': 1.13,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.22,
            'contrast': 1.15,
            'white_overlay': 0.06,
            'sharpness': 1.20,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.12
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16,
            'contrast': 1.09,
            'white_overlay': 0.05,
            'sharpness': 1.14,
            'color_temp_a': 3,
            'color_temp_b': 2,
            'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.06,
            'white_overlay': 0.03,
            'sharpness': 1.11,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.12,
            'white_overlay': 0.07,
            'sharpness': 1.18,
            'color_temp_a': 5,
            'color_temp_b': 4,
            'original_blend': 0.18
        }
    }
}

# 28쌍 AFTER 파일에서 추출한 검증된 배경색
AFTER_BACKGROUND_COLORS = {
    'natural': [200, 195, 190],
    'warm': [205, 198, 185], 
    'cool': [195, 198, 200]
}

class WeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        
    def detect_black_lines(self, image):
        """정밀한 검은색 선 테두리 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 매우 어두운 픽셀만 검은색으로 간주 (threshold=15)
            _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
                
            # 가장 큰 사각형 모양 컨투어 찾기
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 최소 크기 조건
                    # 사각형 근사화
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # 사각형 형태
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.5 < aspect_ratio < 2.0:  # 합리적인 종횡비
                            valid_contours.append((contour, area, (x, y, w, h)))
            
            if not valid_contours:
                return None, None
                
            # 가장 큰 유효한 컨투어 선택
            largest_contour = max(valid_contours, key=lambda x: x[1])
            contour, _, bbox = largest_contour
            
            # 검은색 선 마스크 생성
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)
            
            return mask, bbox
            
        except Exception as e:
            print(f"검은색 선 감지 에러: {e}")
            return None, None
    
    def detect_wedding_ring_in_area(self, image, bbox):
        """검은색 선 안에서 웨딩링만 정확히 감지"""
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]  # 검은색 선 안쪽 영역만
            
            # HSV로 변환
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # 금속 재질 감지 (밝고 채도 있는 영역)
            # 1. 밝기 기준 (V > 100)
            # 2. 채도 기준 (S > 30, 너무 회색 제외)
            # 3. 색상 기준 (금속 계열)
            
            # 금속 마스크 생성
            lower_metal = np.array([0, 30, 100])     # 최소 채도와 밝기
            upper_metal = np.array([180, 255, 255]) # 모든 색상 허용
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
            
            # 형태학적 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
            metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
            
            # 가장 큰 연결된 영역만 선택 (웨딩링)
            contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                ring_mask = np.zeros_like(metal_mask)
                cv2.fillPoly(ring_mask, [largest_contour], 255)
                
                # 웨딩링 바운딩 박스 (ROI 기준)
                ring_x, ring_y, ring_w, ring_h = cv2.boundingRect(largest_contour)
                
                # 전체 이미지 기준으로 좌표 변환
                global_ring_bbox = (x + ring_x, y + ring_y, ring_w, ring_h)
                
                # 전체 이미지 크기로 마스크 확장
                full_ring_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                full_ring_mask[y:y+h, x:x+w] = ring_mask
                
                return full_ring_mask, global_ring_bbox
            
            return None, None
            
        except Exception as e:
            print(f"웨딩링 감지 에러: {e}")
            return None, None
    
    def detect_metal_type(self, image, ring_mask):
        """웨딩링 영역에서만 금속 타입 감지"""
        try:
            # 웨딩링 영역의 픽셀만 추출
            mask_indices = np.where(ring_mask > 0)
            if len(mask_indices[0]) == 0:
                return 'white_gold'
            
            rgb_values = image[mask_indices[0], mask_indices[1], :]
            hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            avg_hue = np.mean(hsv_values[:, 0])
            avg_sat = np.mean(hsv_values[:, 1])
            
            # 금속 타입 분류
            if avg_hue < 15 or avg_hue > 165:  # 빨간색 계열
                if avg_sat > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_hue <= 35:  # 황색 계열
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'white_gold'  # 기본값
                
        except Exception as e:
            print(f"금속 감지 에러: {e}")
            return 'white_gold'
    
    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_channel = lab[:, :, 2]
            b_mean = np.mean(b_channel)
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            print(f"조명 감지 에러: {e}")
            return 'natural'
    
    def enhance_wedding_ring_light(self, image, ring_mask, metal_type, lighting):
        """웨딩링만 가볍게 보정 (v13.3 파라미터의 50% 강도)"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
            
            # 가벼운 보정을 위해 파라미터 강도 50% 감소
            light_params = {
                'brightness': 1.0 + (params['brightness'] - 1.0) * 0.5,
                'contrast': 1.0 + (params['contrast'] - 1.0) * 0.5,
                'white_overlay': params['white_overlay'] * 0.5,
                'sharpness': 1.0 + (params['sharpness'] - 1.0) * 0.5,
                'color_temp_a': params['color_temp_a'] * 0.5,
                'color_temp_b': params['color_temp_b'] * 0.5,
                'original_blend': params['original_blend'] + 0.1  # 원본 더 많이 보존
            }
            
            result = image.copy()
            
            # 웨딩링 영역만 추출
            ring_indices = np.where(ring_mask > 0)
            if len(ring_indices[0]) == 0:
                return result
            
            ring_pixels = image[ring_indices]
            
            # PIL로 가벼운 보정
            ring_image = Image.fromarray(ring_pixels.reshape(-1, 1, 3).squeeze())
            
            # 밝기 조정
            enhancer = ImageEnhance.Brightness(ring_image)
            enhanced = enhancer.enhance(light_params['brightness'])
            
            # 대비 조정  
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(light_params['contrast'])
            
            # 선명도 조정
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(light_params['sharpness'])
            
            enhanced_pixels = np.array(enhanced)
            
            # 하얀색 오버레이 (가볍게)
            white_overlay = np.full_like(enhanced_pixels, 255)
            enhanced_pixels = cv2.addWeighted(
                enhanced_pixels, 1 - light_params['white_overlay'],
                white_overlay, light_params['white_overlay'], 0
            )
            
            # LAB 색온도 조정 (가볍게)
            lab_pixels = cv2.cvtColor(enhanced_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
            lab_pixels[:, :, 1] = np.clip(lab_pixels[:, :, 1] + light_params['color_temp_a'], 0, 255)
            lab_pixels[:, :, 2] = np.clip(lab_pixels[:, :, 2] + light_params['color_temp_b'], 0, 255)
            enhanced_pixels = cv2.cvtColor(lab_pixels, cv2.COLOR_LAB2RGB).squeeze()
            
            # 원본과 블렌딩 (더 보수적)
            original_pixels = image[ring_indices]
            final_pixels = cv2.addWeighted(
                enhanced_pixels, 1 - light_params['original_blend'],
                original_pixels, light_params['original_blend'], 0
            )
            
            # 결과에 웨딩링 픽셀 적용
            result[ring_indices] = final_pixels.astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"웨딩링 보정 에러: {e}")
            return image
    
    def apply_background_except_ring(self, image, black_line_mask, ring_mask, lighting):
        """웨딩링 제외한 모든 영역을 28쌍 AFTER 배경색으로 덮기"""
        try:
            result = image.copy()
            
            # 28쌍 AFTER 파일에서 검증된 배경색 사용
            target_bg_color = np.array(AFTER_BACKGROUND_COLORS[lighting], dtype=np.uint8)
            
            # 웨딩링이 아닌 모든 영역 마스크 생성
            # (검은색 선 + 검은색 선 안의 배경) 그러나 웨딩링은 제외
            background_mask = black_line_mask.copy()  # 검은색 선 포함
            
            # 검은색 선 안쪽 영역 중 웨딩링이 아닌 부분도 배경으로 처리
            black_line_inner = black_line_mask > 0
            ring_area = ring_mask > 0
            
            # 검은색 선 안쪽이지만 웨딩링이 아닌 영역
            inner_background = black_line_inner & (~ring_area)
            
            # 전체 배경 마스크 = 검은색 선 + 검은색 선 안의 비웨딩링 영역
            total_background_mask = (black_line_mask > 0) | inner_background
            
            # 채널별로 배경색 적용 (NumPy 에러 방지)
            result[total_background_mask, 0] = target_bg_color[0]  # R 채널
            result[total_background_mask, 1] = target_bg_color[1]  # G 채널
            result[total_background_mask, 2] = target_bg_color[2]  # B 채널
            
            return result
            
        except Exception as e:
            print(f"배경 적용 에러: {e}")
            return image
    
    def upscale_image(self, image, scale=2):
        """LANCZOS 보간법으로 안전한 업스케일링"""
        try:
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return upscaled
        except Exception as e:
            print(f"업스케일링 에러: {e}")
            return image
    
    def create_thumbnail_ring_centered(self, image, ring_bbox, target_size=(1000, 1300)):
        """웨딩링 중심으로 정확한 1000×1300 썸네일 생성"""
        try:
            ring_x, ring_y, ring_w, ring_h = ring_bbox
            
            # 웨딩링 중심점 계산
            ring_center_x = ring_x + ring_w // 2
            ring_center_y = ring_y + ring_h // 2
            
            # 1000x1300 비율에 맞는 크롭 영역 계산
            target_w, target_h = target_size
            crop_ratio = target_w / target_h  # 0.769
            
            # 웨딩링 크기 기준으로 크롭 크기 결정
            ring_size = max(ring_w, ring_h)
            crop_size = int(ring_size * 2.5)  # 웨딩링 주변에 충분한 여백
            
            # 정사각형 크롭 영역을 1000:1300 비율로 조정
            if crop_ratio < 1:  # 세로가 더 긴 경우
                crop_w = int(crop_size * crop_ratio)
                crop_h = crop_size
            else:
                crop_w = crop_size
                crop_h = int(crop_size / crop_ratio)
            
            # 웨딩링 중심으로 크롭 영역 설정
            x1 = max(0, ring_center_x - crop_w // 2)
            y1 = max(0, ring_center_y - crop_h // 2)
            x2 = min(image.shape[1], x1 + crop_w)
            y2 = min(image.shape[0], y1 + crop_h)
            
            # 실제 크롭 크기 재계산 (경계 처리)
            actual_w = x2 - x1
            actual_h = y2 - y1
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            # 1000×1300으로 정확히 리사이즈
            thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return thumbnail
            
        except Exception as e:
            print(f"썸네일 생성 에러: {e}")
            # 에러 시 기본 크롭
            return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.10 연결 성공: {input_data['prompt']}",
                "status": "ready_for_processing",
                "version": "v14.10_correct_algorithm",
                "features": [
                    "웨딩링만 정확히 감지 및 가볍게 보정",
                    "웨딩링 제외한 모든 영역 배경색 적용",
                    "28쌍 AFTER 파일 기준 배경색",
                    "웨딩링 중심 정확한 썸네일",
                    "NumPy 에러 완전 해결"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            processor = WeddingRingProcessor()
            
            # 1. 검은색 선 테두리 감지
            black_line_mask, black_bbox = processor.detect_black_lines(image_array)
            if black_line_mask is None:
                return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
            
            # 2. 검은색 선 안에서 웨딩링만 정확히 감지
            ring_mask, ring_bbox = processor.detect_wedding_ring_in_area(image_array, black_bbox)
            if ring_mask is None:
                return {"error": "웨딩링을 찾을 수 없습니다."}
            
            # 3. 금속 타입 및 조명 감지 (웨딩링 영역 기준)
            metal_type = processor.detect_metal_type(image_array, ring_mask)
            lighting = processor.detect_lighting(image_array)
            
            # 4. 웨딩링만 가볍게 보정
            enhanced_image = processor.enhance_wedding_ring_light(
                image_array, ring_mask, metal_type, lighting
            )
            
            # 5. 웨딩링 제외한 모든 영역을 28쌍 AFTER 배경색으로 덮기
            final_image = processor.apply_background_except_ring(
                enhanced_image, black_line_mask, ring_mask, lighting
            )
            
            # 6. 2x 업스케일링
            upscaled = processor.upscale_image(final_image, scale=2)
            
            # 7. 웨딩링 중심으로 정확한 1000×1300 썸네일 생성
            # ring_bbox를 2배로 스케일링
            scaled_ring_bbox = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
            thumbnail = processor.create_thumbnail_ring_centered(upscaled, scaled_ring_bbox)
            
            # 8. 결과 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(final_image)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v14.10",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "black_bbox": black_bbox,
                    "ring_bbox": ring_bbox,
                    "background_color": AFTER_BACKGROUND_COLORS[lighting],
                    "scale_factor": 2,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "method": "ring_only_enhancement_background_replacement"
                }
            }
        
        return {"error": "image_base64 또는 prompt가 필요합니다."}
        
    except Exception as e:
        return {"error": f"v14.10 처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
