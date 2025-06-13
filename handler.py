import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import json

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반)
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
            'brightness': 1.12,
            'contrast': 1.05,
            'white_overlay': 0.08,
            'sharpness': 1.12,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.22
        },
        'cool': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.04,
            'sharpness': 1.18,
            'color_temp_a': 4,
            'color_temp_b': 2,
            'original_blend': 0.18
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
            'brightness': 1.14,
            'contrast': 1.08,
            'white_overlay': 0.10,
            'sharpness': 1.14,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.17
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.06,
            'sharpness': 1.18,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.13
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
            'brightness': 1.13,
            'contrast': 1.06,
            'white_overlay': 0.07,
            'sharpness': 1.12,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.24
        },
        'cool': {
            'brightness': 1.19,
            'contrast': 1.12,
            'white_overlay': 0.03,
            'sharpness': 1.16,
            'color_temp_a': 5,
            'color_temp_b': 3,
            'original_blend': 0.20
        }
    }
}

class WeddingRingProcessorV14:
    def __init__(self):
        self.debug_info = []
        
    def log_debug(self, message):
        """디버깅 정보 로깅"""
        self.debug_info.append(message)
        print(f"[DEBUG] {message}")
    
    def detect_black_border_lines(self, image):
        """
        v14.0 핵심: 검은색 선(테두리)만 정확히 감지
        웨딩링은 절대 건드리지 않음
        """
        self.log_debug("검은색 선 테두리 감지 시작")
        
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 2. 다중 threshold로 검은색 선만 정확히 감지
        # 매우 어두운 픽셀만 선택 (threshold < 30)
        _, black_lines = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 3. 선 형태만 추출 (morphological operations)
        # 수직선 감지
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(black_lines, cv2.MORPH_OPEN, vertical_kernel)
        
        # 수평선 감지
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(black_lines, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 4. 선들 결합
        border_lines = cv2.add(vertical_lines, horizontal_lines)
        
        # 5. 선을 두껍게 만들어 제거 효과 향상 (3픽셀)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        border_lines = cv2.dilate(border_lines, kernel, iterations=1)
        
        # 6. 테두리 사각형 영역 찾기
        contours, _ = cv2.findContours(border_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어를 테두리로 판단
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 내부 영역 계산 (여백 5픽셀)
            inner_x = max(0, x + 5)
            inner_y = max(0, y + 5)
            inner_w = max(1, w - 10)
            inner_h = max(1, h - 10)
            inner_bbox = (inner_x, inner_y, inner_w, inner_h)
            
            self.log_debug(f"검은색 선 테두리 발견: ({x}, {y}, {w}, {h})")
            self.log_debug(f"내부 웨딩링 영역: ({inner_x}, {inner_y}, {inner_w}, {inner_h})")
            
            return border_lines, (x, y, w, h), inner_bbox
        
        self.log_debug("검은색 선 테두리를 찾을 수 없음")
        return None, None, None
    
    def detect_metal_type(self, image, inner_bbox=None):
        """HSV 색공간 분석으로 금속 타입 감지 (웨딩링 영역만)"""
        if inner_bbox:
            x, y, w, h = inner_bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # 평균 색상 계산
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        self.log_debug(f"색상 분석: H={avg_hue:.1f}, S={avg_sat:.1f}, V={avg_val:.1f}")
        
        # 금속 타입 분류
        if avg_sat < 25:
            metal_type = 'white_gold'
        elif 5 <= avg_hue <= 25:
            metal_type = 'yellow_gold' if avg_sat > 60 else 'champagne_gold'
        elif avg_hue < 5 or avg_hue > 165:
            metal_type = 'rose_gold'
        else:
            metal_type = 'white_gold'
        
        self.log_debug(f"감지된 금속 타입: {metal_type}")
        return metal_type
    
    def detect_lighting(self, image, inner_bbox=None):
        """LAB 색공간 B채널로 조명 환경 감지 (웨딩링 영역만)"""
        if inner_bbox:
            x, y, w, h = inner_bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # LAB 변환
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
        self.log_debug(f"조명 분석: B채널 평균={b_mean:.1f}")
        
        if b_mean < 125:
            lighting = 'warm'
        elif b_mean > 135:
            lighting = 'cool'
        else:
            lighting = 'natural'
        
        self.log_debug(f"감지된 조명: {lighting}")
        return lighting
    
    def enhance_wedding_ring_v13_3(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (28쌍 학습 데이터 기반)"""
        self.log_debug(f"v13.3 보정 시작: {metal_type} - {lighting}")
        
        # 파라미터 가져오기
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                        WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # PIL ImageEnhance로 기본 보정
        pil_image = Image.fromarray(image)
        
        # 1. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        self.log_debug(f"밝기 조정: {params['brightness']}")
        
        # 2. 대비 조정
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        self.log_debug(f"대비 조정: {params['contrast']}")
        
        # 3. 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        self.log_debug(f"선명도 조정: {params['sharpness']}")
        
        # 4. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
        enhanced_array = np.array(enhanced)
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        self.log_debug(f"하얀색 오버레이: {params['white_overlay']}")
        
        # 5. LAB 색공간에서 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        self.log_debug(f"색온도 조정: A={params['color_temp_a']}, B={params['color_temp_b']}")
        
        # 6. 원본과 블렌딩 (자연스러움 보장)
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        self.log_debug(f"원본 블렌딩: {params['original_blend']}")
        
        # 7. 추가 고급 보정
        # 노이즈 제거
        final = cv2.bilateralFilter(final, 9, 75, 75)
        
        # CLAHE 적용 (명료도 향상)
        lab_final = cv2.cvtColor(final, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_final[:, :, 0] = clahe.apply(lab_final[:, :, 0])
        final = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
        
        # 감마 보정
        gamma = 1.05
        final = np.power(final / 255.0, gamma) * 255.0
        
        self.log_debug("v13.3 보정 완료")
        return final.astype(np.uint8)
    
    def remove_black_lines_only(self, image, black_lines_mask):
        """
        v14.0 핵심: 검은색 선만 제거, 웨딩링은 절대 건드리지 않음
        """
        self.log_debug("검은색 선 제거 시작 (웨딩링 보존)")
        
        # 1. 검은색 선 영역만 inpainting으로 제거
        # TELEA 방식 (더 자연스러운 결과)
        inpainted = cv2.inpaint(image, black_lines_mask, 3, cv2.INPAINT_TELEA)
        
        # 2. 가장자리 부드럽게 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(black_lines_mask, kernel, iterations=1)
        edge_mask = dilated_mask - black_lines_mask
        
        # 3. 가장자리만 블러 처리
        if np.any(edge_mask):
            blurred = cv2.GaussianBlur(inpainted, (5, 5), 0)
            result = np.where(edge_mask[..., None] > 0, blurred, inpainted)
        else:
            result = inpainted
        
        self.log_debug("검은색 선 제거 완료")
        return result.astype(np.uint8)
    
    def upscale_image(self, image, scale=2):
        """LANCZOS 2x 업스케일링"""
        self.log_debug(f"{scale}x 업스케일링 시작")
        
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        self.log_debug(f"업스케일링 완료: {width}x{height} → {new_width}x{new_height}")
        return upscaled
    
    def create_thumbnail_1000x1300(self, image, inner_bbox):
        """정확한 1000x1300 썸네일 생성 (웨딩링 중심)"""
        self.log_debug("1000x1300 썸네일 생성 시작")
        
        if inner_bbox:
            x, y, w, h = inner_bbox
            
            # 여유 공간 추가 (15% 마진)
            margin_w = int(w * 0.15)
            margin_h = int(h * 0.15)
            
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image.shape[1], x + w + margin_w)
            y2 = min(image.shape[0], y + h + margin_h)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            self.log_debug(f"크롭 영역: ({x1}, {y1}) → ({x2}, {y2})")
        else:
            # 중앙 영역 크롭
            h, w = image.shape[:2]
            crop_size = min(h, w) // 2
            center_x, center_y = w // 2, h // 2
            x1 = max(0, center_x - crop_size)
            y1 = max(0, center_y - crop_size)
            x2 = min(w, center_x + crop_size)
            y2 = min(h, center_y + crop_size)
            cropped = image[y1:y2, x1:x2]
            self.log_debug("기본 중앙 크롭 적용")
        
        # 1000x1300 비율에 맞게 조정
        target_w, target_h = 1000, 1300
        crop_h, crop_w = cropped.shape[:2]
        
        # 비율 계산
        ratio_w = target_w / crop_w
        ratio_h = target_h / crop_h
        ratio = min(ratio_w, ratio_h)
        
        # 리사이즈
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 1000x1300 캔버스에 중앙 배치
        canvas = np.full((target_h, target_w, 3), 240, dtype=np.uint8)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        self.log_debug(f"썸네일 완성: {new_w}x{new_h} → 1000x1300")
        return canvas
    
    def process_wedding_ring_v14(self, image_array):
        """v14.0 메인 처리 함수"""
        self.log_debug("=== v14.0 웨딩링 처리 시작 ===")
        
        # 1. 검은색 선 테두리 감지 (웨딩링은 건드리지 않음)
        black_lines_mask, outer_bbox, inner_bbox = self.detect_black_border_lines(image_array)
        
        if black_lines_mask is None:
            self.log_debug("검은색 선을 찾을 수 없음 - 전체 이미지 처리")
            # 검은색 선이 없으면 전체 이미지를 웨딩링으로 간주
            metal_type = self.detect_metal_type(image_array)
            lighting = self.detect_lighting(image_array)
            enhanced = self.enhance_wedding_ring_v13_3(image_array, metal_type, lighting)
            upscaled = self.upscale_image(enhanced, scale=2)
            thumbnail = self.create_thumbnail_1000x1300(upscaled, None)
            
            return {
                'success': True,
                'enhanced_image': upscaled,
                'thumbnail': thumbnail,
                'processing_info': {
                    'metal_type': metal_type,
                    'lighting': lighting,
                    'black_lines_detected': False,
                    'debug_log': self.debug_info
                }
            }
        
        # 2. 웨딩링 영역에서 금속 타입 및 조명 감지
        metal_type = self.detect_metal_type(image_array, inner_bbox)
        lighting = self.detect_lighting(image_array, inner_bbox)
        
        # 3. 웨딩링 영역만 추출하여 v13.3 보정
        if inner_bbox:
            x, y, w, h = inner_bbox
            ring_region = image_array[y:y+h, x:x+w].copy()
            
            # v13.3 보정 적용
            enhanced_ring = self.enhance_wedding_ring_v13_3(ring_region, metal_type, lighting)
            
            # 보정된 웨딩링을 원본에 다시 배치
            enhanced_full = image_array.copy()
            enhanced_full[y:y+h, x:x+w] = enhanced_ring
        else:
            enhanced_full = self.enhance_wedding_ring_v13_3(image_array, metal_type, lighting)
        
        # 4. 검은색 선만 제거 (웨딩링은 보존)
        lines_removed = self.remove_black_lines_only(enhanced_full, black_lines_mask)
        
        # 5. 2x 업스케일링
        upscaled = self.upscale_image(lines_removed, scale=2)
        
        # 6. 1000x1300 썸네일 생성
        # 업스케일된 inner_bbox 계산
        if inner_bbox:
            scaled_inner_bbox = (inner_bbox[0] * 2, inner_bbox[1] * 2, 
                               inner_bbox[2] * 2, inner_bbox[3] * 2)
        else:
            scaled_inner_bbox = None
        
        thumbnail = self.create_thumbnail_1000x1300(upscaled, scaled_inner_bbox)
        
        self.log_debug("=== v14.0 웨딩링 처리 완료 ===")
        
        return {
            'success': True,
            'enhanced_image': upscaled,
            'thumbnail': thumbnail,
            'processing_info': {
                'metal_type': metal_type,
                'lighting': lighting,
                'black_lines_detected': True,
                'outer_bbox': outer_bbox,
                'inner_bbox': inner_bbox,
                'debug_log': self.debug_info
            }
        }

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.0 연결 성공: {input_data['prompt']}",
                "status": "ready_for_image_processing",
                "capabilities": [
                    "검은색 선 테두리 정확 감지",
                    "웨딩링 v13.3 보정 (28쌍 데이터)",
                    "검은색 선만 제거 (웨딩링 보존)",
                    "2x 업스케일링",
                    "1000x1300 썸네일 생성"
                ],
                "version": "14.0"
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # v14.0 프로세서 생성
            processor = WeddingRingProcessorV14()
            
            # 메인 처리
            result = processor.process_wedding_ring_v14(image_array)
            
            if not result['success']:
                return {"error": "처리 중 오류 발생"}
            
            # 결과 이미지들을 base64로 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(result['enhanced_image'])
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(result['thumbnail'])
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": result['processing_info']
            }
        
        return {"error": "image_base64 또는 prompt가 필요합니다"}
        
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
