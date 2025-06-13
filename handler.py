import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'original_blend': 0.15,
            'saturation': 1.05,
            'gamma': 1.02
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18,
            'saturation': 1.03,
            'gamma': 1.01
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12,
            'saturation': 1.08,
            'gamma': 1.03
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
            'original_blend': 0.20,
            'saturation': 1.15,
            'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.05,
            'white_overlay': 0.08,
            'sharpness': 1.10,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.22,
            'saturation': 1.10,
            'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25,
            'contrast': 1.15,
            'white_overlay': 0.04,
            'sharpness': 1.25,
            'color_temp_a': 4,
            'color_temp_b': 2,
            'original_blend': 0.18,
            'saturation': 1.25,
            'gamma': 1.02
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
            'original_blend': 0.15,
            'saturation': 1.08,
            'gamma': 1.00
        },
        'warm': {
            'brightness': 1.14,
            'contrast': 1.08,
            'white_overlay': 0.10,
            'sharpness': 1.14,
            'color_temp_a': -3,
            'color_temp_b': -2,
            'original_blend': 0.17,
            'saturation': 1.05,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22,
            'contrast': 1.15,
            'white_overlay': 0.06,
            'sharpness': 1.20,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.13,
            'saturation': 1.12,
            'gamma': 1.02
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
            'original_blend': 0.22,
            'saturation': 1.20,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.06,
            'white_overlay': 0.07,
            'sharpness': 1.12,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.25,
            'saturation': 1.12,
            'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28,
            'contrast': 1.20,
            'white_overlay': 0.03,
            'sharpness': 1.25,
            'color_temp_a': 5,
            'color_temp_b': 3,
            'original_blend': 0.20,
            'saturation': 1.28,
            'gamma': 1.03
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        logger.info("WeddingRingProcessor 초기화 완료")
        
    def detect_black_rectangle_border(self, image):
        """검은색 선으로 된 사각형 테두리 정밀 감지"""
        logger.info("검은색 사각형 테두리 감지 시작")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 검은색 선 감지 (threshold 25 이하)
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 선 강화
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("검은색 테두리를 찾을 수 없습니다")
            return None, None, None
            
        # 가장 큰 사각형 컨투어 찾기
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 사각형 근사화
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 사각형인지 확인 (4개 꼭짓점)
        if len(approx) == 4:
            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(approx)
            logger.info(f"검은색 사각형 테두리 감지 완료: ({x}, {y}, {w}, {h})")
            
            # 마스크 생성 (테두리만)
            border_mask = np.zeros_like(gray)
            cv2.drawContours(border_mask, [approx], -1, 255, thickness=cv2.FILLED)
            
            # 내부 영역 마스크 (웨딩링 영역)
            inner_mask = np.zeros_like(gray)
            # 테두리 두께 추정 (약 5-10픽셀)
            inner_contour = cv2.erode(border_mask, np.ones((10, 10), np.uint8), iterations=1)
            inner_mask[inner_contour > 0] = 255
            
            return border_mask, inner_mask, (x, y, w, h)
        else:
            logger.warning("사각형 테두리가 아닙니다")
            return None, None, None
    
    def detect_metal_type(self, image, inner_mask):
        """웨딩링 영역에서 금속 타입 감지"""
        logger.info("금속 타입 감지 시작")
        
        if inner_mask is None:
            return 'white_gold'
            
        # 마스킹된 영역에서만 HSV 분석
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask_indices = np.where(inner_mask > 0)
        
        if len(mask_indices[0]) == 0:
            return 'white_gold'
            
        hsv_values = hsv[mask_indices[0], mask_indices[1], :]
        avg_hue = np.mean(hsv_values[:, 0])
        avg_sat = np.mean(hsv_values[:, 1])
        
        logger.info(f"평균 Hue: {avg_hue}, 평균 Saturation: {avg_sat}")
        
        # 금속 타입 분류
        if avg_hue < 15 or avg_hue > 165:
            if avg_sat > 50:
                metal_type = 'rose_gold'
            else:
                metal_type = 'white_gold'
        elif 15 <= avg_hue <= 35:
            if avg_sat > 80:
                metal_type = 'yellow_gold'
            else:
                metal_type = 'champagne_gold'
        else:
            metal_type = 'white_gold'
            
        logger.info(f"감지된 금속 타입: {metal_type}")
        return metal_type
    
    def detect_lighting(self, image):
        """조명 환경 감지"""
        logger.info("조명 환경 감지 시작")
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:, :, 2]
        b_mean = np.mean(b_channel)
        
        if b_mean < 125:
            lighting = 'warm'
        elif b_mean > 135:
            lighting = 'cool'
        else:
            lighting = 'natural'
            
        logger.info(f"감지된 조명 환경: {lighting}")
        return lighting
    
    def enhance_wedding_ring_region(self, image, inner_mask, metal_type, lighting):
        """웨딩링 영역만 확대하여 v13.3 보정 후 원래 크기로 복귀"""
        logger.info("웨딩링 영역 확대 보정 시작")
        
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                                           WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 웨딩링 영역 추출
        ring_region = cv2.bitwise_and(image, image, mask=inner_mask)
        
        # 웨딩링 영역 바운딩 박스
        coords = np.where(inner_mask > 0)
        if len(coords[0]) == 0:
            return image
            
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # 웨딩링 영역 크롭
        ring_crop = ring_region[y_min:y_max, x_min:x_max]
        mask_crop = inner_mask[y_min:y_max, x_min:x_max]
        
        # 2x 확대
        scale_factor = 2
        h, w = ring_crop.shape[:2]
        enlarged = cv2.resize(ring_crop, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_LANCZOS4)
        enlarged_mask = cv2.resize(mask_crop, (w * scale_factor, h * scale_factor), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # v13.3 보정 적용
        enhanced_enlarged = self.apply_v13_enhancement(enlarged, params)
        
        # 원래 크기로 축소
        enhanced_original = cv2.resize(enhanced_enlarged, (w, h), 
                                     interpolation=cv2.INTER_LANCZOS4)
        
        # 원본 이미지에 합성
        result = image.copy()
        result[y_min:y_max, x_min:x_max] = enhanced_original
        
        logger.info("웨딩링 영역 보정 완료")
        return result
    
    def apply_v13_enhancement(self, image, params):
        """v13.3 보정 적용 (모든 기능 포함)"""
        logger.info("v13.3 보정 시작")
        
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. PIL 기본 보정
        pil_image = Image.fromarray(denoised)
        
        # 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 대비 조정
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # 채도 조정
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(params['saturation'])
        
        enhanced_array = np.array(enhanced)
        
        # 3. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 4. LAB 색공간 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 5. CLAHE 적용 (명료도 향상)
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 6. 감마 보정
        gamma = params['gamma']
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced_array = cv2.LUT(enhanced_array, table)
        
        # 7. 원본과 블렌딩 (자연스러움 보장)
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        
        logger.info("v13.3 보정 완료")
        return final.astype(np.uint8)
    
    def apply_global_color_harmony(self, image, enhanced_ring_image):
        """전체 색감 조화 (학습파일 참고)"""
        logger.info("전체 색감 조화 시작")
        
        # 전체 이미지에 미묘한 보정 적용
        pil_image = Image.fromarray(enhanced_ring_image)
        
        # 전체적인 밝기 향상 (10%)
        enhancer = ImageEnhance.Brightness(pil_image)
        harmonized = enhancer.enhance(1.10)
        
        # 전체적인 대비 향상 (5%)
        enhancer = ImageEnhance.Contrast(harmonized)
        harmonized = enhancer.enhance(1.05)
        
        # 배경 색온도 조화
        harmonized_array = np.array(harmonized)
        lab = cv2.cvtColor(harmonized_array, cv2.COLOR_RGB2LAB)
        
        # 배경 영역의 A, B 채널 미세 조정
        lab[:, :, 1] = np.clip(lab[:, :, 1] - 2, 0, 255)  # 약간 차갑게
        lab[:, :, 2] = np.clip(lab[:, :, 2] - 1, 0, 255)  # 약간 덜 황색
        
        harmonized_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        logger.info("전체 색감 조화 완료")
        return harmonized_array
    
    def remove_black_border_completely(self, image, border_mask):
        """검은색 테두리 완전 제거 (고급 inpainting)"""
        logger.info("검은색 테두리 완전 제거 시작")
        
        # 테두리 마스크 확장 (경계 부드럽게)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded_mask = cv2.dilate(border_mask, kernel, iterations=2)
        
        # Telea 인페인팅
        inpainted = cv2.inpaint(image, expanded_mask, 5, cv2.INPAINT_TELEA)
        
        # NS 인페인팅으로 추가 보정
        inpainted = cv2.inpaint(inpainted, border_mask, 3, cv2.INPAINT_NS)
        
        # 가장자리 블러 처리
        edge_mask = cv2.dilate(border_mask, kernel, iterations=1) - border_mask
        if np.any(edge_mask):
            blurred = cv2.GaussianBlur(inpainted, (7, 7), 0)
            inpainted = np.where(edge_mask[..., None] > 0, 
                               cv2.addWeighted(inpainted, 0.7, blurred, 0.3, 0), 
                               inpainted)
        
        logger.info("검은색 테두리 제거 완료")
        return inpainted.astype(np.uint8)
    
    def create_thumbnail_1000x1300(self, image, bbox):
        """검은색 선 기준으로 정확한 1000x1300 썸네일 생성"""
        logger.info("1000x1300 썸네일 생성 시작")
        
        x, y, w, h = bbox
        
        # 검은색 선 영역에서 10% 마진 추가
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # 웨딩링 중심 영역 크롭
        cropped = image[y1:y2, x1:x2]
        
        # 1000x1300 비율에 맞게 조정
        target_w, target_h = 1000, 1300
        crop_h, crop_w = cropped.shape[:2]
        
        # 비율 계산 (큰 쪽에 맞춤)
        ratio_w = target_w / crop_w
        ratio_h = target_h / crop_h
        ratio = max(ratio_w, ratio_h)  # 큰 쪽에 맞춰서 크롭되지 않게
        
        # 리사이즈
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 1000x1300 캔버스에 중앙 배치
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
        
        # 크롭 처리 (너무 큰 경우)
        if new_w > target_w:
            start_x_crop = (new_w - target_w) // 2
            resized = resized[:, start_x_crop:start_x_crop + target_w]
            new_w = target_w
            
        if new_h > target_h:
            start_y_crop = (new_h - target_h) // 2
            resized = resized[start_y_crop:start_y_crop + target_h, :]
            new_h = target_h
        
        # 캔버스에 배치
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        # 썸네일 전용 보정 (더 선명하고 밝게)
        thumbnail_enhanced = self.enhance_thumbnail(canvas)
        
        logger.info("1000x1300 썸네일 생성 완료")
        return thumbnail_enhanced
    
    def enhance_thumbnail(self, image):
        """썸네일 전용 보정 (디테일 강화)"""
        logger.info("썸네일 보정 시작")
        
        pil_image = Image.fromarray(image)
        
        # 밝기 강화 (25%)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(1.25)
        
        # 대비 강화 (20%)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.20)
        
        # 선명도 강화 (30%)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.30)
        
        # 채도 강화 (15%)
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.15)
        
        enhanced_array = np.array(enhanced)
        
        # CLAHE로 디테일 극대화
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        logger.info("썸네일 보정 완료")
        return enhanced_array.astype(np.uint8)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI 연결 성공: {input_data['prompt']}",
                "status": "ready_for_processing",
                "version": "v13.3_complete_fixed",
                "capabilities": [
                    "검은색 선 테두리 정밀 감지",
                    "웨딩링 확대 보정 후 복귀",
                    "v13.3 완전 보정 (28쌍 데이터)",
                    "검은색 선 완전 제거",
                    "전체 색감 조화",
                    "1000x1300 썸네일 생성"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64가 필요합니다"}
            
        logger.info("이미지 처리 시작")
        
        # Base64 디코딩
        image_data = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        processor = WeddingRingProcessor()
        
        # 1. 검은색 사각형 테두리 감지
        border_mask, inner_mask, bbox = processor.detect_black_rectangle_border(image_array)
        
        if border_mask is None:
            return {
                "error": "검은색 사각형 테두리를 찾을 수 없습니다",
                "suggestion": "검은색 선으로 된 사각형 테두리가 명확한지 확인해주세요"
            }
        
        # 2. 웨딩링 금속 타입 및 조명 감지
        metal_type = processor.detect_metal_type(image_array, inner_mask)
        lighting = processor.detect_lighting(image_array)
        
        # 3. 웨딩링 영역 확대 → v13.3 보정 → 축소 복귀
        enhanced_ring = processor.enhance_wedding_ring_region(
            image_array, inner_mask, metal_type, lighting
        )
        
        # 4. 전체 색감 조화 (학습파일 참고)
        harmonized_image = processor.apply_global_color_harmony(
            image_array, enhanced_ring
        )
        
        # 5. 검은색 테두리 완전 제거
        final_image = processor.remove_black_border_completely(
            harmonized_image, border_mask
        )
        
        # 6. 1000x1300 썸네일 생성
        thumbnail = processor.create_thumbnail_1000x1300(final_image, bbox)
        
        # 7. 결과 인코딩
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
                "metal_type": metal_type,
                "lighting": lighting,
                "border_detected": True,
                "border_removed": True,
                "bbox": bbox,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "thumbnail_size": "1000x1300",
                "version": "v13.3_complete_fixed"
            },
            "success": True
        }
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        return {
            "error": f"처리 중 오류 발생: {str(e)}",
            "success": False
        }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
