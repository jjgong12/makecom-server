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
        
    def detect_black_masking(self, image):
        """검은색 마스킹 감지 (기존 작동하던 방식 유지)"""
        logger.info("검은색 마스킹 감지 시작")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 검은색 영역 감지 (threshold < 20)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 마스킹 영역 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 영역을 마스킹으로 판단
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 마스크 생성
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # 웨딩링 영역 바운딩 박스
            x, y, w, h = cv2.boundingRect(largest_contour)
            logger.info(f"검은색 마스킹 감지 완료: ({x}, {y}, {w}, {h})")
            
            return mask, largest_contour, (x, y, w, h)
        
        return None, None, None
    
    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 감지 (기존 방식 유지)"""
        logger.info("금속 타입 감지 시작")
        
        if mask is not None:
            # 마스킹 영역 내에서만 분석
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 마스킹된 영역의 평균 색상 계산
        if mask is not None:
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
            else:
                return 'white_gold'
        else:
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
        
        logger.info(f"평균 Hue: {avg_hue}, 평균 Saturation: {avg_sat}")
        
        # 금속 타입 분류 (기존 방식)
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
    
    def enhance_wedding_ring(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (기존 잘 작동하던 방식 유지)"""
        logger.info("v13.3 웨딩링 보정 시작")
        
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                                           WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. PIL ImageEnhance로 기본 보정
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
        
        # 4. LAB 색공간에서 색온도 조정
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
        
        logger.info("v13.3 웨딩링 보정 완료")
        return final.astype(np.uint8)
    
    def remove_black_border_simple(self, image, mask):
        """검은색 테두리 간단한 제거 (새로 추가)"""
        logger.info("검은색 테두리 제거 시작")
        
        # 마스크 약간 확장
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Telea 인페인팅으로 제거
        inpainted = cv2.inpaint(image, expanded_mask, 3, cv2.INPAINT_TELEA)
        
        logger.info("검은색 테두리 제거 완료")
        return inpainted.astype(np.uint8)
    
    def create_thumbnail_1000x1300_fixed(self, image, bbox):
        """검은색 영역 기준 정확한 1000x1300 썸네일 생성 (수정됨)"""
        logger.info("1000x1300 썸네일 생성 시작")
        
        x, y, w, h = bbox
        
        # 웨딩링 중심 영역만 크롭 (마진 최소화)
        margin = 20  # 20픽셀만 마진
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # 웨딩링 영역 크롭
        cropped = image[y1:y2, x1:x2]
        
        # 정확한 1000x1300 리사이즈
        resized = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        # 썸네일 전용 보정
        thumbnail_enhanced = self.enhance_thumbnail_simple(resized)
        
        logger.info("1000x1300 썸네일 생성 완료")
        return thumbnail_enhanced
    
    def enhance_thumbnail_simple(self, image):
        """썸네일 간단한 보정 (새로 추가)"""
        pil_image = Image.fromarray(image)
        
        # 밝기 향상 (15%)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(1.15)
        
        # 대비 향상 (10%)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.10)
        
        # 선명도 향상 (20%)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.20)
        
        return np.array(enhanced)

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
                "version": "v13.3_fixed_minimal",
                "capabilities": [
                    "검은색 마스킹 감지 (기존 방식)",
                    "v13.3 웨딩링 보정 (28쌍 데이터)",
                    "검은색 선 간단 제거 (새로 추가)",
                    "1000x1300 썸네일 (수정됨)"
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
        
        # 1. 검은색 마스킹 감지 (기존 방식)
        mask, contour, bbox = processor.detect_black_masking(image_array)
        
        if mask is None:
            return {
                "error": "검은색 마스킹을 찾을 수 없습니다",
                "suggestion": "검은색 영역이 명확한지 확인해주세요"
            }
        
        # 2. 웨딩링 금속 타입 및 조명 감지 (기존 방식)
        metal_type = processor.detect_metal_type(image_array, mask)
        lighting = processor.detect_lighting(image_array)
        
        # 3. v13.3 웨딩링 보정 (기존 방식, 잘 작동하던 것)
        enhanced_image = processor.enhance_wedding_ring(image_array, metal_type, lighting)
        
        # 4. 검은색 선 제거 (새로 추가)
        final_image = processor.remove_black_border_simple(enhanced_image, mask)
        
        # 5. 1000x1300 썸네일 생성 (수정됨)
        thumbnail = processor.create_thumbnail_1000x1300_fixed(final_image, bbox)
        
        # 6. 결과 인코딩
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
                "masking_detected": True,
                "border_removed": True,
                "bbox": bbox,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "thumbnail_size": "1000x1300",
                "version": "v13.3_fixed_minimal"
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
