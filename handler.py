import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
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
        """검은색 마스킹 감지 (원래 잘 작동하던 방식)"""
        logger.info("검은색 마스킹 감지 시작")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 검은색 영역 감지 (threshold < 25)
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 마스킹 영역 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기 및 바운딩 박스 추출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            x, y, w, h = cv2.boundingRect(largest_contour)
            logger.info(f"검은색 마스킹 감지 완료: ({x}, {y}, {w}, {h})")
            return mask, (x, y, w, h)
        
        return None, None
    
    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 자동 감지"""
        logger.info("금속 타입 감지 시작")
        
        if mask is not None:
            # 마스킹 영역 내에서만 분석
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return 'white_gold'
            rgb_values = image[mask_indices[0], mask_indices[1], :]
            hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            avg_hue = np.mean(hsv_values[:, 0])
            avg_sat = np.mean(hsv_values[:, 1])
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
        
        # 금속 타입 분류
        if avg_sat < 30:
            return 'white_gold'
        elif 5 <= avg_hue <= 25:
            return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
        elif avg_hue < 5 or avg_hue > 170:
            return 'rose_gold'
        else:
            return 'white_gold'
    
    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        logger.info("조명 환경 감지 시작")
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'
    
    def enhance_wedding_ring(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (원래 잘 작동하던 방식)"""
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
        
        # 3. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
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
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
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
    
    def basic_upscale(self, image, scale=2):
        """기본 업스케일링 (2x)"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def inpaint_masking(self, image, mask):
        """검은색 마스킹 영역 인페인팅으로 제거"""
        logger.info("검은색 마스킹 제거 시작")
        
        # 마스크를 약간 확장해서 완전히 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Telea 인페인팅으로 제거
        inpainted = cv2.inpaint(image, expanded_mask, 3, cv2.INPAINT_TELEA)
        
        # 가장자리 부드럽게 처리
        edge_mask = expanded_mask - mask
        if np.any(edge_mask):
            blurred = cv2.GaussianBlur(inpainted, (5, 5), 0)
            result = np.where(edge_mask[..., None] > 0, blurred, inpainted)
        else:
            result = inpainted
        
        logger.info("검은색 마스킹 제거 완료")
        return result.astype(np.uint8)
    
    def create_thumbnail(self, image, bbox, target_size=(1000, 1300)):
        """검은색 마스킹 기준으로 1000x1300 썸네일 생성"""
        logger.info("1000x1300 썸네일 생성 시작")
        
        x, y, w, h = bbox
        
        # 웨딩링 중심으로 크롭 (최소 마진)
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # 크롭 및 1000x1300 리사이즈
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 썸네일 전용 보정
        pil_thumbnail = Image.fromarray(resized)
        
        # 밝기 향상 (20%)
        enhancer = ImageEnhance.Brightness(pil_thumbnail)
        enhanced = enhancer.enhance(1.20)
        
        # 대비 향상 (15%)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.15)
        
        # 선명도 향상 (25%)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.25)
        
        logger.info("1000x1300 썸네일 생성 완료")
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
                "version": "v13.3_original_working",
                "capabilities": [
                    "검은색 마스킹 감지",
                    "v13.3 웨딩링 보정 (28쌍 데이터)",
                    "검은색 마스킹 제거",
                    "2x 업스케일링",
                    "1000x1300 썸네일"
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
        
        # 1. 검은색 마스킹 감지
        mask, bbox = processor.detect_black_masking(image_array)
        
        if mask is None:
            return {"error": "검은색 마스킹을 찾을 수 없습니다."}
        
        # 2. 금속 타입 및 조명 감지
        metal_type = processor.detect_metal_type(image_array, mask)
        lighting = processor.detect_lighting(image_array)
        
        # 3. v13.3 웨딩링 보정 (원래 잘 작동하던 방식)
        enhanced = processor.enhance_wedding_ring(image_array, metal_type, lighting)
        
        # 4. 2x 업스케일링
        upscaled = processor.basic_upscale(enhanced, scale=2)
        upscaled_mask = processor.basic_upscale(mask, scale=2)
        upscaled_mask = np.where(upscaled_mask > 127, 255, 0).astype(np.uint8)
        
        # 5. 인페인팅으로 검은색 마스킹 제거
        final_image = processor.inpaint_masking(upscaled, upscaled_mask)
        
        # 6. 1000x1300 썸네일 생성
        scaled_bbox = (bbox[0]*2, bbox[1]*2, bbox[2]*2, bbox[3]*2)
        thumbnail = processor.create_thumbnail(final_image, scaled_bbox)
        
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
                "bbox": bbox,
                "scale_factor": 2,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                "thumbnail_size": "1000x1300"
            }
        }
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
