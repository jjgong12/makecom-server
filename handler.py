import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 파라미터 (28쌍 학습 데이터 기반)
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
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        print("웨딩링 프로세서 초기화 - LANCZOS 업스케일링 사용")
        
    def detect_black_masking(self, image):
        """정밀한 검은색 마스킹 감지 및 웨딩링 영역 추출"""
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
            return mask, largest_contour, (x, y, w, h)
        
        return None, None, None

    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 감지"""
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
            return 'white_gold'

    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:, :, 2]
        b_mean = np.mean(b_channel)
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'

    def enhance_wedding_ring(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (28쌍 학습 데이터 기반)"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting,
                WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # PIL ImageEnhance로 기본 보정
        pil_image = Image.fromarray(image)
        
        # 1. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 2. 대비 조정
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 3. 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # 4. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
        enhanced_array = np.array(enhanced)
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 5. LAB 색공간에서 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 6. 원본과 블렌딩 (자연스러움 보장)
        original_blend = params['original_blend']
        final = cv2.addWeighted(
            enhanced_array, 1 - original_blend,
            image, original_blend, 0
        )
        
        return final.astype(np.uint8)

    def lanczos_upscale(self, image, scale=2):
        """LANCZOS 2x 업스케일링"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def inpaint_masking(self, image, mask):
        """검은색 마스킹 영역 인페인팅으로 제거"""
        # OpenCV TELEA 인페인팅
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # 추가적으로 가장자리 부드럽게 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        edge_mask = dilated_mask - mask
        
        # 가장자리 영역 가우시안 블러 적용
        blurred = cv2.GaussianBlur(inpainted, (7, 7), 0)
        result = np.where(edge_mask[..., None] > 0, blurred, inpainted)
        
        return result.astype(np.uint8)

    def create_thumbnail(self, image, bbox, target_size=(1000, 1300)):
        """검은색 마스킹 기준으로 썸네일 생성"""
        x, y, w, h = bbox
        
        # 여유 공간 추가 (20% 마진)
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # 크롭
        cropped = image[y1:y2, x1:x2]
        
        # 1000x1300 비율에 맞게 조정
        target_w, target_h = target_size
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
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas

def handler(event):
    """RunPod Serverless 메인 핸들러 - 바이너리 직접 반환"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI 연결 성공: {input_data['prompt']}",
                "status": "ready_for_image_processing"
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 프로세서 초기화
            processor = WeddingRingProcessor()
            
            # 1. 검은색 마스킹 감지
            mask, contour, bbox = processor.detect_black_masking(image_array)
            if mask is None:
                return {"error": "검은색 마스킹을 찾을 수 없습니다."}
            
            # 2. 금속 타입 및 조명 감지 (마스킹 영역 기준)
            metal_type = processor.detect_metal_type(image_array, mask)
            lighting = processor.detect_lighting(image_array)
            
            # 3. 웨딩링 영역 추출 (마스킹 내부만)
            ring_mask = mask.copy()
            ring_region = cv2.bitwise_and(image_array, image_array, mask=ring_mask)
            
            # 4. v13.3 웨딩링 보정
            enhanced_ring = processor.enhance_wedding_ring(ring_region, metal_type, lighting)
            
            # 5. LANCZOS 2x 업스케일링
            upscaled = processor.lanczos_upscale(enhanced_ring, scale=2)
            
            # 마스크도 같은 비율로 확대
            scale_factor = upscaled.shape[0] / enhanced_ring.shape[0]
            upscaled_mask = cv2.resize(mask,
                (int(mask.shape[1] * scale_factor), int(mask.shape[0] * scale_factor)),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 6. 인페인팅으로 검은색 마스킹 제거
            final_image = processor.inpaint_masking(upscaled, upscaled_mask)
            
            # 7. 썸네일 생성 (원본 bbox 기준)
            thumbnail = processor.create_thumbnail(final_image, bbox)
            
            # 8. 바이너리 직접 반환 (base64 변환 없음!)
            # 메인 이미지
            main_pil = Image.fromarray(final_image)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_binary = main_buffer.getvalue()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_binary = thumb_buffer.getvalue()
            
            return {
                "enhanced_image": main_binary,  # 바이너리 직접 반환!
                "thumbnail": thumb_binary,      # 바이너리 직접 반환!
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "masking_detected": True,
                    "scale_factor": scale_factor,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300"
                }
            }
    
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 설정
runpod.serverless.start({"handler": handler})
