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
    }
}

class WeddingRingProcessor:
    def detect_black_masking(self, image):
        """검은색 마킹 영역 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 마킹 영역 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return mask, (x, y, w, h)
        
        return None, None

    def detect_metal_type(self, image, mask=None):
        """금속 타입 감지 (HSV 기반)"""
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
            else:
                return 'white_gold'
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
        
        # 금속 분류 로직
        if avg_hue < 15 or avg_hue > 165:
            return 'rose_gold' if avg_sat > 50 else 'white_gold'
        elif 15 <= avg_hue <= 35:
            return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
        else:
            return 'white_gold'

    def enhance_full_image(self, image, metal_type):
        """원본 전체 이미지 v13.3 보정"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get('natural', 
                                        WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # PIL ImageEnhance로 전체 보정
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
        
        # 4. 하얀색 오버레이 적용
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
        
        # 6. 원본과 블렌딩
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        
        return final.astype(np.uint8)

    def enhance_ring_area(self, image, mask, metal_type):
        """마킹 내 커플링 영역 확대 보정"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get('natural', 
                                        WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 마킹 영역만 추출
        ring_area = cv2.bitwise_and(image, image, mask=mask)
        
        # 더 강한 보정 (확대 보정)
        pil_ring = Image.fromarray(ring_area)
        
        # 밝기 +20% 추가
        enhancer = ImageEnhance.Brightness(pil_ring)
        enhanced = enhancer.enhance(params['brightness'] * 1.2)
        
        # 선명도 +30% 추가
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'] * 1.3)
        
        # 대비 +15% 추가
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'] * 1.15)
        
        return np.array(enhanced)

    def remove_black_masking(self, image, mask):
        """검은색 마킹 인페인팅으로 제거"""
        # OpenCV 인페인팅으로 마킹 제거
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # 가장자리 부드럽게 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        edge_mask = dilated_mask - mask
        
        # 가장자리 가우시안 블러
        if np.any(edge_mask):
            blurred = cv2.GaussianBlur(inpainted, (7, 7), 0)
            result = np.where(edge_mask[..., None] > 0, blurred, inpainted)
        else:
            result = inpainted
            
        return result.astype(np.uint8)

    def create_thumbnail(self, image, bbox):
        """마킹 영역 크롭 → 1000×1300 썸네일 생성"""
        x, y, w, h = bbox
        
        # 20% 마진 추가
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # 마킹 영역 크롭
        cropped = image[y1:y2, x1:x2]
        
        # 1000×1300 리사이즈
        target_w, target_h = 1000, 1300
        crop_h, crop_w = cropped.shape[:2]
        
        ratio = min(target_w / crop_w, target_h / crop_h)
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        
        # 업스케일링 (품질 저하 방지)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 1000×1300 캔버스에 중앙 배치
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # 썸네일 추가 보정
        pil_thumb = Image.fromarray(canvas)
        enhancer = ImageEnhance.Sharpness(pil_thumb)
        enhanced_thumb = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(enhanced_thumb)
        final_thumb = enhancer.enhance(1.1)
        
        return np.array(final_thumb)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        image_base64 = input_data["image_base64"]
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        original_image = np.array(image.convert('RGB'))
        
        processor = WeddingRingProcessor()
        
        # 1. 검은색 마킹 감지
        mask, bbox = processor.detect_black_masking(original_image)
        if mask is None:
            return {"error": "검은색 마킹을 찾을 수 없습니다."}
        
        # 2. 금속 타입 감지
        metal_type = processor.detect_metal_type(original_image, mask)
        
        # === A_001 컨셉샷 생성 ===
        # 3. 원본 전체 v13.3 보정
        full_enhanced = processor.enhance_full_image(original_image, metal_type)
        
        # 4. 마킹 내 커플링 확대 보정
        enhanced_ring = processor.enhance_ring_area(full_enhanced, mask, metal_type)
        
        # 5. 확대 보정된 커플링을 전체 이미지에 합성
        mask_3d = np.stack([mask] * 3, axis=-1)
        full_with_enhanced_ring = np.where(mask_3d > 0, enhanced_ring, full_enhanced)
        
        # 6. 검은색 마킹 제거 → A_001 컨셉샷 완성
        a001_result = processor.remove_black_masking(full_with_enhanced_ring, mask)
        
        # === 썸네일 생성 ===
        # 7. 마킹 영역 크롭 → 1000×1300 썸네일
        thumbnail_result = processor.create_thumbnail(original_image, bbox)
        
        # 8. 결과 인코딩
        # A_001 컨셉샷
        a001_pil = Image.fromarray(a001_result)
        a001_buffer = io.BytesIO()
        a001_pil.save(a001_buffer, format='JPEG', quality=95, progressive=True)
        a001_base64 = base64.b64encode(a001_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail_result)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        return {
            "enhanced_image": a001_base64,      # A_001 컨셉샷
            "thumbnail": thumb_base64,          # 썸네일
            "processing_info": {
                "metal_type": metal_type,
                "masking_detected": True,
                "bbox": bbox,
                "original_size": f"{original_image.shape[1]}x{original_image.shape[0]}",
                "a001_size": f"{a001_result.shape[1]}x{a001_result.shape[0]}",
                "thumbnail_size": "1000x1300"
            }
        }
        
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 설정
runpod.serverless.start({"handler": handler})
