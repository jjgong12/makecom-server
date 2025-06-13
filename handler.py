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
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.08,
            'sharpness': 1.16, 'color_temp_a': -1, 'color_temp_b': -1,
            'original_blend': 0.15
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22
        }
    }
}

class WeddingRingProcessor:
    def detect_black_line_border(self, image):
        """검은색 선으로 된 사각형 테두리 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. 검은색 선 감지 (매우 어두운 픽셀)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 3. 가장 큰 사각형 모양의 컨투어 찾기
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                # 컨투어를 사각형으로 근사화
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 4개의 점을 가진 사각형이고, 면적이 충분히 큰 경우
                if len(approx) == 4 and cv2.contourArea(contour) > 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 5% 안쪽 마진으로 내부 영역만 반환 (선 두께 고려)
                    margin_x = int(w * 0.05)
                    margin_y = int(h * 0.05)
                    
                    return (x + margin_x, y + margin_y, 
                           w - 2*margin_x, h - 2*margin_y), contour
        
        return None, None

    def detect_metal_type(self, image):
        """HSV 색공간 분석으로 금속 타입 감지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        
        if avg_hue < 15 or avg_hue > 165:
            return 'rose_gold' if avg_sat > 50 else 'white_gold'
        elif 15 <= avg_hue <= 35:
            return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
        else:
            return 'white_gold'

    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
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

    def apply_global_color_adjustment(self, image, metal_type, lighting):
        """전체 색감 조정 (50% 강도)"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting,
                                                              WEDDING_RING_PARAMS['white_gold']['natural'])
        
        pil_image = Image.fromarray(image)
        
        # 전체적인 미묘한 조정 (50% 강도)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(1 + (params['brightness'] - 1) * 0.5)
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1 + (params['contrast'] - 1) * 0.5)
        
        return np.array(enhanced)

    def remove_black_lines(self, image, line_contour):
        """검은색 선 제거 (inpainting)"""
        # 1. 선 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [line_contour], -1, 255, thickness=10)  # 선 두께 고려
        
        # 2. 가장자리만 남기고 내부는 제거 (inpainting이 자연스럽게)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # 3. inpainting으로 선 제거
        result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        
        return result

    def create_thumbnail_from_border(self, image, border_rect, target_size=(1000, 1300)):
        """검은색 선 기준 썸네일 생성"""
        x, y, w, h = border_rect
        
        # 1. 15% 마진 추가해서 크롭
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)
        
        crop_x1 = max(0, x - margin_w)
        crop_y1 = max(0, y - margin_h)
        crop_x2 = min(image.shape[1], x + w + margin_w)
        crop_y2 = min(image.shape[0], y + h + margin_h)
        
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 2. 1000×1300 비율에 맞게 조정
        target_w, target_h = target_size
        crop_h, crop_w = cropped.shape[:2]
        
        ratio = min(target_w / crop_w, target_h / crop_h)
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        
        # 3. 2x 업스케일링
        upscaled = cv2.resize(cropped, (new_w*2, new_h*2), interpolation=cv2.INTER_LANCZOS4)
        
        # 4. 최종 캔버스에 중앙 배치
        canvas = np.full((target_h*2, target_w*2, 3), 240, dtype=np.uint8)
        start_y = (target_h*2 - new_h*2) // 2
        start_x = (target_w*2 - new_w*2) // 2
        canvas[start_y:start_y+new_h*2, start_x:start_x+new_w*2] = upscaled
        
        return canvas

    def process_complete_workflow(self, image):
        """완전한 웨딩링 처리 워크플로우"""
        
        # 1. 검은색 선 테두리 감지
        border_rect, line_contour = self.detect_black_line_border(image)
        if border_rect is None:
            return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
        
        x, y, w, h = border_rect
        
        # 2. 테두리 내부 웨딩링 영역 추출
        ring_region = image[y:y+h, x:x+w].copy()
        
        # 3. 금속 타입 및 조명 감지
        metal_type = self.detect_metal_type(ring_region)
        lighting = self.detect_lighting(ring_region)
        
        # 4. 웨딩링 영역 확대 → 보정 → 축소
        # 4-1. 2x 확대
        enlarged = cv2.resize(ring_region, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
        
        # 4-2. v13.3 보정 적용
        enhanced_enlarged = self.enhance_wedding_ring(enlarged, metal_type, lighting)
        
        # 4-3. 원래 크기로 축소
        enhanced_region = cv2.resize(enhanced_enlarged, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 5. 전체 이미지에 합성
        result_image = image.copy()
        result_image[y:y+h, x:x+w] = enhanced_region
        
        # 6. 전체 색감 조정 (28쌍 학습파일 기준)
        result_image = self.apply_global_color_adjustment(result_image, metal_type, lighting)
        
        # 7. 검은색 선 제거 (inpainting)
        final_image = self.remove_black_lines(result_image, line_contour)
        
        # 8. 썸네일 생성 (검은색 선 기준 크롭 → 1000×1300 → 업스케일링)
        thumbnail = self.create_thumbnail_from_border(final_image, border_rect)
        
        return {
            "enhanced_image": final_image,
            "thumbnail": thumbnail,
            "border_rect": border_rect,
            "metal_type": metal_type,
            "lighting": lighting,
            "processing_steps": [
                "검은색 선 테두리 감지",
                f"웨딩링 영역 확대 → {metal_type}/{lighting} 보정 → 축소",
                "전체 색감 조정",
                "검은색 선 제거",
                "썸네일 생성 (1000×1300 업스케일링)"
            ]
        }

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI 연결 성공: {input_data['prompt']}",
                "status": "ready_for_image_processing",
                "capabilities": [
                    "검은색 선 테두리 감지",
                    "웨딩링 확대 보정",
                    "전체 색감 조정",
                    "검은색 선 제거",
                    "썸네일 생성 (1000×1300 업스케일링)"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 프로세서 초기화 및 처리
            processor = WeddingRingProcessor()
            result = processor.process_complete_workflow(image_array)
            
            if "error" in result:
                return result
            
            # 결과 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(result["enhanced_image"])
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(result["thumbnail"])
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": result["metal_type"],
                    "lighting": result["lighting"],
                    "border_rect": result["border_rect"],
                    "processing_steps": result["processing_steps"],
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "thumbnail_size": "2000x2600",  # 1000×1300을 2x 업스케일링
                    "status": "완전 처리 완료"
                }
            }
            
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
