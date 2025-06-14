import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

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
            'white_overlay': 0.04,
            'sharpness': 1.12,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.08,
            'sharpness': 1.18,
            'color_temp_a': 4,
            'color_temp_b': 2,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17,
            'contrast': 1.11,
            'white_overlay': 0.12,  # v14.2에서 화이트골드 방향으로 조정
            'sharpness': 1.16,
            'color_temp_a': -4,     # v14.2에서 -1 → -4
            'color_temp_b': -4,     # v14.2에서 -1 → -4
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.14,
            'contrast': 1.08,
            'white_overlay': 0.10,
            'sharpness': 1.13,
            'color_temp_a': -6,
            'color_temp_b': -6,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.14,
            'sharpness': 1.19,
            'color_temp_a': -2,
            'color_temp_b': -2,
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
            'brightness': 1.13,
            'contrast': 1.06,
            'white_overlay': 0.03,
            'sharpness': 1.11,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.28
        },
        'cool': {
            'brightness': 1.19,
            'contrast': 1.12,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': 5,
            'color_temp_b': 3,
            'original_blend': 0.18
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        
    def detect_black_border_v145(self, image):
        """v14.5 강화된 검은색 테두리 감지"""
        print("Starting v14.5 enhanced black border detection...")
        
        # 1. 더 정밀한 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. 다중 threshold로 정확한 검은색 감지
        _, binary1 = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY_INV)  # 더 낮은 threshold
        _, binary2 = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY_INV)  # 중간 threshold
        _, binary3 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)  # 높은 threshold
        
        # 3. 다중 threshold 결과 조합
        combined_binary = cv2.bitwise_and(binary1, binary2)
        combined_binary = cv2.bitwise_and(combined_binary, binary3)
        
        print(f"Multi-threshold detection completed")
        
        # 4. 노이즈 제거 (작은 점들 제거)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_small)
        
        # 5. 직사각형 컨투어 찾기
        contours, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        best_area = 0
        
        for contour in contours:
            # 컨투어 면적 체크
            area = cv2.contourArea(contour)
            if area < 5000:  # 최소 면적
                continue
                
            # 직사각형 근사
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4개 꼭짓점인지 확인 (직사각형)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # 종횡비 체크 (정사각형에 가까운 직사각형)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # 더 넓은 범위
                    if area > best_area:
                        best_area = area
                        best_rect = (x, y, w, h)
                        
        if best_rect is None:
            print("No valid rectangular border found")
            return None, None, None
            
        x, y, w, h = best_rect
        print(f"Found rectangular border: {best_rect}, area: {best_area}")
        
        # 6. 검은색 선 마스크 생성 (더 정밀하게)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # 테두리 선만 마스킹 (내부는 제외)
        border_thickness = max(3, min(w, h) // 50)  # 동적 두께 계산
        
        # 바깥쪽 사각형
        cv2.rectangle(mask, (x-border_thickness, y-border_thickness), 
                     (x+w+border_thickness, y+h+border_thickness), 255, -1)
        
        # 안쪽 사각형 제거 (웨딩링 영역 보호)
        cv2.rectangle(mask, (x+border_thickness, y+border_thickness), 
                     (x+w-border_thickness, y+h-border_thickness), 0, -1)
        
        # 웨딩링 영역 bbox
        ring_bbox = (x+border_thickness*2, y+border_thickness*2, 
                    w-border_thickness*4, h-border_thickness*4)
        
        print(f"Created precise border mask, ring bbox: {ring_bbox}")
        return mask, best_rect, ring_bbox
    
    def advanced_inpainting_v146(self, image, mask, ring_bbox):
        """v14.6 - 25번 성공 방식 완전 재현"""
        print("Starting v14.6 inpainting - 25번 성공 방식 재현...")
        
        # 1. 웨딩링 완전 보호 마스크 생성 (25번 방식)
        ring_protection_mask = np.zeros_like(mask)
        if ring_bbox:
            x, y, w, h = ring_bbox
            # 웨딩링 영역을 255로 설정 (보호)
            ring_protection_mask[y:y+h, x:x+w] = 255
            print(f"Ring protection area: {ring_bbox}")
        
        # 2. 검은색 선만 제거할 마스크 (웨딩링 영역 제외)
        line_only_mask = mask.copy()
        line_only_mask[ring_protection_mask > 0] = 0  # 웨딩링 영역은 inpainting 금지
        
        # 3. 강력한 inpainting으로 검은색 선 완전 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        expanded_line_mask = cv2.dilate(line_only_mask, kernel, iterations=3)
        
        # NS 방식으로 더 강력한 inpainting
        inpainted = cv2.inpaint(image, expanded_line_mask, 15, cv2.INPAINT_NS)
        
        # 4. 25번 성공 방식: 부드러운 그라데이션 블렌딩
        # 31x31 가우시간 블러로 15픽셀 자연스러운 그라데이션
        smooth_mask = cv2.GaussianBlur(expanded_line_mask.astype(np.float32), (31, 31), 10)
        smooth_mask = smooth_mask / 255.0
        
        # 5. 웨딩링 절대 보호 + 자연스러운 블렌딩
        result = image.copy().astype(np.float32)
        
        # RGB 채널별로 부드러운 블렌딩
        for c in range(3):
            result[:,:,c] = (
                image[:,:,c].astype(np.float32) * (1 - smooth_mask) +
                inpainted[:,:,c].astype(np.float32) * smooth_mask
            )
        
        # 6. 웨딩링 영역 강제 복원 (절대 보호)
        if ring_bbox:
            x, y, w, h = ring_bbox
            result[y:y+h, x:x+w] = image[y:y+h, x:x+w].astype(np.float32)
            print("Ring area forcibly protected")
        
        print("v14.6 inpainting completed - 25번 방식 재현")
        return result.astype(np.uint8)
    
    def detect_metal_type(self, image, ring_bbox=None):
        """금속 타입 자동 감지"""
        if ring_bbox:
            x, y, w, h = ring_bbox
            ring_region = image[y:y+h, x:x+w]
            hsv = cv2.cvtColor(ring_region, cv2.COLOR_RGB2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 평균 색상 계산
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        print(f"Color analysis - Hue: {avg_hue:.1f}, Sat: {avg_sat:.1f}, Val: {avg_val:.1f}")
        
        # 금속 타입 분류 (v13.3 기준)
        if avg_sat < 25:  # 채도가 낮으면 화이트골드/플래티넘
            return 'white_gold'
        elif 5 <= avg_hue <= 25:  # 황색 계열
            if avg_sat > 80:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        elif avg_hue < 5 or avg_hue > 170:  # 적색 계열
            return 'rose_gold'
        else:
            return 'white_gold'  # 기본값
    
    def detect_lighting(self, image):
        """조명 환경 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:, :, 2]
        b_mean = np.mean(b_channel)
        
        print(f"Lighting analysis - B channel mean: {b_mean:.1f}")
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'
    
    def enhance_wedding_ring_v145(self, image, metal_type, lighting, ring_bbox=None):
        """v14.5 강화된 웨딩링 보정 (v13.3 파라미터 기반)"""
        print(f"Enhancing wedding ring - Metal: {metal_type}, Lighting: {lighting}")
        
        # v13.3 파라미터 가져오기
        params = self.params.get(metal_type, {}).get(lighting, 
                                self.params['white_gold']['natural'])
        
        print(f"Using parameters: {params}")
        
        # 1. 노이즈 제거 (고품질)
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
        
        # 3. NumPy 배열로 변환
        enhanced_array = np.array(enhanced)
        
        # 4. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
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
        
        # 6. CLAHE 적용 (고급 명료도)
        lab_clahe = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        lab_clahe[:,:,0] = clahe.apply(lab_clahe[:,:,0])
        enhanced_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # 7. 감마 보정
        gamma = 1.02
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                               for i in np.arange(0, 256)]).astype("uint8")
        enhanced_array = cv2.LUT(enhanced_array, gamma_table)
        
        # 8. 원본과 블렌딩 (자연스러움 보장)
        original_blend = params['original_blend']
        final = cv2.addWeighted(
            enhanced_array, 1 - original_blend,
            image, original_blend, 0
        )
        
        print("Wedding ring enhancement completed")
        return final.astype(np.uint8)
    
    def lanczos_upscale(self, image, scale=2):
        """LANCZOS 2x 업스케일링"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def create_enhanced_thumbnail_v145(self, image, ring_bbox, target_size=(1000, 1300)):
        """v14.5 강화된 썸네일 생성 (웨딩링 중심)"""
        print(f"Creating v14.5 enhanced thumbnail from ring bbox: {ring_bbox}")
        
        if ring_bbox is None:
            print("No ring bbox provided, using center crop")
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            crop_size = min(w, h) // 2
            x = center_x - crop_size // 2
            y = center_y - crop_size // 2
            ring_bbox = (x, y, crop_size, crop_size)
        
        x, y, w, h = ring_bbox
        
        # 1. 웨딩링 중심점 계산
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 2. 적절한 마진 추가 (웨딩링이 너무 크게 나오지 않도록)
        margin_factor = 1.8  # 웨딩링 주변 여백
        crop_w = int(w * margin_factor)
        crop_h = int(h * margin_factor)
        
        # 3. 크롭 영역 계산 (이미지 경계 체크)
        crop_x1 = max(0, center_x - crop_w // 2)
        crop_y1 = max(0, center_y - crop_h // 2)
        crop_x2 = min(image.shape[1], center_x + crop_w // 2)
        crop_y2 = min(image.shape[0], center_y + crop_h // 2)
        
        # 4. 크롭 실행
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        print(f"Cropped region: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
        
        # 5. 1000x1300 비율에 맞게 리사이즈
        target_w, target_h = target_size
        crop_h_actual, crop_w_actual = cropped.shape[:2]
        
        # 비율 계산 (꽉 채우기)
        ratio_w = target_w / crop_w_actual
        ratio_h = target_h / crop_h_actual
        ratio = max(ratio_w, ratio_h)  # 더 큰 비율로 확대
        
        # 리사이즈
        new_w = int(crop_w_actual * ratio)
        new_h = int(crop_h_actual * ratio)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 6. 1000x1300 캔버스에 중앙 배치
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)  # 연한 회색 배경
        
        # 중앙 위치 계산
        start_y = max(0, (target_h - new_h) // 2)
        start_x = max(0, (target_w - new_w) // 2)
        end_y = min(target_h, start_y + new_h)
        end_x = min(target_w, start_x + new_w)
        
        # 캔버스 크기에 맞게 자르기
        resized_h, resized_w = resized.shape[:2]
        crop_start_y = max(0, -start_y)
        crop_start_x = max(0, -start_x)
        crop_end_y = min(resized_h, crop_start_y + (end_y - start_y))
        crop_end_x = min(resized_w, crop_start_x + (end_x - start_x))
        
        # 이미지 배치
        canvas[start_y:end_y, start_x:end_x] = resized[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        
        print(f"Thumbnail created: {target_w}x{target_h}")
        return canvas

def handler(event):
    """RunPod Serverless 메인 핸들러 - v14.5 Ultimate"""
    try:
                print("=== Wedding Ring AI v14.6 Ultimate Starting ===")        
        
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"v14.6 Ultimate 연결 성공: {input_data['prompt']}",
                "version": "v14.6",
                "status": "ready_for_image_processing",
                "capabilities": [
                    "정밀 검은색 선 감지 및 완전 제거",
                    "강화된 웨딩링 보정 (v13.3 28쌍 데이터)",
                    "자연스러운 배경 블렌딩",
                    "완벽한 1000x1300 썸네일",
                    "4가지 금속 × 3가지 조명 자동 감지"
                ]
            }
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64가 필요합니다"}
        
        # Base64 디코딩
        print("Decoding base64 image...")
        image_data = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        print(f"Input image shape: {image_array.shape}")
        
        # 프로세서 초기화
        processor = WeddingRingProcessor()
        
        # 1. v14.5 강화된 검은색 테두리 감지
        border_mask, border_bbox, ring_bbox = processor.detect_black_border_v145(image_array)
        
        if border_mask is None:
            return {"error": "검은색 테두리를 찾을 수 없습니다. 이미지를 확인해주세요."}
        
        # 2. 웨딩링 영역에서 금속 타입 및 조명 감지
        metal_type = processor.detect_metal_type(image_array, ring_bbox)
        lighting = processor.detect_lighting(image_array)
        
        print(f"Detected - Metal: {metal_type}, Lighting: {lighting}")
        
        # 3. v14.5 강화된 웨딩링 보정
        enhanced_image = processor.enhance_wedding_ring_v145(
            image_array, metal_type, lighting, ring_bbox
        )
        
        # 4. v14.6 강화된 inpainting으로 검은색 선 완전 제거 (25번 성공 방식)
        inpainted_image = processor.advanced_inpainting_v146(enhanced_image, border_mask, ring_bbox)
        
        # 5. 2x 업스케일링
        upscaled_image = processor.lanczos_upscale(inpainted_image, scale=2)
        
        # 업스케일링된 ring_bbox 계산
        if ring_bbox:
            ring_bbox_scaled = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
        else:
            ring_bbox_scaled = None
        
        # 6. v14.5 강화된 썸네일 생성
        thumbnail = processor.create_enhanced_thumbnail_v145(upscaled_image, ring_bbox_scaled)
        
        # 7. 결과 인코딩
        print("Encoding results...")
        
        # 메인 이미지
        main_pil = Image.fromarray(upscaled_image)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        print("=== Processing completed successfully ===")
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "processing_info": {
                "version": "v14.6 Ultimate",
                "metal_type": metal_type,
                "lighting": lighting,
                "border_detected": True,
                "border_bbox": border_bbox,
                "ring_bbox": ring_bbox,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                "thumbnail_size": "1000x1300",
                "enhancements_applied": [
                    "다중 threshold 검은색 선 감지",
                    "25번 성공 방식 재현 inpainting",
                    "웨딩링 완전 보호 + 자연스러운 블렌딩",
                    "v13.3 웨딩링 보정 (28쌍 데이터 기반)",
                    "LANCZOS 2x 업스케일링",
                    "웨딩링 중심 1000x1300 썸네일"
                ]
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"처리 중 오류 발생: {str(e)}",
            "version": "v14.6",
            "timestamp": "error_occurred"
        }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
