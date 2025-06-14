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
            'white_overlay': 0.12,
            'sharpness': 1.16,
            'color_temp_a': -4,
            'color_temp_b': -4,
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
        
    def detect_black_lines_and_background_v148(self, image):
        """v14.8 혁신적 방식: 검은색 선 감지 + 배경 색상 분석"""
        print("Starting v14.8 revolutionary black line detection...")
        
        # 1. 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 2. 극도로 정밀한 검은색 픽셀 감지
        _, black_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        
        print(f"Black pixels detected: {np.sum(black_mask > 0)} pixels")
        
        # 3. 컨투어로 사각형 테두리 찾기
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("No contours found")
            return None, None, None
        
        # 가장 큰 컨투어가 테두리
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print(f"Found border rectangle: ({x}, {y}, {w}, {h})")
        
        # 4. 웨딩링 영역 계산 (테두리 내부)
        ring_margin = 10
        ring_bbox = (x + ring_margin, y + ring_margin, 
                    w - ring_margin*2, h - ring_margin*2)
        
        # 5. 배경 색상 분석 (검은색 선이 없는 영역에서)
        background_regions = []
        
        # 좌측 배경
        if x > 50:
            bg_left = image[y:y+h, 0:x-10]
            background_regions.append(bg_left.reshape(-1, 3))
        
        # 우측 배경  
        if x + w < image.shape[1] - 50:
            bg_right = image[y:y+h, x+w+10:image.shape[1]]
            background_regions.append(bg_right.reshape(-1, 3))
        
        # 상단 배경
        if y > 50:
            bg_top = image[0:y-10, x:x+w]
            background_regions.append(bg_top.reshape(-1, 3))
        
        # 하단 배경
        if y + h < image.shape[0] - 50:
            bg_bottom = image[y+h+10:image.shape[0], x:x+w]
            background_regions.append(bg_bottom.reshape(-1, 3))
        
        # 배경 색상 평균 계산
        if background_regions:
            all_bg_pixels = np.vstack(background_regions)
            avg_bg_color = np.mean(all_bg_pixels, axis=0).astype(np.uint8)
            print(f"Average background color: {avg_bg_color}")
        else:
            # 기본 회색
            avg_bg_color = np.array([200, 195, 190], dtype=np.uint8)
            print("Using default background color")
        
        return black_mask, ring_bbox, avg_bg_color
    
    def revolutionary_line_removal_v148(self, image, black_mask, avg_bg_color):
        """v14.8 혁신적 검은색 선 제거: 배경색 직접 덮어쓰기"""
        print("Starting v14.8 revolutionary line removal...")
        
        result = image.copy()
        
        # 1. 검은색 픽셀을 배경 색상으로 직접 교체
        mask_3channel = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2RGB)
        mask_indices = mask_3channel > 0
        
        # 배경 색상으로 직접 덮어쓰기
        result[mask_indices] = avg_bg_color
        
        # 2. 가장자리 부드럽게 블렌딩
        # 검은색 마스크 확장
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(black_mask, kernel, iterations=1)
        
        # 확장된 부분만 부드럽게 블렌딩
        edge_mask = dilated_mask - black_mask
        edge_indices = edge_mask > 0
        
        if np.any(edge_indices):
            # 가장자리는 50% 블렌딩
            result[edge_indices] = (
                image[edge_indices] * 0.5 + 
                avg_bg_color * 0.5
            ).astype(np.uint8)
        
        # 3. 전체적으로 약간의 가우시안 블러 (자연스러움)
        smooth_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32), (5, 5), 1)
        smooth_mask = np.stack([smooth_mask/255.0] * 3, axis=2)
        
        # 최종 블렌딩
        final_result = (
            image.astype(np.float32) * (1 - smooth_mask) +
            result.astype(np.float32) * smooth_mask
        ).astype(np.uint8)
        
        print("Revolutionary line removal completed")
        return final_result
    
    def detect_metal_type(self, image, ring_bbox=None):
        """금속 타입 자동 감지"""
        if ring_bbox:
            x, y, w, h = ring_bbox
            ring_region = image[y:y+h, x:x+w]
            hsv = cv2.cvtColor(ring_region, cv2.COLOR_RGB2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        
        print(f"Metal detection - Hue: {avg_hue:.1f}, Sat: {avg_sat:.1f}")
        
        if avg_sat < 25:
            return 'white_gold'
        elif 5 <= avg_hue <= 25:
            if avg_sat > 80:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        elif avg_hue < 5 or avg_hue > 170:
            return 'rose_gold'
        else:
            return 'white_gold'
    
    def detect_lighting(self, image):
        """조명 환경 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
        print(f"Lighting analysis - B channel: {b_mean:.1f}")
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'
    
    def enhance_wedding_ring_v148(self, image, metal_type, lighting):
        """v14.8 웨딩링 보정 (v13.3 파라미터 완전 유지)"""
        print(f"Enhancing wedding ring - Metal: {metal_type}, Lighting: {lighting}")
        
        params = self.params.get(metal_type, {}).get(lighting, 
                                self.params['white_gold']['natural'])
        
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. PIL ImageEnhance 보정
        pil_image = Image.fromarray(denoised)
        
        # 밝기
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 대비
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 선명도
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        enhanced_array = np.array(enhanced)
        
        # 3. 하얀색 오버레이
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 4. LAB 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 5. CLAHE
        lab_clahe = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        lab_clahe[:,:,0] = clahe.apply(lab_clahe[:,:,0])
        enhanced_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # 6. 감마 보정
        gamma = 1.02
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                               for i in np.arange(0, 256)]).astype("uint8")
        enhanced_array = cv2.LUT(enhanced_array, gamma_table)
        
        # 7. 원본과 블렌딩
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        
        print("Wedding ring enhancement completed")
        return final.astype(np.uint8)
    
    def lanczos_upscale(self, image, scale=2):
        """LANCZOS 2x 업스케일링"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def create_perfect_thumbnail_v148(self, image, ring_bbox, target_size=(1000, 1300)):
        """v14.8 완벽한 썸네일: 웨딩링 중심, 정확한 1000x1300"""
        print(f"Creating v14.8 perfect thumbnail from ring bbox: {ring_bbox}")
        
        if ring_bbox is None:
            print("No ring bbox, using center")
            h, w = image.shape[:2]
            ring_bbox = (w//4, h//4, w//2, h//2)
        
        x, y, w, h = ring_bbox
        
        # 1. 웨딩링 중심점
        center_x = x + w // 2
        center_y = y + h // 2
        
        print(f"Ring center: ({center_x}, {center_y})")
        
        # 2. 1000x1300 비율 계산 (0.769 비율)
        target_ratio = 1000 / 1300  # 0.769
        
        # 웨딩링이 썸네일의 60%를 차지하도록
        ring_size = max(w, h)
        thumbnail_size = int(ring_size / 0.6)
        
        # 3. 1000x1300 비율에 맞는 크롭 영역 계산
        if thumbnail_size * target_ratio > thumbnail_size:
            # 세로가 더 긴 경우
            crop_w = int(thumbnail_size * target_ratio)
            crop_h = thumbnail_size
        else:
            # 가로가 더 긴 경우
            crop_w = thumbnail_size
            crop_h = int(thumbnail_size / target_ratio)
        
        # 4. 크롭 영역 설정 (웨딩링 중심)
        crop_x1 = max(0, center_x - crop_w // 2)
        crop_y1 = max(0, center_y - crop_h // 2)
        crop_x2 = min(image.shape[1], crop_x1 + crop_w)
        crop_y2 = min(image.shape[0], crop_y1 + crop_h)
        
        # 경계 체크 및 조정
        if crop_x2 - crop_x1 < crop_w:
            crop_x1 = max(0, crop_x2 - crop_w)
        if crop_y2 - crop_y1 < crop_h:
            crop_y1 = max(0, crop_y2 - crop_h)
        
        # 5. 크롭 실행
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        print(f"Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # 6. 정확히 1000x1300으로 리사이즈
        resized = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        print("Perfect 1000x1300 thumbnail created")
        return resized

def handler(event):
    """RunPod Serverless 메인 핸들러 - v14.8 Revolutionary"""
    try:
        print("=== Wedding Ring AI v14.8 Revolutionary Starting ===")
        
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"v14.8 Revolutionary 연결 성공: {input_data['prompt']}",
                "version": "v14.8",
                "status": "ready_for_image_processing",
                "capabilities": [
                    "혁신적 검은색 선 제거 (배경색 직접 덮어쓰기)",
                    "배경 색상 자동 분석 및 적용",
                    "완벽한 1000x1300 썸네일 (웨딩링 중심)",
                    "강화된 웨딩링 보정 (v13.3 28쌍 데이터)",
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
        
        # 1. v14.8 혁신적 검은색 선 감지 + 배경 분석
        black_mask, ring_bbox, avg_bg_color = processor.detect_black_lines_and_background_v148(image_array)
        
        if black_mask is None:
            return {"error": "검은색 선을 찾을 수 없습니다. 이미지를 확인해주세요."}
        
        # 2. 웨딩링 영역에서 금속 타입 및 조명 감지
        metal_type = processor.detect_metal_type(image_array, ring_bbox)
        lighting = processor.detect_lighting(image_array)
        
        print(f"Detected - Metal: {metal_type}, Lighting: {lighting}")
        
        # 3. v14.8 웨딩링 보정
        enhanced_image = processor.enhance_wedding_ring_v148(image_array, metal_type, lighting)
        
        # 4. v14.8 혁신적 검은색 선 완전 제거
        line_removed_image = processor.revolutionary_line_removal_v148(
            enhanced_image, black_mask, avg_bg_color
        )
        
        # 5. 2x 업스케일링
        upscaled_image = processor.lanczos_upscale(line_removed_image, scale=2)
        
        # 업스케일링된 ring_bbox 계산
        if ring_bbox:
            ring_bbox_scaled = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
        else:
            ring_bbox_scaled = None
        
        # 6. v14.8 완벽한 1000x1300 썸네일
        thumbnail = processor.create_perfect_thumbnail_v148(upscaled_image, ring_bbox_scaled)
        
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
        
        print("=== v14.8 Processing completed successfully ===")
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "processing_info": {
                "version": "v14.8 Revolutionary",
                "metal_type": metal_type,
                "lighting": lighting,
                "line_detected": True,
                "ring_bbox": ring_bbox,
                "background_color": avg_bg_color.tolist(),
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                "thumbnail_size": "1000x1300",
                "enhancements_applied": [
                    "혁신적 검은색 선 제거 (배경색 직접 덮어쓰기)",
                    "배경 색상 자동 분석 및 매칭",
                    "v13.3 웨딩링 보정 (28쌍 데이터 기반)",
                    "LANCZOS 2x 업스케일링",
                    "완벽한 1000x1300 썸네일 (웨딩링 중심 60%)"
                ]
            }
        }
        
    except Exception as e:
        print(f"Error in v14.8 handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"처리 중 오류 발생: {str(e)}",
            "version": "v14.8",
            "timestamp": "error_occurred"
        }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
