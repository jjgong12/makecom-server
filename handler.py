import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 파라미터 (28쌍 학습 데이터 기반) - 모든 금속 × 조명 조합
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
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.03,
            'sharpness': 1.10, 'color_temp_a': 0, 'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 2,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.10,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.14,
            'sharpness': 1.25, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.03,
            'sharpness': 1.15, 'color_temp_a': 1, 'color_temp_b': 1,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.07,
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 3,
            'original_blend': 0.18
        }
    }
}

class WeddingRingAIv14_4:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        self.black_line_coords = None
        self.background_color = None
        
    def detect_and_remember_black_lines(self, image):
        """정밀한 검은색 선 테두리 감지 및 좌표 기억 (25번 성공 방식)"""
        print("🔍 Step 1: 정밀한 검은색 선 감지 시작")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 25번에서 성공했던 threshold=15 방식
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("❌ 검은색 선을 찾을 수 없습니다")
            return None, None, None
            
        # 가장 큰 사각형 모양 컨투어 찾기
        best_contour = None
        best_bbox = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 너무 작은 영역 제외
                continue
                
            # 사각형 근사
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4개 꼭짓점인 사각형 확인
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # 정사각형에 가까운 비율 확인 (0.5 < ratio < 2.0)
                if 0.5 < aspect_ratio < 2.0:
                    best_contour = contour
                    best_bbox = (x, y, w, h)
                    break
        
        if best_contour is None:
            print("❌ 적절한 사각형 테두리를 찾을 수 없습니다")
            return None, None, None
            
        # 좌표 기억
        self.black_line_coords = best_bbox
        
        # 검은색 선 마스크 생성
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        
        # 웨딩링 영역 마스크 (선 내부)
        x, y, w, h = best_bbox
        ring_mask = np.zeros_like(gray)
        ring_mask[y+2:y+h-2, x+2:x+w-2] = 255  # 안쪽 영역만
        
        print(f"✅ 검은색 선 감지 완료: {best_bbox}")
        return mask, ring_mask, best_bbox
    
    def analyze_simple_background(self, image, exclude_mask):
        """v14.4: 간단한 배경 색상 분석 (sklearn 없이)"""
        print("🎨 Step 2: 간단한 배경 분석 시작")
        
        # 배경 영역만 추출
        background_mask = 255 - exclude_mask
        background_pixels = image[background_mask > 0]
        
        if len(background_pixels) == 0:
            print("⚠️ 배경 영역이 너무 작습니다. 기본값 사용")
            self.background_color = [240, 240, 240]
            return [240, 240, 240]
        
        # 간단한 평균 색상 계산
        avg_color = np.mean(background_pixels, axis=0).astype(int)
        
        # 배경 균일성 체크
        std_color = np.std(background_pixels, axis=0)
        is_uniform = np.all(std_color < 20)  # 표준편차가 20 이하면 균일
        
        self.background_color = avg_color.tolist()
        
        print(f"✅ 배경 분석 완료: {avg_color}, 균일함: {is_uniform}")
        return avg_color
    
    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 감지 (기존 유지)"""
        if mask is not None:
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
        
        # 금속 타입 분류 (25번 대화 기준)
        if avg_hue < 15 or avg_hue > 165:
            if avg_sat > 50:
                return 'rose_gold'
            else:
                return 'white_gold'
        elif 15 <= avg_hue <= 35:
            if avg_sat > 80:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        else:
            return 'white_gold'
    
    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지 (기존 유지)"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'
    
    def enhance_wedding_ring_v13_3(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (28쌍 학습 데이터 기반) - 기존 그대로 유지"""
        params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
        
        print(f"🔧 v13.3 보정 적용: {metal_type} - {lighting}")
        
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
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        
        return final.astype(np.uint8)
    
    def apply_noise_reduction(self, image):
        """노이즈 제거 (기존 유지)"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def apply_clahe(self, image, clip_limit=1.3):
        """CLAHE 적용 (기존 유지)"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def apply_gamma_correction(self, image, gamma=1.02):
        """감마 보정 (기존 유지)"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def improved_background_removal_v14_4(self, image, line_mask, ring_bbox):
        """v14.4: 25번 성공 방식 기반 개선된 배경 제거"""
        print("🎨 Step 3: 개선된 배경 제거 시작 (25번 성공 방식)")
        
        x, y, w, h = ring_bbox
        
        # 웨딩링 완전 보호 마스크 (25번 성공 방식)
        ring_protection_mask = np.zeros_like(line_mask)
        ring_protection_mask[y+3:y+h-3, x+3:x+w-3] = 255
        
        # 실제 제거할 마스크 (웨딩링 영역 완전 제외)
        removal_mask = cv2.bitwise_and(line_mask, 255 - ring_protection_mask)
        
        # 고급 inpainting으로 검은색 선 제거
        inpainted = cv2.inpaint(image, removal_mask, 5, cv2.INPAINT_NS)
        
        # 웨딩링 영역 원본 완전 복원
        result = inpainted.copy()
        result[ring_protection_mask > 0] = image[ring_protection_mask > 0]
        
        # 25번 성공했던 자연스러운 블렌딩 (31×31 가우시안 블러)
        blend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        ring_protection_mask_float = ring_protection_mask.astype(np.float32) / 255.0
        
        # 31×31 가우시안 블러로 부드러운 블렌딩 (25번 성공 방식)
        ring_protection_mask_float = cv2.GaussianBlur(ring_protection_mask_float, (31, 31), 10)
        
        # RGB 채널별로 자연스러운 블렌딩
        for c in range(3):
            result[:,:,c] = (
                image[:,:,c].astype(np.float32) * ring_protection_mask_float +
                inpainted[:,:,c].astype(np.float32) * (1 - ring_protection_mask_float)
            )
        
        print("✅ 25번 방식 배경 제거 완료")
        return result.astype(np.uint8)
    
    def create_background_seamless_thumbnail_v14_4(self, image, ring_bbox, target_size=(1000, 1300)):
        """v14.4: 배경 색상 기반 자연스러운 썸네일 생성"""
        print("🖼️ Step 4: 배경 연속성 썸네일 생성")
        
        x, y, w, h = ring_bbox
        
        # 웨딩링 중심 계산
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 1000×1300 크롭 영역 계산
        target_w, target_h = target_size
        
        # 웨딩링을 중심으로 한 크롭 영역
        crop_x1 = max(0, center_x - target_w // 2)
        crop_y1 = max(0, center_y - target_h // 2)
        crop_x2 = min(image.shape[1], crop_x1 + target_w)
        crop_y2 = min(image.shape[0], crop_y1 + target_h)
        
        # 실제 크롭 크기 계산
        actual_w = crop_x2 - crop_x1
        actual_h = crop_y2 - crop_y1
        
        # 배경 색상 기반 캔버스 생성
        if self.background_color is not None:
            bg_color = self.background_color
        else:
            bg_color = [240, 240, 240]  # 기본 밝은 회색
        
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        
        # 부드러운 그라디언트 추가 (자연스러운 배경)
        for i in range(target_h):
            factor = i / target_h
            gradient_color = [int(c * (0.98 + 0.04 * factor)) for c in bg_color]
            canvas[i, :] = gradient_color
        
        # 크롭된 이미지를 캔버스 중앙에 배치
        paste_x = (target_w - actual_w) // 2
        paste_y = (target_h - actual_h) // 2
        
        # 크롭된 영역 배치
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        canvas[paste_y:paste_y+actual_h, paste_x:paste_x+actual_w] = cropped
        
        # 가장자리 자연스럽게 블렌딩
        if actual_w < target_w or actual_h < target_h:
            # 부드러운 블렌딩 마스크
            blend_margin = 30
            blend_mask = np.zeros((target_h, target_w), dtype=np.float32)
            
            # 중앙은 1.0, 가장자리는 0.0
            if actual_h > 2*blend_margin and actual_w > 2*blend_margin:
                blend_mask[
                    paste_y + blend_margin:paste_y + actual_h - blend_margin,
                    paste_x + blend_margin:paste_x + actual_w - blend_margin
                ] = 1.0
                
                # 가우시안 블러로 부드러운 전환
                blend_mask = cv2.GaussianBlur(blend_mask, (61, 61), 20)
                
                # 3채널로 확장하여 블렌딩
                blend_mask_3d = np.stack([blend_mask] * 3, axis=2)
                
                canvas = (
                    canvas.astype(np.float32) * (1 - blend_mask_3d) +
                    canvas.astype(np.float32) * blend_mask_3d
                ).astype(np.uint8)
        
        print(f"✅ 썸네일 생성 완료: {target_size}")
        return canvas
    
    def basic_upscale(self, image, scale=2):
        """기본 업스케일링 (기존 유지)"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러 - v14.4 Stable"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.4 Stable 연결 성공: {input_data['prompt']}",
                "version": "v14.4",
                "features": [
                    "v13.3 파라미터 (28쌍 학습 데이터)",
                    "25번 성공 방식 기반",
                    "안정된 배경 제거",
                    "배경 연속성 썸네일",
                    "sklearn 없는 안정적 처리"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            print("🚀 웨딩링 AI v14.4 Stable 처리 시작")
            
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 프로세서 초기화
            processor = WeddingRingAIv14_4()
            
            # 1. 검은색 선 감지 및 좌표 기억 (25번 성공 방식)
            line_mask, ring_mask, ring_bbox = processor.detect_and_remember_black_lines(image_array)
            
            if line_mask is None:
                return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
            
            # 2. 간단한 배경 색상 분석 (sklearn 없이)
            background_color = processor.analyze_simple_background(image_array, line_mask)
            
            # 3. 웨딩링 영역에서 금속 타입 및 조명 감지
            metal_type = processor.detect_metal_type(image_array, ring_mask)
            lighting = processor.detect_lighting(image_array)
            
            print(f"📊 감지 결과: {metal_type} / {lighting}")
            
            # 4. 웨딩링 영역 추출 및 보정
            x, y, w, h = ring_bbox
            ring_region = image_array[y:y+h, x:x+w].copy()
            
            # 노이즈 제거
            ring_region = processor.apply_noise_reduction(ring_region)
            
            # v13.3 웨딩링 보정
            enhanced_ring = processor.enhance_wedding_ring_v13_3(ring_region, metal_type, lighting)
            
            # CLAHE 적용
            enhanced_ring = processor.apply_clahe(enhanced_ring)
            
            # 감마 보정
            enhanced_ring = processor.apply_gamma_correction(enhanced_ring)
            
            # 보정된 웨딩링을 원본에 다시 배치
            result_image = image_array.copy()
            result_image[y:y+h, x:x+w] = enhanced_ring
            
            # 5. v14.4 개선된 배경 제거 (25번 성공 방식)
            main_result = processor.improved_background_removal_v14_4(result_image, line_mask, ring_bbox)
            
            # 6. 2x 업스케일링
            upscaled = processor.basic_upscale(main_result, scale=2)
            
            # 7. v14.4 배경 연속성 썸네일 생성
            # 원본 bbox를 업스케일링 비율에 맞게 조정
            scaled_bbox = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
            thumbnail = processor.create_background_seamless_thumbnail_v14_4(upscaled, scaled_bbox)
            
            # 8. 결과 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            processing_info = {
                "version": "v14.4 Stable",
                "metal_type": metal_type,
                "lighting": lighting,
                "background_color": background_color.tolist() if hasattr(background_color, 'tolist') else background_color,
                "ring_bbox": ring_bbox,
                "scale_factor": 2,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                "thumbnail_size": "1000x1300",
                "features_used": [
                    "v13.3 파라미터",
                    "25번 성공 방식",
                    "안정적 배경 제거",
                    "배경 연속성 썸네일"
                ]
            }
            
            print("✅ v14.4 Stable 처리 완료")
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": processing_info
            }
        
        return {"error": "image_base64 파라미터가 필요합니다."}
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {str(e)}")
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
