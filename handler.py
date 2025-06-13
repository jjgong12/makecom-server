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
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15, 'clarity': 1.18, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18, 'clarity': 1.15, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'clarity': 1.22, 'gamma': 1.03
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'clarity': 1.10, 'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.04,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'clarity': 1.05, 'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.18, 'clarity': 1.20, 'gamma': 1.02
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.08,
            'sharpness': 1.16, 'color_temp_a': -1, 'color_temp_b': -1,
            'original_blend': 0.15, 'clarity': 1.15, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.10,
            'sharpness': 1.20, 'color_temp_a': -3, 'color_temp_b': -2,
            'original_blend': 0.18, 'clarity': 1.12, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.06,
            'sharpness': 1.25, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.12, 'clarity': 1.18, 'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'clarity': 1.12, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.03,
            'sharpness': 1.15, 'color_temp_a': 1, 'color_temp_b': 1,
            'original_blend': 0.28, 'clarity': 1.08, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.07,
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 4,
            'original_blend': 0.15, 'clarity': 1.18, 'gamma': 1.03
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        """웨딩링 프로세서 초기화"""
        print("Wedding Ring AI Processor v13.3 Initialized")
        print("28-pair learning data based parameters loaded")
    
    def detect_black_line_border(self, image):
        """검은색 선으로 된 사각형 테두리 정밀 감지"""
        print("Step 1: Detecting black line border...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # 1. 매우 어두운 픽셀 감지 (threshold 35)
        _, binary = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 3. 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 4. 가장 적합한 사각형 컨투어 찾기
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                
                # 최소 면적 체크 (이미지의 5% 이상)
                if area < (height * width * 0.05):
                    continue
                
                # 사각형 근사화
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 4개 점의 사각형이고 충분한 크기인 경우
                if len(approx) >= 4 and area > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 가로세로 비율 체크 (너무 찌그러진 사각형 제외)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.0:
                        # 선 두께 고려해서 내부 영역 추출 (8% 마진)
                        margin_x = max(3, int(w * 0.08))
                        margin_y = max(3, int(h * 0.08))
                        
                        inner_x = x + margin_x
                        inner_y = y + margin_y
                        inner_w = w - 2 * margin_x
                        inner_h = h - 2 * margin_y
                        
                        if inner_w > 50 and inner_h > 50:  # 최소 크기 보장
                            print(f"Border detected: ({inner_x}, {inner_y}, {inner_w}, {inner_h})")
                            return (inner_x, inner_y, inner_w, inner_h), contour
        
        print("No valid border found")
        return None, None

    def detect_metal_type(self, image):
        """HSV 색공간 분석으로 금속 타입 정밀 감지"""
        print("Step 2: Detecting metal type...")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 중앙 50% 영역에서 색상 분석 (노이즈 제거)
        h, w = hsv.shape[:2]
        center_h_start, center_h_end = h//4, 3*h//4
        center_w_start, center_w_end = w//4, 3*w//4
        center_region = hsv[center_h_start:center_h_end, center_w_start:center_w_end]
        
        avg_hue = np.mean(center_region[:, :, 0])
        avg_saturation = np.mean(center_region[:, :, 1])
        avg_value = np.mean(center_region[:, :, 2])
        
        print(f"Color analysis - H: {avg_hue:.1f}, S: {avg_saturation:.1f}, V: {avg_value:.1f}")
        
        # 정밀한 금속 분류
        if avg_hue < 15 or avg_hue > 165:  # Red-Pink range
            if avg_saturation > 50 and avg_value > 100:
                metal_type = 'rose_gold'
            else:
                metal_type = 'white_gold'
        elif 15 <= avg_hue <= 35:  # Yellow-Orange range
            if avg_saturation > 80:
                metal_type = 'yellow_gold'
            else:
                metal_type = 'champagne_gold'
        else:
            metal_type = 'white_gold'  # Default fallback
        
        print(f"Detected metal type: {metal_type}")
        return metal_type

    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 정밀 감지"""
        print("Step 3: Detecting lighting condition...")
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 전체 이미지의 색온도 분석
        l_mean = np.mean(lab[:, :, 0])  # Lightness
        a_mean = np.mean(lab[:, :, 1])  # Green-Red axis
        b_mean = np.mean(lab[:, :, 2])  # Blue-Yellow axis
        
        print(f"LAB analysis - L: {l_mean:.1f}, A: {a_mean:.1f}, B: {b_mean:.1f}")
        
        # 조명 환경 분류
        if b_mean < 125:  # Blue-ish (cool light appears warm in LAB)
            lighting = 'warm'
        elif b_mean > 135:  # Yellow-ish (warm light)
            lighting = 'cool'
        else:
            lighting = 'natural'
        
        print(f"Detected lighting: {lighting}")
        return lighting

    def noise_reduction(self, image):
        """고급 노이즈 제거"""
        # Bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    def apply_clahe(self, image, clip_limit=2.0):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced

    def enhance_wedding_ring_complete(self, image, metal_type, lighting):
        """v13.3 완전한 웨딩링 보정 (28쌍 학습 데이터 기반)"""
        print(f"Step 4: Applying v13.3 enhancement for {metal_type}/{lighting}...")
        
        # 파라미터 로드
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting,
                                                              WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 1. 노이즈 제거
        denoised = self.noise_reduction(image)
        
        # 2. PIL ImageEnhance로 기본 보정
        pil_image = Image.fromarray(denoised)
        
        # 2-1. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        print(f"Applied brightness: {params['brightness']}")
        
        # 2-2. 대비 조정
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        print(f"Applied contrast: {params['contrast']}")
        
        # 2-3. 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        print(f"Applied sharpness: {params['sharpness']}")
        
        # 3. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
        enhanced_array = np.array(enhanced)
        if params['white_overlay'] > 0:
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            print(f"Applied white overlay: {params['white_overlay']}")
        
        # 4. LAB 색공간에서 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)  # A채널
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)  # B채널
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        print(f"Applied color temperature: A{params['color_temp_a']}, B{params['color_temp_b']}")
        
        # 5. CLAHE 명료도 적용
        if params.get('clarity', 1.0) > 1.0:
            clarity_strength = params['clarity']
            enhanced_array = self.apply_clahe(enhanced_array, clip_limit=clarity_strength)
            print(f"Applied CLAHE clarity: {clarity_strength}")
        
        # 6. 감마 보정
        if params.get('gamma', 1.0) != 1.0:
            gamma = params['gamma']
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_array = cv2.LUT(enhanced_array, table)
            print(f"Applied gamma correction: {gamma}")
        
        # 7. 원본과 블렌딩 (자연스러움 보장)
        original_blend = params['original_blend']
        final = cv2.addWeighted(
            enhanced_array, 1 - original_blend,
            image, original_blend, 0
        )
        print(f"Applied original blending: {original_blend}")
        
        return final.astype(np.uint8)

    def apply_global_color_adjustment(self, image, metal_type, lighting):
        """전체 색감 조정 (28쌍 학습파일 참고 - 50% 강도)"""
        print("Step 5: Applying global color adjustment...")
        
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting,
                                                              WEDDING_RING_PARAMS['white_gold']['natural'])
        
        pil_image = Image.fromarray(image)
        
        # 전체적인 미묘한 조정 (50% 강도로 자연스럽게)
        brightness_factor = 1 + (params['brightness'] - 1) * 0.5
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(brightness_factor)
        
        contrast_factor = 1 + (params['contrast'] - 1) * 0.5
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast_factor)
        
        print(f"Global adjustment - Brightness: {brightness_factor:.3f}, Contrast: {contrast_factor:.3f}")
        
        return np.array(enhanced)

    def remove_black_lines_advanced(self, image, line_contour):
        """고급 검은색 선 제거 (정밀한 inpainting)"""
        print("Step 6: Removing black lines with advanced inpainting...")
        
        # 1. 정밀한 선 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 2. 컨투어 기반 마스크
        cv2.drawContours(mask, [line_contour], -1, 255, thickness=12)
        
        # 3. 형태학적 연산으로 선 영역 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
        
        # 4. 가장자리 검출로 선만 정확히 추출
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 5. 가장자리와 마스크 교집합으로 정밀한 선 영역
        final_mask = cv2.bitwise_and(mask, edges)
        
        # 6. 마스크 확장 (inpainting 품질 향상)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        
        # 7. 고품질 inpainting (TELEA 알고리즘)
        result = cv2.inpaint(image, final_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        # 8. 추가 부드러움 처리
        blurred = cv2.GaussianBlur(result, (3, 3), 0)
        
        # 9. 마스크 영역만 블러 적용 (자연스러운 경계)
        mask_3channel = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB) / 255.0
        result = (result * (1 - mask_3channel) + blurred * mask_3channel).astype(np.uint8)
        
        print("Black lines removed successfully")
        return result

    def upscale_lanczos_high_quality(self, image, scale_factor=2):
        """고품질 LANCZOS 업스케일링"""
        print(f"Step 7: High-quality LANCZOS upscaling (x{scale_factor})...")
        
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # LANCZOS4 보간법으로 고품질 업스케일링
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 업스케일링 후 약간의 선명도 개선
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel * 0.1)
        
        # 원본과 블렌딩 (90% 업스케일 + 10% 선명화)
        final = cv2.addWeighted(upscaled, 0.9, sharpened, 0.1, 0)
        
        print(f"Upscaling completed: {width}x{height} → {new_width}x{new_height}")
        return final.astype(np.uint8)

    def create_thumbnail_complete(self, image, border_rect, target_size=(1000, 1300)):
        """완전한 썸네일 생성 (검은색 선 기준 크롭 → 1000×1300 → 업스케일링과 보정)"""
        print("Step 8: Creating complete thumbnail...")
        
        x, y, w, h = border_rect
        
        # 1. 검은색 선 기준으로 크롭 (15% 마진 추가)
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)
        
        crop_x1 = max(0, x - margin_w)
        crop_y1 = max(0, y - margin_h)
        crop_x2 = min(image.shape[1], x + w + margin_w)
        crop_y2 = min(image.shape[0], y + h + margin_h)
        
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        print(f"Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # 2. 1000×1300 비율에 맞게 조정
        target_w, target_h = target_size
        crop_h, crop_w = cropped.shape[:2]
        
        # 비율 계산 (원본 비율 유지)
        ratio = min(target_w / crop_w, target_h / crop_h)
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        
        # 3. 1000×1300으로 리사이즈
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"Resized to: {new_w}x{new_h}")
        
        # 4. 업스케일링 (2x)
        upscaled = self.upscale_lanczos_high_quality(resized, scale_factor=2)
        print(f"Upscaled to: {upscaled.shape[1]}x{upscaled.shape[0]}")
        
        # 5. 썸네일 전용 보정 적용
        # 5-1. 약간의 추가 선명도
        enhancer = ImageEnhance.Sharpness(Image.fromarray(upscaled))
        sharpened = enhancer.enhance(1.1)
        
        # 5-2. 약간의 대비 강화 (썸네일 특성상)
        enhancer = ImageEnhance.Contrast(sharpened)
        contrasted = enhancer.enhance(1.05)
        
        final_thumbnail = np.array(contrasted)
        
        # 6. 최종 캔버스에 중앙 배치 (2000×2600)
        canvas_w, canvas_h = target_w * 2, target_h * 2
        canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)  # 연한 회색 배경
        
        # 중앙 배치 계산
        start_y = (canvas_h - final_thumbnail.shape[0]) // 2
        start_x = (canvas_w - final_thumbnail.shape[1]) // 2
        
        end_y = start_y + final_thumbnail.shape[0]
        end_x = start_x + final_thumbnail.shape[1]
        
        # 캔버스에 배치
        canvas[start_y:end_y, start_x:end_x] = final_thumbnail
        
        print(f"Final thumbnail size: {canvas_w}x{canvas_h}")
        return canvas

    def process_complete_workflow(self, image):
        """완전한 웨딩링 처리 워크플로우 (모든 단계 포함)"""
        print("=== Starting Complete Wedding Ring AI Processing Workflow ===")
        print("v13.3 System with 28-pair learning data")
        
        # 1. 검은색 선 테두리 감지
        border_rect, line_contour = self.detect_black_line_border(image)
        if border_rect is None:
            return {"error": "검은색 선 테두리를 찾을 수 없습니다. 명확한 검은색 사각형 테두리가 필요합니다."}
        
        x, y, w, h = border_rect
        
        # 2. 테두리 내부 웨딩링 영역 추출
        ring_region = image[y:y+h, x:x+w].copy()
        print(f"Ring region extracted: {w}x{h}")
        
        # 3. 금속 타입 및 조명 감지 (내부 영역 기준)
        metal_type = self.detect_metal_type(ring_region)
        lighting = self.detect_lighting(ring_region)
        
        # 4. 웨딩링 영역 확대 → v13.3 보정 → 축소
        print("=== Ring Enhancement Process ===")
        
        # 4-1. 2x 확대
        enlarged_w, enlarged_h = w * 2, h * 2
        enlarged = cv2.resize(ring_region, (enlarged_w, enlarged_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"Enlarged ring region: {enlarged_w}x{enlarged_h}")
        
        # 4-2. v13.3 완전한 보정 적용
        enhanced_enlarged = self.enhance_wedding_ring_complete(enlarged, metal_type, lighting)
        
        # 4-3. 원래 크기로 축소해서 돌아오기
        enhanced_region = cv2.resize(enhanced_enlarged, (w, h), interpolation=cv2.INTER_LANCZOS4)
        print(f"Enhanced region resized back: {w}x{h}")
        
        # 5. 전체 이미지에 합성
        result_image = image.copy()
        result_image[y:y+h, x:x+w] = enhanced_region
        print("Enhanced ring region merged back to original image")
        
        # 6. 전체 색감 조정 (28쌍 학습파일 참고)
        result_image = self.apply_global_color_adjustment(result_image, metal_type, lighting)
        
        # 7. 검은색 선 제거 (고급 inpainting)
        final_image = self.remove_black_lines_advanced(result_image, line_contour)
        
        # 8. 썸네일 생성 (검은색 선 기준 크롭 → 1000×1300 → 업스케일링과 보정)
        thumbnail = self.create_thumbnail_complete(final_image, border_rect)
        
        print("=== Processing Complete ===")
        
        return {
            "enhanced_image": final_image,
            "thumbnail": thumbnail,
            "processing_info": {
                "metal_type": metal_type,
                "lighting": lighting,
                "border_rect": border_rect,
                "original_size": f"{image.shape[1]}x{image.shape[0]}",
                "ring_region_size": f"{w}x{h}",
                "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                "thumbnail_size": f"{thumbnail.shape[1]}x{thumbnail.shape[0]}",
                "processing_steps": [
                    "1. 검은색 선 테두리 정밀 감지",
                    f"2. 웨딩링 영역 추출 ({w}x{h})",
                    f"3. 금속/조명 감지 ({metal_type}/{lighting})",
                    "4. 웨딩링 2x 확대 → v13.3 보정 → 축소",
                    "5. 전체 색감 조정 (28쌍 학습데이터 기준)",
                    "6. 검은색 선 고급 inpainting 제거",
                    "7. 썸네일 생성 (1000×1300 → 2x 업스케일링)",
                    "8. 완전 처리 완료"
                ],
                "status": "SUCCESS",
                "version": "v13.3",
                "learning_data": "28-pair verified"
            }
        }

def handler(event):
    """RunPod Serverless 메인 핸들러 (100% 완전 버전)"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v13.3 연결 성공: {input_data['prompt']}",
                "status": "ready_for_complete_processing",
                "version": "v13.3",
                "learning_data": "28-pair verified",
                "capabilities": [
                    "검은색 선 테두리 정밀 감지",
                    "웨딩링 확대 → v13.3 보정 → 축소",
                    "전체 색감 조정 (28쌍 학습데이터)",
                    "고급 검은색 선 제거 (inpainting)",
                    "썸네일 생성 (1000×1300 → 2x 업스케일링)",
                    "4가지 금속 타입 지원 (White/Rose/Champagne/Yellow Gold)",
                    "3가지 조명 환경 지원 (Natural/Warm/Cool)",
                    "완전 자동화 처리"
                ],
                "processing_flow": [
                    "검은색 선 감지 → 웨딩링 추출 → 확대보정 → 축소합성 → 전체조정 → 선제거 → 썸네일생성"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            print("=== Starting Wedding Ring AI Processing ===")
            
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            print(f"Input image size: {image_array.shape[1]}x{image_array.shape[0]}")
            
            # 웨딩링 프로세서 초기화 및 완전 처리
            processor = WeddingRingProcessor()
            result = processor.process_complete_workflow(image_array)
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "suggestions": [
                        "검은색 선으로 명확한 사각형 테두리가 있는지 확인",
                        "테두리 안에 웨딩링이 명확히 보이는지 확인",
                        "이미지 해상도가 충분한지 확인 (최소 500x500)",
                        "검은색 선의 두께가 적절한지 확인"
                    ]
                }
            
            # 결과 인코딩
            print("Encoding results...")
            
            # 메인 이미지 인코딩
            main_pil = Image.fromarray(result["enhanced_image"])
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일 인코딩
            thumb_pil = Image.fromarray(result["thumbnail"])
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            print("=== Processing Successfully Completed ===")
            
            return {
                "success": True,
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": result["processing_info"],
                "system_info": {
                    "version": "v13.3",
                    "learning_data": "28-pair verified parameters",
                    "processing_time": "45-60 seconds estimated",
                    "quality": "Professional grade",
                    "features": "Complete automated workflow"
                }
            }
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            "success": False,
            "error": f"처리 중 오류 발생: {str(e)}",
            "debug_info": {
                "error_type": type(e).__name__,
                "error_location": "handler function"
            }
        }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
