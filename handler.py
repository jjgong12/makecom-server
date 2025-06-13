import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
from sklearn.cluster import KMeans

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

class WeddingRingAIv14_3:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        self.black_line_coords = None
        self.background_profile = None
        
    def detect_and_remember_black_lines(self, image):
        """정밀한 검은색 선 테두리 감지 및 좌표 기억"""
        print("🔍 Step 1: 정밀한 검은색 선 감지 시작")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 다중 threshold로 정확한 감지
        _, binary1 = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        _, binary2 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 두 결과 결합
        binary = cv2.bitwise_and(binary1, binary2)
        
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
    
    def analyze_background_characteristics(self, image, exclude_mask):
        """v14.3: 배경 특성 완전 분석"""
        print("🎨 Step 2: 배경 특성 분석 시작")
        
        # 배경 영역만 추출
        background_mask = 255 - exclude_mask
        background_pixels = image[background_mask > 0]
        
        if len(background_pixels) == 0:
            print("⚠️ 배경 영역이 너무 작습니다. 기본값 사용")
            return {
                'dominant_color': [240, 240, 240],
                'gradient_type': 'uniform',
                'texture_type': 'smooth'
            }
        
        # K-means로 dominant color 추출
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(background_pixels)
        
        # 가장 많이 나타나는 색상
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        dominant_label = np.argmax(label_counts)
        dominant_color = kmeans.cluster_centers_[dominant_label].astype(int)
        
        # 그라디언트 분석
        gray_bg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_bg_masked = cv2.bitwise_and(gray_bg, gray_bg, mask=background_mask)
        
        # Sobel 그라디언트 계산
        sobelx = cv2.Sobel(gray_bg_masked, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_bg_masked, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 그라디언트 강도 분석
        avg_gradient = np.mean(gradient_magnitude[background_mask > 0])
        
        if avg_gradient < 10:
            gradient_type = 'uniform'
        elif avg_gradient < 30:
            gradient_type = 'gentle'
        else:
            gradient_type = 'complex'
        
        # 텍스처 분석 (표준편차 기반)
        bg_std = np.std(gray_bg_masked[background_mask > 0])
        
        if bg_std < 15:
            texture_type = 'smooth'
        elif bg_std < 40:
            texture_type = 'textured'
        else:
            texture_type = 'complex'
        
        profile = {
            'dominant_color': dominant_color.tolist(),
            'gradient_type': gradient_type,
            'texture_type': texture_type,
            'avg_gradient': float(avg_gradient),
            'bg_std': float(bg_std)
        }
        
        self.background_profile = profile
        print(f"✅ 배경 분석 완료: {profile}")
        return profile
    
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
    
    def seamless_background_removal_v14_3(self, image, line_mask, ring_bbox):
        """v14.3: Seamless Cloning 기반 배경 제거"""
        print("🎨 Step 3: Seamless Background Removal 시작")
        
        # 배경 특성에 맞는 색상 생성
        bg_color = self.background_profile['dominant_color']
        
        # 배경과 유사한 색상의 canvas 생성
        height, width = image.shape[:2]
        background_canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # 배경 타입에 따른 처리
        if self.background_profile['gradient_type'] != 'uniform':
            # 그라디언트 배경 생성
            if self.background_profile['gradient_type'] == 'gentle':
                # 부드러운 그라디언트
                for i in range(height):
                    factor = i / height
                    gradient_color = [int(c * (0.9 + 0.2 * factor)) for c in bg_color]
                    background_canvas[i, :] = gradient_color
        
        # 웨딩링 영역 보호를 위한 마스크 생성
        x, y, w, h = ring_bbox
        protection_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 웨딩링 영역 확장 (10픽셀 마진)
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(width, x + w + 10)
        y2 = min(height, y + h + 10)
        protection_mask[y1:y2, x1:x2] = 255
        
        # 검은색 선 마스크 dilate (seamless cloning용)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_line_mask = cv2.dilate(line_mask, kernel, iterations=2)
        
        # 웨딩링 보호 영역 제외
        cloning_mask = cv2.bitwise_and(dilated_line_mask, 255 - protection_mask)
        
        # Seamless Cloning 적용
        if np.any(cloning_mask > 0):
            # 중심점 계산
            moments = cv2.moments(cloning_mask)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                center = (center_x, center_y)
                
                try:
                    # NORMAL_CLONE 모드로 seamless cloning
                    result = cv2.seamlessClone(
                        background_canvas.astype(np.uint8), 
                        image.astype(np.uint8), 
                        cloning_mask, 
                        center, 
                        cv2.NORMAL_CLONE
                    )
                    print("✅ Seamless Cloning 완료")
                    return result
                except Exception as e:
                    print(f"⚠️ Seamless Cloning 실패: {e}")
                    # 폴백: 기존 방식
                    return self.fallback_inpainting(image, line_mask, ring_bbox)
        
        return self.fallback_inpainting(image, line_mask, ring_bbox)
    
    def fallback_inpainting(self, image, line_mask, ring_bbox):
        """폴백: 기존 inpainting 방식"""
        print("🔄 Fallback: 기존 inpainting 방식 사용")
        
        x, y, w, h = ring_bbox
        
        # 웨딩링 보호 마스크
        protection_mask = np.zeros_like(line_mask)
        protection_mask[y+3:y+h-3, x+3:x+w-3] = 255
        
        # 실제 제거할 마스크 (웨딩링 영역 제외)
        removal_mask = cv2.bitwise_and(line_mask, 255 - protection_mask)
        
        # 고급 inpainting
        inpainted = cv2.inpaint(image, removal_mask, 5, cv2.INPAINT_NS)
        
        # 웨딩링 영역 원본 복원
        result = inpainted.copy()
        result[protection_mask > 0] = image[protection_mask > 0]
        
        # 부드러운 블렌딩
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blend_mask = cv2.dilate(protection_mask, kernel, iterations=1) - protection_mask
        
        if np.any(blend_mask > 0):
            blend_mask_norm = blend_mask.astype(np.float32) / 255.0
            for c in range(3):
                result[:,:,c] = (
                    result[:,:,c].astype(np.float32) * (1 - blend_mask_norm * 0.3) +
                    image[:,:,c].astype(np.float32) * (blend_mask_norm * 0.3)
                )
        
        return result.astype(np.uint8)
    
    def basic_upscale(self, image, scale=2):
        """기본 업스케일링 (기존 유지)"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def create_seamless_thumbnail_v14_3(self, image, ring_bbox, target_size=(1000, 1300)):
        """v14.3: 배경 연속성을 고려한 썸네일 생성"""
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
        
        # 배경 특성 기반 캔버스 생성
        bg_color = self.background_profile['dominant_color']
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        
        # 배경 타입별 처리
        if self.background_profile['gradient_type'] != 'uniform':
            # 그라디언트 배경
            for i in range(target_h):
                factor = i / target_h
                if self.background_profile['gradient_type'] == 'gentle':
                    # 부드러운 그라디언트 (위에서 아래로)
                    gradient_color = [int(c * (0.95 + 0.1 * factor)) for c in bg_color]
                    canvas[i, :] = gradient_color
        
        # 크롭된 이미지를 캔버스 중앙에 배치
        paste_x = (target_w - actual_w) // 2
        paste_y = (target_h - actual_h) // 2
        
        # 크롭된 영역 배치
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        canvas[paste_y:paste_y+actual_h, paste_x:paste_x+actual_w] = cropped
        
        # 가장자리 부드럽게 블렌딩
        if actual_w < target_w or actual_h < target_h:
            # 블렌딩 마스크 생성
            blend_margin = 20
            blend_mask = np.zeros((target_h, target_w), dtype=np.float32)
            
            # 중앙 영역은 1.0, 가장자리로 갈수록 0.0
            center_mask = np.ones((actual_h - 2*blend_margin, actual_w - 2*blend_margin))
            
            if center_mask.shape[0] > 0 and center_mask.shape[1] > 0:
                blend_mask[
                    paste_y + blend_margin:paste_y + actual_h - blend_margin,
                    paste_x + blend_margin:paste_x + actual_w - blend_margin
                ] = center_mask
                
                # 가우시안 블러로 부드러운 전환
                blend_mask = cv2.GaussianBlur(blend_mask, (41, 41), 15)
                
                # 3채널로 확장
                blend_mask_3d = np.stack([blend_mask] * 3, axis=2)
                
                # 최종 블렌딩
                final_canvas = canvas.astype(np.float32)
                canvas_content = canvas.astype(np.float32)
                
                final_canvas = (
                    canvas_content * (1 - blend_mask_3d) +
                    canvas.astype(np.float32) * blend_mask_3d
                )
                
                canvas = final_canvas.astype(np.uint8)
        
        print(f"✅ 썸네일 생성 완료: {target_size}")
        return canvas

def handler(event):
    """RunPod Serverless 메인 핸들러 - v14.3 Ultimate"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.3 Ultimate 연결 성공: {input_data['prompt']}",
                "version": "v14.3",
                "features": [
                    "v13.3 파라미터 (28쌍 학습 데이터)",
                    "Seamless Cloning 배경 제거",
                    "배경 특성 분석",
                    "썸네일 배경 연속성",
                    "좌표 기억 시스템"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            print("🚀 웨딩링 AI v14.3 Ultimate 처리 시작")
            
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 프로세서 초기화
            processor = WeddingRingAIv14_3()
            
            # 1. 검은색 선 감지 및 좌표 기억
            line_mask, ring_mask, ring_bbox = processor.detect_and_remember_black_lines(image_array)
            
            if line_mask is None:
                return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
            
            # 2. 배경 특성 분석
            background_profile = processor.analyze_background_characteristics(image_array, line_mask)
            
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
            
            # 5. v14.3 Seamless Background Removal
            main_result = processor.seamless_background_removal_v14_3(result_image, line_mask, ring_bbox)
            
            # 6. 2x 업스케일링
            upscaled = processor.basic_upscale(main_result, scale=2)
            
            # 7. v14.3 배경 연속성 썸네일 생성
            # 원본 bbox를 업스케일링 비율에 맞게 조정
            scaled_bbox = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
            thumbnail = processor.create_seamless_thumbnail_v14_3(upscaled, scaled_bbox)
            
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
                "version": "v14.3 Ultimate",
                "metal_type": metal_type,
                "lighting": lighting,
                "background_profile": background_profile,
                "ring_bbox": ring_bbox,
                "scale_factor": 2,
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                "thumbnail_size": "1000x1300",
                "features_used": [
                    "v13.3 파라미터",
                    "배경 특성 분석",
                    "Seamless Cloning",
                    "배경 연속성 썸네일"
                ]
            }
            
            print("✅ v14.3 Ultimate 처리 완료")
            
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
