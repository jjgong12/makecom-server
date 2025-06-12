from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import time
from datetime import datetime
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class SegmentedWeddingRingEnhancer:
    def __init__(self):
        # 웨딩링 기본 보정 파라미터 (노이즈 제거 + 선명도 위주)
        self.ring_basic_params = {
            'brightness': 1.02,     # 최소한의 밝기 조정
            'contrast': 1.05,       # 살짝 대비 향상
            'sharpness': 1.25,      # 유무광 재질 확인 가능한 선명도
            'noise_reduction': 1.15, # 노이즈 제거
            'clarity': 1.08         # 기본적인 명료도
        }
        
        # 배경 파라미터 = 28쌍 학습 데이터 그대로 (완전한 after 결과)
        self.background_after_params = {
            'white_gold': {
                'natural': {'brightness': 1.22, 'contrast': 1.12, 'warmth': 0.95, 'saturation': 1.00, 'sharpness': 1.30, 'clarity': 1.18, 'gamma': 1.01},
                'warm': {'brightness': 1.28, 'contrast': 1.18, 'warmth': 0.80, 'saturation': 0.95, 'sharpness': 1.35, 'clarity': 1.22, 'gamma': 1.03},
                'cool': {'brightness': 1.18, 'contrast': 1.08, 'warmth': 1.00, 'saturation': 1.03, 'sharpness': 1.25, 'clarity': 1.15, 'gamma': 0.99}
            },
            'rose_gold': {
                'natural': {'brightness': 1.15, 'contrast': 1.08, 'warmth': 1.20, 'saturation': 1.15, 'sharpness': 1.15, 'clarity': 1.10, 'gamma': 0.98},
                'warm': {'brightness': 1.10, 'contrast': 1.05, 'warmth': 1.05, 'saturation': 1.10, 'sharpness': 1.10, 'clarity': 1.05, 'gamma': 0.95},
                'cool': {'brightness': 1.25, 'contrast': 1.15, 'warmth': 1.35, 'saturation': 1.25, 'sharpness': 1.25, 'clarity': 1.20, 'gamma': 1.02}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.18, 'contrast': 1.12, 'warmth': 1.08, 'saturation': 1.08, 'sharpness': 1.22, 'clarity': 1.15, 'gamma': 1.00},
                'warm': {'brightness': 1.15, 'contrast': 1.10, 'warmth': 0.95, 'saturation': 1.05, 'sharpness': 1.20, 'clarity': 1.12, 'gamma': 0.98},
                'cool': {'brightness': 1.22, 'contrast': 1.15, 'warmth': 1.18, 'saturation': 1.12, 'sharpness': 1.25, 'clarity': 1.18, 'gamma': 1.02}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.20, 'contrast': 1.15, 'warmth': 1.25, 'saturation': 1.20, 'sharpness': 1.18, 'clarity': 1.12, 'gamma': 1.01},
                'warm': {'brightness': 1.12, 'contrast': 1.08, 'warmth': 1.10, 'saturation': 1.12, 'sharpness': 1.15, 'clarity': 1.08, 'gamma': 0.97},
                'cool': {'brightness': 1.28, 'contrast': 1.20, 'warmth': 1.40, 'saturation': 1.28, 'sharpness': 1.25, 'clarity': 1.18, 'gamma': 1.03}
            }
        }
    
    def detect_ring_type(self, image):
        """28쌍 학습 데이터 기반 금속 타입 자동 감지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 금속 영역 추출 (밝기 기반)
        metal_mask = cv2.inRange(hsv[:,:,2], 100, 255)
        metal_pixels = hsv[metal_mask > 0]
        
        if len(metal_pixels) == 0:
            return 'white_gold'
        
        # 색상(Hue) 분석
        avg_hue = np.mean(metal_pixels[:, 0])
        avg_saturation = np.mean(metal_pixels[:, 1])
        
        # 28쌍 분석 결과 기반 임계값
        if avg_saturation < 30:
            return 'white_gold'
        elif 5 <= avg_hue <= 25 and avg_saturation > 40:
            if avg_saturation > 80:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        elif 160 <= avg_hue <= 180 and avg_saturation > 30:
            return 'rose_gold'
        else:
            return 'white_gold'
    
    def detect_lighting_environment(self, image):
        """조명 환경 자동 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128
        
        avg_a = np.mean(a_channel)
        avg_b = np.mean(b_channel)
        
        if avg_b > 8:
            return 'warm'
        elif avg_b < -5:
            return 'cool'
        else:
            return 'natural'
    
    def extract_ring_mask(self, image):
        """웨딩링 영역 자동 추출 (28쌍 학습 패턴 기반)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1단계: 금속 반사 영역 감지 (높은 밝기)
        brightness_mask = cv2.inRange(hsv[:,:,2], 120, 255)
        
        # 2단계: 색상 범위로 금속 감지
        low_saturation = cv2.inRange(hsv[:,:,1], 0, 60)  # 무채색 금속
        
        gold_hue_1 = cv2.inRange(hsv[:,:,0], 10, 30)  # 황색 범위
        gold_hue_2 = cv2.inRange(hsv[:,:,0], 0, 10)   # 주황-황색
        gold_mask = cv2.bitwise_or(gold_hue_1, gold_hue_2)
        
        # 전체 금속 마스크
        metal_mask = cv2.bitwise_and(brightness_mask, 
                                   cv2.bitwise_or(low_saturation, gold_mask))
        
        # 3단계: 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
        
        # 4단계: 컨투어 감지로 링 모양 필터링
        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ring_mask = np.zeros_like(metal_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 300 and area < 80000:
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        cv2.fillPoly(ring_mask, [contour], 255)
        
        # 5단계: 마스크 부드럽게 처리
        ring_mask = cv2.GaussianBlur(ring_mask, (5,5), 2)
        
        return ring_mask
    
    def _prepare_image(self, image):
        """이미지 전처리 및 메모리 최적화"""
        height, width = image.shape[:2]
        
        if width > 2048 or height > 2048:
            scale = min(2048/width, 2048/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def enhance_ring_region(self, image, mask, ring_type, lighting_env):
        """웨딩링 영역 전용 집중 보정"""
        params = self.ring_focused_params[ring_type][lighting_env]
        
        # 마스크 영역만 추출
        ring_region = cv2.bitwise_and(image, image, mask=mask)
        
        # 강력한 금속 보정 적용
        enhanced = self._apply_ring_enhancement(ring_region, params)
        
    def _apply_basic_ring_enhancement(self, image):
        """웨딩링 기본 보정 (노이즈 제거 + 선명도 + 유무광 재질 확인)"""
        params = self.ring_basic_params
        
        # 1. 노이즈 제거 (가우시안 블러 → 선명화)
        denoised = cv2.GaussianBlur(image, (3,3), 0.5)
        
        # 2. 기본 밝기/대비 조정
        enhanced = cv2.convertScaleAbs(denoised, 
                                     alpha=params['contrast'], 
                                     beta=(params['brightness']-1)*20)
        
        # 3. 선명도 향상 (유무광 재질 확인 가능하게)
        blurred = cv2.GaussianBlur(enhanced, (3,3), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1 + (params['sharpness']-1), 
                                 blurred, -(params['sharpness']-1), 0)
        
        # 4. 기본 명료도 (CLAHE 약하게)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        clip_limit = 1.0 + (params['clarity'] - 1) * 1.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_full_enhancement(self, image, params):
        """28쌍 학습 데이터 전체 보정 (완전한 after 결과)"""
        # 밝기/대비 조정
        enhanced = cv2.convertScaleAbs(image, 
                                     alpha=params['contrast'], 
                                     beta=(params['brightness']-1)*50)
        
        # 색온도 조정
        if params['warmth'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,2] = lab[:,:,2] * params['warmth']
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 채도 조정
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * params['saturation']
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 선명도 향상
        blurred = cv2.GaussianBlur(enhanced, (5,5), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1 + (params['sharpness']-1), 
                                 blurred, -(params['sharpness']-1), 0)
        
        # 명료도 강화
        if params['clarity'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            clip_limit = 2.0 + (params['clarity'] - 1) * 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 감마 보정
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_strong_tone_harmony(self, original_image, enhanced_ring, enhanced_background, ring_mask):
        """강력한 톤 조화: 배경 색감 변화를 웨딩링에 70% 묻어나게"""
        # 배경 마스크
        background_mask = cv2.bitwise_not(ring_mask)
        
        # 원본과 보정된 배경의 색감 변화량 계산
        original_bg = cv2.bitwise_and(original_image, original_image, mask=background_mask)
        enhanced_bg = cv2.bitwise_and(enhanced_background, enhanced_background, mask=background_mask)
        
        # LAB 색공간에서 색감 변화량 분석
        original_bg_lab = cv2.cvtColor(original_bg, cv2.COLOR_RGB2LAB).astype(np.float32)
        enhanced_bg_lab = cv2.cvtColor(enhanced_bg, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 유효한 픽셀만 분석 (배경 영역)
        valid_pixels = background_mask > 0
        if np.sum(valid_pixels) > 0:
            # 배경의 평균 색감 변화량
            original_mean_a = np.mean(original_bg_lab[valid_pixels, 1])
            original_mean_b = np.mean(original_bg_lab[valid_pixels, 2])
            enhanced_mean_a = np.mean(enhanced_bg_lab[valid_pixels, 1])
            enhanced_mean_b = np.mean(enhanced_bg_lab[valid_pixels, 2])
            
            # 색감 변화량 (A, B 채널)
            delta_a = enhanced_mean_a - original_mean_a
            delta_b = enhanced_mean_b - original_mean_b
            
            # 밝기 변화량 (L 채널)
            original_mean_l = np.mean(original_bg_lab[valid_pixels, 0])
            enhanced_mean_l = np.mean(enhanced_bg_lab[valid_pixels, 0])
            delta_l = enhanced_mean_l - original_mean_l
            
            # 웨딩링에 배경 변화량의 70% 적용 (배경색이 묻어나게)
            ring_lab = cv2.cvtColor(enhanced_ring, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # 웨딩링 영역에만 강한 톤 조화 적용
            ring_pixels = ring_mask > 0
            if np.sum(ring_pixels) > 0:
                ring_lab[ring_pixels, 0] += delta_l * 0.5  # 밝기 50% 반영
                ring_lab[ring_pixels, 1] += delta_a * 0.7  # A 채널 70% 반영  
                ring_lab[ring_pixels, 2] += delta_b * 0.7  # B 채널 70% 반영 (배경색 묻어나게)
                
                # 범위 제한
                ring_lab[:,:,0] = np.clip(ring_lab[:,:,0], 0, 100)
                ring_lab[:,:,1] = np.clip(ring_lab[:,:,1], -127, 128)
                ring_lab[:,:,2] = np.clip(ring_lab[:,:,2], -127, 128)
            
            # RGB로 다시 변환
            harmonized_ring = cv2.cvtColor(ring_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            return harmonized_ring
        
        return enhanced_ring
    
    def enhance_background_region(self, image, mask, lighting_env):
        """배경 영역 전용 분위기 보정"""
        background_mask = cv2.bitwise_not(mask)
        background_region = cv2.bitwise_and(image, image, mask=background_mask)
        
        params = self.background_focused_params[lighting_env]
        enhanced = self._apply_background_enhancement(background_region, params)
        
        return enhanced
    
    def _apply_ring_enhancement(self, image, params):
        """웨딩링 전용 집중 보정 (금속 질감 극대화)"""
        # 밝기/대비 조정
        enhanced = cv2.convertScaleAbs(image, 
                                     alpha=params['contrast'], 
                                     beta=(params['brightness']-1)*50)
        
        # 색온도 조정
        if params['warmth'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,2] = lab[:,:,2] * params['warmth']
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 채도 조정
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * params['saturation']
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 언샤프 마스킹으로 선명도 극대화
        blurred = cv2.GaussianBlur(enhanced, (5,5), 1.5)
        sharpness = params['sharpness']
        enhanced = cv2.addWeighted(enhanced, 1 + (sharpness-1), 
                                 blurred, -(sharpness-1), 0)
        
        # CLAHE로 명료도 강화
        if params['clarity'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            clip_limit = 2.0 + (params['clarity'] - 1) * 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 감마 보정
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_background_enhancement(self, image, params):
        """배경 전용 분위기 보정 (은은한 조화)"""
        # 부드러운 밝기 조정
        enhanced = cv2.convertScaleAbs(image,
                                     alpha=params['contrast'],
                                     beta=(params['brightness']-1)*30)
        
        # 색온도 조정
        if params['warmth'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,2] = lab[:,:,2] * params['warmth']
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 채도 조정
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * params['saturation']
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 감마 보정
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def enhance_wedding_ring_segmented(self, image_data):
        """메인 함수: 28쌍 기반 영역별 차별 보정 + 글로벌 톤 조화"""
        start_time = time.time()
        
        try:
            # Base64 디코딩
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image.convert('RGB'))
            
            # 이미지 전처리
            image_array = self._prepare_image(image_array)
            original_image = image_array.copy()
            
            # AI 분석
            ring_type = self.detect_ring_type(image_array)
            lighting_env = self.detect_lighting_environment(image_array)
            
            # 웨딩링 영역 자동 추출
            ring_mask = self.extract_ring_mask(image_array)
            
            # 각 영역별 보정
            enhanced_ring_basic = self.enhance_ring_basic(image_array, ring_mask)
            enhanced_background_after = self.enhance_background_after(image_array, ring_mask, ring_type, lighting_env)
            
            # 웨딩링에 배경 톤 70% 묻어나게 적용
            enhanced_ring_final = self._apply_strong_tone_harmony(
                original_image, enhanced_ring_basic, enhanced_background_after, ring_mask
            )
            
            # 두 영역 자연스럽게 합성
            ring_mask_norm = ring_mask.astype(np.float32) / 255.0
            ring_mask_3d = np.dstack([ring_mask_norm] * 3)
            
            final_result = (enhanced_ring_final.astype(np.float32) * ring_mask_3d + 
                           enhanced_background_after.astype(np.float32) * (1 - ring_mask_3d))
            
            # PIL로 변환 및 JPG 저장
            enhanced_pil = Image.fromarray(final_result.astype(np.uint8))
            enhanced_buffer = io.BytesIO()
            enhanced_pil.save(enhanced_buffer, format='JPEG', quality=95, progressive=True)
            enhanced_buffer.seek(0)
            
            processing_time = time.time() - start_time
            
            # 사용된 파라미터
            ring_params = self.ring_basic_params
            bg_params = self.background_after_params[ring_type][lighting_env]
            
            logging.info(f"Basic ring enhancement with strong tone harmony: {ring_type} ring under {lighting_env} lighting in {processing_time:.2f}s")
            
            return enhanced_buffer, ring_type, lighting_env, ring_params, bg_params, processing_time
            
        except Exception as e:
            logging.error(f"Segmented enhancement failed: {str(e)}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'segmented_wedding_ring_enhancer'})

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """웨딩링 영역별 차별 보정 - 바이너리 직접 반환"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        enhancer = SegmentedWeddingRingEnhancer()
        enhanced_buffer, ring_type, lighting_env, ring_params, bg_params, processing_time = enhancer.enhance_wedding_ring_segmented(
            data['image_base64']
        )
        
        # 타임스탬프 기반 영어 파일명
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f'segmented_enhanced_{timestamp}.jpg'
        
        return Response(
            enhanced_buffer.getvalue(),
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'attachment; filename={safe_filename}',
                'Content-Type': 'image/jpeg',
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting_env,
                'X-Processing-Time': str(round(processing_time, 2))
            }
        )
        
    except Exception as e:
        logging.error(f"Error in enhance_wedding_ring_segmented: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
