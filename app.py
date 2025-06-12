from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import time
from datetime import datetime

app = Flask(__name__)

class SegmentedWeddingRingEnhancer:
    def __init__(self):
        # 28쌍 학습 데이터 기반 최적 파라미터
        self.metal_params = {
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

    def detect_metal_type(self, image):
        """HSV 색공간 분석으로 금속 타입 자동 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 밝은 영역만 분석 (반사광 영역)
            bright_mask = v > 180
            bright_h = h[bright_mask]
            bright_s = s[bright_mask]
            
            if len(bright_h) == 0:
                return 'white_gold'  # 기본값
            
            avg_h = np.mean(bright_h)
            avg_s = np.mean(bright_s)
            
            # 색상값 기반 분류
            if avg_h < 15 or avg_h > 165:  # 빨간색 계열
                if avg_s > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_h <= 35:  # 황색 계열
                if avg_s > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'white_gold'
                
        except Exception as e:
            print(f"Metal detection error: {e}")
            return 'white_gold'

    def detect_lighting(self, image):
        """LAB 색공간 A,B 채널로 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            avg_a = np.mean(a)
            avg_b = np.mean(b)
            
            # A채널: 초록-빨강, B채널: 파랑-노랑
            if avg_b > 135:  # 노란빛이 강함
                return 'warm'
            elif avg_b < 115:  # 파란빛이 강함
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            print(f"Lighting detection error: {e}")
            return 'natural'

    def extract_ring_region(self, image):
        """웨딩링 영역 자동 추출"""
        try:
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 금속 특성 기반 마스크 생성
            # 1. 밝기 기반 (금속 반사)
            bright_mask = v > 120
            
            # 2. 채도 기반 (금속 특성)
            metal_mask = (s > 20) & (s < 200)
            
            # 3. 색상 기반 (금속 색상 범위)
            color_mask1 = (h < 30) | (h > 150)  # 금색/은색/로즈골드
            color_mask2 = (h >= 30) & (h <= 150) & (s < 100)  # 화이트골드
            color_mask = color_mask1 | color_mask2
            
            # 마스크 결합
            ring_mask = bright_mask & metal_mask & color_mask
            
            # 형태학적 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ring_mask = cv2.morphologyEx(ring_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
            
            # 가장 큰 연결 컴포넌트만 유지
            contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                ring_mask = np.zeros_like(ring_mask)
                cv2.fillPoly(ring_mask, [largest_contour], 255)
            
            return ring_mask.astype(bool)
            
        except Exception as e:
            print(f"Ring extraction error: {e}")
            # 실패시 중앙 영역을 링으로 가정
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            cv2.circle(mask, (w//2, h//2), min(w, h)//4, True, -1)
            return mask

    def enhance_ring_basic(self, ring_region):
        """웨딩링 영역 기본 보정 - 노이즈 제거 + 선명도 + 최소한의 보정"""
        if ring_region is None or ring_region.size == 0:
            return ring_region
        
        # 웨딩링 기본 보정 파라미터 (문서 기준)
        ring_basic_params = {
            'brightness': 1.02,      # 최소한의 밝기 조정
            'contrast': 1.05,        # 살짝 대비 향상
            'sharpness': 1.25,       # 유무광 재질 확인 가능한 선명도
            'noise_reduction': 1.15, # 노이즈 제거
            'clarity': 1.08          # 기본적인 명료도
        }
        
        try:
            # 1. 노이즈 제거 (bilateral filter)
            denoised = cv2.bilateralFilter(ring_region, 9, 75, 75)
            
            # 2. 밝기/대비 조정
            brightness_factor = ring_basic_params['brightness']
            contrast_factor = ring_basic_params['contrast']
            
            # 밝기/대비 적용
            enhanced = cv2.convertScaleAbs(denoised, 
                                         alpha=contrast_factor, 
                                         beta=(brightness_factor - 1.0) * 50)
            
            # 3. 선명도 향상 (언샤프 마스킹)
            sharpness_factor = ring_basic_params['sharpness']
            gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            sharpened = cv2.addWeighted(enhanced, sharpness_factor, 
                                       gaussian_blur, -(sharpness_factor - 1.0), 0)
            
            # 4. CLAHE 적용 (명료도 향상)
            clarity_factor = ring_basic_params['clarity']
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 클립 리미트 조정
            clip_limit = 2.0 * clarity_factor
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l, a, b])
            final_ring = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return final_ring
            
        except Exception as e:
            print(f"Error in enhance_ring_basic: {e}")
            return ring_region

    def apply_tone_harmony(self, ring_region, background_enhanced, original_bg):
        """배경 톤 조화 - 배경 색감을 웨딩링에 70% 반영"""
        try:
            # 배경 색감 변화량 계산
            bg_lab = cv2.cvtColor(original_bg, cv2.COLOR_BGR2LAB)
            bg_enhanced_lab = cv2.cvtColor(background_enhanced, cv2.COLOR_BGR2LAB)
            
            # 평균 색감 변화
            delta_l = np.mean(bg_enhanced_lab[:,:,0]) - np.mean(bg_lab[:,:,0])
            delta_a = np.mean(bg_enhanced_lab[:,:,1]) - np.mean(bg_lab[:,:,1])
            delta_b = np.mean(bg_enhanced_lab[:,:,2]) - np.mean(bg_lab[:,:,2])
            
            # 웨딩링을 LAB로 변환
            ring_lab = cv2.cvtColor(ring_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(ring_lab)
            
            # 배경 톤 변화를 70% 반영 (문서 기준)
            l = np.clip(l + delta_l * 0.5, 0, 255)  # 밝기는 50%만
            a = np.clip(a + delta_a * 0.7, 0, 255)  # A채널 70%
            b = np.clip(b + delta_b * 0.7, 0, 255)  # B채널 70%
            
            # 다시 합치고 BGR로 변환
            harmonized_lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
            harmonized_ring = cv2.cvtColor(harmonized_lab, cv2.COLOR_LAB2BGR)
            
            return harmonized_ring
            
        except Exception as e:
            print(f"Tone harmony error: {e}")
            return ring_region

    def enhance_background_28pairs(self, background_region, metal_type, lighting):
        """배경 영역을 28쌍 after 수준으로 보정"""
        try:
            params = self.metal_params[metal_type][lighting]
            
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(background_region, cv2.COLOR_BGR2RGB))
            
            # 1. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(params['brightness'])
            
            # 2. 대비 조정
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(params['contrast'])
            
            # 3. 색상 조정 (warmth)
            if params['warmth'] != 1.0:
                img_array = np.array(pil_image)
                img_array = img_array.astype(np.float32)
                
                warmth_factor = params['warmth']
                if warmth_factor > 1.0:  # 따뜻하게
                    img_array[:,:,0] *= warmth_factor  # R 증가
                    img_array[:,:,2] *= (2.0 - warmth_factor)  # B 감소
                else:  # 차갑게
                    img_array[:,:,0] *= warmth_factor  # R 감소
                    img_array[:,:,2] *= (2.0 - warmth_factor)  # B 증가
                
                pil_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            
            # 4. 채도 조정
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(params['saturation'])
            
            # 5. 선명도 조정
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(params['sharpness'])
            
            # OpenCV로 변환
            enhanced_bg = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 6. CLAHE (명료도)
            if params['clarity'] != 1.0:
                lab = cv2.cvtColor(enhanced_bg, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0*params['clarity'], tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced_bg = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            # 7. 감마 보정
            if params['gamma'] != 1.0:
                gamma = params['gamma']
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                enhanced_bg = cv2.LUT(enhanced_bg, lookup_table)
            
            return enhanced_bg
            
        except Exception as e:
            print(f"Background enhancement error: {e}")
            return background_region

    def process_image_segmented(self, image):
        """영역별 차별 보정 메인 프로세스"""
        try:
            start_time = time.time()
            
            # 메모리 최적화를 위한 크기 조정
            h, w = image.shape[:2]
            if max(h, w) > 2048:
                scale = 2048 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1. 자동 분석
            metal_type = self.detect_metal_type(image)
            lighting = self.detect_lighting(image)
            
            print(f"Detected: {metal_type}, {lighting}")
            
            # 2. 웨딩링 영역 추출
            ring_mask = self.extract_ring_region(image)
            
            # 3. 영역 분리
            ring_region = image.copy()
            ring_region[~ring_mask] = 0
            
            background_region = image.copy()
            background_region[ring_mask] = 0
            
            # 4. 배경 영역 28쌍 after 수준 보정
            background_enhanced = self.enhance_background_28pairs(background_region, metal_type, lighting)
            
            # 5. 웨딩링 기본 보정
            ring_basic_enhanced = self.enhance_ring_basic(ring_region)
            
            # 6. 배경 톤을 웨딩링에 70% 반영
            ring_harmonized = self.apply_tone_harmony(ring_basic_enhanced, background_enhanced, background_region)
            
            # 7. 두 영역 합성
            result = background_enhanced.copy()
            result[ring_mask] = ring_harmonized[ring_mask]
            
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f}s")
            
            return result, {
                'metal_type': metal_type,
                'lighting': lighting,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"Segmented processing error: {e}")
            return image, {'error': str(e)}

# Flask 앱 엔드포인트들
enhancer = SegmentedWeddingRingEnhancer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v3.0_segmented'
    })

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """메인 영역별 차별 보정 엔드포인트"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 영역별 차별 보정 처리
        enhanced_image, metadata = enhancer.process_image_segmented(image)
        
        # JPEG로 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        # 바이너리 데이터로 직접 반환
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'segmented_enhanced_{int(time.time())}.jpg'
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """백업용 통합 보정 엔드포인트"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 메모리 최적화를 위한 크기 조정
        h, w = image.shape[:2]
        if max(h, w) > 2048:
            scale = 2048 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 간단한 통합 보정 (기존 방식)
        metal_type = enhancer.detect_metal_type(image)
        lighting = enhancer.detect_lighting(image)
        params = enhancer.metal_params[metal_type][lighting]
        
        # PIL 보정
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 밝기, 대비, 채도, 선명도 조정
        enhancer_brightness = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer_brightness.enhance(params['brightness'])
        
        enhancer_contrast = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer_contrast.enhance(params['contrast'])
        
        enhancer_color = ImageEnhance.Color(pil_image)
        pil_image = enhancer_color.enhance(params['saturation'])
        
        enhancer_sharpness = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer_sharpness.enhance(params['sharpness'])
        
        # OpenCV로 변환
        enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # JPEG로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'enhanced_{int(time.time())}.jpg'
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
