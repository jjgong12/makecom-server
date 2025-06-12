from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class WeddingRingEnhancerV625:
    def __init__(self):
        # v6.2.5 적정선 조정 파라미터 (v6.3 문제 해결)
        self.metal_params = {
            'white_gold': {
                'natural': {
                    'bg_brightness': 1.22, 'bg_contrast': 1.12, 'bg_sharpness': 1.15, 'bg_clarity': 1.08,
                    'ring_brightness': 1.30, 'ring_contrast': 1.18, 'ring_sharpness': 1.25, 'ring_clarity': 1.15  # 1.65→1.30으로 감소
                },
                'warm': {
                    'bg_brightness': 1.18, 'bg_contrast': 1.08, 'bg_sharpness': 1.12, 'bg_clarity': 1.05,
                    'ring_brightness': 1.28, 'ring_contrast': 1.15, 'ring_sharpness': 1.22, 'ring_clarity': 1.12
                },
                'cool': {
                    'bg_brightness': 1.25, 'bg_contrast': 1.15, 'bg_sharpness': 1.18, 'bg_clarity': 1.10,
                    'ring_brightness': 1.32, 'ring_contrast': 1.20, 'ring_sharpness': 1.28, 'ring_clarity': 1.18
                }
            },
            'rose_gold': {
                'natural': {
                    'bg_brightness': 1.15, 'bg_contrast': 1.08, 'bg_sharpness': 1.10, 'bg_clarity': 1.05,
                    'ring_brightness': 1.25, 'ring_contrast': 1.12, 'ring_sharpness': 1.18, 'ring_clarity': 1.10
                },
                'warm': {
                    'bg_brightness': 1.10, 'bg_contrast': 1.05, 'bg_sharpness': 1.08, 'bg_clarity': 1.02,
                    'ring_brightness': 1.22, 'ring_contrast': 1.08, 'ring_sharpness': 1.15, 'ring_clarity': 1.08
                },
                'cool': {
                    'bg_brightness': 1.25, 'bg_contrast': 1.15, 'bg_sharpness': 1.15, 'bg_clarity': 1.08,
                    'ring_brightness': 1.35, 'ring_contrast': 1.18, 'ring_sharpness': 1.25, 'ring_clarity': 1.15
                }
            },
            'champagne_gold': {
                'natural': {
                    'bg_brightness': 1.18, 'bg_contrast': 1.12, 'bg_sharpness': 1.12, 'bg_clarity': 1.08,
                    'ring_brightness': 1.28, 'ring_contrast': 1.15, 'ring_sharpness': 1.22, 'ring_clarity': 1.12
                },
                'warm': {
                    'bg_brightness': 1.15, 'bg_contrast': 1.10, 'bg_sharpness': 1.10, 'bg_clarity': 1.05,
                    'ring_brightness': 1.25, 'ring_contrast': 1.12, 'ring_sharpness': 1.20, 'ring_clarity': 1.10
                },
                'cool': {
                    'bg_brightness': 1.22, 'bg_contrast': 1.15, 'bg_sharpness': 1.15, 'bg_clarity': 1.10,
                    'ring_brightness': 1.32, 'ring_contrast': 1.18, 'ring_sharpness': 1.25, 'ring_clarity': 1.15
                }
            },
            'yellow_gold': {
                'natural': {
                    'bg_brightness': 1.20, 'bg_contrast': 1.15, 'bg_sharpness': 1.12, 'bg_clarity': 1.08,
                    'ring_brightness': 1.30, 'ring_contrast': 1.18, 'ring_sharpness': 1.25, 'ring_clarity': 1.15
                },
                'warm': {
                    'bg_brightness': 1.12, 'bg_contrast': 1.08, 'bg_sharpness': 1.08, 'bg_clarity': 1.05,
                    'ring_brightness': 1.22, 'ring_contrast': 1.12, 'ring_sharpness': 1.18, 'ring_clarity': 1.10
                },
                'cool': {
                    'bg_brightness': 1.28, 'bg_contrast': 1.20, 'bg_sharpness': 1.18, 'bg_clarity': 1.12,
                    'ring_brightness': 1.38, 'ring_contrast': 1.22, 'ring_sharpness': 1.30, 'ring_clarity': 1.18
                }
            }
        }
    
    def detect_metal_type(self, image):
        """보수적 금속 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 보수적 기준 - 애매하면 champagne_gold
            avg_hue = np.mean(h[v > 50])
            avg_sat = np.mean(s[v > 50])
            
            if avg_hue < 15 or avg_hue > 165:
                if avg_sat < 50:
                    return 'white_gold'
                else:
                    return 'rose_gold'
            elif 15 <= avg_hue <= 35:
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'champagne_gold'
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            avg_b = np.mean(b)
            
            if avg_b < 120:
                return 'cool'
            elif avg_b > 140:
                return 'warm'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def create_wedding_ring_mask(self, image):
        """정밀한 웨딩링 마스크 생성"""
        height, width = image.shape[:2]
        
        # HSV 기반 금속 감지
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 금속 색상 범위 (더 넓게)
        lower_metal1 = np.array([0, 20, 100])
        upper_metal1 = np.array([30, 255, 255])
        lower_metal2 = np.array([15, 15, 80])
        upper_metal2 = np.array([35, 100, 255])
        
        mask1 = cv2.inRange(hsv, lower_metal1, upper_metal1)
        mask2 = cv2.inRange(hsv, lower_metal2, upper_metal2)
        metal_mask = cv2.bitwise_or(mask1, mask2)
        
        # 밝기 기반 보완
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        
        # 결합 및 정제
        combined_mask = cv2.bitwise_and(metal_mask, bright_mask)
        
        # 형태학적 연산으로 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 가우시안 블러로 부드러운 경계
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        return combined_mask
    
    def enhance_region(self, image, params, mask=None):
        """영역별 보정 적용"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # PIL 기반 안전 보정
        if 'brightness' in params:
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(params['brightness'])
        
        if 'contrast' in params:
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(params['contrast'])
        
        if 'sharpness' in params:
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(params['sharpness'])
        
        enhanced = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # CLAHE 적용 (완화된 설정)
        if 'clarity' in params and params['clarity'] > 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # v6.3 문제 해결: 16x16 타일 (8x8→16x16으로 완화)
            clip_limit = min(2.0, (params['clarity'] - 1.0) * 4.0 + 1.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def smooth_background(self, image, ring_mask):
        """배경 음영/그림자 제거"""
        # 배경 영역만 추출
        bg_mask = cv2.bitwise_not(ring_mask)
        bg_only = cv2.bitwise_and(image, image, mask=bg_mask)
        
        # 배경의 평균 색상 계산
        bg_pixels = image[bg_mask > 0]
        if len(bg_pixels) > 0:
            avg_color = np.mean(bg_pixels, axis=0)
            
            # 가우시안 블러로 부드럽게
            blurred_bg = cv2.GaussianBlur(image, (51, 51), 0)
            
            # 배경을 평균 색상으로 점진적 교체
            smooth_bg = image.copy()
            bg_mask_3d = cv2.merge([bg_mask, bg_mask, bg_mask]) / 255.0
            
            # 30% 평균 색상, 70% 블러된 배경으로 조합
            uniform_bg = np.full_like(image, avg_color)
            smooth_background = cv2.addWeighted(blurred_bg, 0.7, uniform_bg, 0.3, 0)
            
            # 배경만 교체 (웨딩링은 보존)
            result = image * (1 - bg_mask_3d) + smooth_background * bg_mask_3d
            return result.astype(np.uint8)
        
        return image

    def enhance_wedding_ring(self, image_data):
        """메인 보정 함수"""
        try:
            # 이미지 디코딩 및 리샘플링
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            height, width = original.shape[:2]
            if height > 2048 or width > 2048:
                scale = min(2048/height, 2048/width)
                new_height, new_width = int(height * scale), int(width * scale)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 노이즈 제거
            denoised = cv2.bilateralFilter(original, 9, 75, 75)
            
            # 웨딩링 마스크 먼저 생성 (배경 평활화용)
            ring_mask = self.create_wedding_ring_mask(denoised)
            
            # 배경 음영/그림자 제거
            smoothed = self.smooth_background(denoised, ring_mask)
            
            # 금속/조명 감지
            metal_type = self.detect_metal_type(smoothed)
            lighting = self.detect_lighting(smoothed)
            
            params = self.metal_params.get(metal_type, self.metal_params['champagne_gold'])[lighting]
            
            # 웨딩링 마스크 재생성 (평활화된 이미지 기준)
            ring_mask = self.create_wedding_ring_mask(smoothed)
            bg_mask = cv2.bitwise_not(ring_mask)
            
            # 배경 보정 (보수적)
            bg_params = {
                'brightness': params['bg_brightness'],
                'contrast': params['bg_contrast'],
                'sharpness': params['bg_sharpness'],
                'clarity': params['bg_clarity']
            }
            bg_enhanced = self.enhance_region(smoothed, bg_params)
            
            # 웨딩링 보정 (적정선 조정)
            ring_params = {
                'brightness': params['ring_brightness'],
                'contrast': params['ring_contrast'],
                'sharpness': params['ring_sharpness'],
                'clarity': params['ring_clarity']
            }
            ring_enhanced = self.enhance_region(smoothed, ring_params)
            
            # 마스크 적용하여 결합
            ring_mask_3d = cv2.merge([ring_mask, ring_mask, ring_mask]) / 255.0
            bg_mask_3d = cv2.merge([bg_mask, bg_mask, bg_mask]) / 255.0
            
            combined = (ring_enhanced * ring_mask_3d + bg_enhanced * bg_mask_3d).astype(np.uint8)
            
            # 보수적 블렌딩 (85% 보정 + 15% 평활화된 원본)
            final_result = cv2.addWeighted(combined, 0.85, smoothed, 0.15, 0)
            
            # 미묘한 하이라이트 부스팅
            gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
            highlight_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
            highlight_boost = cv2.addWeighted(final_result, 1.0, final_result, 0.05, 0)
            highlight_mask_3d = cv2.merge([highlight_mask, highlight_mask, highlight_mask]) / 255.0
            final_result = (highlight_boost * highlight_mask_3d + final_result * (1 - highlight_mask_3d)).astype(np.uint8)
            
            return final_result, metal_type, lighting
            
        except Exception as e:
            logging.error(f"Enhancement error: {str(e)}")
            return None, "error", "error"

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "v6.2.5-smooth",
        "message": "Wedding Ring Enhancement API - 배경 음영/그림자 제거 기능 추가",
        "endpoints": [
            "/health",
            "/enhance_wedding_ring_v625_smooth"
        ]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "v6.2.5-smooth"})

@app.route('/enhance_wedding_ring_v625_smooth', methods=['POST'])
def enhance_wedding_ring_v625_smooth():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 보정 수행
        enhancer = WeddingRingEnhancerV625()
        enhanced_image, metal_type, lighting = enhancer.enhance_wedding_ring(image_data)
        
        if enhanced_image is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
