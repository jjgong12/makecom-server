from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class SmartBackgroundEnhancer:
    def __init__(self):
        # 28쌍 After 기반 배경 템플릿 (조명별)
        self.background_templates = {
            'natural': {
                'base_color': [245, 240, 235],  # 자연광 배경
                'gradient_factor': 0.95,        # 상단→하단 그라데이션
                'brightness_range': (240, 250)
            },
            'warm': {
                'base_color': [250, 245, 235],  # 따뜻한 조명 배경
                'gradient_factor': 0.93,
                'brightness_range': (235, 255)
            },
            'cool': {
                'base_color': [240, 242, 250],  # 차가운 조명 배경
                'gradient_factor': 0.96,
                'brightness_range': (240, 250)
            }
        }
        
        # 웨딩링 보정 파라미터 (기존 유지)
        self.metal_params = {
            'white_gold': {
                'natural': {'brightness': 1.30, 'contrast': 1.18, 'sharpness': 1.25, 'clarity': 1.15},
                'warm': {'brightness': 1.28, 'contrast': 1.15, 'sharpness': 1.22, 'clarity': 1.12},
                'cool': {'brightness': 1.32, 'contrast': 1.20, 'sharpness': 1.28, 'clarity': 1.18}
            },
            'rose_gold': {
                'natural': {'brightness': 1.25, 'contrast': 1.12, 'sharpness': 1.18, 'clarity': 1.10},
                'warm': {'brightness': 1.22, 'contrast': 1.08, 'sharpness': 1.15, 'clarity': 1.08},
                'cool': {'brightness': 1.35, 'contrast': 1.18, 'sharpness': 1.25, 'clarity': 1.15}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.28, 'contrast': 1.15, 'sharpness': 1.22, 'clarity': 1.12},
                'warm': {'brightness': 1.25, 'contrast': 1.12, 'sharpness': 1.20, 'clarity': 1.10},
                'cool': {'brightness': 1.32, 'contrast': 1.18, 'sharpness': 1.25, 'clarity': 1.15}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.30, 'contrast': 1.18, 'sharpness': 1.25, 'clarity': 1.15},
                'warm': {'brightness': 1.22, 'contrast': 1.12, 'sharpness': 1.18, 'clarity': 1.10},
                'cool': {'brightness': 1.38, 'contrast': 1.22, 'sharpness': 1.30, 'clarity': 1.18}
            }
        }
    
    def detect_metal_type(self, image):
        """보수적 금속 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            avg_hue = np.mean(h[v > 50])
            avg_sat = np.mean(s[v > 50])
            
            if avg_hue < 15 or avg_hue > 165:
                return 'white_gold' if avg_sat < 50 else 'rose_gold'
            elif 15 <= avg_hue <= 35:
                return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
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
    
    def analyze_background_complexity(self, image):
        """배경 복잡도 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 표준편차로 복잡도 측정
        std_dev = np.std(gray)
        
        # 엣지 밀도 계산
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 복잡도 점수 (0-100)
        complexity = min(100, (std_dev / 50 * 60) + (edge_density * 1000 * 40))
        
        return complexity
    
    def create_soft_ring_mask(self, image):
        """부드러운 웨딩링 마스크 생성 (경계선 문제 해결)"""
        height, width = image.shape[:2]
        
        # HSV 기반 금속 감지
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 더 관대한 금속 범위
        lower_metal1 = np.array([0, 10, 80])   # 더 넓게
        upper_metal1 = np.array([35, 255, 255])
        lower_metal2 = np.array([10, 5, 60])   # 더 관대하게
        upper_metal2 = np.array([40, 120, 255])
        
        mask1 = cv2.inRange(hsv, lower_metal1, upper_metal1)
        mask2 = cv2.inRange(hsv, lower_metal2, upper_metal2)
        metal_mask = cv2.bitwise_or(mask1, mask2)
        
        # 밝기 기반 보완 (더 관대하게)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]  # 100으로 낮춤
        
        # 결합
        combined_mask = cv2.bitwise_and(metal_mask, bright_mask)
        
        # 형태학적 연산 (더 부드럽게)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 크게
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        
        # 팽창 → 침식으로 자연스러운 경계
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel_small, iterations=1)
        
        # 강한 가우시안 블러로 매우 부드러운 경계
        combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)  # 21x21로 증가
        
        return combined_mask
    
    def create_background_template(self, image, lighting, complexity):
        """조명별 배경 템플릿 생성"""
        height, width = image.shape[:2]
        template = self.background_templates[lighting]
        
        # 기본 색상
        base_color = np.array(template['base_color'])
        
        # 복잡도에 따른 조정
        if complexity > 70:  # 복잡한 배경 → 강제 단순화
            intensity = 1.0
        elif complexity > 30:  # 보통 배경 → 부드러운 교체
            intensity = 0.7
        else:  # 단순한 배경 → 미묘한 조정
            intensity = 0.3
        
        # 그라데이션 생성 (상단 밝고 → 하단 어둡게)
        gradient_factor = template['gradient_factor']
        background = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            # 상단(0) → 하단(height)으로 갈수록 어두워짐
            gradient_ratio = 1.0 - (y / height) * (1.0 - gradient_factor)
            color = base_color * gradient_ratio
            background[y, :] = np.clip(color, 0, 255)
        
        return background, intensity
    
    def enhance_ring_only(self, image, params):
        """웨딩링만 보정"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # PIL 기반 안전 보정
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(params['brightness'])
        
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(params['contrast'])
        
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(params['sharpness'])
        
        enhanced = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 제한적 CLAHE
        if params['clarity'] > 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clip_limit = min(2.0, (params['clarity'] - 1.0) * 3.0 + 1.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def smart_background_enhance(self, image_data):
        """스마트 배경 템플릿 시스템"""
        try:
            # 이미지 디코딩 및 리샘플링
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            height, width = original.shape[:2]
            if height > 2048 or width > 2048:
                scale = min(2048/height, 2048/width)
                new_height, new_width = int(height * scale), int(width * scale)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 기본 노이즈 제거
            denoised = cv2.bilateralFilter(original, 9, 75, 75)
            
            # 1단계: 분석
            metal_type = self.detect_metal_type(denoised)
            lighting = self.detect_lighting(denoised)
            complexity = self.analyze_background_complexity(denoised)
            
            # 2단계: 부드러운 웨딩링 마스크 생성
            ring_mask = self.create_soft_ring_mask(denoised)
            
            # 3단계: 배경 템플릿 생성
            bg_template, intensity = self.create_background_template(denoised, lighting, complexity)
            
            # 4단계: 웨딩링 보정
            params = self.metal_params.get(metal_type, self.metal_params['champagne_gold'])[lighting]
            ring_enhanced = self.enhance_ring_only(denoised, params)
            
            # 5단계: 스마트 "딸깍" 배경 처리
            # 현재 배경과 템플릿 블렌딩
            current_bg = denoised.copy()
            smart_bg = cv2.addWeighted(current_bg, 1.0 - intensity, bg_template, intensity, 0)
            
            # 6단계: 자연스러운 합성 (경계선 문제 해결)
            ring_mask_3d = cv2.merge([ring_mask, ring_mask, ring_mask]) / 255.0
            
            # 웨딩링 영역과 배경 영역 부드럽게 합성
            final_result = (ring_enhanced * ring_mask_3d + smart_bg * (1.0 - ring_mask_3d)).astype(np.uint8)
            
            # 7단계: 최종 블렌딩 (원본과 5% 블렌딩으로 자연스럽게)
            final_result = cv2.addWeighted(final_result, 0.95, denoised, 0.05, 0)
            
            return final_result, metal_type, lighting, complexity
            
        except Exception as e:
            logging.error(f"Enhancement error: {str(e)}")
            return None, "error", "error", 0

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "Smart Background v1.0",
        "message": "스마트 배경 템플릿 시스템 + 마스크 경계 문제 해결",
        "endpoints": [
            "/health",
            "/enhance_wedding_ring_smart"
        ]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "Smart Background v1.0"})

@app.route('/enhance_wedding_ring_smart', methods=['POST'])
def enhance_wedding_ring_smart():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 스마트 보정 수행
        enhancer = SmartBackgroundEnhancer()
        enhanced_image, metal_type, lighting, complexity = enhancer.smart_background_enhance(image_data)
        
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
