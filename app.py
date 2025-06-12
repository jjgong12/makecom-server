from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import time
from datetime import datetime

app = Flask(__name__)

class NaturalWeddingRingEnhancer:
    def __init__(self):
        # 28쌍 학습 데이터 기반 - 자연스러운 보정만
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
            bright_mask = v > 150
            bright_h = h[bright_mask]
            bright_s = s[bright_mask]
            
            if len(bright_h) == 0:
                return 'champagne_gold'  # 기본값을 champagne_gold로 (가장 안전)
            
            avg_h = np.mean(bright_h)
            avg_s = np.mean(bright_s)
            
            # 색상값 기반 분류 - 보수적으로
            if avg_h < 15 or avg_h > 165:  # 빨간색 계열
                if avg_s > 60:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_h <= 35:  # 황색 계열
                if avg_s > 90:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'champagne_gold'  # 애매하면 중성적인 champagne_gold
                
        except Exception as e:
            print(f"Metal detection error: {e}")
            return 'champagne_gold'

    def detect_lighting(self, image):
        """LAB 색공간 A,B 채널로 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            avg_a = np.mean(a)
            avg_b = np.mean(b)
            
            # A채널: 초록-빨강, B채널: 파랑-노랑
            if avg_b > 140:  # 노란빛이 확실히 강함
                return 'warm'
            elif avg_b < 110:  # 파란빛이 확실히 강함
                return 'cool'
            else:
                return 'natural'  # 애매하면 natural
                
        except Exception as e:
            print(f"Lighting detection error: {e}")
            return 'natural'

    def apply_natural_enhancement(self, image, metal_type, lighting):
        """28쌍 after 수준의 자연스러운 보정만"""
        try:
            params = self.metal_params[metal_type][lighting]
            
            # 파라미터 강도 조절 (원본 유지 위해 더 약하게)
            safe_params = {
                'brightness': 1.0 + (params['brightness'] - 1.0) * 0.7,  # 70%로 감소
                'contrast': 1.0 + (params['contrast'] - 1.0) * 0.7,
                'warmth': 1.0 + (params['warmth'] - 1.0) * 0.6,  # 60%로 감소
                'saturation': 1.0 + (params['saturation'] - 1.0) * 0.8,
                'sharpness': 1.0 + (params['sharpness'] - 1.0) * 0.5,  # 50%로 감소
                'clarity': 1.0 + (params['clarity'] - 1.0) * 0.6,
                'gamma': params['gamma']
            }
            
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 1. 밝기 조정 (매우 미묘하게)
            if abs(safe_params['brightness'] - 1.0) > 0.02:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(safe_params['brightness'])
            
            # 2. 대비 조정 (매우 미묘하게)
            if abs(safe_params['contrast'] - 1.0) > 0.02:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(safe_params['contrast'])
            
            # 3. 색온도 조정 (매우 조심스럽게)
            if abs(safe_params['warmth'] - 1.0) > 0.03:
                img_array = np.array(pil_image)
                img_array = img_array.astype(np.float32)
                
                warmth_factor = safe_params['warmth']
                warmth_strength = min(abs(warmth_factor - 1.0), 0.15)  # 최대 15%까지만
                
                if warmth_factor > 1.0:  # 따뜻하게
                    img_array[:,:,0] *= (1.0 + warmth_strength)  # R 증가
                    img_array[:,:,2] *= (1.0 - warmth_strength * 0.5)  # B 감소
                else:  # 차갑게
                    img_array[:,:,0] *= (1.0 - warmth_strength * 0.5)  # R 감소
                    img_array[:,:,2] *= (1.0 + warmth_strength)  # B 증가
                
                pil_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            
            # 4. 채도 조정 (매우 미묘하게)
            if abs(safe_params['saturation'] - 1.0) > 0.02:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(safe_params['saturation'])
            
            # 5. 선명도 조정 (매우 약하게)
            if abs(safe_params['sharpness'] - 1.0) > 0.05:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(safe_params['sharpness'])
            
            # OpenCV로 변환
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 6. CLAHE (매우 약하게)
            if abs(safe_params['clarity'] - 1.0) > 0.03:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 클립 리미트 매우 낮게
                clip_limit = max(1.0, 2.0 * safe_params['clarity'])
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            # 7. 감마 보정 (매우 미묘하게)
            if abs(safe_params['gamma'] - 1.0) > 0.02:
                gamma = safe_params['gamma']
                # 감마 변화량 제한
                gamma = max(0.95, min(gamma, 1.05))
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, lookup_table)
            
            return enhanced
            
        except Exception as e:
            print(f"Natural enhancement error: {e}")
            return image

    def process_image_natural(self, image):
        """자연스러운 보정 프로세스 - 원본 최대한 유지"""
        try:
            start_time = time.time()
            
            # 메모리 최적화를 위한 크기 조정
            h, w = image.shape[:2]
            if max(h, w) > 2048:
                scale = 2048 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1. 자동 분석 (보수적으로)
            metal_type = self.detect_metal_type(image)
            lighting = self.detect_lighting(image)
            
            print(f"Detected: {metal_type}, {lighting}")
            
            # 2. 노이즈 제거 (매우 약하게)
            denoised = cv2.bilateralFilter(image, 5, 30, 30)
            
            # 3. 28쌍 after 수준의 자연스러운 보정
            enhanced = self.apply_natural_enhancement(denoised, metal_type, lighting)
            
            # 4. 원본과 블렌딩 (안전 장치)
            # 80% 보정 + 20% 원본으로 더 자연스럽게
            final_result = cv2.addWeighted(enhanced, 0.8, image, 0.2, 0)
            
            processing_time = time.time() - start_time
            print(f"Natural processing time: {processing_time:.2f}s")
            
            return final_result, {
                'metal_type': metal_type,
                'lighting': lighting,
                'processing_time': processing_time,
                'enhancement_level': 'natural'
            }
            
        except Exception as e:
            print(f"Natural processing error: {e}")
            return image, {'error': str(e)}

# Flask 앱 엔드포인트들
enhancer = NaturalWeddingRingEnhancer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v5.0_natural'
    })

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """자연스러운 보정 엔드포인트 - 28쌍 after 수준"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 자연스러운 보정 처리
        enhanced_image, metadata = enhancer.process_image_natural(image)
        
        # JPEG로 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        # 바이너리 데이터로 직접 반환
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'natural_enhanced_{int(time.time())}.jpg'
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """기존 엔드포인트 - 자연스러운 보정으로 교체"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 자연스러운 보정 처리 (동일)
        enhanced_image, metadata = enhancer.process_image_natural(image)
        
        # JPEG로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'segmented_enhanced_{int(time.time())}.jpg'
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
