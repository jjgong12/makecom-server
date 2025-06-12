from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import os

app = Flask(__name__)

class WeddingRingEnhancerV13_3:
    def __init__(self):
        # v13.3: 버전10 쪽으로 조정된 자연스러운 파라미터
        self.metal_params = {
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
                    'white_overlay': 0.10,
                    'sharpness': 1.13,
                    'color_temp_a': -2,
                    'color_temp_b': -4,
                    'original_blend': 0.15
                },
                'cool': {
                    'brightness': 1.20,
                    'contrast': 1.14,
                    'white_overlay': 0.08,
                    'sharpness': 1.17,
                    'color_temp_a': -4,
                    'color_temp_b': -2,
                    'original_blend': 0.15
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.15,
                    'contrast': 1.08,
                    'white_overlay': 0.07,
                    'sharpness': 1.12,
                    'color_temp_a': -1,
                    'color_temp_b': -2,
                    'original_blend': 0.15
                },
                'warm': {
                    'brightness': 1.12,
                    'contrast': 1.06,
                    'white_overlay': 0.06,
                    'sharpness': 1.10,
                    'color_temp_a': 0,
                    'color_temp_b': -1,
                    'original_blend': 0.15
                },
                'cool': {
                    'brightness': 1.18,
                    'contrast': 1.12,
                    'white_overlay': 0.09,
                    'sharpness': 1.15,
                    'color_temp_a': -2,
                    'color_temp_b': -3,
                    'original_blend': 0.15
                }
            },
            'champagne_gold': {
                'natural': {
                    'brightness': 1.16,
                    'contrast': 1.10,
                    'white_overlay': 0.08,
                    'sharpness': 1.14,
                    'color_temp_a': -2,
                    'color_temp_b': -2,
                    'original_blend': 0.15
                },
                'warm': {
                    'brightness': 1.14,
                    'contrast': 1.08,
                    'white_overlay': 0.07,
                    'sharpness': 1.12,
                    'color_temp_a': -1,
                    'color_temp_b': -3,
                    'original_blend': 0.15
                },
                'cool': {
                    'brightness': 1.19,
                    'contrast': 1.13,
                    'white_overlay': 0.09,
                    'sharpness': 1.16,
                    'color_temp_a': -3,
                    'color_temp_b': -1,
                    'original_blend': 0.15
                }
            },
            'yellow_gold': {
                'natural': {
                    'brightness': 1.17,
                    'contrast': 1.11,
                    'white_overlay': 0.06,
                    'sharpness': 1.13,
                    'color_temp_a': 0,
                    'color_temp_b': -1,
                    'original_blend': 0.15
                },
                'warm': {
                    'brightness': 1.13,
                    'contrast': 1.07,
                    'white_overlay': 0.05,
                    'sharpness': 1.11,
                    'color_temp_a': 1,
                    'color_temp_b': 0,
                    'original_blend': 0.15
                },
                'cool': {
                    'brightness': 1.21,
                    'contrast': 1.15,
                    'white_overlay': 0.08,
                    'sharpness': 1.17,
                    'color_temp_a': -1,
                    'color_temp_b': -2,
                    'original_blend': 0.15
                }
            }
        }

    def _detect_metal_type(self, image):
        """간소화된 금속 타입 감지"""
        try:
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 밝은 영역만 분석
            bright_mask = v > 100
            if np.sum(bright_mask) < 1000:
                return 'champagne_gold'
            
            avg_hue = np.mean(h[bright_mask])
            avg_sat = np.mean(s[bright_mask])
            
            # 보수적인 판정
            if avg_hue < 15 and avg_sat < 30:
                return 'white_gold'
            elif 8 <= avg_hue <= 25 and avg_sat > 40:
                return 'rose_gold'
            elif 20 <= avg_hue <= 35:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
                
        except:
            return 'champagne_gold'

    def _detect_lighting(self, image):
        """간소화된 조명 감지"""
        try:
            # LAB 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            avg_b = np.mean(b)
            
            if avg_b < 125:
                return 'cool'
            elif avg_b > 135:
                return 'warm'
            else:
                return 'natural'
                
        except:
            return 'natural'

    def _apply_white_overlay(self, image, strength):
        """하얀색 오버레이"""
        if strength <= 0:
            return image
            
        height, width = image.shape[:2]
        white_overlay = np.full((height, width, 3), 255, dtype=np.uint8)
        
        result = cv2.addWeighted(image, 1 - strength, white_overlay, strength, 0)
        return result

    def _adjust_color_temperature(self, image, delta_a, delta_b):
        """색온도 조정"""
        if delta_a == 0 and delta_b == 0:
            return image
            
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:, :, 1] += delta_a
            lab[:, :, 2] += delta_b
            
            lab = np.clip(lab, 0, 255)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return result
        except:
            return image

    def enhance_image(self, image_data):
        """메인 보정 함수"""
        try:
            # 이미지 읽기
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            # 크기 조정 (메모리 최적화)
            height, width = original.shape[:2]
            if width > 2048:
                ratio = 2048 / width
                new_width = 2048
                new_height = int(height * ratio)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # 노이즈 제거
            try:
                denoised = cv2.bilateralFilter(original, 9, 75, 75)
            except:
                denoised = original.copy()

            # 금속 타입과 조명 감지
            metal_type = self._detect_metal_type(denoised)
            lighting = self._detect_lighting(denoised)
            
            # 파라미터 가져오기
            params = self.metal_params.get(metal_type, self.metal_params['champagne_gold'])
            current_params = params.get(lighting, params['natural'])

            # PIL 보정
            pil_image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            # Brightness
            if current_params['brightness'] != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(current_params['brightness'])
            
            # Contrast
            if current_params['contrast'] != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(current_params['contrast'])
            
            # Sharpness
            if current_params['sharpness'] != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(current_params['sharpness'])

            # OpenCV로 변환
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 하얀색 오버레이
            if current_params['white_overlay'] > 0:
                enhanced = self._apply_white_overlay(enhanced, current_params['white_overlay'])

            # 색온도 조정
            if current_params['color_temp_a'] != 0 or current_params['color_temp_b'] != 0:
                enhanced = self._adjust_color_temperature(
                    enhanced, 
                    current_params['color_temp_a'], 
                    current_params['color_temp_b']
                )

            # 원본과 블렌딩
            blend_ratio = current_params['original_blend']
            if blend_ratio > 0:
                enhanced = cv2.addWeighted(enhanced, 1 - blend_ratio, denoised, blend_ratio, 0)

            # 간단한 하이라이트 부스팅
            try:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                
                threshold = np.percentile(l_channel, 85)
                bright_mask = l_channel >= threshold
                l_channel[bright_mask] = np.clip(l_channel[bright_mask] * 1.08, 0, 255)
                
                lab[:, :, 0] = l_channel
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except:
                pass

            # JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded_img = cv2.imencode('.jpg', enhanced, encode_param)
            
            return {
                'success': True,
                'image_data': encoded_img.tobytes(),
                'metal_type': metal_type,
                'lighting': lighting
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Flask 앱
enhancer = WeddingRingEnhancerV13_3()

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'version': 'v13.3-simplified',
        'description': '웨딩링 특화 AI 보정 시스템 (Railway 최적화)',
        'endpoints': [
            '/health',
            '/enhance_wedding_ring_advanced'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': 'v13.3-simplified',
        'message': 'Railway 최적화 완료'
    })

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'image_base64 필드가 필요합니다'}), 400

        image_data = base64.b64decode(data['image_base64'])
        result = enhancer.enhance_image(image_data)
        
        if result['success']:
            return result['image_data'], 200, {
                'Content-Type': 'image/jpeg',
                'X-Metal-Type': result['metal_type'],
                'X-Lighting': result['lighting'],
                'X-Version': 'v13.3-simplified'
            }
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 모든 백업 엔드포인트
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    return enhance_wedding_ring_advanced()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
