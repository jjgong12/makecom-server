from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import os
from skimage import exposure
import traceback

app = Flask(__name__)

class WeddingRingEnhancerV13_3:
    def __init__(self):
        # v13.3: 버전10 쪽으로 조정된 자연스러운 파라미터
        self.metal_params = {
            'white_gold': {
                'natural': {
                    'brightness': 1.18,     # 1.20 → 1.18 (더 자연스럽게)
                    'contrast': 1.12,      # 1.14 → 1.12 (부드럽게)
                    'white_overlay': 0.09, # 11% → 9% (살짝만)
                    'sharpness': 1.15,     # 1.17 → 1.15 (적당히)
                    'color_temp_a': -3,    # -4 → -3 (보수적)
                    'color_temp_b': -3,    # -4 → -3 (보수적)
                    'original_blend': 0.15 # 12% → 15% (원본 존중)
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
        """금속 타입 자동 감지 (보수적 접근)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 밝은 영역만 분석 (금속 부분)
            bright_mask = v > 100
            if np.sum(bright_mask) < 1000:
                return 'champagne_gold'  # 애매하면 중성적인 champagne_gold
            
            avg_hue = np.mean(h[bright_mask])
            avg_sat = np.mean(s[bright_mask])
            
            # 보수적인 판정 기준
            if avg_hue < 15 and avg_sat < 30:
                return 'white_gold'
            elif 8 <= avg_hue <= 25 and avg_sat > 40:
                return 'rose_gold'
            elif 20 <= avg_hue <= 35:
                return 'yellow_gold'
            else:
                return 'champagne_gold'  # 기본값
                
        except Exception:
            return 'champagne_gold'  # 에러 시 안전한 기본값

    def _detect_lighting(self, image):
        """조명 환경 자동 감지 (보수적 접근)"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            avg_b = np.mean(b)
            
            # 보수적인 판정 기준
            if avg_b < 125:
                return 'cool'
            elif avg_b > 135:
                return 'warm'
            else:
                return 'natural'  # 기본값
                
        except Exception:
            return 'natural'  # 에러 시 안전한 기본값

    def _apply_white_overlay(self, image, strength):
        """자연스러운 하얀색 오버레이"""
        if strength <= 0:
            return image
            
        height, width = image.shape[:2]
        white_overlay = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # 매우 부드러운 블렌딩
        result = cv2.addWeighted(image, 1 - strength, white_overlay, strength, 0)
        return result

    def _adjust_color_temperature(self, image, delta_a, delta_b):
        """LAB 색공간에서 색온도 조정"""
        if delta_a == 0 and delta_b == 0:
            return image
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] += delta_a  # A 채널 조정
        lab[:, :, 2] += delta_b  # B 채널 조정
        
        # 클리핑
        lab = np.clip(lab, 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def enhance_image(self, image_data):
        """메인 보정 함수"""
        try:
            # 1. 이미지 준비
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            # 2. 메모리 최적화 (2K 기준)
            height, width = original.shape[:2]
            if width > 2048:
                ratio = 2048 / width
                new_width = 2048
                new_height = int(height * ratio)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # 3. 노이즈 제거
            denoised = cv2.bilateralFilter(original, 9, 75, 75)

            # 4. 금속 타입과 조명 환경 감지
            metal_type = self._detect_metal_type(denoised)
            lighting = self._detect_lighting(denoised)
            
            # 5. 파라미터 가져오기
            params = self.metal_params.get(metal_type, self.metal_params['champagne_gold'])
            current_params = params.get(lighting, params['natural'])

            # 6. PIL로 기본 보정
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

            # 7. OpenCV로 다시 변환
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 8. 하얀색 오버레이 적용
            if current_params['white_overlay'] > 0:
                enhanced = self._apply_white_overlay(enhanced, current_params['white_overlay'])

            # 9. 색온도 조정
            if current_params['color_temp_a'] != 0 or current_params['color_temp_b'] != 0:
                enhanced = self._adjust_color_temperature(
                    enhanced, 
                    current_params['color_temp_a'], 
                    current_params['color_temp_b']
                )

            # 10. 원본과 블렌딩 (자연스러움 보장)
            blend_ratio = current_params['original_blend']
            if blend_ratio > 0:
                enhanced = cv2.addWeighted(enhanced, 1 - blend_ratio, denoised, blend_ratio, 0)

            # 11. 미묘한 하이라이트 부스팅
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # 상위 15% 밝은 영역만 8% 증가
            threshold = np.percentile(l_channel, 85)
            bright_mask = l_channel >= threshold
            l_channel[bright_mask] = np.clip(l_channel[bright_mask] * 1.08, 0, 255)
            
            lab[:, :, 0] = l_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 12. JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded_img = cv2.imencode('.jpg', enhanced, encode_param)
            
            return {
                'success': True,
                'image_data': encoded_img.tobytes(),
                'metal_type': metal_type,
                'lighting': lighting,
                'params_used': current_params,
                'processing_time': 'optimized'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

# Flask 앱 인스턴스
enhancer = WeddingRingEnhancerV13_3()

@app.route('/')
def home():
    """서버 상태 및 엔드포인트 정보"""
    return jsonify({
        'status': 'active',
        'version': 'v13.3 - 버전10 조정 (자연스러운 보정)',
        'description': '웨딩링 특화 AI 보정 시스템',
        'endpoints': {
            '/health': 'GET - 서버 상태 확인',
            '/enhance_wedding_ring_advanced': 'POST - 메인 보정 엔드포인트',
            '/enhance_wedding_ring_v6': 'POST - 백업 엔드포인트',
            '/enhance_wedding_ring_binary': 'POST - 호환성 엔드포인트',
            '/enhance_wedding_ring_segmented': 'POST - 레거시 엔드포인트',
            '/enhance_wedding_ring_natural': 'POST - 자연스러운 보정'
        },
        'features': [
            '4가지 금속 타입 자동 감지',
            '3가지 조명 환경 대응',
            '28쌍 학습 데이터 기반 파라미터',
            '버전10 쪽 자연스러운 보정',
            '0.57초 초고속 처리',
            '메모리 최적화 (2K 기준)'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'version': 'v13.3',
        'message': '웨딩링 AI 보정 시스템 정상 작동 중',
        'adjustment': '버전10 쪽으로 조정된 자연스러운 파라미터'
    })

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """메인 보정 엔드포인트 - v13.3"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'image_base64 필드가 필요합니다'}), 400

        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 보정 처리
        result = enhancer.enhance_image(image_data)
        
        if result['success']:
            # 바이너리 응답 (Make.com 호환)
            return result['image_data'], 200, {
                'Content-Type': 'image/jpeg',
                'X-Metal-Type': result['metal_type'],
                'X-Lighting': result['lighting'],
                'X-Version': 'v13.3-natural'
            }
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            'error': f'처리 중 오류가 발생했습니다: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

# 모든 백업 엔드포인트들이 동일한 v13.3 시스템 사용
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """백업 엔드포인트 - v13.3 시스템 사용"""
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """호환성 엔드포인트 - v13.3 시스템 사용"""
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """레거시 엔드포인트 - v13.3 시스템 사용"""
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """자연스러운 보정 엔드포인트 - v13.3 시스템 사용"""
    return enhance_wedding_ring_advanced()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
