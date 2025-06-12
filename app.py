from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import time
from datetime import datetime

app = Flask(__name__)

class AdvancedWeddingRingEnhancer:
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

    def extract_ring_region_enhanced(self, image):
        """커플링 영역 확대 - 더 정확한 감지"""
        try:
            h, w = image.shape[:2]
            
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)
            
            # 1. 금속 반사 영역 감지 (확대)
            bright_mask = v_channel > 100  # 기존 120에서 100으로 낮춤
            
            # 2. 금속 색상 범위 확대
            metal_mask1 = (h_channel < 40) | (h_channel > 140)  # 금색/은색/로즈골드 범위 확대
            metal_mask2 = (h_channel >= 40) & (h_channel <= 140) & (s_channel < 120)  # 화이트골드 범위 확대
            metal_mask = metal_mask1 | metal_mask2
            
            # 3. 채도 조건 완화
            saturation_mask = (s_channel > 10) & (s_channel < 250)  # 범위 확대
            
            # 4. 모든 조건 결합
            ring_mask = bright_mask & metal_mask & saturation_mask
            
            # 5. 형태학적 연산으로 노이즈 제거 및 영역 확대
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 크기 증가
            ring_mask = cv2.morphologyEx(ring_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
            
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel_open)
            
            # 6. 팽창 연산으로 영역 더 확대
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            ring_mask = cv2.dilate(ring_mask, kernel_dilate, iterations=2)
            
            # 7. 연결 컴포넌트 분석 - 가장 큰 영역들 유지 (여러 개 링 고려)
            contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 면적 기준으로 정렬
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                ring_mask = np.zeros_like(ring_mask)
                
                # 상위 2개 윤곽선 유지 (커플링 2개)
                for i, contour in enumerate(contours[:2]):
                    if cv2.contourArea(contour) > 1000:  # 최소 면적 조건
                        cv2.fillPoly(ring_mask, [contour], 255)
            
            # 8. 경계 부드럽게 처리 (테두리 아티팩트 해결)
            ring_mask = cv2.GaussianBlur(ring_mask.astype(np.float32), (5, 5), 2)
            ring_mask = (ring_mask > 0.3).astype(np.uint8) * 255
            
            return ring_mask.astype(bool)
            
        except Exception as e:
            print(f"Enhanced ring extraction error: {e}")
            # 실패시 중앙 영역들을 링으로 가정
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            # 2개 링 가정
            cv2.circle(mask, (w//3, h//2), min(w, h)//6, True, -1)
            cv2.circle(mask, (2*w//3, h//2), min(w, h)//6, True, -1)
            return mask

    def detect_surface_finish(self, ring_region):
        """유무광 재질 감지"""
        try:
            if ring_region.size == 0:
                return 'polished'
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
            
            # Laplacian 분산으로 텍스처 분석
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 히스토그램 분석
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # 밝은 픽셀 비율 계산
            bright_ratio = np.sum(hist[200:]) / np.sum(hist)
            
            # 표준편차 계산
            std_dev = np.std(gray)
            
            # 유광/무광 판단
            if bright_ratio > 0.15 and std_dev > 50:
                return 'polished'  # 유광 (반사가 강함)
            else:
                return 'matte'     # 무광 (반사가 약함)
                
        except Exception as e:
            print(f"Surface finish detection error: {e}")
            return 'polished'

    def enhance_cubic_details(self, ring_region):
        """큐빅/밀그레인 디테일 강화"""
        try:
            if ring_region.size == 0:
                return ring_region
            
            # 1. 매우 밝은 영역 감지 (큐빅/다이아몬드)
            gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
            cubic_mask = gray > 220
            
            # 2. 작은 연결 컴포넌트 찾기 (작은 큐빅들)
            contours, _ = cv2.findContours(cubic_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detail_enhanced = ring_region.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 500:  # 작은 큐빅 크기 범위
                    # 마스크 생성
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    
                    # 해당 영역만 추출
                    cubic_region = ring_region.copy()
                    cubic_region[mask == 0] = 0
                    
                    # 큐빅 영역 강화
                    enhanced_cubic = self.enhance_cubic_sparkle(cubic_region, mask)
                    
                    # 원본에 적용
                    detail_enhanced[mask > 0] = enhanced_cubic[mask > 0]
            
            # 3. 밀그레인 패턴 강화 (가장자리 세밀한 패턴)
            detail_enhanced = self.enhance_milgrain_pattern(detail_enhanced)
            
            return detail_enhanced
            
        except Exception as e:
            print(f"Cubic detail enhancement error: {e}")
            return ring_region

    def enhance_cubic_sparkle(self, cubic_region, mask):
        """개별 큐빅 반짝임 강화"""
        try:
            # LAB 색공간에서 밝기 강화
            lab = cv2.cvtColor(cubic_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # L 채널 강화 (밝기)
            l[mask > 0] = np.clip(l[mask > 0] * 1.3, 0, 255)
            
            # 대비 강화
            l[mask > 0] = np.clip((l[mask > 0] - 128) * 1.4 + 128, 0, 255)
            
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 언샤프 마스킹으로 선명도 극대화
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Cubic sparkle error: {e}")
            return cubic_region

    def enhance_milgrain_pattern(self, ring_region):
        """밀그레인 패턴 강화"""
        try:
            # 가장자리 감지
            gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 가장자리 주변 선명도 강화
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(ring_region, -1, kernel)
            
            # 가장자리에만 적용
            edge_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            enhanced = ring_region.copy()
            enhanced[edge_dilated > 0] = sharpened[edge_dilated > 0]
            
            return enhanced
            
        except Exception as e:
            print(f"Milgrain enhancement error: {e}")
            return ring_region

    def enhance_ring_professional(self, ring_region, surface_finish):
        """유무광 재질에 따른 프로페셔널 보정"""
        try:
            if ring_region.size == 0:
                return ring_region
            
            # 1. 노이즈 제거 (bilateral filter)
            denoised = cv2.bilateralFilter(ring_region, 9, 75, 75)
            
            # 2. 유무광에 따른 차별 보정
            if surface_finish == 'polished':
                # 유광: 반사와 대비 강화
                enhanced = self.enhance_polished_surface(denoised)
            else:
                # 무광: 질감과 균일성 강화
                enhanced = self.enhance_matte_surface(denoised)
            
            # 3. 밝은 조명을 받은 것처럼 보정
            brightened = self.simulate_bright_lighting(enhanced)
            
            # 4. 큐빅/밀그레인 디테일 강화
            detail_enhanced = self.enhance_cubic_details(brightened)
            
            # 5. 적당한 선명도 조절
            final = self.adjust_product_sharpness(detail_enhanced)
            
            return final
            
        except Exception as e:
            print(f"Professional ring enhancement error: {e}")
            return ring_region

    def enhance_polished_surface(self, image):
        """유광 표면 강화"""
        try:
            # 대비 강화
            enhanced = cv2.convertScaleAbs(image, alpha=1.15, beta=5)
            
            # 반사 영역 강화
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 밝은 부분 더 밝게
            l = np.where(l > 180, np.clip(l * 1.1, 0, 255), l)
            
            enhanced_lab = cv2.merge([l.astype(np.uint8), a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"Polished enhancement error: {e}")
            return image

    def enhance_matte_surface(self, image):
        """무광 표면 강화"""
        try:
            # 균일성 강화 (CLAHE)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 질감 보존하면서 밝기 조정
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.08, beta=8)
            
            return enhanced
            
        except Exception as e:
            print(f"Matte enhancement error: {e}")
            return image

    def simulate_bright_lighting(self, image):
        """밝은 조명을 받은 것처럼 시뮬레이션"""
        try:
            # 전체적인 밝기 향상
            brightened = cv2.convertScaleAbs(image, alpha=1.12, beta=15)
            
            # 하이라이트 영역 강화
            lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 감마 보정으로 자연스러운 밝기
            gamma = 0.9
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
            l = cv2.LUT(l, lookup_table)
            
            enhanced_lab = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"Bright lighting simulation error: {e}")
            return image

    def adjust_product_sharpness(self, image):
        """제품 선명도 적당히 조절"""
        try:
            # 언샤프 마스킹
            gaussian = cv2.GaussianBlur(image, (0, 0), 1.5)
            sharpened = cv2.addWeighted(image, 1.4, gaussian, -0.4, 0)
            
            return sharpened
            
        except Exception as e:
            print(f"Sharpness adjustment error: {e}")
            return image

    def apply_final_color_grading(self, image, metal_type, lighting):
        """전체 이미지 after 수준 색감 조정"""
        try:
            params = self.metal_params[metal_type][lighting]
            
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 1. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(params['brightness'])
            
            # 2. 대비 조정
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(params['contrast'])
            
            # 3. 색온도 조정 (warmth)
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
            
            # OpenCV로 변환
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 5. CLAHE (명료도)
            if params['clarity'] != 1.0:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0*params['clarity'], tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            # 6. 감마 보정
            if params['gamma'] != 1.0:
                gamma = params['gamma']
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, lookup_table)
            
            return enhanced
            
        except Exception as e:
            print(f"Final color grading error: {e}")
            return image

    def process_image_advanced(self, image):
        """새로운 고급 보정 프로세스"""
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
            
            # 2. 커플링 영역 확대 감지
            ring_mask = self.extract_ring_region_enhanced(image)
            
            # 3. 링 영역 추출
            ring_region = image.copy()
            ring_region[~ring_mask] = 0
            
            # 4. 유무광 재질 분석
            surface_finish = self.detect_surface_finish(ring_region)
            print(f"Surface finish: {surface_finish}")
            
            # 5. 커플링 프로페셔널 보정
            ring_enhanced = self.enhance_ring_professional(ring_region, surface_finish)
            
            # 6. 링 영역을 원본에 합성 (부드러운 블렌딩)
            result = image.copy()
            
            # 마스크를 부드럽게 처리 (테두리 아티팩트 완전 제거)
            mask_float = ring_mask.astype(np.float32)
            mask_blurred = cv2.GaussianBlur(mask_float, (7, 7), 2)
            
            for c in range(3):
                result[:,:,c] = (ring_enhanced[:,:,c] * mask_blurred + 
                               image[:,:,c] * (1 - mask_blurred))
            
            # 7. 전체 이미지 after 수준 색감 조정
            final_result = self.apply_final_color_grading(result, metal_type, lighting)
            
            processing_time = time.time() - start_time
            print(f"Advanced processing time: {processing_time:.2f}s")
            
            return final_result, {
                'metal_type': metal_type,
                'lighting': lighting,
                'surface_finish': surface_finish,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"Advanced processing error: {e}")
            return image, {'error': str(e)}

# Flask 앱 엔드포인트들
enhancer = AdvancedWeddingRingEnhancer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v4.0_advanced'
    })

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """새로운 고급 보정 엔드포인트"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 고급 보정 처리
        enhanced_image, metadata = enhancer.process_image_advanced(image)
        
        # JPEG로 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        # 바이너리 데이터로 직접 반환
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'advanced_enhanced_{int(time.time())}.jpg'
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """기존 영역별 차별 보정 엔드포인트 (백업용)"""
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 고급 보정 처리 (동일한 로직)
        enhanced_image, metadata = enhancer.process_image_advanced(image)
        
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
