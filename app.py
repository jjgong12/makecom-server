from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import io
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class WeddingRingEnhancer:
    """Claude 분석 기반 웨딩링 보정 엔진"""
    
    def __init__(self):
        # 기본 파라미터 (Claude 분석 후 업데이트 예정)
        self.default_params = {
            'brightness': 1.2,
            'contrast': 1.1,
            'warmth': 1.05,
            'saturation': 1.0,
            'sharpness': 1.3,
            'clarity': 1.1,
            'gamma': 1.1
        }
        
        # 링 타입별 특화 파라미터
        self.ring_specific_params = {
            'gold': {
                'warmth': 1.15,
                'brightness': 1.2,
                'saturation': 1.1
            },
            'silver': {
                'warmth': 0.95,
                'brightness': 1.25,
                'clarity': 1.2
            },
            'rose_gold': {
                'warmth': 1.1,
                'brightness': 1.15,
                'saturation': 1.05
            },
            'platinum': {
                'warmth': 1.0,
                'brightness': 1.2,
                'clarity': 1.25
            }
        }
        
        # 학습 데이터 저장
        self.learning_data = []
    
    def detect_ring_type(self, image):
        """링 타입 자동 감지"""
        try:
            # HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 색상 히스토그램 분석
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # 채도 평균
            avg_saturation = np.mean(hsv[:,:,1])
            
            # 주요 색상 범위 분석
            yellow_range = np.sum(hist_h[15:35])  # 노란색 계열
            red_range = np.sum(hist_h[0:15])      # 빨간색 계열
            
            # 분류 로직
            if avg_saturation < 50:
                return 'silver' if np.mean(hsv[:,:,2]) > 100 else 'platinum'
            elif yellow_range > red_range:
                return 'gold'
            elif red_range > 0:
                return 'rose_gold'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Ring type detection failed: {e}")
            return 'unknown'
    
    def detect_lighting_condition(self, image):
        """조명 환경 감지"""
        try:
            # 색온도 추정
            b, g, r = cv2.split(image)
            
            avg_r = np.mean(r)
            avg_b = np.mean(b)
            
            # R/B 비율로 색온도 추정
            if avg_r / avg_b > 1.2:
                return 'warm'  # 따뜻한 조명 (텅스텐)
            elif avg_r / avg_b < 0.8:
                return 'cool'  # 차가운 조명 (형광등)
            else:
                return 'natural'  # 자연광
                
        except Exception as e:
            logger.error(f"Lighting detection failed: {e}")
            return 'unknown'
    
    def enhance(self, image, custom_params=None, ring_type=None):
        """메인 보정 함수"""
        try:
            # 링 타입 자동 감지 (제공되지 않은 경우)
            if ring_type is None:
                ring_type = self.detect_ring_type(image)
            
            # 조명 환경 감지
            lighting = self.detect_lighting_condition(image)
            
            # 파라미터 결정
            if custom_params:
                params = custom_params
            else:
                params = self.get_optimal_params(ring_type, lighting)
            
            # 단계별 보정 실행
            enhanced = self._prepare_image(image)
            enhanced = self._adjust_brightness_contrast(enhanced, params)
            enhanced = self._adjust_warmth(enhanced, params)
            enhanced = self._adjust_saturation(enhanced, params)
            enhanced = self._enhance_sharpness(enhanced, params)
            enhanced = self._enhance_clarity(enhanced, params)
            enhanced = self._apply_gamma_correction(enhanced, params)
            
            return enhanced, params, ring_type, lighting
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image, self.default_params, 'unknown', 'unknown'
    
    def get_optimal_params(self, ring_type, lighting):
        """최적 파라미터 조합 계산"""
        # 기본 파라미터에서 시작
        params = self.default_params.copy()
        
        # 링 타입별 조정
        if ring_type in self.ring_specific_params:
            ring_params = self.ring_specific_params[ring_type]
            for key, value in ring_params.items():
                params[key] = value
        
        # 조명별 미세 조정
        if lighting == 'warm':
            params['warmth'] *= 0.9  # 따뜻한 조명에서는 warmth 줄이기
        elif lighting == 'cool':
            params['warmth'] *= 1.1  # 차가운 조명에서는 warmth 높이기
        
        return params
    
    def _prepare_image(self, image):
        """이미지 전처리"""
        return cv2.convertScaleAbs(image)
    
    def _adjust_brightness_contrast(self, image, params):
        """밝기/대비 조정"""
        brightness = params.get('brightness', 1.0)
        contrast = params.get('contrast', 1.0)
        
        # alpha: 대비, beta: 밝기
        alpha = contrast
        beta = (brightness - 1.0) * 30
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def _adjust_warmth(self, image, params):
        """색온도 조정"""
        warmth = params.get('warmth', 1.0)
        
        if abs(warmth - 1.0) < 0.01:
            return image
        
        b, g, r = cv2.split(image.astype(np.float32))
        
        if warmth > 1.0:  # 따뜻하게
            r = np.clip(r * warmth, 0, 255)
            b = np.clip(b * (2.0 - warmth), 0, 255)
        else:  # 차갑게
            r = np.clip(r * warmth, 0, 255)
            b = np.clip(b * (2.0 - warmth), 0, 255)
        
        return cv2.merge([b, g, r]).astype(np.uint8)
    
    def _adjust_saturation(self, image, params):
        """채도 조정"""
        saturation = params.get('saturation', 1.0)
        
        if abs(saturation - 1.0) < 0.01:
            return image
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _enhance_sharpness(self, image, params):
        """선명도 향상 (언샤프 마스킹)"""
        sharpness = params.get('sharpness', 1.0)
        
        if abs(sharpness - 1.0) < 0.01:
            return image
        
        # 가우시안 블러 적용
        sigma = max(0.5, 2.0 - sharpness)
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 언샤프 마스킹
        amount = sharpness
        unsharp = cv2.addWeighted(image, 1 + amount, gaussian, -amount, 0)
        
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def _enhance_clarity(self, image, params):
        """국지적 대비 향상 (CLAHE)"""
        clarity = params.get('clarity', 1.0)
        
        if abs(clarity - 1.0) < 0.01:
            return image
        
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L 채널에 CLAHE 적용
        clip_limit = max(1.0, clarity * 2.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_gamma_correction(self, image, params):
        """감마 보정"""
        gamma = params.get('gamma', 1.0)
        
        if abs(gamma - 1.0) < 0.01:
            return image
        
        # 감마 보정 룩업 테이블 생성
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(image, table)

# 글로벌 인스턴스
enhancer = WeddingRingEnhancer()

# 구글시트 연동 (선택사항 - 크리덴셜 파일 필요)
def init_google_sheets():
    """구글시트 초기화 (크리덴셜 파일이 있는 경우)"""
    try:
        # service_account.json 파일이 있는 경우
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file('service_account.json', scopes=scope)
        client = gspread.authorize(creds)
        return client.open('Wedding_Ring_Enhancement_Data').sheet1
    except:
        logger.warning("Google Sheets not initialized - using local storage")
        return None

# 구글시트 인스턴스 (선택사항)
try:
    sheet = init_google_sheets()
except:
    sheet = None

def log_enhancement_data(image_id, params, ring_type, lighting, quality_score=None, user_satisfaction=None):
    """보정 데이터 로깅"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'image_id': image_id,
        'ring_type': ring_type,
        'lighting': lighting,
        'brightness': params.get('brightness'),
        'contrast': params.get('contrast'),
        'warmth': params.get('warmth'),
        'saturation': params.get('saturation'),
        'sharpness': params.get('sharpness'),
        'clarity': params.get('clarity'),
        'gamma': params.get('gamma'),
        'quality_score': quality_score,
        'user_satisfaction': user_satisfaction
    }
    
    # 구글시트에 저장 (가능한 경우)
    if sheet:
        try:
            row = [
                log_data['timestamp'], log_data['image_id'], log_data['ring_type'],
                log_data['lighting'], log_data['brightness'], log_data['contrast'],
                log_data['warmth'], log_data['saturation'], log_data['sharpness'],
                log_data['clarity'], log_data['gamma'], log_data['quality_score'],
                log_data['user_satisfaction']
            ]
            sheet.append_row(row)
        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")
    
    # 로컬에도 저장
    enhancer.learning_data.append(log_data)
    
    return log_data

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "message": "Claude 분석 기반 웨딩링 보정 서버 가동 중",
        "version": "1.0.0",
        "total_processed": len(enhancer.learning_data)
    })

@app.route('/enhance_wedding_ring', methods=['POST'])
def enhance_wedding_ring():
    """웨딩링 이미지 보정 메인 API"""
    try:
        data = request.get_json()
        
        # 필수 데이터 확인
        if 'image_base64' not in data:
            return jsonify({"error": "image_base64 필드가 필요합니다"}), 400
        
        # 이미지 디코딩
        try:
            image_data = base64.b64decode(data['image_base64'])
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "이미지 디코딩 실패"}), 400
                
        except Exception as e:
            return jsonify({"error": f"이미지 처리 실패: {str(e)}"}), 400
        
        # 메타데이터 추출
        image_id = data.get('image_id', f'img_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        custom_params = data.get('custom_parameters', None)
        ring_type = data.get('ring_type', None)
        
        # 보정 실행
        enhanced_image, used_params, detected_ring_type, detected_lighting = enhancer.enhance(
            image, custom_params, ring_type
        )
        
        # 업스케일링 (옵션)
        if data.get('upscale', False):
            scale_factor = data.get('scale_factor', 2.0)
            height, width = enhanced_image.shape[:2]
            new_size = (int(width * scale_factor), int(height * scale_factor))
            enhanced_image = cv2.resize(enhanced_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 결과 인코딩
        _, buffer = cv2.imencode('.jpg', enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 데이터 로깅
        log_data = log_enhancement_data(
            image_id, used_params, detected_ring_type, detected_lighting
        )
        
        return jsonify({
            "enhanced_image": result_base64,
            "parameters_used": used_params,
            "detected_ring_type": detected_ring_type,
            "detected_lighting": detected_lighting,
            "image_id": image_id,
            "log_data": log_data,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Enhancement API error: {e}")
        return jsonify({"error": f"서버 내부 오류: {str(e)}"}), 500

@app.route('/analyze_b001_style', methods=['POST'])
def analyze_b001_style():
    """B_001 완성본들의 스타일 분석 (선택적 기능)"""
    try:
        data = request.get_json()
        
        # 여러 B_001 완성본들 분석
        b001_images = data.get('b001_images', [])
        
        if not b001_images:
            return jsonify({"error": "분석할 B_001 이미지들이 필요합니다"}), 400
        
        style_analysis = []
        
        for img_data in b001_images:
            image_data = base64.b64decode(img_data)
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            if image is not None:
                # 스타일 특성 분석
                brightness_level = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                contrast_level = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation_level = np.mean(hsv[:,:,1])
                
                style_analysis.append({
                    'brightness': float(brightness_level),
                    'contrast': float(contrast_level),
                    'saturation': float(saturation_level)
                })
        
        if style_analysis:
            # 평균 스타일 특성 계산
            avg_style = {
                'target_brightness': np.mean([s['brightness'] for s in style_analysis]),
                'target_contrast': np.mean([s['contrast'] for s in style_analysis]),
                'target_saturation': np.mean([s['saturation'] for s in style_analysis])
            }
            
            return jsonify({
                "b001_style_analysis": style_analysis,
                "average_style_target": avg_style,
                "success": True
            })
        else:
            return jsonify({"error": "유효한 이미지가 없습니다"}), 400
        
    except Exception as e:
        logger.error(f"B001 style analysis error: {e}")
        return jsonify({"error": f"B001 스타일 분석 실패: {str(e)}"}), 500

@app.route('/batch_test_parameters', methods=['POST'])
def batch_test_parameters():
    """여러 파라미터로 동시 테스트 (Claude 분석용)"""
    try:
        data = request.get_json()
        
        # 이미지 디코딩
        image_data = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "이미지 디코딩 실패"}), 400
        
        # 여러 파라미터 세트
        parameter_sets = data['parameter_sets']
        results = {}
        
        for i, params in enumerate(parameter_sets):
            enhanced, used_params, ring_type, lighting = enhancer.enhance(image, params)
            
            _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            results[f'result_{i+1}'] = {
                'image': result_base64,
                'parameters': used_params,
                'ring_type': ring_type,
                'lighting': lighting
            }
        
        return jsonify({
            "results": results,
            "success": True,
            "message": f"{len(parameter_sets)}개 파라미터 조합으로 테스트 완료"
        })
        
    except Exception as e:
        logger.error(f"Batch test error: {e}")
        return jsonify({"error": f"배치 테스트 실패: {str(e)}"}), 500

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    """Claude 분석 결과로 파라미터 업데이트"""
    try:
        data = request.get_json()
        
        # 새로운 기본 파라미터
        if 'default_params' in data:
            enhancer.default_params.update(data['default_params'])
        
        # 링 타입별 파라미터
        if 'ring_specific_params' in data:
            enhancer.ring_specific_params.update(data['ring_specific_params'])
        
        return jsonify({
            "message": "파라미터가 성공적으로 업데이트되었습니다",
            "current_default_params": enhancer.default_params,
            "current_ring_params": enhancer.ring_specific_params,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Parameter update error: {e}")
        return jsonify({"error": f"파라미터 업데이트 실패: {str(e)}"}), 500

@app.route('/get_learning_data', methods=['GET'])
def get_learning_data():
    """축적된 학습 데이터 조회 (Claude 분석용)"""
    try:
        # 최근 30일 데이터만
        recent_data = []
        thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        
        for data_point in enhancer.learning_data:
            try:
                timestamp = datetime.fromisoformat(data_point['timestamp']).timestamp()
                if timestamp > thirty_days_ago:
                    recent_data.append(data_point)
            except:
                continue
        
        # 통계 계산
        if recent_data:
            total_images = len(recent_data)
            avg_satisfaction = np.mean([d.get('user_satisfaction', 0) for d in recent_data if d.get('user_satisfaction')])
            
            # 성공 사례 (만족도 8 이상)
            success_cases = [d for d in recent_data if d.get('user_satisfaction', 0) >= 8]
            success_rate = len(success_cases) / total_images if total_images > 0 else 0
        else:
            total_images = 0
            avg_satisfaction = 0
            success_rate = 0
        
        return jsonify({
            "recent_data": recent_data,
            "statistics": {
                "total_images": total_images,
                "avg_satisfaction": round(avg_satisfaction, 2),
                "success_rate": round(success_rate * 100, 1)
            },
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Get learning data error: {e}")
        return jsonify({"error": f"학습 데이터 조회 실패: {str(e)}"}), 500

@app.route('/feedback', methods=['POST'])
def record_feedback():
    """사용자 피드백 기록"""
    try:
        data = request.get_json()
        
        image_id = data.get('image_id')
        quality_score = data.get('quality_score')  # 1-10
        user_satisfaction = data.get('user_satisfaction')  # 1-10
        selected = data.get('selected', False)
        
        # 기존 데이터 업데이트
        for i, log_data in enumerate(enhancer.learning_data):
            if log_data.get('image_id') == image_id:
                enhancer.learning_data[i]['quality_score'] = quality_score
                enhancer.learning_data[i]['user_satisfaction'] = user_satisfaction
                enhancer.learning_data[i]['selected'] = selected
                break
        
        return jsonify({
            "message": "피드백이 성공적으로 기록되었습니다",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Feedback recording error: {e}")
        return jsonify({"error": f"피드백 기록 실패: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
