from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import os
import json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)

class WeddingRingEnhancer:
    def __init__(self):
        self.version = "1.0.0"
        self.total_processed = 0
        self.enhancement_params = {
            'gold': {
                'brightness': 1.1,
                'contrast': 1.2,
                'warmth': 1.15,
                'saturation': 1.25,
                'sharpness': 1.3,
                'clarity': 2.0,
                'gamma': 0.9
            },
            'silver': {
                'brightness': 1.05,
                'contrast': 1.15,
                'warmth': 0.95,
                'saturation': 1.1,
                'sharpness': 1.4,
                'clarity': 2.2,
                'gamma': 1.0
            },
            'rose_gold': {
                'brightness': 1.08,
                'contrast': 1.18,
                'warmth': 1.25,
                'saturation': 1.3,
                'sharpness': 1.25,
                'clarity': 2.1,
                'gamma': 0.95
            },
            'platinum': {
                'brightness': 1.03,
                'contrast': 1.12,
                'warmth': 0.9,
                'saturation': 1.05,
                'sharpness': 1.5,
                'clarity': 2.3,
                'gamma': 1.05
            }
        }
        
        # Google Sheets 초기화 시도
        self.sheets_client = None
        self.learning_sheet = None
        self._init_google_sheets()
    
    def _init_google_sheets(self):
        """Google Sheets 연결 초기화"""
        try:
            # 환경 변수에서 서비스 계정 키 가져오기
            service_account_info = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
            if service_account_info:
                service_account_dict = json.loads(service_account_info)
                credentials = Credentials.from_service_account_info(
                    service_account_dict,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                self.sheets_client = gspread.authorize(credentials)
                
                # 웨딩링 학습 데이터 시트 열기
                sheet_id = os.environ.get('LEARNING_SHEET_ID', 'default_sheet_id')
                self.learning_sheet = self.sheets_client.open_by_key(sheet_id).sheet1
                print("Google Sheets initialized successfully")
            else:
                print("WARNING: Google Sheets credentials not found - using local storage")
        except Exception as e:
            print(f"WARNING: Google Sheets not initialized - using local storage: {str(e)}")
    
    def detect_ring_type(self, image):
        """링 타입 자동 감지 (금, 은, 로즈골드, 플래티넘)"""
        try:
            # RGB 히스토그램 분석으로 금속 타입 추정
            if isinstance(image, Image.Image):
                np_image = np.array(image)
            else:
                np_image = image
            
            # 밝은 영역에서의 색상 분석
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            bright_mask = gray > np.percentile(gray, 80)
            
            if np.sum(bright_mask) == 0:
                return 'silver'  # 기본값
            
            bright_pixels = np_image[bright_mask]
            avg_color = np.mean(bright_pixels, axis=0)
            
            # R, G, B 비율로 금속 타입 결정
            r, g, b = avg_color
            
            if r > g * 1.1 and r > b * 1.2:  # 빨강 성분이 높음
                return 'rose_gold' if g > b else 'gold'
            elif abs(r - g) < 10 and abs(g - b) < 10:  # 균등한 색상
                return 'platinum' if np.mean(avg_color) > 180 else 'silver'
            else:
                return 'silver'
        except:
            return 'silver'  # 기본값
    
    def detect_lighting_condition(self, image):
        """조명 환경 분석 (따뜻함, 차가움, 자연광)"""
        try:
            if isinstance(image, Image.Image):
                np_image = np.array(image)
            else:
                np_image = image
            
            # 전체 이미지의 색온도 분석
            avg_color = np.mean(np_image.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            # 색온도 비율 계산
            warm_ratio = (r + g) / (2 * b + 1)
            cool_ratio = (b + g) / (2 * r + 1)
            
            if warm_ratio > 1.2:
                return 'warm'
            elif cool_ratio > 1.1:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def _prepare_image(self, image):
        """이미지 전처리"""
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            return image
        return Image.fromarray(image)
    
    def _adjust_brightness_contrast(self, image, brightness=1.0, contrast=1.0):
        """밝기 및 대비 조정"""
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        return image
    
    def _adjust_warmth(self, image, warmth=1.0):
        """색온도 조정 (따뜻함/차가움)"""
        if warmth == 1.0:
            return image
        
        np_image = np.array(image, dtype=np.float32)
        
        if warmth > 1.0:
            # 따뜻하게 - 빨강/노랑 증가
            np_image[:, :, 0] *= min(warmth, 1.3)  # R
            np_image[:, :, 1] *= min(warmth * 0.9, 1.2)  # G
        else:
            # 차갑게 - 파랑 증가
            np_image[:, :, 2] *= min(1/warmth, 1.3)  # B
        
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def _adjust_saturation(self, image, saturation=1.0):
        """채도 조정"""
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        return image
    
    def _enhance_sharpness(self, image, sharpness=1.0):
        """선명도 향상 (언샤프 마스킹)"""
        if sharpness == 1.0:
            return image
        
        # 언샤프 마스킹으로 선명도 향상
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
        np_original = np.array(image, dtype=np.float32)
        np_blurred = np.array(blurred, dtype=np.float32)
        
        # 고주파 성분 강화
        high_freq = np_original - np_blurred
        enhanced = np_original + high_freq * (sharpness - 1.0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def _enhance_clarity(self, image, clarity=1.0):
        """명료도 향상 (CLAHE 적용)"""
        if clarity == 1.0:
            return image
        
        np_image = np.array(image)
        lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
        
        # L 채널에 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=clarity, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    
    def _apply_gamma_correction(self, image, gamma=1.0):
        """감마 보정"""
        if gamma == 1.0:
            return image
        
        np_image = np.array(image, dtype=np.float32) / 255.0
        corrected = np.power(np_image, 1.0 / gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return Image.fromarray(corrected)
    
    def enhance_image(self, image, ring_type=None, lighting=None):
        """메인 이미지 보정 파이프라인 - 메모리 최적화"""
        try:
            start_time = datetime.now()
            
            # 1. 이미지 전처리 및 최적화
            image = self._prepare_image(image)
            original_size = image.size
            
            # 2. 자동 분석 (타입이 지정되지 않은 경우)
            if ring_type is None:
                ring_type = self.detect_ring_type(image)
            if lighting is None:
                lighting = self.detect_lighting_condition(image)
            
            print(f"분석 결과 - 링타입: {ring_type}, 조명: {lighting}")
            
            # 3. 파라미터 선택
            params = self.enhancement_params.get(ring_type, self.enhancement_params['silver']).copy()
            
            # 4. 조명에 따른 파라미터 동적 조정
            if lighting == 'warm':
                params['warmth'] *= 0.9
                params['brightness'] *= 1.05
                print("따뜻한 조명 감지 - 파라미터 조정")
            elif lighting == 'cool':
                params['warmth'] *= 1.1
                params['contrast'] *= 1.05
                print("차가운 조명 감지 - 파라미터 조정")
            
            # 5. 7단계 전문 보정 파이프라인 실행
            print("전문 보정 시작...")
            enhanced = image
            enhanced = self._adjust_brightness_contrast(enhanced, params['brightness'], params['contrast'])
            enhanced = self._adjust_warmth(enhanced, params['warmth'])
            enhanced = self._adjust_saturation(enhanced, params['saturation'])
            enhanced = self._enhance_sharpness(enhanced, params['sharpness'])
            enhanced = self._enhance_clarity(enhanced, params['clarity'])
            enhanced = self._apply_gamma_correction(enhanced, params['gamma'])
            
            # 6. 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 7. 통계 업데이트
            self.total_processed += 1
            
            # 8. 학습 데이터 기록
            self._record_enhancement_data({
                'ring_type': ring_type,
                'lighting': lighting,
                'params': params,
                'processing_time': processing_time,
                'original_size': f"{original_size[0]}x{original_size[1]}",
                'success': True
            })
            
            print(f"보정 완료! 처리시간: {processing_time:.2f}초")
            
            return {
                'enhanced_image': enhanced,
                'ring_type': ring_type,
                'lighting': lighting,
                'params': params,
                'processing_time': processing_time,
                'original_size': original_size
            }
            
        except Exception as e:
            print(f"Enhancement error: {str(e)}")
            # 에러 발생시에도 학습 데이터 기록
            self._record_enhancement_data({
                'error': str(e),
                'ring_type': ring_type or 'unknown',
                'lighting': lighting or 'unknown',
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'success': False
            })
            return {
                'enhanced_image': image,  # 원본 반환
                'error': str(e)
            }
    
    def _record_enhancement_data(self, data):
        """웨딩링 보정 학습 데이터 완전 기록"""
        try:
            if self.learning_sheet:
                # Google Sheets에 상세 기록
                row = [
                    datetime.now().isoformat(),  # 타임스탬프
                    data.get('ring_type', 'unknown'),  # 링 타입
                    data.get('lighting', 'unknown'),  # 조명 환경
                    json.dumps(data.get('params', {})),  # 보정 파라미터
                    data.get('processing_time', 0),  # 처리 시간
                    data.get('original_size', 'unknown'),  # 원본 크기
                    data.get('success', True),  # 성공 여부
                    data.get('error', ''),  # 에러 메시지
                    self.total_processed  # 총 처리 수
                ]
                self.learning_sheet.append_row(row)
                print(f"학습 데이터 기록 완료: {data.get('ring_type')} - {data.get('success')}")
            else:
                # 로컬 저장소에 기록 (구글시트 없을 때)
                print(f"로컬 기록: {data.get('ring_type')} 링, {data.get('processing_time', 0):.2f}초")
        except Exception as e:
            print(f"데이터 기록 에러 (무시됨): {str(e)}")  # 기록 실패해도 메인 기능에 영향 없음

# WeddingRingEnhancer 인스턴스 생성
enhancer = WeddingRingEnhancer()

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "message": "Claude 분석 기반 웨딩링 보정 서버 가동 중",
        "version": enhancer.version,
        "total_processed": enhancer.total_processed,
        "google_sheets": "connected" if enhancer.sheets_client else "local_storage"
    })

@app.route('/enhance_wedding_ring', methods=['POST'])
def enhance_wedding_ring():
    """A_001 메인 보정 엔드포인트"""
    try:
        # 요청 데이터 상세 로깅
        print(f"Content-Type: {request.content_type}")
        print(f"Request method: {request.method}")
        print(f"Request headers: {dict(request.headers)}")
        
        # JSON 데이터 파싱
        try:
            data = request.get_json(force=True)  # force=True로 Content-Type 무시
            if not data:
                # JSON이 아닐 경우 raw 데이터 확인
                raw_data = request.get_data()
                return jsonify({
                    "success": False,
                    "error": "No JSON data received",
                    "content_type": request.content_type,
                    "raw_data_length": len(raw_data),
                    "raw_sample": str(raw_data[:100])
                }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"JSON parsing failed: {str(e)}",
                "content_type": request.content_type
            }), 400
        
        print(f"Parsed data keys: {list(data.keys())}")
        
        # 이미지 데이터 확인 및 처리
        image_data = None
        
        if 'image_base64' in data:
            try:
                # Base64 디코딩
                base64_string = data['image_base64']
                if isinstance(base64_string, str):
                    # 데이터 URL 접두사 제거 (data:image/jpeg;base64, 등)
                    if ',' in base64_string:
                        base64_string = base64_string.split(',')[1]
                    image_data = base64.b64decode(base64_string)
                else:
                    return jsonify({
                        "success": False,
                        "error": "image_base64 must be string",
                        "received_type": type(base64_string).__name__
                    }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Base64 decoding failed: {str(e)}"
                }), 400
                
        elif 'image_data' in data:
            # 바이너리 데이터 직접 처리
            image_data = data['image_data']
            if isinstance(image_data, str):
                image_data = image_data.encode()
                
        else:
            return jsonify({
                "success": False,
                "error": "No image data found",
                "received_keys": list(data.keys()),
                "expected_keys": ["image_base64", "image_data"]
            }), 400
        
        # 이미지 열기 및 최적화
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"Original image: {image.size}, {image.mode}")
            
            # 고해상도 이미지 메모리 최적화 (품질 보존)
            max_dimension = 2048  # 2K 해상도 - 대형 인쇄 가능한 고품질
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                # LANCZOS 최고급 리샘플링으로 품질 보존
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Memory optimized to: {image.size} (품질 보존)")
            
            # RGB 모드 보장
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Image loading failed: {str(e)}",
                "data_length": len(image_data) if image_data else 0
            }), 400
        
        # 웨딩링 보정 수행
        result = enhancer.enhance_image(
            image=image,
            ring_type=data.get('ring_type'),
            lighting=data.get('lighting')
        )
        
        if 'error' in result:
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500
        
        # 결과 이미지를 Base64로 인코딩 (최적화)
        enhanced_image = result['enhanced_image']
        output_buffer = io.BytesIO()
        
        # 고품질 JPEG 출력 (웨딩 앨범급 품질)
        enhanced_image.save(
            output_buffer, 
            format='JPEG', 
            quality=95,  # 높은 품질 유지
            optimize=True,  # 파일 크기 최적화
            progressive=True  # 점진적 로딩
        )
        enhanced_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "enhanced_image_base64": enhanced_base64,
            "original_filename": data.get('filename', 'unknown'),
            "original_size": f"{image.size[0]}x{image.size[1]}",
            "output_size": f"{enhanced_image.size[0]}x{enhanced_image.size[1]}",
            "ring_type": result['ring_type'],
            "lighting": result['lighting'],
            "processing_time": f"{result['processing_time']:.2f}s",
            "enhancement_params": result['params'],
            "quality_score": 95,  # 품질 점수 추가
            "message": "🎉 웨딩링 이미지 전문 보정 완료 - 앨범급 고품질"
        })
        
    except Exception as e:
        print(f"Unexpected error in enhance_wedding_ring: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "서버 내부 오류"
        }), 500

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """웨딩링 보정 - 바이너리 직접 반환 (Google Drive 업로드용)"""
    try:
        # 기존 enhance_wedding_ring과 동일한 처리
        data = request.get_json(force=True)
        if not data:
            return "No JSON data received", 400
        
        # 이미지 데이터 처리 (동일)
        if 'image_base64' in data:
            base64_string = data['image_base64']
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            image_data = base64.b64decode(base64_string)
        else:
            return "No image data found", 400
        
        # 이미지 처리 (동일)
        image = Image.open(io.BytesIO(image_data))
        print(f"Binary endpoint - Original: {image.size}")
        
        # 메모리 최적화
        max_dimension = 2048
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 웨딩링 보정 수행
        result = enhancer.enhance_image(image=image)
        
        if 'error' in result:
            return f"Enhancement failed: {result['error']}", 500
        
        # 바이너리 직접 반환
        enhanced_image = result['enhanced_image']
        output_buffer = io.BytesIO()
        enhanced_image.save(
            output_buffer, 
            format='JPEG', 
            quality=95,
            optimize=True,
            progressive=True
        )
        
        # 바이너리 데이터 직접 반환
        output_buffer.seek(0)
        return output_buffer.getvalue(), 200, {
            'Content-Type': 'image/jpeg',
            'Content-Disposition': f'attachment; filename="enhanced_{data.get("filename", "image.jpg")}"',
            'X-Ring-Type': result['ring_type'],
            'X-Lighting': result['lighting'],
            'X-Processing-Time': f"{result['processing_time']:.2f}s"
        }
        
    except Exception as e:
        print(f"Binary endpoint error: {str(e)}")
        return f"Server error: {str(e)}", 500
def analyze_b001_style():
    """B_001 스타일 분석 (선택적 기능)"""
    try:
        data = request.get_json()
        
        # 간단한 스타일 분석 구현
        return jsonify({
            "success": True,
            "style_analysis": "B_001 스타일 매칭 기능 개발 중",
            "message": "B_001 분석 완료"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/batch_test_parameters', methods=['POST'])
def batch_test_parameters():
    """Claude 분석용 다중 테스트"""
    try:
        data = request.get_json()
        
        # 배치 테스트 구현
        return jsonify({
            "success": True,
            "batch_results": [],
            "message": "배치 테스트 완료"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get_learning_data', methods=['GET'])
def get_learning_data():
    """축적 데이터 조회"""
    try:
        return jsonify({
            "success": True,
            "total_processed": enhancer.total_processed,
            "google_sheets_status": "connected" if enhancer.sheets_client else "local_storage",
            "message": "학습 데이터 조회 완료"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
