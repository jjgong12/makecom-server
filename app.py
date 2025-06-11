from flask import Flask, request, jsonify
import base64
import io
import cv2
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        # PIL Image로 변환
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Base64 to Image 변환 오류: {e}")
        return None

def image_to_base64(image):
    """PIL Image를 Base64 문자열로 변환"""
    try:
        # 이미지를 바이트로 변환
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()
        
        # Base64 인코딩
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Image to Base64 변환 오류: {e}")
        return None

def detect_black_marking_in_image(image):
    """이미지에서 검은색 마킹 영역 탐지"""
    try:
        # PIL Image를 OpenCV 형식으로 변환
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # 검은색 영역 탐지 (임계값 조정 가능)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 연결된 구성 요소 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_coordinates = []
        
        for contour in contours:
            # 작은 노이즈 제거 (면적 기준)
            area = cv2.contourArea(contour)
            if area > 100:  # 최소 면적 100픽셀
                # 경계 상자 계산
                x, y, w, h = cv2.boundingRect(contour)
                
                # ROI 좌표 저장 (x, y, width, height)
                roi_coordinates.append({
                    "x": int(x),
                    "y": int(y), 
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })
        
        # 면적 기준으로 정렬 (큰 것부터)
        roi_coordinates.sort(key=lambda x: x['area'], reverse=True)
        
        return roi_coordinates, contours
        
    except Exception as e:
        print(f"검은색 마킹 탐지 오류: {e}")
        return [], []

def generate_thumbnail(image, roi_coords, size):
    """지정된 ROI 영역의 썸네일 생성 (정확한 크롭)"""
    try:
        # ROI 영역 추출
        if roi_coords:
            # 첫 번째 ROI 사용 (가장 큰 영역)
            roi = roi_coords[0] if isinstance(roi_coords, list) else roi_coords
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            
            # 이미지 크기 확인
            img_width, img_height = image.size
            
            # ROI 좌표 검증 및 조정
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # 🔥 정확한 ROI 영역 크롭 (링만 정확히 추출)
            cropped = image.crop((x, y, x + w, y + h))
        else:
            # ROI가 없으면 전체 이미지 사용
            cropped = image
        
        # 썸네일 크기 파싱 (예: "1000x1300")
        if 'x' in size:
            width, height = map(int, size.split('x'))
        else:
            width = height = int(size)
        
        # 🔥 고품질 리사이즈 (LANCZOS4 사용)
        resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
        
        return resized
        
    except Exception as e:
        print(f"썸네일 생성 오류: {e}")
        return None

@app.route('/detect_black_marking', methods=['POST'])
def detect_black_marking():
    """검은색 마킹 탐지 API (마스크 포함)"""
    try:
        # JSON 데이터 받기
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 필드가 필요합니다'
            }), 400
        
        # Base64 이미지 디코딩
        image = base64_to_image(data['image_base64'])
        if image is None:
            return jsonify({
                'success': False,
                'error': '이미지 디코딩에 실패했습니다'
            }), 400
        
        # 검은색 마킹 탐지
        roi_coordinates, contours = detect_black_marking_in_image(image)
        
        # 🔥 마스크 이미지 생성 (Topaz inpainting용)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = np.zeros(opencv_image.shape[:2], dtype=np.uint8)
        
        # 검은색 영역을 흰색(255)으로 채우기 (inpainting 마스크)
        for contour in contours:
            cv2.fillPoly(mask, [contour], 255)
        
        # 마스크를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'roi_coordinates': roi_coordinates,
            'mask_base64': mask_base64,  # 🔥 Topaz inpainting용 마스크 추가!
            'total_markings': len(roi_coordinates)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }), 500

@app.route('/generate_thumbnails', methods=['POST'])
def generate_thumbnails():
    """썸네일 생성 API (정확한 크롭)"""
    try:
        # JSON 데이터 받기
        data = request.get_json()
        
        required_fields = ['enhanced_image', 'roi_coords', 'sizes']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} 필드가 필요합니다'
                }), 400
        
        # Base64 이미지 디코딩
        image = base64_to_image(data['enhanced_image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': '이미지 디코딩에 실패했습니다'
            }), 400
        
        # ROI 좌표 파싱
        roi_coords = data['roi_coords']
        sizes = data['sizes']
        
        # 🔥 정확한 크롭을 위한 썸네일 생성
        thumbnails = {}
        
        for size in sizes:
            thumbnail = generate_thumbnail(image, roi_coords, size)
            if thumbnail:
                thumbnail_base64 = image_to_base64(thumbnail)
                if thumbnail_base64:
                    thumbnails[f'thumbnail_{size}'] = thumbnail_base64
        
        # 🔥 크롭된 영역의 마스크도 생성 (필요시)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = np.zeros(opencv_image.shape[:2], dtype=np.uint8)
        
        if roi_coords:
            roi = roi_coords[0] if isinstance(roi_coords, list) else roi_coords
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'thumbnails': thumbnails,
            'mask_base64': mask_base64,  # 🔥 크롭 영역 마스크 추가
            'generated_count': len(thumbnails)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'message': 'Make.com 워크플로우 서버가 정상 작동 중입니다'
    })

@app.route('/', methods=['GET'])
def home():
    """홈페이지"""
    return jsonify({
        'service': 'Make.com 워크플로우 API 서버',
        'version': '1.0.0',
        'endpoints': {
            '/detect_black_marking': 'POST - 검은색 마킹 탐지 (마스크 포함)',
            '/generate_thumbnails': 'POST - 썸네일 생성 (정확한 크롭)',
            '/health': 'GET - 서버 상태 확인'
        }
    })

if __name__ == '__main__':
    print("🚀 Make.com 워크플로우 API 서버 시작!")
    print("📍 엔드포인트:")
    print("   POST /detect_black_marking - 검은색 마킹 탐지 (마스크 포함)")
    print("   POST /generate_thumbnails - 썸네일 생성 (정확한 크롭)")
    print("   GET  /health - 서버 상태 확인")
    print("   GET  / - 서비스 정보")
    
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
