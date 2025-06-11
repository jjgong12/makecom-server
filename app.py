from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from PIL import Image
import io
import traceback

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Make.com 워크플로우 서버가 정상 작동 중입니다"})

@app.route('/detect_black_marking', methods=['POST'])
def detect_black_marking():
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 검은색 영역 감지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_list = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 크기 필터
                x, y, w, h = cv2.boundingRect(contour)
                roi_list.append({
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h),
                    "area": int(area)
                })
                cv2.fillPoly(mask, [contour], 255)
        
        # 마스크를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "roi_coordinates": roi_list,
            "mask_base64": mask_base64,
            "success": True,
            "total_markings": len(roi_list)
        })
        
    except Exception as e:
        print(f"detect_black_marking 에러: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/generate_thumbnails', methods=['POST'])
def generate_thumbnails():
    try:
        print("=== generate_thumbnails 시작 ===")
        
        data = request.get_json()
        print(f"받은 데이터 키: {list(data.keys())}")
        
        enhanced_image = data.get('enhanced_image', '')
        roi_coords = data.get('roi_coords', {})
        sizes = data.get('sizes', ['1000x1300'])
        
        print(f"enhanced_image 길이: {len(enhanced_image)}")
        print(f"roi_coords 타입: {type(roi_coords)}, 값: {roi_coords}")
        print(f"sizes: {sizes}")
        
        # ROI 좌표 처리 - 배열인지 객체인지 확인
        if isinstance(roi_coords, list):
            if len(roi_coords) > 0:
                roi = roi_coords[0]
            else:
                return jsonify({"error": "ROI 좌표 배열이 비어있습니다"}), 400
        else:
            roi = roi_coords
        
        print(f"사용할 ROI: {roi}")
        
        # 필수 키 확인
        required_keys = ['x', 'y', 'width', 'height']
        for key in required_keys:
            if key not in roi:
                return jsonify({"error": f"ROI에 {key} 필드가 없습니다"}), 400
        
        # Base64 디코딩
        if not enhanced_image:
            return jsonify({"error": "enhanced_image가 비어있습니다"}), 400
            
        try:
            image_data = base64.b64decode(enhanced_image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            print(f"이미지 디코딩 에러: {decode_error}")
            return jsonify({"error": f"이미지 디코딩 실패: {decode_error}"}), 400
        
        if image is None:
            return jsonify({"error": "이미지 디코딩 결과가 None입니다"}), 400
        
        print(f"원본 이미지 크기: {image.shape}")
        
        # ROI 좌표로 크롭
        x = int(roi['x'])
        y = int(roi['y'])
        w = int(roi['width'])
        h = int(roi['height'])
        
        print(f"원래 ROI: x={x}, y={y}, w={w}, h={h}")
        
        # 경계 확인 및 조정
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        print(f"조정된 ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"이미지 크기: width={img_width}, height={img_height}")
        
        if w <= 0 or h <= 0:
            return jsonify({"error": f"잘못된 ROI 크기: width={w}, height={h}"}), 400
        
        # 크롭 실행
        try:
            cropped = image[y:y+h, x:x+w]
            print(f"크롭된 이미지 크기: {cropped.shape}")
        except Exception as crop_error:
            print(f"크롭 에러: {crop_error}")
            return jsonify({"error": f"크롭 실패: {crop_error}"}), 400
        
        # 썸네일 생성
        thumbnails = {}
        for size in sizes:
            try:
                width, height = map(int, size.split('x'))
                resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
                
                _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
                thumbnails[f'thumbnail_{size}'] = thumbnail_base64
                
                print(f"썸네일 {size} 생성 완료")
            except Exception as thumb_error:
                print(f"썸네일 {size} 생성 에러: {thumb_error}")
                continue
        
        print("=== generate_thumbnails 완료 ===")
        
        return jsonify({
            "thumbnails": thumbnails,
            "success": True,
            "roi_used": roi,
            "original_image_size": f"{img_width}x{img_height}",
            "cropped_size": f"{w}x{h}"
        })
        
    except Exception as e:
        print(f"generate_thumbnails 전체 에러: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"서버 에러: {str(e)}", "traceback": traceback.format_exc()}), 500

@app.route('/generate_thumbnail_binary', methods=['POST'])
def generate_thumbnail_binary():
    try:
        data = request.get_json()
        enhanced_image = data['enhanced_image']
        roi_coords = data['roi_coords']
        size = data.get('size', '1000x1300')
        
        # ROI 좌표 처리
        if isinstance(roi_coords, list):
            roi = roi_coords[0] if len(roi_coords) > 0 else {}
        else:
            roi = roi_coords
        
        # Base64 디코딩
        image_data = base64.b64decode(enhanced_image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "이미지 디코딩 실패"}), 500
        
        # ROI 좌표로 크롭
        x, y, w, h = int(roi['x']), int(roi['y']), int(roi['width']), int(roi['height'])
        
        # 경계 확인
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        cropped = image[y:y+h, x:x+w]
        
        # 리사이즈
        width, height = map(int, size.split('x'))
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # 바이너리 반환
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
