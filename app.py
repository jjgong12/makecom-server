from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import json
import os
import re

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "message": "Make.com 워크플로우 서버가 정상 작동 중입니다"
    })

@app.route('/detect_red_marking', methods=['POST'])
def detect_red_marking():
    """빨간색 마킹 감지 및 ROI 좌표 반환"""
    try:
        print("=== detect_red_marking 시작 ===")
        
        # JSON 데이터 받기
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "image_base64 필드가 필요합니다"}), 400
        
        # Base64 이미지 디코딩
        image_base64 = data['image_base64']
        print(f"Base64 데이터 길이: {len(image_base64)}")
        
        try:
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "이미지 디코딩 실패"}), 400
                
        except Exception as decode_error:
            return jsonify({"error": "Base64 디코딩 실패", "details": str(decode_error)}), 400
        
        print(f"이미지 크기: {image.shape}")
        
        # HSV 색공간으로 변환 (빨간색 감지에 더 효과적)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 빨간색 범위 정의 (HSV) - #ff0000에 최적화
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 빨간색 마스크 생성
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # 노이즈 제거 및 연결
        kernel = np.ones((7,7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ROI 좌표 계산 - 가장 큰 사각형 영역만 선택
        roi_list = []
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if contours:
            # 가장 큰 윤곽선 찾기
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 1000:  # 최소 면적 체크
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 여유 공간 추가 (빨간색 테두리 안쪽 영역)
                padding = 20
                x += padding
                y += padding
                w -= padding * 2
                h -= padding * 2
                
                # 경계 체크
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                roi_list.append({
                    "x": int(x),
                    "y": int(y), 
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })
                
                # 마스크에 영역 추가 (inpainting용)
                cv2.fillPoly(mask, [largest_contour], 255)
        
        print(f"감지된 빨간색 영역 수: {len(roi_list)}")
        if roi_list:
            print(f"링 영역: {roi_list[0]}")
        
        # 마스크를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "roi_coordinates": roi_list,
            "mask_base64": mask_base64,
            "total_markings": len(roi_list),
            "detection_type": "red_color"
        })
        
    except Exception as e:
        print(f"detect_red_marking 에러: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "서버 에러", "details": str(e)}), 500

@app.route('/detect_black_marking', methods=['POST'])
def detect_black_marking():
    """검은색 마킹 감지 및 ROI 좌표 반환 (기존 버전)"""
    try:
        print("=== detect_black_marking 시작 ===")
        
        # JSON 데이터 받기
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "image_base64 필드가 필요합니다"}), 400
        
        # Base64 이미지 디코딩
        image_base64 = data['image_base64']
        print(f"Base64 데이터 길이: {len(image_base64)}")
        
        try:
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "이미지 디코딩 실패"}), 400
                
        except Exception as decode_error:
            return jsonify({"error": "Base64 디코딩 실패", "details": str(decode_error)}), 400
        
        print(f"이미지 크기: {image.shape}")
        
        # 검은색 영역 감지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 검은색 임계값 (0-30 정도의 매우 어두운 픽셀)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ROI 좌표 계산
        roi_list = []
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # 최소 면적 필터 (너무 작은 노이즈 제거)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                roi_list.append({
                    "x": int(x),
                    "y": int(y), 
                    "width": int(w),
                    "height": int(h)
                })
                
                # 마스크에 검은색 영역 추가
                cv2.fillPoly(mask, [contour], 255)
        
        print(f"감지된 검은색 영역 수: {len(roi_list)}")
        
        # 마스크를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "roi_coordinates": roi_list,
            "mask_base64": mask_base64,
            "total_markings": len(roi_list)
        })
        
    except Exception as e:
        print(f"detect_black_marking 에러: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "서버 에러", "details": str(e)}), 500

@app.route('/generate_thumbnails', methods=['POST'])
def generate_thumbnails():
    """ROI 기반 썸네일 생성 (강화된 JSON 파싱)"""
    try:
        print("=== generate_thumbnails 시작 ===")
        
        # 강화된 JSON 파싱
        raw_data = request.get_data(as_text=True)
        print(f"Raw 데이터 길이: {len(raw_data)}")
        print(f"Raw 데이터 시작: {raw_data[:100]}")
        
        # 여러 방법으로 JSON 파싱 시도
        data = None
        
        # 방법 1: 기본 JSON 파싱
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            print(f"기본 JSON 파싱 실패: {e}")
            
            # 방법 2: 정규표현식으로 필드 추출
            try:
                # enhanced_image 필드 추출
                enhanced_image_match = re.search(r'"enhanced_image":\s*"([^"]+)"', raw_data)
                roi_coords_match = re.search(r'"roi_coords":\s*(\{[^}]+\})', raw_data)
                sizes_match = re.search(r'"sizes":\s*(\[[^\]]+\])', raw_data)
                
                if enhanced_image_match:
                    data = {
                        "enhanced_image": enhanced_image_match.group(1),
                        "roi_coords": json.loads(roi_coords_match.group(1)) if roi_coords_match else {},
                        "sizes": json.loads(sizes_match.group(1)) if sizes_match else ["1000x1300"]
                    }
                    print("정규표현식 파싱 성공!")
                    
            except Exception as regex_error:
                print(f"정규표현식 파싱도 실패: {regex_error}")
                return jsonify({"error": "JSON 파싱 불가능", "details": str(e)}), 400
        
        if not data:
            return jsonify({"error": "모든 JSON 파싱 방법 실패"}), 400
            
        print(f"파싱된 데이터 키들: {list(data.keys())}")
        
        # 필드 추출
        enhanced_image_b64 = data.get('enhanced_image', '')
        roi_coords = data.get('roi_coords', {})
        sizes = data.get('sizes', ['1000x1300'])
        
        if not enhanced_image_b64:
            return jsonify({"error": "enhanced_image 필드가 필요합니다"}), 400
        
        # Base64 디코딩
        try:
            image_data = base64.b64decode(enhanced_image_b64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "이미지 디코딩 실패"}), 400
                
        except Exception as decode_error:
            return jsonify({"error": "Base64 디코딩 실패", "details": str(decode_error)}), 400
        
        print(f"원본 이미지 크기: {image.shape}")
        
        # ROI 크롭
        if roi_coords and all(k in roi_coords for k in ['x', 'y', 'width', 'height']):
            x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['width'], roi_coords['height']
            
            # 이미지 경계 체크
            img_h, img_w = image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            print(f"크롭 좌표: x={x}, y={y}, w={w}, h={h}")
            cropped_image = image[y:y+h, x:x+w]
            print(f"크롭된 이미지 크기: {cropped_image.shape}")
        else:
            print("ROI 좌표 없음 - 전체 이미지 사용")
            cropped_image = image
        
        # 썸네일 생성
        thumbnails = {}
        for size in sizes:
            try:
                width, height = map(int, size.split('x'))
                resized = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                
                # 품질 높은 JPEG 인코딩
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                _, buffer = cv2.imencode('.jpg', resized, encode_params)
                
                thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
                thumbnails[f'thumbnail_{size}'] = thumbnail_b64
                
            except Exception as thumb_error:
                print(f"썸네일 {size} 생성 실패: {thumb_error}")
        
        print("=== generate_thumbnails 성공 ===")
        return jsonify({
            "success": True,
            "thumbnails": thumbnails,
            "roi_used": roi_coords,
            "original_size": f"{image.shape[1]}x{image.shape[0]}",
            "cropped_size": f"{cropped_image.shape[1]}x{cropped_image.shape[0]}"
        })
        
    except Exception as e:
        print(f"generate_thumbnails 전체 에러: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "서버 에러", "details": str(e)}), 500

@app.route('/generate_thumbnail_binary', methods=['POST'])
def generate_thumbnail_binary():
    """ROI 크롭 후 바이너리 파일 직접 반환 (강화된 JSON 파싱)"""
    try:
        print("=== generate_thumbnail_binary 시작 ===")
        
        # 강화된 JSON 파싱 (generate_thumbnails와 동일)
        raw_data = request.get_data(as_text=True)
        print(f"Raw 데이터 길이: {len(raw_data)}")
        print(f"Raw 데이터 시작: {raw_data[:100]}")
        
        # 여러 방법으로 JSON 파싱 시도
        data = None
        
        # 방법 1: 기본 JSON 파싱
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            print(f"기본 JSON 파싱 실패: {e}")
            
            # 방법 2: 정규표현식으로 필드 추출
            try:
                enhanced_image_match = re.search(r'"enhanced_image":\s*"([^"]+)"', raw_data)
                roi_coords_match = re.search(r'"roi_coords":\s*(\{[^}]+\})', raw_data)
                
                if enhanced_image_match:
                    data = {
                        "enhanced_image": enhanced_image_match.group(1),
                        "roi_coords": json.loads(roi_coords_match.group(1)) if roi_coords_match else {}
                    }
                    print("정규표현식 파싱 성공!")
                    
            except Exception as regex_error:
                print(f"정규표현식 파싱도 실패: {regex_error}")
                return "JSON 파싱 불가능", 400
        
        if not data:
            return "모든 JSON 파싱 방법 실패", 400
            
        print(f"파싱된 데이터 키들: {list(data.keys())}")
        
        enhanced_image_b64 = data.get('enhanced_image', '')
        roi_coords = data.get('roi_coords', {})
        
        if not enhanced_image_b64:
            print("enhanced_image 필드 없음")
            return "enhanced_image 필드가 필요합니다", 400
        
        print(f"Base64 데이터 길이: {len(enhanced_image_b64)}")
        print(f"ROI 좌표: {roi_coords}")
        
        # Base64 디코딩
        try:
            image_data = base64.b64decode(enhanced_image_b64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("이미지 디코딩 실패")
                return "이미지 디코딩 실패", 400
                
            print(f"원본 이미지 크기: {image.shape}")
            
        except Exception as decode_error:
            print(f"Base64 디코딩 에러: {decode_error}")
            return f"Base64 디코딩 실패: {str(decode_error)}", 400
        
        # ROI 크롭
        if roi_coords and all(k in roi_coords for k in ['x', 'y', 'width', 'height']):
            x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['width'], roi_coords['height']
            
            # 경계 체크
            img_h, img_w = image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            print(f"크롭 좌표: x={x}, y={y}, w={w}, h={h}")
            cropped_image = image[y:y+h, x:x+w]
            print(f"크롭된 이미지 크기: {cropped_image.shape}")
        else:
            print("ROI 좌표 없음 - 전체 이미지 사용")
            cropped_image = image
        
        # 1000x1300으로 리사이즈
        resized = cv2.resize(cropped_image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        print(f"리사이즈된 이미지 크기: {resized.shape}")
        
        # 고품질 JPEG로 인코딩
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        success, buffer = cv2.imencode('.jpg', resized, encode_params)
        
        if not success:
            print("JPEG 인코딩 실패")
            return "JPEG 인코딩 실패", 500
        
        print(f"인코딩된 이미지 크기: {len(buffer)} bytes")
        
        # 바이너리 직접 반환
        response = app.response_class(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': 'attachment; filename=thumbnail.jpg',
                'Content-Length': str(len(buffer))
            }
        )
        
        print("=== generate_thumbnail_binary 성공 ===")
        return response
        
    except Exception as e:
        print(f"generate_thumbnail_binary 전체 에러: {e}")
        import traceback
        traceback.print_exc()
        return f"서버 에러: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
