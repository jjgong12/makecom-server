import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 28쌍 학습 데이터 기반 파라미터 (절대 제거 금지)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18, 'saturation': 0.98, 'gamma': 1.03
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 0.99
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.04,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.15, 'saturation': 1.25, 'gamma': 1.02
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.15,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.18, 'saturation': 0.98, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.10,
            'sharpness': 1.25, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.03,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.07,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.18, 'saturation': 1.28, 'gamma': 1.03
        }
    }
}

# 28쌍 AFTER 파일 배경색 (대화 28번 성과)
AFTER_BACKGROUND_COLORS = {
    'natural': [248, 245, 242],
    'warm': [252, 248, 240], 
    'cool': [245, 247, 250]
}

class UltimateWeddingRingProcessor:
    def __init__(self):
        print("UltimateWeddingRingProcessor 초기화 완료")

    def detect_metal_type_ultimate(self, image):
        """금속 타입 감지 (29번 대화 적응형 방식)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            # 샴페인골드 화이트화 (25번 대화: 무도금 화이트골드 수준)
            if s_mean < 35:
                return 'champagne_gold'  # 화이트골드 방향으로
            elif 5 <= h_mean <= 25:
                return 'yellow_gold' if s_mean > 85 else 'champagne_gold'
            elif h_mean < 5 or h_mean > 170:
                return 'rose_gold'
            else:
                return 'champagne_gold'  # 기본값도 화이트골드 방향
        except:
            return 'champagne_gold'

    def detect_lighting_ultimate(self, image):
        """조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'

    def detect_black_border_ultimate(self, image):
        """적응형 검은색 테두리 감지 (29번 대화 100픽셀 두께 대응)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 다중 threshold로 정확한 감지
            masks = []
            for thresh in [15, 20, 25]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                masks.append(binary)
            
            # 모든 마스크 결합
            combined_mask = np.zeros_like(gray)
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
            
            # 가장 큰 사각형 형태 찾기
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < height * width * 0.1:  # 너무 작으면 제외
                    continue
                    
                # 사각형 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4 and area > best_area:
                    best_contour = contour
                    best_area = area
            
            if best_contour is None:
                return None, None
            
            # 바운딩 박스 생성
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # 적응형 두께 감지 (100픽셀 두께 대응)
            border_thickness = self.measure_border_thickness_safe(combined_mask, x, y, w, h)
            
            # 웨딩링 보호 영역 (30픽셀 안전 마진)
            inner_margin = max(30, border_thickness // 2)
            inner_x = x + inner_margin
            inner_y = y + inner_margin  
            inner_w = w - 2 * inner_margin
            inner_h = h - 2 * inner_margin
            
            # 최소 크기 보장
            if inner_w < 100 or inner_h < 100:
                inner_margin = min(20, w//4, h//4)
                inner_x = x + inner_margin
                inner_y = y + inner_margin
                inner_w = w - 2 * inner_margin  
                inner_h = h - 2 * inner_margin
            
            return (x, y, w, h), (inner_x, inner_y, inner_w, inner_h)
            
        except Exception as e:
            print(f"검은색 테두리 감지 실패: {e}")
            return None, None

    def measure_border_thickness_safe(self, mask, x, y, w, h):
        """안전한 테두리 두께 측정"""
        try:
            # 4방향에서 두께 측정
            thicknesses = []
            
            # 상단
            if y > 10 and y + 20 < mask.shape[0]:
                top_slice = mask[y:y+20, x:x+w]
                if top_slice.size > 0:
                    thickness = np.sum(top_slice > 0) // max(1, w)
                    if thickness > 0:
                        thicknesses.append(thickness)
            
            # 하단  
            if y + h - 20 > 0 and y + h < mask.shape[0]:
                bottom_slice = mask[y+h-20:y+h, x:x+w]
                if bottom_slice.size > 0:
                    thickness = np.sum(bottom_slice > 0) // max(1, w)
                    if thickness > 0:
                        thicknesses.append(thickness)
            
            # 좌측
            if x > 10 and x + 20 < mask.shape[1]:
                left_slice = mask[y:y+h, x:x+20]
                if left_slice.size > 0:
                    thickness = np.sum(left_slice > 0) // max(1, h)
                    if thickness > 0:
                        thicknesses.append(thickness)
            
            # 우측
            if x + w - 20 > 0 and x + w < mask.shape[1]:
                right_slice = mask[y:y+h, x+w-20:x+w]
                if right_slice.size > 0:
                    thickness = np.sum(right_slice > 0) // max(1, h)
                    if thickness > 0:
                        thicknesses.append(thickness)
            
            if thicknesses:
                measured = int(np.median(thicknesses))
                # 100픽셀 두께 대응: 측정값 × 1.5 + 20
                return min(150, max(20, measured * 1.5 + 20))
            else:
                return 40  # 기본값
                
        except:
            return 40

    def enhance_wedding_ring_v13_3_complete(self, image, metal_type, lighting):
        """v13.3 완전한 10단계 웨딩링 보정 (절대 제거 금지)"""
        try:
            params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                     WEDDING_RING_PARAMS['champagne_gold']['natural'])
            
            # PIL로 변환
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # 1단계: 노이즈 제거
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            pil_image = Image.fromarray(denoised)
            
            # 2단계: 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 3단계: 대비 조정  
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 4단계: 선명도 조정
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # 5단계: 채도 조정
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(params.get('saturation', 1.0))
            
            enhanced_array = np.array(enhanced)
            
            # 6단계: 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 7단계: LAB 색공간 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 8단계: CLAHE 명료도 향상
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9단계: 감마 보정
            gamma = params.get('gamma', 1.0)
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                enhanced_array = cv2.LUT(enhanced_array, table)
            
            # 10단계: 원본과 블렌딩 (자연스러움 보장)
            original_blend = params['original_blend']
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                image, original_blend, 0
            )
            
            return final.astype(np.uint8)
            
        except Exception as e:
            print(f"v13.3 보정 실패, 기본 처리: {e}")
            # 기본 보정이라도 수행
            enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            return enhanced

    def remove_black_border_ultimate(self, image, border_bbox, inner_bbox, lighting):
        """검은색 테두리 완전 제거 (27번 대화 고급 inpainting + 28번 배경색)"""
        try:
            if border_bbox is None:
                return image
            
            x, y, w, h = border_bbox
            inner_x, inner_y, inner_w, inner_h = inner_bbox
            
            # 마스크 생성 (웨딩링 영역은 보호)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 테두리 영역 마킹
            mask[y:y+h, x:x+w] = 255
            
            # 웨딩링 영역 보호 (절대 건드리지 않음)
            mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = 0
            
            # 28쌍 AFTER 배경색 적용
            bg_color = AFTER_BACKGROUND_COLORS.get(lighting, [248, 245, 242])
            
            # TELEA inpainting (27번 대화 고급 방식)
            inpainted = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
            
            # 배경색으로 추가 보정
            result = inpainted.copy()
            mask_indices = np.where(mask > 0)
            for c in range(3):
                result[mask_indices[0], mask_indices[1], c] = bg_color[c]
            
            # 자연스러운 블렌딩 (31×31 가우시안 블러)
            blur_mask = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 10) / 255.0
            
            final_result = image.copy().astype(np.float32)
            for c in range(3):
                final_result[:,:,c] = (
                    image[:,:,c].astype(np.float32) * (1 - blur_mask) +
                    result[:,:,c].astype(np.float32) * blur_mask
                )
            
            return final_result.astype(np.uint8)
            
        except Exception as e:
            print(f"테두리 제거 실패: {e}")
            return image

    def upscale_image_2x(self, image):
        """2x 업스케일링"""
        try:
            height, width = image.shape[:2]
            new_size = (width * 2, height * 2)
            upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
            return upscaled
        except:
            return image

    def create_perfect_thumbnail_ultimate(self, image, inner_bbox):
        """완벽한 썸네일 생성 (위아래 여백 완전 제거)"""
        try:
            if inner_bbox is None:
                # 웨딩링 영역 자동 감지
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    inner_bbox = (x, y, w, h)
                else:
                    # 중앙 영역 사용
                    h, w = image.shape[:2]
                    inner_bbox = (w//4, h//4, w//2, h//2)
            
            inner_x, inner_y, inner_w, inner_h = inner_bbox
            
            # 웨딩링 중심으로 넉넉한 영역 크롭
            margin = max(inner_w, inner_h) // 4
            crop_x = max(0, inner_x - margin)
            crop_y = max(0, inner_y - margin)
            crop_x2 = min(image.shape[1], inner_x + inner_w + margin)
            crop_y2 = min(image.shape[0], inner_y + inner_h + margin)
            
            cropped = image[crop_y:crop_y2, crop_x:crop_x2]
            
            if cropped.size == 0:
                # 전체 이미지 사용
                cropped = image
            
            # 1000×1300 비율로 리사이즈 (위아래 여백 최소화)
            target_w, target_h = 1000, 1300
            crop_h, crop_w = cropped.shape[:2]
            
            # 종횡비 계산
            ratio = min(target_w / crop_w, target_h / crop_h)
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            # 웨딩링이 화면 가득 차도록 크기 조정
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000×1300 캔버스에 배치 (위아래 여백 최소화)
            canvas = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
            
            # 상단에 조금 더 가깝게 배치 (위아래 여백 줄이기)
            start_y = max(0, (target_h - new_h) // 3)  # 1/3 지점에 배치
            start_x = (target_w - new_w) // 2
            
            # 범위 체크
            end_y = min(target_h, start_y + new_h)
            end_x = min(target_w, start_x + new_w)
            actual_h = end_y - start_y
            actual_w = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = resized[:actual_h, :actual_w]
            
            return canvas
            
        except Exception as e:
            print(f"썸네일 생성 실패: {e}")
            # 기본 썸네일 생성
            try:
                resized = cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
                return resized
            except:
                return np.full((1300, 1000, 3), 248, dtype=np.uint8)

def handler(event):
    """RunPod Serverless 메인 핸들러 - v16.5 Ultimate"""
    try:
        input_data = event.get("input", {})
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "message": f"v16.5 Ultimate 연결 성공: {input_data['prompt']}",
                "status": "ready",
                "version": "v16.5_ultimate",
                "features": ["검은색_선_완전제거", "v13.3_완전보정", "썸네일_여백제거", "샴페인골드_화이트화"]
            }
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64가 필요합니다"}
        
        # Base64 디코딩
        image_data = base64.b64decode(input_data["image_base64"])
        pil_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(pil_image.convert('RGB'))
        
        print(f"이미지 로딩 완료: {image_array.shape}")
        
        # 프로세서 초기화
        processor = UltimateWeddingRingProcessor()
        
        # 1단계: 금속 타입 및 조명 감지 (무조건 실행)
        metal_type = processor.detect_metal_type_ultimate(image_array)
        lighting = processor.detect_lighting_ultimate(image_array)
        print(f"감지 완료 - 금속: {metal_type}, 조명: {lighting}")
        
        # 2단계: v13.3 완전한 10단계 웨딩링 보정 (무조건 실행)
        enhanced_image = processor.enhance_wedding_ring_v13_3_complete(
            image_array, metal_type, lighting)
        print("v13.3 보정 완료")
        
        # 3단계: 검은색 테두리 감지 및 제거 (선택적)
        border_bbox, inner_bbox = processor.detect_black_border_ultimate(enhanced_image)
        if border_bbox is not None:
            enhanced_image = processor.remove_black_border_ultimate(
                enhanced_image, border_bbox, inner_bbox, lighting)
            print("검은색 테두리 제거 완료")
        else:
            print("검은색 테두리 없음 - 전체 이미지 보정 완료")
        
        # 4단계: 2x 업스케일링 (무조건 실행)
        upscaled_image = processor.upscale_image_2x(enhanced_image)
        print("2x 업스케일링 완료")
        
        # 5단계: 썸네일 생성 (무조건 실행)
        # 업스케일된 좌표로 조정
        upscaled_inner_bbox = None
        if inner_bbox is not None:
            ix, iy, iw, ih = inner_bbox
            upscaled_inner_bbox = (ix*2, iy*2, iw*2, ih*2)
        
        thumbnail = processor.create_perfect_thumbnail_ultimate(
            upscaled_image, upscaled_inner_bbox)
        print("썸네일 생성 완료")
        
        # 결과 인코딩
        # 메인 이미지
        main_pil = Image.fromarray(upscaled_image)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        print("인코딩 완료")
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "processing_info": {
                "metal_type": metal_type,
                "lighting": lighting,
                "border_detected": border_bbox is not None,
                "version": "v16.5_ultimate",
                "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                "thumbnail_size": "1000x1300",
                "features_applied": [
                    "v13.3_완전보정",
                    "검은색선제거" if border_bbox else "전체보정",
                    "2x업스케일링", 
                    "썸네일여백제거"
                ]
            }
        }
        
    except Exception as e:
        print(f"처리 중 에러: {e}")
        
        # 에러 시에도 무조건 실제 처리 시도 (Emergency 완전 금지)
        try:
            # 최소한의 처리라도 수행
            image_data = base64.b64decode(input_data.get("image_base64", ""))
            pil_image = Image.open(io.BytesIO(image_data))
            image_array = np.array(pil_image.convert('RGB'))
            
            # 기본 보정
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=15)
            upscaled = cv2.resize(enhanced, (enhanced.shape[1]*2, enhanced.shape[0]*2), 
                                 interpolation=cv2.INTER_LANCZOS4)
            
            # 기본 썸네일
            thumbnail = cv2.resize(upscaled, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            # 인코딩
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=90)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=90)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "status": "basic_processing_fallback",
                    "error": str(e),
                    "version": "v16.5_fallback"
                }
            }
        except:
            return {
                "error": f"완전 실패: {str(e)}",
                "version": "v16.5_ultimate"
            }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
