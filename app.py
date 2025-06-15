import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# v13.3 파라미터 (28쌍 학습 데이터 기반)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3, 'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5, 'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2, 'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1, 'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.05, 'white_overlay': 0.04,
            'sharpness': 1.12, 'color_temp_a': 1, 'color_temp_b': 0, 'original_blend': 0.22
        },
        'cool': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.08,
            'sharpness': 1.18, 'color_temp_a': 3, 'color_temp_b': 2, 'original_blend': 0.18
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,  # v15.3 화이트화
            'sharpness': 1.16, 'color_temp_a': -6, 'color_temp_b': -6, 'original_blend': 0.15  # 화이트골드 방향
        },
        'warm': {
            'brightness': 1.14, 'contrast': 1.08, 'white_overlay': 0.10,
            'sharpness': 1.14, 'color_temp_a': -4, 'color_temp_b': -4, 'original_blend': 0.17
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.14,
            'sharpness': 1.18, 'color_temp_a': -8, 'color_temp_b': -8, 'original_blend': 0.13
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2, 'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.13, 'contrast': 1.06, 'white_overlay': 0.03,
            'sharpness': 1.12, 'color_temp_a': 2, 'color_temp_b': 1, 'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.19, 'contrast': 1.12, 'white_overlay': 0.07,
            'sharpness': 1.16, 'color_temp_a': 4, 'color_temp_b': 3, 'original_blend': 0.20
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        logger.info("WeddingRingProcessor 초기화 완료")

    def detect_black_lines_safe(self, image):
        """안전한 검은색 선 감지 (배열 경계 체크 포함)"""
        try:
            h, w = image.shape[:2]
            logger.info(f"이미지 크기: {w}x{h}")
            
            # 이미지 크기 최소 요구사항 체크
            if h < 100 or w < 100:
                logger.warning("이미지가 너무 작음")
                return None, None
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 다중 threshold로 검은색 감지
            mask1 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]
            mask2 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)[1]
            mask3 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)[1]
            
            # 마스크 결합
            combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("검은색 영역을 찾을 수 없음")
                return None, None
            
            # 가장 큰 직사각형 영역 찾기
            best_rect = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # 너무 작은 영역 제외
                    continue
                
                # 외곽 사각형 구하기
                x, y, w, h = cv2.boundingRect(contour)
                
                # 경계 체크
                if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                    continue
                
                # 종횡비 체크 (너무 이상한 모양 제외)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue
                
                if area > best_area:
                    best_area = area
                    best_rect = (x, y, w, h)
            
            if best_rect is None:
                logger.warning("적절한 검은색 사각형을 찾을 수 없음")
                return None, None
            
            # 선택된 영역의 마스크 생성
            x, y, w, h = best_rect
            mask = np.zeros_like(gray)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            logger.info(f"검은색 선 감지 완료: {best_rect}")
            return mask, best_rect
            
        except Exception as e:
            logger.error(f"검은색 선 감지 중 오류: {str(e)}")
            return None, None

    def detect_actual_line_thickness_safe(self, combined_mask, bbox):
        """안전한 선 두께 감지 (경계 체크 및 예외 처리)"""
        try:
            x, y, w, h = bbox
            img_h, img_w = combined_mask.shape
            
            # 안전한 경계 체크
            thickness_samples = []
            
            # 상단 라인 (안전한 범위에서만)
            if y >= 10 and y < img_h - 10:
                top_start = max(0, y - 5)
                top_end = min(img_h, y + 15)
                left_bound = max(0, x)
                right_bound = min(img_w, x + w)
                
                if top_end > top_start and right_bound > left_bound:
                    top_line = combined_mask[top_start:top_end, left_bound:right_bound]
                    if top_line.size > 0:
                        top_thickness = np.sum(top_line > 0, axis=0)
                        if len(top_thickness) > 0:
                            thickness_samples.extend(top_thickness[top_thickness > 0])
            
            # 하단 라인
            if y + h >= 10 and y + h < img_h - 10:
                bottom_start = max(0, y + h - 15)
                bottom_end = min(img_h, y + h + 5)
                left_bound = max(0, x)
                right_bound = min(img_w, x + w)
                
                if bottom_end > bottom_start and right_bound > left_bound:
                    bottom_line = combined_mask[bottom_start:bottom_end, left_bound:right_bound]
                    if bottom_line.size > 0:
                        bottom_thickness = np.sum(bottom_line > 0, axis=0)
                        if len(bottom_thickness) > 0:
                            thickness_samples.extend(bottom_thickness[bottom_thickness > 0])
            
            # 좌측 라인
            if x >= 10 and x < img_w - 10:
                top_bound = max(0, y)
                bottom_bound = min(img_h, y + h)
                left_start = max(0, x - 5)
                left_end = min(img_w, x + 15)
                
                if bottom_bound > top_bound and left_end > left_start:
                    left_line = combined_mask[top_bound:bottom_bound, left_start:left_end]
                    if left_line.size > 0:
                        left_thickness = np.sum(left_line > 0, axis=1)
                        if len(left_thickness) > 0:
                            thickness_samples.extend(left_thickness[left_thickness > 0])
            
            # 우측 라인
            if x + w >= 10 and x + w < img_w - 10:
                top_bound = max(0, y)
                bottom_bound = min(img_h, y + h)
                right_start = max(0, x + w - 15)
                right_end = min(img_w, x + w + 5)
                
                if bottom_bound > top_bound and right_end > right_start:
                    right_line = combined_mask[top_bound:bottom_bound, right_start:right_end]
                    if right_line.size > 0:
                        right_thickness = np.sum(right_line > 0, axis=1)
                        if len(right_thickness) > 0:
                            thickness_samples.extend(right_thickness[right_thickness > 0])
            
            # 두께 계산 (안전한 방식)
            if len(thickness_samples) == 0:
                logger.warning("두께 샘플을 얻을 수 없음, 기본값 사용")
                return 80  # 기본값
            
            # 이상치 제거 (너무 크거나 작은 값)
            thickness_samples = [t for t in thickness_samples if 20 <= t <= 200]
            
            if len(thickness_samples) == 0:
                logger.warning("유효한 두께 샘플 없음, 기본값 사용")
                return 80
            
            thickness = int(np.median(thickness_samples))
            logger.info(f"감지된 선 두께: {thickness}픽셀 (샘플 수: {len(thickness_samples)})")
            
            # 범위 제한
            thickness = max(50, min(150, thickness))
            return thickness
            
        except Exception as e:
            logger.error(f"선 두께 감지 중 오류: {str(e)}")
            return 80  # 기본값 반환

    def detect_metal_type(self, image, mask=None):
        """금속 타입 감지"""
        try:
            if mask is not None:
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) == 0:
                    return 'white_gold'
                
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
            
            # 금속 분류
            if avg_hue < 15 or avg_hue > 165:
                if avg_sat > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_hue <= 35:
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'white_gold'
                
        except Exception as e:
            logger.error(f"금속 타입 감지 중 오류: {str(e)}")
            return 'white_gold'

    def detect_lighting(self, image):
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
                
        except Exception as e:
            logger.error(f"조명 감지 중 오류: {str(e)}")
            return 'natural'

    def enhance_wedding_ring_v13_3(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (28쌍 학습 데이터 기반)"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
            logger.info(f"보정 파라미터: {metal_type}-{lighting}")
            
            # PIL 변환
            pil_image = Image.fromarray(image)
            
            # 1. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 2. 대비 조정
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 3. 선명도 조정
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # NumPy 변환
            enhanced_array = np.array(enhanced)
            
            # 4. 하얀색 오버레이
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 5. LAB 색공간 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 6. 원본과 블렌딩
            original_blend = params['original_blend']
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                image, original_blend, 0
            )
            
            # 7. 노이즈 제거 (양방향 필터)
            final = cv2.bilateralFilter(final, 5, 80, 80)
            
            # 8. CLAHE 적용 (대비 제한 적응 히스토그램 평활화)
            lab = cv2.cvtColor(final, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9. 감마 보정
            gamma = 1.05
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            final = cv2.LUT(final, table)
            
            # 10. 미세 하이라이트 부스팅
            gray = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
            highlight_threshold = np.percentile(gray, 85)
            highlight_mask = (gray > highlight_threshold).astype(np.float32)
            for c in range(3):
                final[:, :, c] = np.clip(final[:, :, c] * (1 + highlight_mask * 0.08), 0, 255)
            
            return final.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"웨딩링 보정 중 오류: {str(e)}")
            return image

    def extract_and_enhance_ring_safe(self, image, line_mask, bbox):
        """안전한 웨딩링 추출 및 보정"""
        try:
            x, y, w, h = bbox
            img_h, img_w = image.shape[:2]
            
            # 실제 선 두께 측정 (안전한 방식)
            border_thickness = self.detect_actual_line_thickness_safe(line_mask, bbox)
            
            # 안전한 마진 계산
            safe_margin = max(30, border_thickness // 2)  # 최소 30픽셀 마진
            
            # 내부 영역 계산 (안전한 경계 체크)
            inner_x = max(0, x + safe_margin)
            inner_y = max(0, y + safe_margin)
            inner_x2 = min(img_w, x + w - safe_margin)
            inner_y2 = min(img_h, y + h - safe_margin)
            
            # 영역 크기 검증
            inner_w = inner_x2 - inner_x
            inner_h = inner_y2 - inner_y
            
            if inner_w <= 50 or inner_h <= 50:
                logger.warning(f"내부 영역이 너무 작음: {inner_w}x{inner_h}")
                # 더 작은 마진으로 재시도
                safe_margin = max(20, border_thickness // 3)
                inner_x = max(0, x + safe_margin)
                inner_y = max(0, y + safe_margin)
                inner_x2 = min(img_w, x + w - safe_margin)
                inner_y2 = min(img_h, y + h - safe_margin)
                inner_w = inner_x2 - inner_x
                inner_h = inner_y2 - inner_y
                
                if inner_w <= 20 or inner_h <= 20:
                    logger.error("웨딩링 영역을 안전하게 추출할 수 없음")
                    return image  # 원본 반환
            
            # 웨딩링 영역 추출
            ring_region = image[inner_y:inner_y2, inner_x:inner_x2]
            
            # 금속 타입 및 조명 감지
            metal_type = self.detect_metal_type(ring_region)
            lighting = self.detect_lighting(ring_region)
            
            logger.info(f"감지된 금속: {metal_type}, 조명: {lighting}")
            
            # v13.3 보정 적용
            enhanced_ring = self.enhance_wedding_ring_v13_3(ring_region, metal_type, lighting)
            
            # 결과 이미지 생성
            result = image.copy()
            result[inner_y:inner_y2, inner_x:inner_x2] = enhanced_ring
            
            return result
            
        except Exception as e:
            logger.error(f"웨딩링 추출 및 보정 중 오류: {str(e)}")
            return image

    def inpaint_black_lines_safe(self, image, line_mask):
        """안전한 검은색 선 제거"""
        try:
            # 마스크 유효성 검증
            if line_mask is None or np.sum(line_mask) == 0:
                logger.warning("유효하지 않은 마스크")
                return image
            
            # 마스크 전처리
            mask_clean = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            
            # 인페인팅 실행
            inpainted = cv2.inpaint(image, mask_clean, 3, cv2.INPAINT_NS)
            
            # 결과 검증
            if inpainted is None or inpainted.shape != image.shape:
                logger.warning("인페인팅 실패, 원본 반환")
                return image
            
            return inpainted
            
        except Exception as e:
            logger.error(f"인페인팅 중 오류: {str(e)}")
            return image

    def create_thumbnail_safe(self, image, bbox):
        """안전한 썸네일 생성"""
        try:
            if bbox is None:
                logger.warning("bbox가 없어 전체 이미지로 썸네일 생성")
                # 전체 이미지를 1000x1300으로 리사이즈
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            x, y, w, h = bbox
            img_h, img_w = image.shape[:2]
            
            # 30% 마진 추가
            margin_w = max(50, int(w * 0.3))
            margin_h = max(50, int(h * 0.3))
            
            # 안전한 경계 계산
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(img_w, x + w + margin_w)
            y2 = min(img_h, y + h + margin_h)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                logger.warning("크롭된 이미지가 비어있음")
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000x1300 리사이즈
            target_w, target_h = 1000, 1300
            crop_h, crop_w = cropped.shape[:2]
            
            if crop_h == 0 or crop_w == 0:
                logger.warning("유효하지 않은 크롭 크기")
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            # 비율 계산
            ratio = min(target_w / crop_w, target_h / crop_h)
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            # 리사이즈
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                logger.warning("유효하지 않은 리사이즈 크기")
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000x1300 캔버스에 중앙 배치
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            # 경계 체크
            if start_y >= 0 and start_x >= 0 and start_y + new_h <= target_h and start_x + new_w <= target_w:
                canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            else:
                logger.warning("캔버스 배치 실패, 전체 리사이즈로 대체")
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            
            return canvas
            
        except Exception as e:
            logger.error(f"썸네일 생성 중 오류: {str(e)}")
            # 안전한 fallback: 전체 이미지 리사이즈
            try:
                return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
            except:
                return image

def handler(event):
    """RunPod Serverless 메인 핸들러 - 완전 안전 버전"""
    try:
        logger.info("=== v15.3.1 안전 버전 시작 ===")
        
        input_data = event.get("input", {})
        
        # 기본 연결 테스트
        if "test" in input_data:
            return {
                "success": True,
                "message": "v15.3.1 안전 버전 연결 성공",
                "version": "v15.3.1-safety",
                "features": ["안전한 검은색 선 감지", "v13.3 웨딩링 보정", "안전 장치 적용"]
            }
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64가 필요합니다"}
        
        try:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            logger.info(f"이미지 로딩 완료: {image_array.shape}")
            
        except Exception as e:
            logger.error(f"이미지 디코딩 실패: {str(e)}")
            return {"error": f"이미지 디코딩 실패: {str(e)}"}
        
        # 프로세서 초기화
        processor = WeddingRingProcessor()
        
        # 1. 검은색 선 감지 (안전한 방식)
        line_mask, bbox = processor.detect_black_lines_safe(image_array)
        
        if line_mask is None or bbox is None:
            logger.warning("검은색 선을 찾을 수 없음 - 전체 이미지 처리")
            # 검은색 선이 없으면 전체 이미지를 기본 보정
            metal_type = processor.detect_metal_type(image_array)
            lighting = processor.detect_lighting(image_array)
            enhanced_image = processor.enhance_wedding_ring_v13_3(image_array, metal_type, lighting)
            
            # 2x 업스케일링
            height, width = enhanced_image.shape[:2]
            upscaled = cv2.resize(enhanced_image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # 기본 썸네일
            thumbnail = processor.create_thumbnail_safe(upscaled, None)
            
        else:
            # 2. 웨딩링 추출 및 보정 (안전한 방식)
            enhanced_image = processor.extract_and_enhance_ring_safe(image_array, line_mask, bbox)
            
            # 3. 2x 업스케일링
            height, width = enhanced_image.shape[:2]
            upscaled = cv2.resize(enhanced_image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # 업스케일된 마스크
            upscaled_mask = cv2.resize(line_mask, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
            upscaled_mask = np.where(upscaled_mask > 127, 255, 0).astype(np.uint8)
            
            # 4. 검은색 선 제거 (안전한 방식)
            final_image = processor.inpaint_black_lines_safe(upscaled, upscaled_mask)
            
            # 5. 썸네일 생성 (업스케일된 bbox 기준)
            scaled_bbox = (bbox[0] * 2, bbox[1] * 2, bbox[2] * 2, bbox[3] * 2)
            thumbnail = processor.create_thumbnail_safe(final_image, scaled_bbox)
            
            enhanced_image = final_image
        
        # 결과 인코딩
        try:
            # 메인 이미지
            main_pil = Image.fromarray(enhanced_image)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            logger.info("=== 처리 완료 ===")
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v15.3.1-safety",
                    "black_lines_detected": bbox is not None,
                    "bbox": bbox,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{enhanced_image.shape[1]}x{enhanced_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "safety_features": "활성화됨"
                }
            }
            
        except Exception as e:
            logger.error(f"결과 인코딩 실패: {str(e)}")
            return {"error": f"결과 인코딩 실패: {str(e)}"}
    
    except Exception as e:
        logger.error(f"핸들러 최상위 오류: {str(e)}")
        return {
            "error": f"처리 중 오류 발생: {str(e)}",
            "version": "v15.3.1-safety"
        }

# RunPod 서버리스 시작
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
