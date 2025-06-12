import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, jsonify, send_file
import base64
import io
import json
import logging
from datetime import datetime
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeddingRingEnhancerV13_3:
    def __init__(self):
        """
        v13.3 Natural Wedding Ring Enhancement System
        - Based on 28-pair learning data
        - Optimized parameters (closer to version 10)
        - 4 metal types: white_gold, rose_gold, champagne_gold, yellow_gold
        - 3 lighting conditions: natural, warm, cool
        """
        
        # v13.3 Parameters (adjusted towards version 10 - more natural)
        self.enhancement_params = {
            'white_gold': {
                'natural': {
                    'brightness': 1.18,      # 1.20 → 1.18 (more natural)
                    'contrast': 1.12,        # 1.14 → 1.12 (softer)
                    'white_overlay': 0.09,   # 11% → 9% (subtle)
                    'sharpness': 1.15,       # 1.17 → 1.15 (moderate)
                    'color_temp_a': -3,      # -4 → -3 (conservative)
                    'color_temp_b': -3,      # -4 → -3 (conservative)
                    'original_blend': 0.15   # 12% → 15% (respect original)
                },
                'warm': {
                    'brightness': 1.22,
                    'contrast': 1.15,
                    'white_overlay': 0.12,
                    'sharpness': 1.18,
                    'color_temp_a': -5,
                    'color_temp_b': -5,
                    'original_blend': 0.12
                },
                'cool': {
                    'brightness': 1.16,
                    'contrast': 1.10,
                    'white_overlay': 0.08,
                    'sharpness': 1.12,
                    'color_temp_a': -2,
                    'color_temp_b': -2,
                    'original_blend': 0.18
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.15,
                    'contrast': 1.08,
                    'white_overlay': 0.06,
                    'sharpness': 1.12,
                    'color_temp_a': 2,
                    'color_temp_b': 8,
                    'original_blend': 0.18
                },
                'warm': {
                    'brightness': 1.12,
                    'contrast': 1.05,
                    'white_overlay': 0.04,
                    'sharpness': 1.08,
                    'color_temp_a': 1,
                    'color_temp_b': 5,
                    'original_blend': 0.20
                },
                'cool': {
                    'brightness': 1.20,
                    'contrast': 1.12,
                    'white_overlay': 0.08,
                    'sharpness': 1.15,
                    'color_temp_a': 4,
                    'color_temp_b': 12,
                    'original_blend': 0.15
                }
            },
            'champagne_gold': {
                'natural': {
                    'brightness': 1.16,
                    'contrast': 1.10,
                    'white_overlay': 0.07,
                    'sharpness': 1.14,
                    'color_temp_a': 1,
                    'color_temp_b': 3,
                    'original_blend': 0.16
                },
                'warm': {
                    'brightness': 1.14,
                    'contrast': 1.08,
                    'white_overlay': 0.05,
                    'sharpness': 1.11,
                    'color_temp_a': -1,
                    'color_temp_b': 2,
                    'original_blend': 0.18
                },
                'cool': {
                    'brightness': 1.19,
                    'contrast': 1.13,
                    'white_overlay': 0.09,
                    'sharpness': 1.16,
                    'color_temp_a': 2,
                    'color_temp_b': 5,
                    'original_blend': 0.14
                }
            },
            'yellow_gold': {
                'natural': {
                    'brightness': 1.17,
                    'contrast': 1.11,
                    'white_overlay': 0.08,
                    'sharpness': 1.13,
                    'color_temp_a': 3,
                    'color_temp_b': 8,
                    'original_blend': 0.17
                },
                'warm': {
                    'brightness': 1.13,
                    'contrast': 1.07,
                    'white_overlay': 0.05,
                    'sharpness': 1.09,
                    'color_temp_a': 1,
                    'color_temp_b': 5,
                    'original_blend': 0.19
                },
                'cool': {
                    'brightness': 1.21,
                    'contrast': 1.14,
                    'white_overlay': 0.10,
                    'sharpness': 1.17,
                    'color_temp_a': 5,
                    'color_temp_b': 12,
                    'original_blend': 0.13
                }
            }
        }
    
    def detect_metal_type(self, image):
        """
        Detect metal type based on HSV color analysis
        Returns: white_gold, rose_gold, champagne_gold, or yellow_gold
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Focus on middle area (50% of image)
            h, w = hsv.shape[:2]
            center_hsv = hsv[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate average hue and saturation
            avg_hue = np.mean(center_hsv[:, :, 0])
            avg_sat = np.mean(center_hsv[:, :, 1])
            avg_val = np.mean(center_hsv[:, :, 2])
            
            # Metal type classification based on HSV values
            if avg_sat < 30:  # Low saturation - white metals
                if avg_val > 180:
                    return 'white_gold'
                else:
                    return 'white_gold'
            elif avg_hue < 15 or avg_hue > 165:  # Red-pink range
                return 'rose_gold'
            elif avg_hue < 30:  # Yellow-orange range
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'champagne_gold'  # Default safe choice
                
        except Exception as e:
            logger.warning(f"Metal detection failed: {e}")
            return 'champagne_gold'  # Safe default
    
    def detect_lighting_condition(self, image):
        """
        Detect lighting condition based on LAB color analysis
        Returns: natural, warm, or cool
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Calculate average A and B channels
            avg_a = np.mean(lab[:, :, 1])  # Green-Red axis
            avg_b = np.mean(lab[:, :, 2])  # Blue-Yellow axis
            
            # Lighting classification
            if avg_b > 132:  # Warm lighting (yellowish)
                return 'warm'
            elif avg_b < 124:  # Cool lighting (bluish)
                return 'cool'
            else:  # Neutral lighting
                return 'natural'
                
        except Exception as e:
            logger.warning(f"Lighting detection failed: {e}")
            return 'natural'  # Safe default
    
    def apply_white_overlay(self, image, strength):
        """
        Apply subtle white overlay for natural brightness enhancement
        """
        if strength <= 0:
            return image
            
        # Create white overlay
        white_overlay = np.full_like(image, 255, dtype=np.uint8)
        
        # Blend with original
        alpha = min(strength, 0.15)  # Maximum 15% overlay
        result = cv2.addWeighted(image, 1 - alpha, white_overlay, alpha, 0)
        
        return result
    
    def adjust_color_temperature(self, image, temp_a, temp_b):
        """
        Adjust color temperature in LAB color space
        """
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32)
            
            # Adjust A and B channels
            lab[:, :, 1] += temp_a  # A channel (green-red)
            lab[:, :, 2] += temp_b  # B channel (blue-yellow)
            
            # Clip values to valid range
            lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
            
            # Convert back to RGB
            lab = lab.astype(np.uint8)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return result
            
        except Exception as e:
            logger.warning(f"Color temperature adjustment failed: {e}")
            return image
    
    def enhance_wedding_ring_v13_3(self, image_data):
        """
        Main enhancement function for v13.3 system
        """
        try:
            start_time = datetime.now()
            
            # Decode image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # Convert to RGB and resize for processing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Memory optimization - resize if too large
            original_size = image.size
            if max(original_size) > 2048:
                ratio = 2048 / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Detect metal type and lighting condition
            metal_type = self.detect_metal_type(img_array)
            lighting = self.detect_lighting_condition(img_array)
            
            logger.info(f"Detected: {metal_type} under {lighting} lighting")
            
            # Get enhancement parameters
            params = self.enhancement_params[metal_type][lighting]
            
            # Step 1: Noise reduction
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # Step 2: Basic enhancements using PIL
            enhanced_image = Image.fromarray(img_array)
            
            # Brightness enhancement
            brightness_enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = brightness_enhancer.enhance(params['brightness'])
            
            # Contrast enhancement
            contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = contrast_enhancer.enhance(params['contrast'])
            
            # Sharpness enhancement
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = sharpness_enhancer.enhance(params['sharpness'])
            
            # Convert back to array for advanced processing
            enhanced_array = np.array(enhanced_image)
            
            # Step 3: White overlay for natural brightness
            enhanced_array = self.apply_white_overlay(enhanced_array, params['white_overlay'])
            
            # Step 4: Color temperature adjustment
            enhanced_array = self.adjust_color_temperature(
                enhanced_array, 
                params['color_temp_a'], 
                params['color_temp_b']
            )
            
            # Step 5: Subtle highlight boosting
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Boost only the brightest 15% of pixels by 8%
            threshold = np.percentile(l_channel, 85)
            mask = l_channel >= threshold
            l_channel[mask] *= 1.08
            l_channel = np.clip(l_channel, 0, 255)
            
            lab[:, :, 0] = l_channel.astype(np.uint8)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Step 6: Blend with original for naturalness
            original_array = np.array(image)
            blend_ratio = params['original_blend']
            final_array = cv2.addWeighted(
                enhanced_array, 1 - blend_ratio, 
                original_array, blend_ratio, 0
            )
            
            # Convert back to PIL Image
            final_image = Image.fromarray(final_array)
            
            # Resize back to original size if needed
            if final_image.size != original_size:
                final_image = final_image.resize(original_size, Image.Resampling.LANCZOS)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"v13.3 Enhancement completed in {processing_time:.2f}s")
            
            return final_image, {
                'metal_type': metal_type,
                'lighting': lighting,
                'processing_time': processing_time,
                'parameters_used': params,
                'version': 'v13.3'
            }
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            raise

# Flask application routes
@app.route('/')
def home():
    """Homepage with system information"""
    return jsonify({
        'service': 'Wedding Ring Enhancement API v13.3',
        'status': 'active',
        'version': 'v13.3',
        'description': 'Natural wedding ring enhancement based on 28-pair learning data',
        'endpoints': {
            '/health': 'Health check',
            '/enhance_wedding_ring_advanced': 'Main enhancement endpoint (v13.3)',
            '/enhance_wedding_ring_v6': 'Legacy endpoint (compatibility)',
            '/enhance_wedding_ring_binary': 'Binary output endpoint',
            '/enhance_wedding_ring_segmented': 'Legacy segmented endpoint',
            '/enhance_wedding_ring_natural': 'Legacy natural endpoint'
        },
        'metals_supported': ['white_gold', 'rose_gold', 'champagne_gold', 'yellow_gold'],
        'lighting_conditions': ['natural', 'warm', 'cool'],
        'features': [
            'Auto metal type detection',
            'Auto lighting condition analysis', 
            'v13.3 natural enhancement parameters',
            'Memory optimized processing',
            'Compatible with Make.com workflow'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': 'v13.3',
        'timestamp': datetime.now().isoformat(),
        'service': 'Wedding Ring Enhancement API'
    })

# Main enhancement endpoint
@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """Main v13.3 enhancement endpoint"""
    return process_enhancement_request()

# Compatibility endpoints (all use v13.3 system)
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """Legacy v6 endpoint - now uses v13.3"""
    return process_enhancement_request()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """Binary output endpoint - uses v13.3"""
    return process_enhancement_request()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """Legacy segmented endpoint - now uses v13.3"""
    return process_enhancement_request()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """Legacy natural endpoint - now uses v13.3"""
    return process_enhancement_request()

def process_enhancement_request():
    """
    Common function to process enhancement requests
    All endpoints now use v13.3 system for consistency
    """
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'Missing image_base64 field'}), 400
        
        # Initialize enhancer
        enhancer = WeddingRingEnhancerV13_3()
        
        # Process enhancement
        enhanced_image, metadata = enhancer.enhance_wedding_ring_v13_3(data['image_base64'])
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        enhanced_image.save(
            buffer, 
            format='JPEG', 
            quality=95, 
            optimize=True,
            progressive=True
        )
        buffer.seek(0)
        
        # Return binary image data (compatible with Make.com)
        return send_file(
            buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'enhanced_v13_3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        )
        
    except Exception as e:
        logger.error(f"Enhancement request failed: {e}")
        return jsonify({
            'error': 'Enhancement failed',
            'message': str(e),
            'version': 'v13.3'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
