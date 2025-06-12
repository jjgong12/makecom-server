import runpod

def handler(event):
    """진짜진짜 간단한 핸들러 - 텍스트만"""
    
    # 입력 받기
    input_data = event.get("input", {})
    
    # 간단한 응답만
    return {
        "message": "RunPod 작동 성공!",
        "received": input_data,
        "status": "OK",
        "test": "간단테스트완료"
    }

# 시작
runpod.serverless.start({"handler": handler})
