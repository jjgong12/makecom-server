import runpod
import json

def handler(event):
    """Debug handler to check input structure"""
    try:
        print(f"Event received: {json.dumps(event, indent=2)}")
        
        # Check various possible input structures
        input_data = event.get("input", {})
        print(f"Input data: {json.dumps(input_data, indent=2)[:500]}")  # First 500 chars
        
        # Check for image in different locations
        image = None
        if "image" in input_data:
            image = input_data["image"]
            print(f"Found image in input.image, length: {len(image) if image else 0}")
        elif "data" in input_data:
            image = input_data["data"]
            print(f"Found image in input.data, length: {len(image) if image else 0}")
        elif "base64" in input_data:
            image = input_data["base64"]
            print(f"Found image in input.base64, length: {len(image) if image else 0}")
        else:
            print("No image found in common locations")
            print(f"Available keys in input: {list(input_data.keys())}")
        
        return {
            "status": "debug_complete",
            "event_keys": list(event.keys()),
            "input_keys": list(input_data.keys()) if input_data else [],
            "image_found": image is not None,
            "image_length": len(image) if image else 0
        }
        
    except Exception as e:
        print(f"Debug error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "debug_error",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})
