import runpod

print("=== HANDLER.PY LOADED ===")

def handler(event):
    print("=== HANDLER FUNCTION CALLED ===")
    return {"status": "ok", "message": "Handler working"}

print("=== STARTING RUNPOD ===")
runpod.serverless.start({"handler": handler})
