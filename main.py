from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import tempfile

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin, nên giới hạn lại trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình Whisper
model = whisper.load_model("base")  # hoặc "small", "medium"

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Lấy phần mở rộng file gốc
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".mp3"  # fallback nếu không có extension
    
    # Lưu file tạm với đúng phần mở rộng
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Chuyển âm thanh thành text
        result = model.transcribe(tmp_path, language="vi")
        text = result["text"]
    except Exception as e:
        os.remove(tmp_path)
        return {"error": str(e)}
    
    # Xoá file tạm
    os.remove(tmp_path)

    return {"text": text}
