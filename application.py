import tempfile
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from models.model_handler import load_model, predict_violence

router = APIRouter()
model = load_model("resources/rgb_model_new01.keras")

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        score = predict_violence(model, tmp_path)
        return JSONResponse(content={"violence_score": round(score * 100, 2)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)