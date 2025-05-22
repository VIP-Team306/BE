import tempfile
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

from models.model_handler import load_model, predict_violence

router = APIRouter()
model = load_model("resources/rgb_model_new001.h5")


@router.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            score = predict_violence(model, tmp_path)
            results.append({"file_name": file.filename, "violence_score": round(score * 100, 2)})
        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
