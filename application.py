import tempfile
from typing import List

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from models.model_handler import *

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

            score_results = predict_violence_per_segment(model, tmp_path, threshold=0.5)
            for res in score_results:
                print(f"File name: {file.filename}")
                print("Detected violent segments:")
                print(f"From {res['start_time']}s to {res['end_time']}s: {round(res['score'], 2)}% violent")
                print(f"With the description: {res['description']}")

                results.append({"file_name": file.filename,
                                "start_time": res['start_time'],
                                "end_time": res['end_time'],
                                "description": res['description'],
                                "violence_score": round(res['score'], 2)})

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
