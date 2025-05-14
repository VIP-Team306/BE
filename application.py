from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models.model_handler import load_model, predict_violence

app = FastAPI()
# model = load_model("resources/my_model.keras")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # score = predict_violence(model, await file.read())
        return JSONResponse(content={"violence_score": round(0.5, 4)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

