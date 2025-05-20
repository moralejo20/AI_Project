from distutils.command.upload import upload

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model import predict_image, update_model

app = FastAPI()

#Permitir solicitudes desde frontend React

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API de clasificación de imágenes funcionando"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await predict_image(file)
    return {"result": result}

@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    correct_label: int = Form(...)
):
    message = await update_model(file, correct_label)
    return {"status": "ok", "message": message}