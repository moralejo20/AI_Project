from PIL import Image
import io
import numpy as np
import joblib
import os
import uuid
from sklearn.linear_model import SGDClassifier

# Ruta del modelo
model_path = os.path.join(os.path.dirname(__file__), "modelo_digits_sgd.pkl")
feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
os.makedirs(feedback_dir, exist_ok=True)

# Cargar modelo inicial o crear uno nuevo si no existe
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.partial_fit(np.zeros((1, 64)), [0], classes=list(range(10)))
    joblib.dump(model, model_path)

def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert('L')
    image = image.resize((8, 8))
    image_array = np.array(image)
    image_array = (image_array / 255.0) * 16
    image_array = 16 - image_array
    image_array = np.clip(image_array, 0, 16).astype(int)
    return image_array.flatten().reshape(1, -1)

async def predict_image(file):
    contents = await file.read()
    image_vector = preprocess_image(contents)
    prediction = model.predict(image_vector)[0]
    return str(prediction)

async def update_model(file, correct_label: int):
    contents = await file.read()
    image_vector = preprocess_image(contents)

    # Generar ID único para el ejemplo
    image_id = str(uuid.uuid4())

    # Guardar el vector de imagen en feedback_data/
    np.save(os.path.join(feedback_dir, f"{image_id}.npy"), image_vector)

    # Guardar etiqueta correspondiente en feedback_data/labels.txt
    with open(os.path.join(feedback_dir, "labels.txt"), "a") as f:
        f.write(f"{image_id},{correct_label}\n")

    # Reentrenar modelo desde cero con todos los ejemplos corregidos
    X, y = [], []
    label_path = os.path.join(feedback_dir, "labels.txt")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                img_id, label = line.strip().split(",")
                vector_path = os.path.join(feedback_dir, f"{img_id}.npy")
                if os.path.exists(vector_path):
                    img_vector = np.load(vector_path)
                    X.append(img_vector.flatten())
                    y.append(int(label))

    if len(X) > 0:
        global model
        model = SGDClassifier(loss="log_loss", random_state=42)
        model.partial_fit(np.zeros((1, 64)), [0], classes=list(range(10)))  # dummy init
        model.partial_fit(np.array(X), np.array(y), classes=list(range(10)))
        joblib.dump(model, model_path)
        return f"Modelo actualizado con {len(X)} ejemplos incluyendo el nuevo"
    else:
        return "No se encontraron datos de entrenamiento."
