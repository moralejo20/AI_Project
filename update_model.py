import joblib
import numpy as np
import uuid
import os

from model import preprocess_image, model_path

feedback_dir = "feedback_data"
os.makedirs(feedback_dir, exist_ok=True)

async def update_model(file, correct_label: int):
    contents = await file.read()
    image_vector = preprocess_image(contents)

    # Guardar imagen y etiqueta
    image_id = str(uuid.uuid4())
    np.save(os.path.join(feedback_dir, f"{image_id}.npy"), image_vector)
    with open(os.path.join(feedback_dir, "labels.txt"), "a") as f:
        f.write(f"{image_id},{correct_label}\n")

    # Reentrenar desde cero con todos los ejemplos
    from sklearn.linear_model import SGDClassifier
    global model
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.partial_fit(np.zeros((1, 64)), [0], classes=list(range(10)))  # boot

    # Cargar ejemplos guardados
    X, y = [], []
    with open(os.path.join(feedback_dir, "labels.txt"), "r") as f:
        for line in f:
            img_id, label = line.strip().split(",")
            img_array = np.load(os.path.join(feedback_dir, f"{img_id}.npy"))
            X.append(img_array.flatten())
            y.append(int(label))

    model.partial_fit(np.array(X), np.array(y), classes=list(range(10)))

    joblib.dump(model, model_path)
    return f"Modelo reentrenado con {len(X)} ejemplos incluyendo el nuevo"
