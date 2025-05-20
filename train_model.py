from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import joblib

# ------------------------------------------------------------------------
# 1. Cargar el dataset
# ------------------------------------------------------------------------

# El dataset 'digits' contiene imágenes de dígitos del 0 al 9 (8x8 píxeles en escala de grises)
digits = datasets.load_digits()

# Extraemos las imágenes y las transformamos de 8x8 a un vector de 64 valores (flattening)
X = digits.images.reshape((len(digits.images), -1)) # Cada imagen se convierte en un vector de 64 características

# Extraemos las etiquetas correspondientes a cada imagen (número del 0 al 9)
y= digits.target

# ------------------------------------------------------------------------
# 2. Dividir en entrenamiento y prueba
# ------------------------------------------------------------------------

# Separamos 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------------------------------------------------------------
# 3. Entrenar el modelo
# ------------------------------------------------------------------------

## Crear modelo incremental
clf = SGDClassifier(loss="log_loss", random_state=42)

# Entrenar usando .partial_fit con todos los datos y pasando las clases posibles

clf.partial_fit(X_train, y_train, classes=list(range(10)))  # ✅ CORRECTO



# ------------------------------------------------------------------------
# 4. Guardar el modelo entrenado
# ------------------------------------------------------------------------

#guardar el modelo entrando en un archivo .pkl(pickle) usando joblib
joblib.dump(clf, "modelo_digits_sgd.pkl")

#Confirmacion visual
print("Modelo SGD entrenado y guardado como 'modelo_digits_sgd.pkl'")
