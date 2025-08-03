from ultralytics import YOLO

# Cargar un modelo pre-entrenado como punto de partida
model = YOLO('yolov8n.pt')

# Entrenar el modelo con tu dataset (minúsculo)
# 'data' apunta a tu archivo .yaml
# 'epochs' lo ponemos muy bajo (ej. 10) porque es solo una prueba
# 'imgsz' es el tamaño de la imagen (640 es un tamaño común para YOLOv8)
# 'batch' se pone bajo porque solo tienes una imagen
results = model.train(data='obstaculos.yaml', epochs=10, imgsz=640, batch=1)

print("\nEntrenamiento de prueba completado. Revisa la carpeta 'runs/detect/train' para ver los logs.")
print("Recuerda: Este modelo NO es útil para detecciones reales con solo una imagen de entrenamiento.")