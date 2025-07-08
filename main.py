import cv2
import numpy as np
from ultralytics import YOLO
from modulocvdsa import Sort


def main() -> None:
    """Executa o sistema de monitoramento de gado com detecção e rastreamento em tempo real."""

    # Inicializa captura de vídeo (pode ser webcam ou vídeo salvo)
    cap = cv2.VideoCapture("data/gado.mp4")
    # Para usar a webcam, comente a linha acima e descomente abaixo:
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 1280)
    # cap.set(4, 720)

    # Carrega modelo YOLOv8
    modelo = YOLO("yolov8n.pt")

    # Dicionário de classes traduzidas
    class_names = {
        'cow': 'Vaca', 'dog': 'Cachorro', 'horse': 'Cavalo', 'sheep': 'Ovelha',
        'pig': 'Porco', 'deer': 'Cervo', 'person': 'Humano', 'bear': 'Urso',
        'zebra': 'Zebra', 'elephant': 'Elefante'
    }

    # Inicializa rastreador SORT
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Loop de leitura dos frames
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = np.empty((0, 5))
        results = modelo(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                class_name = modelo.names.get(cls, 'Desconhecido')
                nome_pt = class_names.get(class_name, class_name)
                label = f"{nome_pt} ({conf * 100:.1f}%)"

                new_det = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_det))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        resultados_tracker = tracker.update(detections)

        for track in resultados_tracker:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.putText(frame, f"ID {track_id}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Monitoramento de Gado", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
