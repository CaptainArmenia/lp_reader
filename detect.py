from ultralytics import YOLO
import cv2

class LPDetector:
    def __init__(self):
        self.model = YOLO('weights/lp/best.pt')

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results
    
if __name__ == '__main__':
    detector = LPDetector()
    source = '/home/andy/Desktop/datasets/lo_valledor/P2 PATENTE_192.168.2.228_CAJA 2_20230824215959_20230824224034_24900911.mp4'
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = detector.detect(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
