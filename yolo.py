import cv2
from ultralytics import YOLO

# Load YOLOv8 model (Nano version is fast, good for webcam)
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 or other index for external cams

# Loop to process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame, stream=True)

    for r in results:
        # Get the result with labels and boxes
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])  # Class ID
                conf = float(box.conf[0])  # Confidence score
                name = model.names[cls_id]  # Class name

                # Print the name in terminal
                print(f"Detected: {name} ({conf:.2f})")

        # Show annotated frame
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:  # 27 is ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
