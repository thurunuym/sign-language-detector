#  Real-Time Sign Language Detection using YOLOv8

This project demonstrates how to **train, export, and deploy a YOLOv8 model** to perform **real-time sign language detection** through the webcam.  

We use **five commonly used signs** for this project:
- 👋 **Hello**
- ✅ **Yes**
- ❌ **No**
- 🙏 **Thank you**
- 🤲 **Please**

---

## ✅ Prerequisites

Install the required Python libraries:
```bash
# Python and dependencies
pip install ultralytics opencv-python flask

# (Optional) Export to ONNX
pip install onnx onnxruntime
```
---

## ✅ Dataset

- The dataset was annotated using Roboflow
### Data Structure
```bash
dataset/
|-- train/
|   |-- images/
|   |-- labels/
|-- val/
|   |-- images/
|   |-- labels/
```
---
## ✅ Model Training 

The model can be trained on Jupyter Notebook, Google Colab, or a local Python environment:
```bash
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Use a pretrained YOLOv8 nano model
model.train(
    data='data.yaml',
    epochs=50,
    batch_size=16,
    imgsz=640,
    name='sign_detection',
    device='0'  # Set to 'cpu' if no GPU available
)
```

Model Export

```bash
model.export(format='onnx')
```

## ✅ Video Output
```bashfrom flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO(r"path to best.pt")
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, save=False)
        frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

```
This script launches a Flask server and streams your webcam with real-time sign detection using the trained YOLOv8 model.

---


- Use GPU for faster real-time inference

---





