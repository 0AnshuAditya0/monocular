import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/depth_anything_v2_small.onnx"
INPUT_SIZE = (518, 518)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_model():
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("Model loaded:", sess.get_inputs()[0].name)
    return sess

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(depth_output, original_shape):
    depth = depth_output[0][0]
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    depth_resized = cv2.resize(depth_uint8, (original_shape[1], original_shape[0]))
    
    depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)
    return depth_colored

def run_on_video(video_path):
    sess = load_model()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    frame_count = 0
    last_depth = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        inp = preprocess(frame)

        outputs = sess.run(None, {"pixel_values": inp})
        depth_colored = postprocess(outputs, (h, w))

        display = np.hstack([frame, depth_colored])
        cv2.imshow("RGB | Depth", display)
        
        frame_count += 1
        print(f"Frame {frame_count}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "street.mp4"   
    run_on_video(VIDEO_PATH)