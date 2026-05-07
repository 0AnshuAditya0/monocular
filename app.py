import cv2
import numpy as np
import torch
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = depth_model.to(device).eval()

yolo = YOLO("yolov8n.pt")
print("Models loaded")

def make_bev(depth_raw, boxes_with_risk, bev_size=300):
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    for i in range(0, bev_size, bev_size // 5):
        cv2.line(bev, (0, i), (bev_size, i), (40, 40, 40), 1)
        cv2.line(bev, (i, 0), (i, bev_size), (40, 40, 40), 1)
    cam_x, cam_y = bev_size // 2, bev_size - 20
    cv2.drawMarker(bev, (cam_x, cam_y), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 15, 2)
    d_min = depth_raw.min()
    d_max = depth_raw.max()
    for (cx, depth_val, color, label) in boxes_with_risk:
        bev_x = int((cx / depth_raw.shape[1]) * bev_size)
        depth_norm = (depth_val - d_min) / (d_max - d_min + 1e-8)
        bev_y = int((1.0 - depth_norm) * (bev_size - 40)) + 10
        cv2.circle(bev, (bev_x, bev_y), 8, color, -1)
        cv2.putText(bev, label[:3], (bev_x + 10, bev_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(bev, "BEV MAP", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return bev

def process_frame(image):
    if image is None:
        return None, None, None

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    with torch.no_grad():
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs  = processor(images=pil_img, return_tensors="pt").to(device)
        outputs = depth_model(**inputs)
        depth_raw = outputs.predicted_depth.squeeze().cpu().numpy()

    depth_raw_resized = cv2.resize(depth_raw, (w, h))
    close_thresh = float(np.percentile(depth_raw_resized, 70))
    far_thresh   = float(np.percentile(depth_raw_resized, 40))

    depth_norm    = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
    depth_uint8   = (depth_norm * 255).astype(np.uint8)
    depth_resized = cv2.resize(depth_uint8, (w, h))
    depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)

    results = yolo(frame, verbose=False)[0]
    annotated = frame.copy()
    obj_count    = len(results.boxes)
    danger_count = 0
    boxes_with_risk = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls   = int(box.cls[0])
        label = yolo.names[cls]
        conf  = float(box.conf[0])
        cx    = (x1 + x2) // 2

        cx1 = x1 + (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cx2 = x2 - (x2 - x1) // 4
        cy2 = y2 - (y2 - y1) // 4
        region    = depth_raw_resized[cy1:cy2, cx1:cx2]
        depth_val = float(np.median(region)) if region.size > 0 else 0.0

        if depth_val > close_thresh:
            color, risk = (0, 0, 255), "DANGER"
            danger_count += 1
        elif depth_val > far_thresh:
            color, risk = (0, 165, 255), "WARNING"
        else:
            color, risk = (0, 255, 0), "SAFE"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {risk} {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        boxes_with_risk.append((cx, depth_val, color, label))

    cv2.putText(annotated, f"Objects: {obj_count}  Hazards: {danger_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    bev = make_bev(depth_raw_resized, boxes_with_risk, bev_size=300)

    annotated_rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    depth_rgb        = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    bev_rgb          = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)

    return annotated_rgb, depth_rgb, bev_rgb

with gr.Blocks(title="DepthGuard") as demo:
    gr.Markdown("# DepthGuard — Monocular Depth-Aware Collision Risk Detection")
    gr.Markdown("Upload an image or use webcam. Each object is labeled DANGER / WARNING / SAFE based on estimated depth.")

    with gr.Row():
        input_image = gr.Image(label="Input", sources=["webcam", "upload"], streaming=False)

    with gr.Row():
        out_detection = gr.Image(label="Detection + Risk")
        out_depth     = gr.Image(label="Depth Map")
        out_bev       = gr.Image(label="Bird's Eye View")

    input_image.change(
        fn=process_frame,
        inputs=input_image,
        outputs=[out_detection, out_depth, out_bev]
    )

    gr.Examples(
        examples=[["sample12.png"]],
        inputs=input_image
    )

demo.launch()

