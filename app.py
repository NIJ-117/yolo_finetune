import yt_dlp
import cv2
import os
import gradio as gr
from ultralytics import YOLO

# Define paths
DOWNLOAD_DIR = "downloads"
OUTPUT_DIR = "output"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO Model
model = YOLO("best.pt")

# Function to download video
def download_video(url):
    output_path = f"{DOWNLOAD_DIR}/%(title)s.%(ext)s"
    ydl_opts = {
        "format": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_filename = ydl.prepare_filename(info)
        video_filename = video_filename.replace(".webm", ".mp4").replace(".mkv", ".mp4")  # Ensure mp4 format
        return video_filename

# Function to process video with YOLO
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video path
    output_path = f"{OUTPUT_DIR}/processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    persistent_objects = {}  # Store bounding boxes with timeout
    timeout_frames = 10  # Frames before object disappears

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.05)
        current_objects = {}

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                current_objects[(x1, y1, x2, y2)] = (label, timeout_frames)

        new_persistent_objects = {}
        for box, (label, _) in current_objects.items():
            new_persistent_objects[box] = (label, timeout_frames)

        for box, (label, remaining_time) in persistent_objects.items():
            if box not in new_persistent_objects:
                if remaining_time > 0:
                    new_persistent_objects[box] = (label, remaining_time - 1)

        persistent_objects = new_persistent_objects

        for box, (label, remaining_time) in persistent_objects.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Gradio function
def process_youtube_video(url):
    try:
        video_path = download_video(url)
        processed_video_path = process_video(video_path)
        return processed_video_path
    except Exception as e:
        return str(e)

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🎥 YOLO Object Detection on YouTube Videos")
    gr.Markdown("Enter a YouTube link to download the video and process it with YOLO.")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video link")
        submit_button = gr.Button("Process Video")

    output_video = gr.Video(label="Processed Video")

    submit_button.click(fn=process_youtube_video, inputs=url_input, outputs=output_video)

# Launch Gradio App
app.launch()
