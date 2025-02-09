import yt_dlp
import cv2
import os
import gradio as gr
import tempfile
import time
from ultralytics import YOLO

# Create directories if they don't exist
DOWNLOAD_DIR = "downloads"
OUTPUT_DIR = "output"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO Model
model = YOLO("best.pt")

# Function to download video from YouTube
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
        # Ensure the file has an .mp4 extension
        video_filename = video_filename.replace(".webm", ".mp4").replace(".mkv", ".mp4")
        return video_filename

# Generator function to process video and yield intermediate output file paths
def process_video_streaming(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "Error: Unable to open video file."
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary output file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_filename = temp_file.name
    temp_file.close()  # Close it so cv2 can write to it

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

    # Storage for persistent bounding boxes with a timeout
    persistent_objects = {}
    timeout_frames = 10  # Number of frames to keep a box visible

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO inference on the current frame
        results = model(frame, conf=0.5)

        # Gather current frame detections (bounding boxes + labels)
        current_objects = {}
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                current_objects[(x1, y1, x2, y2)] = (label, timeout_frames)

        # Update persistent detections:
        new_persistent_objects = {}
        # Add newly detected objects with full timeout
        for box, (label, _) in current_objects.items():
            new_persistent_objects[box] = (label, timeout_frames)
        # For boxes not detected in this frame, reduce timeout
        for box, (label, remaining_time) in persistent_objects.items():
            if box not in new_persistent_objects:
                if remaining_time > 0:
                    new_persistent_objects[box] = (label, remaining_time - 1)
        persistent_objects = new_persistent_objects

        # Draw bounding boxes and labels on the frame
        for box, (label, _) in persistent_objects.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out.write(frame)

        # Every 30 frames (or on the last frame), flush the current output
        if frame_count % 200 == 0 or frame_count == total_frames:
            out.release()  # Ensure data is flushed to disk
            # Yield a plain string (file path) as an update
            yield temp_filename
            time.sleep(0.5)  # Brief pause before continuing
            # Reopen the VideoWriter in append mode (overwrite with new frames)
            out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

    cap.release()
    out.release()
    # Final yield with the completed video file
    yield temp_filename

# Gradio function that ties everything together
def process_youtube_video_stream(url):
    try:
        video_path = download_video(url)
        # Yield from the generator so that Gradio receives intermediate updates
        yield from process_video_streaming(video_path)
    except Exception as e:
        yield f"Error: {str(e)}"

# Gradio interface using Blocks with streaming enabled
with gr.Blocks() as app:
    gr.Markdown("# 🎥 Real-Time YOLO Object Detection on YouTube Videos")
    gr.Markdown("Enter a YouTube link to download and process the video in real-time. "
                "The processed video updates periodically while the processing is ongoing.")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video link")
        submit_button = gr.Button("Start Processing")

    output_video = gr.Video(label="Processed Video", streaming=True)

    submit_button.click(fn=process_youtube_video_stream, inputs=url_input, outputs=output_video)

# Launch the Gradio app (if you still see issues with queueing, try disabling the queue with app.queue(disable_queue=True))
app.launch()
