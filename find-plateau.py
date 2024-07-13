import ffmpeg
from typing import Union
from scipy.ndimage import binary_closing
import numpy as np
import pandas as pd
import cv2
from itertools import zip_longest
from pathlib import Path
import open_clip
import torch
import tqdm
from PIL import Image
from typing import Tuple, List
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="metaclip_400m"
) 
model.to("cuda")


video_path = "driving-480p.mp4"


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Unable to read video")


fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"{fps=:}")

embeddings = []
with tqdm.tqdm(desc="done: ") as pbar:
    while True:
        # Capture frame-by-frame
        ret, bgr = cap.read()
        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        pbar.update(1)
        image = Image.fromarray(bgr[..., ::-1])
        pixel_values = preprocess(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            image_features = model.encode_image(pixel_values)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()
        embeddings.append(embedding)

sims = []
for emb, next_emb in zip_longest(
        embeddings,embeddings[1:]
    ):
    if next_emb is not None:
        sim = np.dot(emb, next_emb)
        sims.append(sim)


ser = pd.Series(sims)

# you can use lower window than min_length. ex. window = fps, min_length = fps * 3

window = min_length = fps * 3 

ser_std = ser.rolling(window=window, center=True).std()

# 10% is good default. you can lower it for better recall
threshold = ser_std.quantile(0.1)

# find plateau
signal = ser_std < threshold

# binary closing to fill the gap. you can refer to 
signal = binary_closing(signal, iterations=window)
signal = pd.Series(signal)


segments = []

# find consecutive true blocks. pandas make this oneline. you can i add explanation in README file
for group, gdf in signal[signal==True].groupby( (signal==False).cumsum()):
    start = gdf.index[0].item()
    length = len(gdf)
    if length > min_length:
        segments.append((start, length))
 


def convert_seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


for start, length in segments:
    print(f"{convert_seconds_to_hhmmss(start//fps)} - {convert_seconds_to_hhmmss((start+length)//fps)}")




# Function to extract video segments
def extract_video_segments(
    video_path: str,
    segments: List[Tuple[int, int]],
    out_dir: Union[str,Path,None] = "./",
    convert_x264: bool = False
):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    Path(out_dir).mkdir(parents=True, exist_ok=True)





    for start_frame, length in segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        prefix = convert_seconds_to_hhmmss(start_frame//fps).replace(':', '-')
        dest = Path(out_dir) / f"{prefix}_segment.mp4"
        dest = str(dest)

        # export video segment
        segment_out = cv2.VideoWriter(
            dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

        for _ in range(length):
            ret, frame = cap.read()
            if not ret:
                break
            segment_out.write(frame)
        segment_out.release()

        # for visualize in browser/vscode, you need to reencode it to x264
        if convert_x264:
            ffmpeg.input(dest).output(
                dest.replace(".mp4", "_x264.mp4"),
                vcodec="libx264",
                preset="fast",
                crf=23,
                an=None,
            ).run(overwrite_output=True)
        Path(dest).unlink()
    cap.release()


out_dir = Path("./artifact")
extract_video_segments(video_path, segments, out_dir=out_dir, convert_x264=True)


