# Lane Line Detection Using OpenCV

## Table of contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Explanation](#code-explanation)
- [Example Output](#example-output)
- [Note](#note)
- [License](#license)

## Introduction

The Python script is designed to detect and draw lane lines in video footage using OpenCV (Open Source Computer vision) library. Lane line detection is a crtitical component in various computer vision application, such as self-driving cars. The script processes video frames to identify and highlight lane markings, making it a valuable tool in the field of computer vision.

## Prerequisites

Before using this code, ensure that you have the following prerequisites:

- **Python**: The script is written in python and requires a python environment.
- **OpenCV (cv2)**: OpenCv must be installed to perform various image processing tasks and lane detection.
- **MoviePy**: MoviePy is used for video editing and processing, so make sure it's installed.

## Getting Started

1. Clone this repository or create a new python script.
2. Place your input video files in the same directory as the script. In the provided code, the input videos are 'challenge_video.mp4' and 'challenge_video1.mp4'.
3. Run the Python script. The code will perform lane line detection on the unknown videos and display the results.

## Code Explanation

The code is divided into the following sections:

### Library Imports

Importing the required Python libraries, including `numpy`, `pandas`, `cv2`, `google`, and `moviepy`.

```bash
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
from moviepy import editor
import moviepy
```

### Video Processing

The function is responsible for processing a video. It takes two parameters, `test` and `output`.

```bash
def process_video(test, output):
  input_video = editor.VideoFileClip(test, audio=False)
  processed = input_video.fl_image(frame_processor)
  processed.write_videofile(output, audio=False)
```

