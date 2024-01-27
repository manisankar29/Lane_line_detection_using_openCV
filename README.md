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
2. Place your input video files in the same directory as the script. In the provided code, the input videos are [challenge_video.mp4](challenge_video.mp4) and [challenge_video1.mp4](challenge_video1.mp4).
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

### Frame Processing

The `frame_processor` function is a critical part of the pipeline for lane detection. It converts the image to grayscale using `cv2.cvtColor`. Applies a region of interest mask using the `region_selection` function. Uses the Hough line transform in the `hough_transform` function to detect lines in the region. Finally, it draws the detected lane lines using the`draw_lane_lines` function.

```bash
def frame_processor(image):
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel_size = 5
  blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
  low_t = 50
  high_t = 150
  edges = cv2.Canny(blur, low_t, high_t)
  region = region_selection(edges)
  hough = hough_transform(region)
  result = draw_lane_lines(image, lane_lines(image, hough))
  return result
```

### Region Selection

This function is responsible for creating a mask for the region of interest in the image.

```bash
def region_selection(image):
  mask = np.zeros_like(image)

  if len(image.shape) > 2:
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
  else:
    ignore_mask_color = 255

  rows, cols = image.shape[:2]
  bottom_left  = [cols * 0.1, rows * 0.95]
  top_left     = [cols * 0.4, rows * 0.6]
  bottom_right = [cols * 0.9, rows * 0.95]
  top_right    = [cols * 0.6, rows * 0.6]
  vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

  cv2.fillPoly(mask, vertices, ignore_mask_color)

  masked_image = cv2.bitwise_and(image, mask)

  return masked_image
```

### Hough Transform

This function performs the Hough line transform to detect lines in the provided image. It takes an edge-detected image as input and returns an array of detected lines. The function uses parameters such as rho, theta, threshold, miniLineLength, and maxLineGap to control the line detection process.

```bash
def hough_transform(image):
  rho = 1
  theta = np.pi/180
  threshold = 20
  minLineLength = 20
  maxLineGap = 500

  return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
```

### Average Slope Intercept

This function processes the lines detected by the Hough transform and calculates the average slope and intercept for the left and right lane lines.

```bash
def average_slope_intercept(lines):
  left_lines    = []
  left_weights  = []
  right_lines   = []
  right_weights = []

  for line in lines:
    for x1, y1, x2, y2 in line:
      if x1 == x2:
        continue

      slope = (y2 - y1) / (x2 - x1)
      intercept = y1 - (slope * x1)
      length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

      if slope < 0:
        left_lines.append((slope, intercept))
        left_weights.append((length))
      else:
        right_lines.append((slope, intercept))
        right_weights.append((length))

  left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
  right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

  return left_lane, right_lane
```

### Pixel Points

This function calculates the pixel coordinates of a line given it's slope, intercept, and y-coordinates(y1 and y2).

```bash
def pixel_points(y1, y2, line):
  if line is None:
    return None

  slope, intercept = line
  x1 = int((y1 - intercept) / slope)
  x2 = int((y2 - intercept) / slope)
  y1 = int(y1)
  y2 = int(y2)
  return ((x1, y1), (x2, y2))
```

### Lane Lines

This function uses the average slope and intercept information to compute the pixel coordinates of the left and right lane lines.

```bash
def lane_lines(image, lines):
  left_lane, right_lane = average_slope_intercept(lines)
  y1 = image.shape[0]
  y2 = y1 * 0.6
  left_line = pixel_points(y1, y2, left_lane)
  right_line = pixel_points(y1, y2, right_lane)
  return left_line, right_line
```

### Drawing Lane Lines

This function takes an image and the pixel coordinates of the lane lines and draws the lines on the image using OpenCV's `cv2.line` function. It then returns the original image with the drawn lane lines.

```bash
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
  line_image = np.zeros_like(image)
  for line in lines:
    if line is not None:
      cv2.line(line_image, *line, color, thickness)

  return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
```

### Processing Videos

Finally, the script processes two input videos, 'challenge_video.mp4' and 'challenge_video1.mp4' by invoking the `process_video` function. 

[TEST VIDEOS](Test_videos)

The processed videos are saved as 'output.mp4' and 'output1.mp4' respectively.

```bash
process_video('challenge_video.mp4','output.mp4')
process_video('challenge_video1.mp4','output1.mp4')
```

## Example Output

output.mp4

https://github.com/manisankar29/Lane_line_detection/assets/138246745/0a4a13a4-95c4-4a89-ab04-f689c623c9f3

output1.mp4

https://github.com/manisankar29/Lane_line_detection/assets/138246745/2cc322b0-02fc-4e12-93ba-263c1906c902

## Note 

- The accuracy of lane line detection depends on the quality and diversity of the test videos. It may not be perfect in all cases.
- Ensure that the video file paths are correctly specified in the code.

Feel free to modify the code to suit your needs and add more lane line videos for detection.

Enjoy using the lane line detection script!

If you encounter any issues or have questions, feel free to reach out for assistance.

```vbnet
You can include this README.md file in your project's repository, and it will serve as a guide for users who want to use the provided lane line detection code.
```

## License

[MIT License](LICENSE)
