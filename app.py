import streamlit as st
import cv2
import numpy as np
from moviepy import editor
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #ff9999, #66ccff);
        font-family: 'Times New Roman', Times, serif;
    }
    .copyright {
        text-align: center;
        margin-top: 2opx;
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def process_video(input_video_path, output_video_path):
    input_video = editor.VideoFileClip(input_video_path, audio=False)
    processed_video = input_video.fl_image(frame_processor)
    processed_video.write_videofile(output_video_path, audio=False)

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

def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500

    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

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

def pixel_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def main():
    st.title("Lane Line Detector 🛣️")

    st.write(
        "Welcome to the Lane Line Detector!"
    )
    st.write(
        "This Streamlit application is designed for detecting lane lines in videos and it only supports MP4 and MPEG4 video file formats."
    )

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()

        st.video(video_path)

        if st.button("Process Video"):
            with st.spinner('Processing video...'):
                output_video_path = 'output.mp4'  # You can change the output video path as needed
                process_video(video_path, output_video_path)
            st.success("Video processed successfully!")

            st.subheader("Processed Video")
            st.video(output_video_path)
    st.markdown('<div class="copyright">&copy; 2024 mani sankar pasala. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
