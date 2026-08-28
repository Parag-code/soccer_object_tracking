import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Soccer Object Tracking",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Multi-Object Detection & Persistent ID Tracking")
st.write(
    "YOLO-based soccer object detection with ByteTrack and "
    "BoT-SORT + Re-ID + GMC."
)

st.divider()

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Detection Model", "YOLO")

with col2:
    st.metric("Baseline Tracker", "ByteTrack")

with col3:
    st.metric("Enhanced Tracker", "BoT-SORT + Re-ID + GMC")

st.divider()

st.header("Final Tracking Result")

video_path = Path("outputs/final_submission_video.mp4")

if video_path.exists():
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()

    st.video(video_bytes)
else:
    st.error(f"Video not found: {video_path.resolve()}")

st.divider()

st.header("Tracking Comparison")

comparison = pd.DataFrame({
    "Metric": [
        "Unique Tracks",
        "Average Observations / ID",
        "Average Track Duration",
        "Median Track Duration",
        "Longest Track"
    ],
    "ByteTrack": [
        73,
        240.62,
        243.60,
        31,
        750
    ],
    "BoT-SORT + Re-ID + GMC": [
        63,
        278.00,
        280.38,
        28,
        750
    ]
})

st.table(comparison)

st.divider()

st.header("Tracking Data")

csv_path = Path("outputs/tracking_data/botsort.csv")

if csv_path.exists():
    df = pd.read_csv(csv_path)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Tracking Records", len(df))

    with col2:
        st.metric("Unique IDs", df["track_id"].nunique())

    st.dataframe(
        df.head(100),
        use_container_width=True
    )
else:
    st.warning("Tracking CSV not found.")

st.divider()

st.header("Tracking Pipeline")

st.code(
    """Input Soccer Video
        ↓
Football YOLO Detection
        ↓
ByteTrack Baseline
        ↓
BoT-SORT + Re-ID + GMC
        ↓
Persistent Track IDs
        ↓
CSV Tracking Data
        ↓
Trajectory Visualization
        ↓
Final Annotated Video""",
    language="text"
)

st.divider()

st.header("Video Source")

st.markdown(
    "Original public video source: "
    "[Bundesliga Soccer Video]"
    "(https://huggingface.co/datasets/dbal0503/Bundesliga/blob/main/Bundesliga/Clips/2e57b9_0.mp4)"
)

st.divider()

st.header("Project Repository")

st.markdown(
    "[GitHub Repository]"
    "(https://github.com/Parag-code/soccer_object_tracking)"
)

st.divider()

st.caption(
    "Computer Vision Assignment — Multi-Object Detection and Persistent ID Tracking"
)