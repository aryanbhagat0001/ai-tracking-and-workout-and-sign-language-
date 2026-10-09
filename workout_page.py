"""Workout Tracker page (Squat).

Pose model: YOLOv8-pose (Ultralytics). Camera: browser webcam via streamlit-webrtc,
so the same code works on a laptop and when deployed online.
"""
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from exercises import EXERCISES, create_detector
from utils.angle_utils import score_label
from utils.drawing import draw_overlay, draw_pose
from utils.pose_engine import PoseEngine


RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class PoseProcessor(VideoProcessorBase):
    """Runs on every webcam frame: pose -> exercise logic -> drawing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.engine = PoseEngine()          # model is created once per camera session
        self.exercise_name = "Squat"
        self.detector = create_detector(self.exercise_name)
        self.result = self.detector.get_result()
        self.fps = 0.0
        self.prev_time = time.time()

    def set_exercise(self, name):
        with self.lock:
            if name != self.exercise_name:
                self.exercise_name = name
                self.detector = create_detector(name)   # new exercise = clean state
                self.result = self.detector.get_result()

    def reset(self):
        with self.lock:
            self.detector.reset()
            self.result = self.detector.get_result()

    def get_result(self):
        with self.lock:
            return self.result

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror view

        landmarks = self.engine.detect(img)

        with self.lock:
            result = self.detector.update(landmarks)
            self.result = result

        if landmarks is not None:
            draw_pose(img, landmarks, result["feedback"][1])

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_time, 1e-3))
        self.prev_time = now
        draw_overlay(img, result, self.fps)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_stats(slots, result):
    """Update the metric cards and the feedback box."""
    slots["reps"].metric("Repetitions", result["reps"])
    score = result["score"]
    score_text = "--" if score is None else str(score) + "%"
    slots["score"].metric("Form Score (prototype)", score_text, score_label(score), delta_color="off")
    slots["exercise"].metric("Exercise", result["exercise"])
    message, status = result["feedback"]
    if status == "good":
        slots["feedback"].success(message)
    elif status == "warning":
        slots["feedback"].warning(message)
    else:
        slots["feedback"].info(message)




def render():
    st.header("Workout Tracker")
    st.caption("12 exercises with live rep counting and form feedback. Single person only.")

    exercise_name = st.selectbox("Select Exercise", list(EXERCISES.keys()))
    st.info(create_detector(exercise_name).info)
    reset_clicked = st.button("Reset Workout")

    st.write("Press START below, allow camera access in your browser, then stand so your "
             "whole body is visible.")

    ctx = webrtc_streamer(
        key="workout",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    c1, c2, c3 = st.columns(3)
    slots = {
        "reps": c1.empty(),
        "score": c2.empty(),
        "exercise": c3.empty(),
        "feedback": st.empty(),
    }
    st.caption("Prototype fitness guidance only. The form score is a simple heuristic, not a medical "
               "or professional assessment. Use proper technique and stop if you feel pain or discomfort.")

    processor = ctx.video_processor
    if processor is not None:
        processor.set_exercise(exercise_name)
        if reset_clicked:
            processor.reset()
        # Keep the stats cards updating while the camera is running.
        while ctx.state.playing:
            processor = ctx.video_processor
            if processor is None:
                break
            render_stats(slots, processor.get_result())
            time.sleep(0.3)
    else:
        idle_result = create_detector(exercise_name).get_result()
        idle_result["feedback"] = ("Press START to begin.", "neutral")
        render_stats(slots, idle_result)
