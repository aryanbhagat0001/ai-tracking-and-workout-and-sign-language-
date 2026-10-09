"""Sign Language page: built-in recognition (no training needed), learn and practice,
and optional custom training to add your own signs."""
import threading
import time
from collections import Counter, deque

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from utils.dataset import HAND_MODEL_PATH, MODEL_PATH, delete_dataset, label_counts, save_samples
from utils.features import landmarks_to_features
from utils.hand_drawing import draw_banner, draw_debug, draw_hand, draw_letter
from utils.sign_rules import SIGNS, TEXT_OUT, analyze_hand, classify_sign, describe_hand
from utils.trainer import load_model, train_model

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
MIN_CONFIDENCE = 0.6   # custom model: prediction must be at least this sure
VOTE_WINDOW = 8        # last N predictions are voted on
VOTE_NEEDED = 6        # this many must agree before a sign is accepted
HOLD_FRAMES = 15       # hold a sign this long to add it to the text


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.engine_error = ""
        try:
            from utils.hand_engine import HandEngine
            self.engine = HandEngine(HAND_MODEL_PATH)
        except Exception as error:
            self.engine_error = "Hand model failed: " + str(error)

        self.mode = "Recognize"      # Recognize, Practice or Collect
        self.recording = False
        self.buffer = []
        self.hand_visible = False
        self.use_custom = False
        self.model = None
        self.model_time = 0.0
        self.history = deque(maxlen=VOTE_WINDOW)
        self.letter = ""
        self.confidence = 0.0
        self.hold = 0
        self.last_added = None
        self.text = ""
        self.target = ""
        self.success = False
        self.debug = ""

    # ----- controls called from the Streamlit page -----
    def set_mode(self, mode):
        with self.lock:
            if mode != self.mode:
                self.mode = mode
                self.recording = False
                self.history.clear()
                self.letter = ""
                self.hold = 0
                self.success = False

    def set_target(self, target):
        with self.lock:
            if target != self.target:
                self.target = target
                self.success = False

    def set_use_custom(self, value):
        with self.lock:
            self.use_custom = bool(value)
            self.history.clear()

    def start_recording(self):
        with self.lock:
            self.buffer = []
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            samples = list(self.buffer)
            self.buffer = []
        return samples

    def reload_model(self):
        import os
        try:
            modified = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
        except OSError:
            modified = 0.0
        with self.lock:
            if modified != self.model_time:
                self.model = load_model()
                self.model_time = modified

    def clear_text(self):
        with self.lock:
            self.text = ""

    def backspace(self):
        with self.lock:
            self.text = self.text[:-1]

    def add_space(self):
        with self.lock:
            self.text += " "

    def snapshot(self):
        with self.lock:
            return {
                "hand": self.hand_visible, "count": len(self.buffer),
                "letter": self.letter, "confidence": self.confidence,
                "text": self.text, "custom_ready": self.model is not None,
                "success": self.success, "debug": self.debug,
                "error": self.engine_error,
            }

    # ----- recognition logic -----
    def _predict(self, points, features):
        """Return (label or None, confidence). Uses built-in rules by default."""
        if self.use_custom and self.model is not None and features is not None:
            try:
                probabilities = self.model.predict_proba([features])[0]
                best = int(probabilities.argmax())
                confidence = float(probabilities[best])
                label = str(self.model.classes_[best])
                return (label if confidence >= MIN_CONFIDENCE else None), confidence
            except Exception:
                return None, 0.0
        analysis = analyze_hand(points)
        self.debug = describe_hand(analysis)
        return classify_sign(analysis), 1.0

    def _recognize(self, points, features):
        if points is None:
            self.history.clear()
            self.letter = ""
            self.confidence = 0.0
            self.hold = 0
            self.last_added = None
            self.debug = ""
            self.success = False
            return
        label, self.confidence = self._predict(points, features)
        self.history.append(label)
        top_label, top_count = Counter(self.history).most_common(1)[0]
        if top_label is not None and top_count >= VOTE_NEEDED:
            self.letter = top_label
            self.hold += 1
        else:
            self.letter = ""
            self.hold = 0
            self.last_added = None
        self.success = bool(self.target) and self.letter == self.target
        if self.mode == "Recognize" and self.letter and self.hold >= HOLD_FRAMES \
                and self.last_added != self.letter:
            out = TEXT_OUT.get(self.letter, self.letter)
            self.text += out if len(out) == 1 else " " + out + " "
            self.last_added = self.letter

    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)  # mirror view

        points = self.engine.detect(img) if self.engine is not None else None
        features = landmarks_to_features(points) if points is not None else None

        with self.lock:
            self.hand_visible = points is not None
            if self.mode == "Collect":
                if self.recording and features is not None:
                    self.buffer.append(features)
            else:
                self._recognize(points, features)
            mode, recording = self.mode, self.recording
            letter, confidence, text = self.letter, self.confidence, self.text
            target, success, debug = self.target, self.success, self.debug
            hand = self.hand_visible

        if points is not None:
            color = (0, 0, 255) if recording else ((0, 200, 0) if success else (255, 170, 0))
            draw_hand(img, points, color)

        if self.engine is None:
            draw_banner(img, "Hand model not loaded. Check internet and restart.", (0, 0, 200))
        elif mode == "Collect":
            draw_banner(img, "RECORDING" if recording else "Collect mode: ready",
                        (0, 0, 200) if recording else (60, 60, 60))
        else:
            draw_letter(img, letter, confidence)
            draw_debug(img, debug)
            if not hand:
                draw_banner(img, "Show your hand to the camera")
            elif mode == "Practice":
                draw_banner(img, ("Correct! " if success else "Target: ") + target,
                            (0, 150, 0) if success else (60, 60, 60))
            else:
                draw_banner(img, "Text: " + text[-35:])
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def sign_table():
    return pd.DataFrame(SIGNS, columns=["Sign", "Text it adds", "How to make it"])


def render():
    st.header("Sign Language Recognition")
    st.caption("Built-in recognition for 19 signs. No training needed. One hand only.")

    section = st.radio("Section", ["Recognize", "Learn and Practice", "Custom Training (optional)"],
                       horizontal=True)
    step = None
    if section == "Custom Training (optional)":
        step = st.radio("Custom step", ["Collect Data", "Train Model"], horizontal=True)

    if step == "Train Model":
        st.subheader("Collected samples")
        counts = label_counts()
        if counts.empty:
            st.info("No samples yet. Go to Collect Data first.")
        else:
            st.dataframe(counts, use_container_width=True)
            st.caption("Aim for 150 to 300 samples per sign, with at least 2 signs.")
        c1, c2 = st.columns(2)
        if c1.button("Train Model"):
            with st.spinner("Training..."):
                message, accuracy = train_model()
            if accuracy is None:
                st.warning(message)
            else:
                st.success(message + " Accuracy: " + str(round(accuracy * 100, 1)) + "%")
        if c2.button("Delete all samples and model"):
            delete_dataset()
            st.warning("All samples and the trained model were deleted.")
        st.caption("After training, use Recognize and tick the custom model option.")
        return

    if section == "Learn and Practice":
        labels = [s[0] for s in SIGNS]
        target = st.selectbox("Choose a sign to practice", labels)
        how = {s[0]: s[2] for s in SIGNS}
        st.info("How to make it: " + how[target])
    elif section == "Recognize":
        use_custom = st.checkbox("Use my custom trained model instead of the built-in rules")

    ctx = webrtc_streamer(
        key="sign",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    proc = ctx.video_processor
    status = st.empty()

    if section == "Custom Training (optional)":
        label = st.text_input("Sign label (for example HELLO)", value="A").strip().upper()
        st.write("Only needed to add your own signs. Show the sign, press Start Recording, "
                 "move the hand slightly, then press Stop and Save.")
        b1, b2 = st.columns(2)
        start = b1.button("Start Recording")
        stop = b2.button("Stop and Save")
        if proc is not None:
            proc.set_mode("Collect")
            if start:
                proc.start_recording()
            if stop:
                samples = proc.stop_recording()
                if label == "":
                    st.warning("Please type a label first.")
                else:
                    st.success("Saved " + str(save_samples(label, samples)) + " samples for " + label + ".")
        st.subheader("Collected samples")
        st.dataframe(label_counts(), use_container_width=True)
        if proc is not None:
            while ctx.state.playing:
                proc = ctx.video_processor
                if proc is None:
                    break
                info = proc.snapshot()
                if info["error"]:
                    status.error(info["error"])
                else:
                    status.info("Hand detected: " + ("yes" if info["hand"] else "no") +
                                "   Samples in this recording: " + str(info["count"]))
                time.sleep(0.3)

    elif section == "Learn and Practice":
        with st.expander("All supported signs"):
            st.dataframe(sign_table(), use_container_width=True, hide_index=True)
        if proc is None:
            st.write("Press START on the video, allow the camera, then make the sign shown above. "
                     "The hand turns green when it is correct.")
        else:
            proc.set_mode("Practice")
            proc.set_use_custom(False)
            proc.set_target(target)
            while ctx.state.playing:
                proc = ctx.video_processor
                if proc is None:
                    break
                info = proc.snapshot()
                if info["error"]:
                    status.error(info["error"])
                elif info["success"]:
                    status.success("Correct! You made: " + target)
                else:
                    status.info("Target: " + target + "   Detected: " + (info["letter"] or "-") +
                                "   " + info["debug"])
                time.sleep(0.3)

    else:
        col1, col2, col3 = st.columns(3)
        clear = col1.button("Clear Text")
        back = col2.button("Backspace")
        space = col3.button("Add Space")
        with st.expander("Supported signs (how to make each one)"):
            st.dataframe(sign_table(), use_container_width=True, hide_index=True)
        if proc is None:
            st.write("Press START on the video, allow the camera, then show a sign. "
                     "Hold a sign for about one second to add it to the text.")
        else:
            proc.set_mode("Recognize")
            proc.reload_model()
            proc.set_use_custom(use_custom)
            if clear:
                proc.clear_text()
            if back:
                proc.backspace()
            if space:
                proc.add_space()
            while ctx.state.playing:
                proc = ctx.video_processor
                if proc is None:
                    break
                info = proc.snapshot()
                if info["error"]:
                    status.error(info["error"])
                elif use_custom and not info["custom_ready"]:
                    status.warning("No custom model found. Train one in Custom Training, or untick the option.")
                else:
                    status.success("Sign: " + (info["letter"] or "-") + "   Text: " + info["text"])
                time.sleep(0.3)

    st.caption("Prototype for learning. Built-in rules use hand geometry; lighting, camera angle and "
               "hand shape affect accuracy. The overlay shows finger states so you can see what it detects.")
