# AI Vision Studio V2

Two computer vision tools in one Streamlit app. Pick a tool in the left sidebar.

1. Workout Tracker: YOLOv8-pose, 12 exercises, live rep counting and form feedback.
2. Sign Language: built-in recognition of 19 signs (no training needed), a learn and
   practice mode, and optional custom training.

The camera is your browser webcam (streamlit-webrtc), so the app works locally and online.

## Run (Windows)
Extract the zip, open the folder (it must contain app.py, requirements.txt, run.bat)
and double-click run.bat. First install takes several minutes. Internet is needed the
first time each tool runs, to download its small model.

Manual alternative, from this folder:
    python -m venv venv
    venv\Scripts\python.exe -m pip install -r requirements.txt
    venv\Scripts\python.exe -m streamlit run app.py

## Workout Tracker exercises
Squat, Bicep Curl, Shoulder Press, Lateral Arm Raise, Push-up, Lunge, High Knees,
Hip Hinge, Sit-up, Side Leg Raise, Tricep Extension, Jumping Jack.
Select one, read the camera tip shown on the page, press START. Reps count when the
movement reaches the target angle and returns to the start. Form score is a simple
heuristic, not a medical or professional assessment.

To add an exercise: add one dictionary to CONFIGS in exercises/library.py
(joints, angle thresholds, optional form checks). It appears in the dropdown automatically.
Angle thresholds can be tuned there if a rep is not counting.

## Sign Language
Recognize: show a sign and hold it about one second to add it to the text.
Learn and Practice: pick a sign, read how to make it, practice until the hand turns green.
Custom Training (optional): record your own samples and train a model to add new signs
or improve accuracy, then tick the custom model option in Recognize.

Built-in signs: A, B (Four), C, D, F (OK), I, L, O, S, U, V (Peace/Two), W (Three),
Y (Call me), Hello/Five, I Love You, Thumbs Up, Thumbs Down, Point (One), Rock On.
Not built in: letters that need motion (J, Z) or very subtle finger positions
(E, G, H, K, M, N, P, Q, R, T, X). Use Custom Training for those.
Built-in recognition uses hand geometry rules. Thresholds are at the top of
utils/sign_rules.py. The overlay shows finger states (E straight, H half, C curled)
so you can see what the app detects.

## Deploy
Browsers allow camera access only on HTTPS or localhost. Use a host with HTTPS
(Docker on Render, Railway, Azure, or a Hugging Face Docker Space, port 8501).
If video works locally but not online, add a TURN server in RTC_CONFIG in both page files.

## Common errors
- Python not found: install Python 3.10 to 3.12 and tick Add to PATH.
- Camera blocked: allow the camera in the browser address bar; close other apps using it.
- Model download failed: check internet and restart.
- Low FPS: lower imgsz in utils/pose_engine.py (for example 256).
- Sign not recognized: good light, hand fully visible, palm toward camera, plain background.
