"""AI Vision Studio V2: Workout Tracker (12 exercises) and Sign Language in one app.

Each tool is loaded only when selected, so a problem in one tool
(for example a missing model download) does not break the other.
"""
import streamlit as st

st.set_page_config(page_title="AI Vision Studio V2", layout="wide")

st.sidebar.title("AI Vision Studio V2")
tool = st.sidebar.radio("Choose a tool", ["Workout Tracker", "Sign Language"])
st.sidebar.caption("Only one camera tool runs at a time. Switching tools stops the camera.")

if tool == "Workout Tracker":
    try:
        import workout_page
    except Exception as error:
        st.error("Could not load the Workout Tracker: " + str(error))
        st.stop()
    workout_page.render()
else:
    try:
        import sign_page
    except Exception as error:
        st.error("Could not load Sign Language: " + str(error))
        st.stop()
    sign_page.render()
