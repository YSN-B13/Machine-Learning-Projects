import streamlit as st
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

model = load_model("handwritten_clf.h5")

st.title("🧠 MNIST Digit Recognizer")
st.write("Draw a digit (0–9) and click **Predict**")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    img = canvas_result.image_data
    if img is None or img.sum() == 0:
        st.warning("Please draw a digit first.")
    else:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0

        img_input = np.expand_dims(img, axis=0)
        prediction = model.predict(img_input)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Processed (28×28)", width=150)
        with col2:
            st.metric("Predicted Digit", digit)
            st.metric("Confidence", f"{confidence:.1%}")

        st.bar_chart(
            pd.DataFrame(prediction[0], index=[str(i) for i in range(10)], columns=["probability"])
        )

