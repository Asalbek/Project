import streamlit as st
from fastai.vision.all import PILImage, load_learner
from torch import multiprocessing
import pathlib
import plotly.express as px


# Ensure compatibility with the WindowsPath for fastai models
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Title
st.title("Transportni klassifikatsiya qiluvchi model")

# File uploader
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])

if file is not None:
    # Display the uploaded image
    st.image(file)

    # Create an image object
    img = PILImage.create(file)

    # Load the pre-trained model
    model = load_learner('transports.pkl')

    # Make a prediction
    pred, pred_id, probs = model.predict(img)

    # Show the prediction result
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id] * 100:.1f}%")
else:
    st.info("Iltimos, rasm yuklang.")

# Plotting
fig = px.bar(x=probs * 100, y=model.dls.vocab)
st.plotly_chart(fig)
