import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

import os
rest_api_url = os.getenv("REST_API_SERVICE_NAME")

st.set_page_config(layout="wide")
st.title("Image Inference App")

# Compact upload section
cols = st.columns([4,1])
with cols[0]:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with cols[1]:
    run_btn = st.button("Run Inference", disabled=not uploaded_file)

# Run inference only when both image and button exist
if uploaded_file:
    with st.spinner("Processing..."):
        inference_url = f"http://{rest_api_url}/v1/infer"
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        
        response = requests.post(inference_url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "annotated_image" in data:
                annotated_image = Image.open(BytesIO(base64.b64decode(data["annotated_image"])))
                uploaded_file.seek(0)
                original_image = Image.open(uploaded_file)
                
                # Display both images SIDE-BY-SIDE after inference
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="Original Image", use_container_width=True)
                with col2:
                    if run_btn:
                        st.image(annotated_image, caption="Annotated Image", use_container_width=True)
            else:
                st.error("No annotated image received")
        else:
            st.error(f"Inference failed: {response.text}")