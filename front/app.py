import requests
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import base64

st.title("My first FastAPI application")

tab1, tab2 = st.tabs(["Image", "Posts"])


def main():
    with tab1:
        # create input form
        image = st.file_uploader("Segment an image", type=["jpg", "jpeg", "png"])
        # st.write(type(image))
        if st.button("Find Forest") and image is not None:
            # format data for input format
            # img = Image.open(image)
            files = {"file": image}
            # send data and get the result
            res = requests.post(
                "http://158.160.128.217:8000/segment_image",
                files=files,
                # "http://127.0.0.1:8000/segment_image", files=files
            ).json()
            st.write("ответ получен")
            mask = Image.open(BytesIO(base64.b64decode(res["mask"])))
            masked_img = Image.open(BytesIO(base64.b64decode(res["masked_image"])))
            # mask_list = res["mask"]
            # combined_img_list = res["masked_image"]
            # mask_np = np.array(mask_list)
            # combined_img_np = np.array(combined_img_list)[:, :, ::-1]
            # mask = Image.fromarray((mask_np * 255).astype(np.uint8))
            # combined_img = Image.fromarray((combined_img_np * 255).astype(np.uint8))
            col1, col2 = st.columns(2)
            with col1:
                st.image(image)
            with col2:
                st.image(masked_img)
            img_byte_arr = BytesIO()
            mask.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            st.download_button(
                label="Download mask",
                data=img_byte_arr,
                file_name="mask.png",
                mime="image/png",
            )

    with tab2:
        txt = st.text_input("Classify posts")
        if st.button("Classify"):
            text = {"text": txt}
            res = requests.post(
                "http://158.160.128.217:8000/clf_post",
                # "http://127.0.0.1:8000/clf_post",
                json=text,
            )
            st.write(txt)
            st.write(f' Class: {res.json()["label"]}')
            st.write(f'Probability: {res.json()["prob"]:.2f}')


if __name__ == "__main__":
    main()
