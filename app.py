# ----- import dependencies -----
import easyocr
import numpy as np
import cv2
import streamlit as st
from textblob import TextBlob

def main():

    # ----- initializing page configuration -----
    st.set_page_config("Textractor", page_icon=":waving_hand:", layout="wide")

    # ----- hinding mainmenu -----
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # ----- page title -----
    #st.title("Textractor")
    st.markdown("<h1 style='text-align: center;'>Textractor</h1>", unsafe_allow_html=True)

    st.write("---")

    user_image = st.file_uploader(label="Upload a text containing image here",type=['png','jpg','jpeg'],accept_multiple_files=False,
                                  key='file uploader',label_visibility="visible")
    image_button = st.button(label="Submit",key='image submission')

    st.write("---")

    if image_button and user_image:
        got_image(user_image)
    else:
        no_image()

# ----- UI without image -----
def no_image():
    st.markdown("<h1 style='text-align: center;'>No image uploaded yet</h1>",unsafe_allow_html=True)

# ----- UI after getting image -----
def got_image(user_image):
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<u><h1 style='text-align: center;'> Image </h1></u>",unsafe_allow_html=True)
            imgcol1, imgcol2, imgcol3 = st.columns([1,5,1])
            with imgcol1:
                pass
            with imgcol2:
                st.image(user_image)
            with imgcol3:
                pass
        with col2:
            final_text = text_extraction(user_image)
            if len(final_text) != 0:
                st.markdown("<u><h1 style='text-align: center;'> Extracted text </h1></u>",unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;color: #FEFBF6;'> {final_text} </h3>", unsafe_allow_html=True)
            else:
                st.markdown("<u><h1 style='text-align: center;'> Oops! </h1></u>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;color: #6C5F5B;'>Sorry, but may be "
                            "the image is either not that much comprehensive to extract something or not containing "
                            "any text because I'm unable to read anything."
                            "Kindly try with some other text containing image. </h3>", unsafe_allow_html=True)

# ----- extracting text -----
def text_extraction(user_image):

    file_bytes = np.asarray(bytearray(user_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ----- applying modifications to the image -----
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 1.2)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    # ----- applying easyocr and getting result -----
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img,paragraph=False)
    output = []
    for bbox, text, prob in result:
        output.append(text)

    # ----- refining the output -----
    text_list = list()
    for word in output:
        if word.isalpha() or word.isnumeric():
            text_list.append(word.lower())

    display_text = " ".join(text_list)

    display_text = TextBlob(display_text)
    word_list = display_text.words

    correct_text = list()

    for word in word_list:
        check = word.spellcheck()
        if check[0][1] == 0.0:
            continue
        else:
            correct_text.append(check[0][0].capitalize())

    final_text = " ".join(correct_text)

    # ----- returning final refined text -----
    return final_text

if __name__ == "__main__":
    main()


