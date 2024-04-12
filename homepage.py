import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Text Style Extraction"
    )

    st.markdown(
        """
        Project Description: \n
        """
    )

    st.image('data_visualisation/home_image.png')

    st.markdown(
        """
        Style extraction is a fascinating area of machine learning that focuses on the automatic identification and extraction of stylistic elements from various forms of media, including text, images, and audio. 

        - These stylistic elements can include characteristics such as color schemes, typography, writing style, and artistic motifs. 
        - By leveraging machine learning techniques, style extraction enables the analysis and understanding of artistic expression, aiding in tasks such as content categorization, recommendation systems, and content creation.
    """
    )


if __name__ == "__main__":
    run()