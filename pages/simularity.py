import numpy as np
import streamlit as st
import torch
from loguru import logger
from peft import PeftConfig, PeftModel

from src.models.embedders import EmbeddingModel


@st.cache_data
def load_embedder(peft_checkpoint_path='checkpoints/cls_cosface_cl_contrastive'):
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PeftConfig.from_pretrained(peft_checkpoint_path)

    model = EmbeddingModel(model_name=config.base_model_name_or_path)
    model.model = PeftModel.from_pretrained(model.model, peft_checkpoint_path, is_trainable=False)
    model.eval()
    model.to(_device)
    logger.info('Model has been loaded')
    return model, _device


@st.cache_resource
def compute_cosine_similarity(text1, text2):
    text_1 = embedder.tokenizer([text1], max_length=512, padding='max_length',
                                truncation=True, return_tensors='pt').to(device)
    text_2 = embedder.tokenizer([text2], max_length=512, padding='max_length',
                                truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_1_emb = embedder(**text_1).cpu().reshape(-1).numpy()
        text_2_emb = embedder(**text_2).cpu().reshape(-1).numpy()
    return np.dot(text_1_emb, text_2_emb)


# Streamlit app
def calculating_similarity():
    # Inputs
    st.header("Text Input")
    text1 = st.text_input("Enter Text 1", "Not a poem by William Shakespeare")
    text2 = st.text_input("Enter Text 2", "Not a poem by anyone, actually")

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(text1, text2)
    st.subheader("Cosine Similarity")
    st.write(cosine_sim)


st.set_page_config(page_title="Cosine Simularity Calculation", page_icon="🧮")
st.markdown("# Cosine Simularity")
st.sidebar.header("Cosine Simularity")
st.write(
    """Here you can calculate similarity between two texts styles"""
)
embedder, device = load_embedder()

calculating_similarity()
