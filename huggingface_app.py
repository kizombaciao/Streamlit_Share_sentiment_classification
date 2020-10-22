'''
https://www.youtube.com/watch?v=B5M_F9dYHOM&t=94s

https://colab.research.google.com/github/RachitBansal/Using-HuggingFace/blob/master/HuggingFace_1.ipynb#scrollTo=XsIp762F44pA

https://medium.com/analytics-vidhya/hugging-face-transformers-how-to-use-pipelines-10775aa3db7e

#! Using HuggingFace

'''
import streamlit as st
from transformers import pipeline

nlp = pipeline('sentiment-analysis')

st.title('Classifies Sentence Sentiment as Positive or Negative.')
st.write('')
sentence = st.text_input('Input your sentence here:  ')
if sentence:
    st.write(sentence)

    #sentiment = nlp('Transformers piplines are easy to use.')
    sentiment = nlp(sentence)
    st.write(f"SENTITMENT:  {sentiment[0]['label']}")
    st.write(f"SCORE:  {sentiment[0]['score']:.2f}")
    

