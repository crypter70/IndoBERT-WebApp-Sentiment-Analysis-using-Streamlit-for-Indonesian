import streamlit as st
from transformers import pipeline
from PIL import Image
# import path
# import sys
# import os

sentiment_pipeline = pipeline("text-classification", model="crypter70/IndoBERT-Sentiment-Analysis")

# dir = path.Path(__file__).absolute()
# sys.path.append(dir.parent.parent)

# file_path = 'https://raw.githubusercontent.com/crypter70/My/main/World-Universities-EDA/world-uni-rankings.csv'

# path_to_image_1 = 'https://github.com/crypter70/IndoBERT-WebApp-Sentiment-Analysis-using-Streamlit-for-Indonesian/blob/main/positive.png'
# path_to_image_2 = 'https://github.com/crypter70/IndoBERT-WebApp-Sentiment-Analysis-using-Streamlit-for-Indonesian/blob/main/neutral.PNG'
# path_to_image_3 = 'https://github.com/crypter70/IndoBERT-WebApp-Sentiment-Analysis-using-Streamlit-for-Indonesian/blob/main/negative.png'
path_to_image_1 = 'positive.png'
path_to_image_2 = 'neutral.PNG'
path_to_image_3 = 'negative.png'

# def get_image_path(image_name):
#     current_dir = os.path.dirname(__file__)
#     return os.path.join(current_dir, "images", image_name)

def getEmoji(label, score):
    if label == "POSITIVE":
        # image = Image.open('./images/positive.PNG')
        image = Image.open(path_to_image_1)
    elif label == "NEUTRAL":
        # image = Image.open('./images/neutral.PNG')
        image = Image.open(path_to_image_2)
    elif label == "NEGATIVE":
        # image = Image.open('./images/negative.PNG')
        image = Image.open(path_to_image_3)
    
    st.text("")
    st.write("Score: ", score)
    st.image(image, caption=label)
    

def getSentiment(text):
    label = sentiment_pipeline(text)[0]['label']
    score = str(sentiment_pipeline(text)[0]['score'])[:5]

    return label, score


def main():
    
    st.title("Sentiment Analysis WebApp 😊😐🙁")
    st.markdown("""Language: Indonesian""")
    st.markdown("This app will predict the sentiment (positive, neutral, negative) of the sentence entered by the user.")   

    example_list = ["Doi asik bgt orangnya", "Ada pengumuman nih gaiss, besok liburr", "Kok gitu sih kelakuannya"]
    options = example_list + ["Input a new text ..."]

    selection = st.selectbox("Examples Text", options=options)

    text = ""

    if selection == "Input a new text ...": 
        otherOption = st.text_input("Enter your text...")
        text = otherOption
    else:
        text = selection

    if st.button("Predict"):
        with st.spinner('Predict the sentiment...'):
            label, score = getSentiment(text)
            getEmoji(label, score)

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Powered by <a href='https://huggingface.co/crypter70/IndoBERT-Sentiment-Analysis' target='_blank'>IndoBERT-Sentiment-Analysis</a></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
