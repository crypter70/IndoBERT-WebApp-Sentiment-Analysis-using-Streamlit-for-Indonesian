import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient


# with streamlit cloud deployment
client = InferenceClient(token=st.secrets["api_token"])

# from dotenv import load_dotenv
# load_dotenv()

# without deployment
# client = InferenceClient()

path_to_image_1 = 'positive.png'
path_to_image_2 = 'neutral.png'
path_to_image_3 = 'negative.png'


def getEmoji(label, score):
    if label == "POSITIVE":
        image = Image.open(path_to_image_1)
    elif label == "NEUTRAL":
        image = Image.open(path_to_image_2)
    elif label == "NEGATIVE":
        image = Image.open(path_to_image_3)
    
    st.text("")
    st.write("Score: ", score)
    st.image(image, caption=label)
    

def getSentiment(text):
    output = client.text_classification(model="crypter70/IndoBERT-Sentiment-Analysis", text=text)

    label = output[0].label
    score = str(output[0].score)[:5]

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
