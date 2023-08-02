import os
import streamlit as st
import gdown
from textSummarization.config.config import ConfigurationManager
from transformers import AutoTokenizer, pipeline

# Google Drive direct download link
GOOGLE_DRIVE_FILE_ID = '106rriA0aYTyNwZx0uCWUhPMkjHulTH8w'
MODEL_PATH = 'pytorch_model.bin'
GOOGLE_DRIVE_DOWNLOAD_LINK = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

# Define the PredictionPipeline class
class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        
    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output

# Download the model file if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(GOOGLE_DRIVE_DOWNLOAD_LINK, MODEL_PATH, quiet=False)

# Define the main function
def main():
    st.title("Text Summarization")
    st.sidebar.title("Navigation")

    page = st.sidebar.selectbox("Choose a page", ["Home", "Model Development", "About"])

    if page == "Home":
        st.header("Text Summarization Prediction")
        user_input = st.text_area("Enter your text here")

        if st.button("Predict"):
            try:
                obj = PredictionPipeline()  # Initialize your prediction pipeline
                result = obj.predict(user_input)  # Make prediction
                st.success("Prediction Successful")
                st.write("Summarized Text:", result)
            except Exception as e:
                st.error(f"Prediction Failed: {e}")

    elif page == "Model Development":
        st.header("Model Development Page")
        st.write("Here you can describe how your model was developed, what techniques were used, etc.")

    elif page == "About":
        st.header("About Page")
        st.write("Here you can write something about yourself or about this project.")

if __name__ == "__main__":
    main()
