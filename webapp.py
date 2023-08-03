import os
import streamlit as st
import gdown
from nbconvert import HTMLExporter
import nbformat
import streamlit.components.v1 as components
from textSummarization.config.config import ConfigurationManager
from transformers import AutoTokenizer, pipeline
from PIL import Image

st.set_page_config(layout="wide")

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

    page = st.sidebar.radio("Choose a page", ["Home", "Model Development", "About Me"])

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
                
        st.info("Note: The prediction may not work properly after some time as the model might be deleted from the cloud to cut costs. This is for demo purposes as a portfolio project.")
        
        text = """
        ### Introduction

        In the era of information overload, the ability to distill essential insights from large volumes of text is more crucial than ever. This is where the field of text summarization comes into play. Text summarization is a branch of Natural Language Processing (NLP) that involves condensing a larger body of text into a short, coherent summary while preserving its key informational elements and overall meaning.

        The primary goal of this project is to build a text summarization model that can take a lengthy piece of text and generate a concise summary, much like a human would. This has wide-ranging applications in numerous fields such as journalism, where it can be used to generate news digests, in academia for literature reviews, or in business for summarizing reports and meetings.

        One popular example of a text summarization tool is Quillbot, which uses advanced NLP techniques to paraphrase and summarize text. Similarly, our project aims to create a model that can understand the context and extract the salient points from a piece of text, thereby saving the reader's time and effort.

        For instance, given a long article about climate change, our model should be able to generate a summary like: "The article discusses the increasing threat of climate change, highlighting the rise in global temperatures and sea levels. It emphasizes the need for urgent action, suggesting renewable energy and sustainable practices as potential solutions."

        In the following sections, we will walk through the steps of building such a model, from data collection and preprocessing to model training and evaluation. We will also discuss the challenges faced during the project and how we overcame them. Let's dive in!
        """

        st.markdown(text)
        
        
        text = """
        ### Objective

        The primary objective of this project is to design and implement an effective text summarization model that can condense lengthy textual information into a concise and coherent summary. The aim is to retain the key points and overall context of the original text in the summary, thereby providing a quick and accurate understanding of the content without the need to read through the entire text.

        In the context of this project, I am particularly interested in exploring the application of advanced Natural Language Processing (NLP) techniques and machine learning algorithms for this task. I aim to build a model that can handle a variety of text types, ranging from news articles and research papers to blog posts and book chapters.

        Another key objective is to ensure that the generated summaries are not only short and informative but also grammatically correct and readable. The model should be able to generate summaries that are smooth, coherent, and can stand on their own as a comprehensive reflection of the original text.

        Furthermore, I aim to evaluate the performance of my model rigorously and objectively, using established evaluation metrics in the field of NLP. This will allow me to understand the strengths and weaknesses of my model and guide future improvements.

        Ultimately, the goal is to contribute to the ongoing efforts in the field of NLP to make information more accessible and manageable in the face of growing data. By creating an effective text summarization model, I hope to provide a tool that can save time and effort for anyone who needs to understand large volumes of text quickly and efficiently.
        """

        st.markdown(text)

        st.image("https://drive.google.com/uc?export=download&id=17Bfgh0xweO6C7B_fBOHRXT_sEiSLwOU1", caption="Model Development Flow Chart")
        
        
        
       

        text = """
        ### Data Collection

        For this project, I used the [SAMSum Corpus](https://huggingface.co/datasets/samsum), a dataset that contains approximately 16,000 messenger-like conversations along with their summaries. These conversations were created and written down by linguists fluent in English, reflecting a variety of topics, styles, and registers. The dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes.

        ### Data Preparation

        The SAMSum Corpus is already structured and annotated, which simplifies the data preparation process. However, it's still necessary to preprocess the data for the specific requirements of the model. This includes tokenizing the text and converting it into a format that can be fed into the model.

        ### Model Selection

        For this project, I chose to use the pre-trained PEGASUS model provided by Google. PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence) is a state-of-the-art model for abstractive text summarization. The specific version of the model I used is '[google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)', which has been trained on both the C4 and HugeNews datasets.

        ### Model Training

        The pre-trained PEGASUS model was then fine-tuned on the SAMSum Corpus. Fine-tuning is a process where the pre-trained model is further trained on a specific task (in this case, text summarization), allowing it to adapt to the specific characteristics of the task.

        ### Model Evaluation

        After the model was fine-tuned, it was evaluated on a separate test set from the SAMSum Corpus. The performance of the model was measured using established evaluation metrics for text summarization, such as ROUGE and BLEU scores.

        ### Model Deployment

        Once the model was trained and evaluated, it was deployed using AWS and GitHub Actions. AWS provides a robust platform for hosting machine learning models, while GitHub Actions allows for automated deployment processes. This ensures that any updates to the model or the code are automatically reflected in the deployed application.
        """

        st.markdown(text)

        st.image("https://drive.google.com/uc?export=download&id=142aA8dZA6lH0gLE_-OZ9Gvw-sgVc4oet", caption="Work Flow Template Flow Chart")
        
        
        
        
        
        text = """
        ### Dependencies and Setup

        Before we start with the actual model training and evaluation, we need to set up our environment and download the necessary dependencies. Here are the libraries and modules we will be using:

        - **Transformers**: This library, developed by Hugging Face, provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, summarization, etc. We will be using it to access the pre-trained PEGASUS model.

        - **Datasets**: Also developed by Hugging Face, this library provides a simple API to download and preprocess datasets. We will be using it to download the SAMSum Corpus.

        - **Matplotlib**: This is a comprehensive library for creating static, animated, and interactive visualizations in Python. We will be using it to plot the training progress.

        - **Pandas**: This library provides high-performance, easy-to-use data structures and data analysis tools for Python. We will be using it to manipulate our data.

        - **NLTK**: The Natural Language Toolkit (NLTK) is a platform used for building Python programs to work with human language data. We will be using it for tokenization.

        - **TQDM**: This library provides a fast, extensible progress bar for Python and CLI. We will be using it to visualize the progress of our loops.

        - **Torch**: PyTorch is an open-source machine learning library based on the Torch library. We will be using it as our main library to train our model.

        - **Warnings**: This is a standard Python module for warning control. We will be using it to ignore any warning messages to keep our output clean.
        """

        st.markdown(text)

        st.code("""
        from transformers import pipeline, set_seed
        from datasets import load_dataset, load_from_disk
        import matplotlib.pyplot as plt
        from datasets import load_dataset
        import pandas as pd
        from datasets import load_dataset, load_metric

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        import nltk
        from nltk.tokenize import sent_tokenize

        from tqdm import tqdm
        import torch

        nltk.download("punkt")


        import warnings

        warnings.filterwarnings("ignore")
        """, language='python')
        
        
        

        text = """
        ### Setting up the Device

        Before we start with the model training, we need to set up our device configuration. In PyTorch, we need to set up our device as either CPU or CUDA (which stands for Compute Unified Device Architecture, a parallel computing platform and application programming interface model created by NVIDIA). If a GPU is available, PyTorch will use it by default, otherwise, it will use the CPU.
        """

        st.markdown(text)

        st.code("""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        device
        """, language='python')

        text = """
        ### Loading the Pre-trained Model and Tokenizer

        We will be using the PEGASUS model pre-trained on the CNN/DailyMail dataset. The model checkpoint is available on the Hugging Face model hub under the name 'google/pegasus-cnn_dailymail'.

        We first load the tokenizer associated with the model. The tokenizer is responsible for preprocessing the text for the model. This includes steps like tokenization, which is the process of converting the text into tokens (smaller parts like words or subwords), and encoding, which is the process of converting these tokens into numbers that the model can understand.

        Next, we load the pre-trained PEGASUS model. We specify that the model should be loaded onto our device (either the CPU or GPU, depending on availability).
        """

        st.markdown(text)

        st.code("""
        model_ckpt = "google/pegasus-cnn_dailymail"

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
        """, language='python')

        text = """
        ### Downloading and Unzipping the Dataset

        The SAMSum dataset is used for this project. It is available on GitHub and can be downloaded directly. After downloading, the dataset is unzipped to access the data files.
        """

        st.markdown(text)

        st.code("""
        #download & unzip data

        !wget https://github.com/nani2357/My_dataset_repo/raw/main/samsumdata.zip
        !unzip summarizer-data.zip


        dataset_samsum = load_from_disk('samsum_dataset')
        dataset_samsum
        """, language='python')

        text = """
        ### Exploring the Dataset

        To understand the dataset better, the length of each split (train, test, validation) is calculated and the features of the dataset are printed. The dataset consists of dialogues and their corresponding summaries.

        The dialogue and summary of a sample from the test set are also printed to get a sense of what the data looks like.
        """

        st.markdown(text)

        st.code("""
        split_lengths = [len(dataset_samsum[split])for split in dataset_samsum]

        print(f"Split lengths: {split_lengths}")
        print(f"Features: {dataset_samsum['train'].column_names}")
        print("\\nDialogue:")

        print(dataset_samsum["test"][1]["dialogue"])

        print("\\nSummary:")

        print(dataset_samsum["test"][1]["summary"])
        """, language='python')

        text = """
        ### Preprocessing the Data

        The data needs to be preprocessed before it can be fed into the model. This involves converting the dialogues and summaries into a format that the model can understand.

        A function `convert_examples_to_features` is defined for this purpose. This function takes a batch of examples and performs the following steps:

        1. It tokenizes the dialogues using the tokenizer's `__call__` method. The dialogues are truncated to a maximum length of 1024 tokens.

        2. It tokenizes the summaries in a similar way, but with a maximum length of 128 tokens. The tokenizer is switched to target mode using the `as_target_tokenizer` context manager. This is because the summaries are the targets that the model will be trained to predict.

        3. It returns a dictionary containing the input IDs, attention masks, and labels. The input IDs and attention masks are derived from the tokenized dialogues, and the labels are derived from the tokenized summaries.
        """

        st.markdown(text)

        st.code("""
        def convert_examples_to_features(example_batch):
        input_encoding = tokenizer(example_batch['dialogue'],
                                    max_length = 1024,
                                    truncation=True
                                    )

        with tokenizer.as_target_tokenizer():
            target_encoding = tokenizer(example_batch['summary'],
                                        max_length=128,
                                        truncation=True
                                        )

        return {
            'input_ids':input_encoding['input_ids'],
            'attention_mask':input_encoding['attention_mask'],
            'labels' : target_encoding['input_ids']
        }

        dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)

        dataset_samsum_pt["train"]
        """, language='python')
        
        
        text = """
        ### Training the Model

        The model is trained using the Hugging Face's `Trainer` class. The `Trainer` requires a number of arguments:

        1. `model`: The model to be trained, which in this case is the Pegasus model.

        2. `args`: Training arguments that specify the training parameters. The `TrainingArguments` class is used to create the training arguments.

        3. `data_collator`: The data collator is responsible for batching the data. The `DataCollatorForSeq2Seq` class is used to create the data collator.

        4. `tokenizer`: The tokenizer used for preprocessing the data.

        5. `train_dataset` and `eval_dataset`: The training and validation datasets.

        The `Trainer` is then used to train the model with the `train` method.
        """

        st.markdown(text)

        st.code("""
        #training

        from transformers import DataCollatorForSeq2Seq

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        """, language='python')
        
        st.code("""
        from transformers import TrainingArguments, Trainer

        trainer_args = TrainingArguments(
            output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        )
        """, language='python')

        st.code("""
        trainer = Trainer(model=model_pegasus, args=trainer_args,
                        tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                        train_dataset=dataset_samsum_pt["test"],
                        eval_dataset=dataset_samsum_pt["validation"])
        trainer.train()
        """, language='python')
        
        
        st.markdown("""
        ### Evaluating the Model

        The model is evaluated using the ROUGE metric, which is a set of metrics used to evaluate automatic summarization and machine translation. The `load_metric` function from the `datasets` library is used to load the ROUGE metric.

        The evaluation process involves the following steps:

        1. **Batching the Data**: The test data is divided into smaller batches using the `generate_batch_sized_chunks` function. This function takes a list of elements and a batch size as input and yields successive batch-sized chunks from the list of elements.

        2. **Generating Summaries**: For each batch of articles, the model generates a batch of summaries. The `generate` method of the model is used for this purpose. The inputs to the model are tokenized versions of the articles. The `generate` method also takes a few additional parameters such as `length_penalty`, `num_beams`, and `max_length` to control the generation process.

        3. **Decoding the Summaries**: The generated summaries are then decoded using the tokenizer. The `decode` method of the tokenizer is used for this purpose. The `skip_special_tokens` and `clean_up_tokenization_spaces` parameters are set to `True` to remove any special tokens and clean up the tokenization spaces.

        4. **Calculating the Metric**: The decoded summaries and the target summaries are then added to the metric using the `add_batch` method. Finally, the metric is computed using the `compute` method.
        """)

        st.code("""
        def generate_batch_sized_chunks(list_of_elements, batch_size):
            \"\"\"split the dataset into smaller batches that we can process simultaneously
            Yield successive batch-sized chunks from list_of_elements.\"\"\"
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i : i + batch_size]

        def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                    batch_size=16, device=device,
                                    column_text=\"article\",
                                    column_summary=\"highlights\"):
            article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
            target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total=len(article_batches)):

                inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                                padding=\"max_length\", return_tensors=\"pt\")

                summaries = model.generate(input_ids=inputs[\"input_ids\"].to(device),
                                attention_mask=inputs[\"attention_mask\"].to(device),
                                length_penalty=0.8, num_beams=8, max_length=128)
                ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

                # Finally, we decode the generated texts,
                # replace the  token, and add the decoded texts with the references to the metric.
                decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
                    for s in summaries]

                decoded_summaries = [d.replace(\"\", \" \") for d in decoded_summaries]


                metric.add_batch(predictions=decoded_summaries, references=target_batch)

            #  Finally compute and return the ROUGE scores.
            score = metric.compute()
            return score
        """, language='python')

        st.code("""
        rouge_names = [\"rouge1\", \"rouge2\", \"rougeL\", \"rougeLsum\"]
        rouge_metric = load_metric('rouge')
        """, language='python')
        
        
        
        
        
        st.markdown("""
        ### Calculating the Score and Saving the Model

        The `calculate_metric_on_test_ds` function is used to calculate the ROUGE score on the test dataset. The function takes the test dataset, the ROUGE metric, the model, the tokenizer, the batch size, and the column names for the dialogue and summary as input. The function returns a dictionary with the ROUGE scores.

        The ROUGE scores are then converted into a dictionary and displayed in a pandas DataFrame for easy visualization. The keys of the dictionary are the names of the ROUGE metrics, and the values are the corresponding scores.

        Finally, the trained model is saved using the `save_pretrained` method of the model. The model is saved in a directory named "pegasus-samsum-model".
        """)

        st.code("""
        score = calculate_metric_on_test_ds(
            dataset_samsum['test'][0:10], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
        )

        rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

        pd.DataFrame(rouge_dict, index = [f'pegasus'] )
        """, language='python')

        st.code("""
        ## Save model
        model_pegasus.save_pretrained("pegasus-samsum-model")
        ## Save tokenizer
        tokenizer.save_pretrained("tokenizer")
        """, language='python')

        st.code("""
        #Load

        tokenizer = AutoTokenizer.from_pretrained("/content/tokenizer")
        """, language='python')
        
        
        st.markdown("""
        ### Prediction

        The model's performance is evaluated by generating a summary for a sample dialogue from the test dataset. The `pipeline` function from the transformers library is used to create a summarization pipeline with the trained model and tokenizer.

        The `length_penalty` parameter in the `gen_kwargs` dictionary is used to control the length of the generated summary. A lower value results in shorter summaries, while a higher value results in longer summaries. The `num_beams` parameter is used for beam search, which is a search algorithm that considers multiple hypotheses at each step. The `max_length` parameter is used to limit the maximum length of the generated summary.

        The dialogue, the reference summary, and the model's generated summary are then printed for comparison.
        """)

        st.code("""
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        sample_text = dataset_samsum["test"][0]["dialogue"]

        reference = dataset_samsum["test"][0]["summary"]

        pipe = pipeline("summarization", model="pegasus-samsum-model",tokenizer=tokenizer)

        print("Dialogue:")
        print(sample_text)

        print("\\nReference Summary:")
        print(reference)

        print("\\nModel Summary:")
        print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])
        """, language='python')
        
        
        st.markdown("""
        ## FastAPI

        This script is used to create a FastAPI application that serves as a web server for the text summarization model. The application provides two endpoints: one for training the model and another for generating summaries.

        Here is a brief explanation of the code:

        1. The script begins by importing the necessary modules and initializing a FastAPI application.

        2. The `@app.get("/")` decorator creates a root endpoint that redirects users to the API documentation.

        3. The `@app.get("/train")` decorator creates an endpoint that trains the model when accessed. The training process is initiated by running the `main.py` script using the `os.system` function. If the training is successful, the endpoint returns a success message. If an error occurs during training, the endpoint returns an error message.

        4. The `@app.post("/predict")` decorator creates an endpoint that generates a summary for a given text. The endpoint receives the text as a POST request, creates an instance of the `PredictionPipeline` class, and calls its `predict` method to generate the summary. If an error occurs during prediction, the endpoint raises an exception.

        5. The `if __name__=="__main__":` block runs the application on a local server at port 8080 when the script is run directly.

        This FastAPI application provides a simple and efficient way to train the model and generate summaries using HTTP requests. It can be easily integrated into other applications or services.
        """)

        st.code("""
        from fastapi import FastAPI
        import uvicorn
        import sys
        import os
        from fastapi.templating import Jinja2Templates
        from starlette.responses import RedirectResponse
        from fastapi.responses import Response
        from textSummarization.pipeline.prediction import PredictionPipeline

        text:str = "What is Text Summarization?"

        app = FastAPI()

        @app.get("/", tags=["authentication"])
        async def index():
            return RedirectResponse(url="/docs")

        @app.get("/train")
        async def training():
            try:
                os.system("python main.py")
                return Response("Training successful !!")
            except Exception as e:
                return Response(f"Error Occurred! {e}")
            
        @app.post("/predict")
        async def predict_route(text):
            try:
                obj = PredictionPipeline()
                text = obj.predict(text)
                return text
            except Exception as e:
                raise e

        if __name__=="__main__":
            uvicorn.run(app, host="0.0.0.0", port=8080)
        """, language='python')

        st.markdown("""
        ![Model Development Flow Chart](https://drive.google.com/uc?export=download&id=1JBGRTowCwOYJ26Qxe659Y8ho8X3sbs-e)
        """)
        
        st.warning("""
        **Disclaimer:** The content provided above serves as a demonstration of a text summarization model. It is intended to provide a high-level overview of the model's development and deployment process. For a more comprehensive understanding, including modular coding, detailed deployment steps, and additional resources, please visit the full project repository on GitHub at [https://github.com/nani2357/text_summarization_project.git](https://github.com/nani2357/text_summarization_project.git).
        """)












                
                
                

    elif page == "Model Development":
        st.warning("""
        **Disclaimer:** The content provided above serves as a demonstration of a text summarization model. It is intended to provide a high-level overview of the model's development and deployment process. For a more comprehensive understanding, including modular coding, detailed deployment steps, and additional resources, please visit the full project repository on GitHub at [https://github.com/nani2357/text_summarization_project.git](https://github.com/nani2357/text_summarization_project.git).
        """)
        
        # Read the Jupyter notebook file
        with open('test_summariazation_demo_colab.ipynb', 'r') as f:
            notebook = nbformat.read(f, as_version=4)

    # Convert the Jupyter notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, _) = html_exporter.from_notebook_node(notebook)
        components.html(body,width=1000, height=1200, scrolling=True)
        
        
        
        
        
        
        
    elif page == "About Me":
        st.title('About Me')

        col1, col2 = st.columns(2)

        with col1:
            st.write("""
            ## Naveen Kumar Kadampally
         
            ## Professional Background

           My career journey started as a **Civil Engineer** at RG Constructions where I honed my skills in construction management, contract management, and AutoCAD. My role here was multidimensional, and I was responsible for everything from supervising construction workers, ensuring site safety, to material quantity management. My stint at RG Constructions provided me with a strong foundation and a broad perspective of problem-solving, which was the cornerstone for my subsequent career transitions.

           
        """)

        with col2:
            image = Image.open("img.jpg")

            st.image(image, use_column_width=True)

        
        st.write("""
                 I then made a bold career switch into the technology domain as a **Software Test Analyst** at Unified Softech. Intrigued by the role of data in decision-making and the potential it had to revolutionize industries, I navigated my way into data analysis and data science.

                 My transformation into a **Data Analyst** was an exciting journey. After acquiring the IBM Data Science Professional Certification, I began leveraging my knowledge in various data analysis tools and Python libraries to derive valuable insights from complex datasets. I played a pivotal role in developing targeted marketing strategies and enhancing customer experience by managing customer complaints more efficiently.

                 Building on my data analysis experience, I am currently diving deeper into the realm of **Artificial Intelligence (AI) and Machine Learning**. I am exploring these technologies to build advanced models that can further streamline decision-making and efficiency in businesses.

                 ## Skills

                 My technical skills range from **AutoCAD** and **Revit** in the field of Civil Engineering to **Python**, **SQL**, **Tableau**, and **PowerBI** in Data Analysis. My knowledge of **Machine Learning algorithms** and **Data Visualization tools** has allowed me to create intuitive dashboards and powerful predictive models.

                 On the soft skills front, I pride myself on my strong **communication skills** and **project management** abilities. These skills have been instrumental in leading teams, managing projects, and effectively coordinating with stakeholders.

                 ## Projects

                 I have been involved in several exciting projects throughout my career journey. My key projects include a **Customer Segmentation Project** at Unified Softech where I used Machine Learning to segment customers based on their purchasing history. The outcome was a more personalized customer experience and improved customer retention rates.

                 I also embarked on a project focused on **Predictive Analysis of Sales Data**. This project involved creating a model that could accurately predict future sales based on historical data, helping the company to strategically plan their production and inventory management.

                 In a freelance capacity, I worked with a construction company to develop a **Machine Learning model for Green Concrete Strength Prediction**. The model made accurate predictions about the strength of the green concrete based on the composition of recycled materials. This innovative approach promoted sustainability in the construction industry.

                 Now, as I venture into AI and Deep Learning, I'm excited to work on more advanced projects that integrate these technologies. My current projects are aimed at exploring new frontiers in AI and I am thrilled about the potential outcomes.
                 """)
                 
                 
    
        st.write("""
        ## Certifications

        I have pursued several certifications to expand my knowledge base and validate my skills, including:

        - **IBM Data Science Professional Certification**: This certification provided a solid grounding in data science methodologies, and tools including Python, SQL, and data visualization.

        - **ISTQB Certified Tester**: This internationally recognized certification has honed my understanding of software testing techniques and principles. It played a pivotal role during my tenure as a Software Test Analyst at Unified Softech.

        - **Microsoft Power BI Certification**: This certification confirmed my expertise in using PowerBI for data analysis and visualization, enabling me to create effective and insightful dashboards.

        - **AutoCAD Certified User**: This certification validated my proficiency in AutoCAD software, which I utilized extensively during my tenure as a Civil Engineer.

        ## Learning Journey and Future Goals

        Transitioning from Civil Engineering to Software Testing, and then Data Science has been an incredible journey of learning and growth. This journey has taught me that technology's potential is boundless, and the more we learn, the more we can harness its power.

        As I venture further into the world of AI and Machine Learning, I aim to create more sophisticated models and solutions that can enhance decision-making and efficiency in businesses.

        I am also eager to contribute to the growing field of AI ethics, as I believe that responsible AI usage is crucial for future technological advancements. In line with this, I am working towards obtaining a certification in AI Ethics in the coming year.

        ## Personal Interests

        I have a profound affinity for the outdoors. I am captivated by nature and relish every opportunity to be outside, whether that's on sun-drenched beaches or in the verdant expanses of a forest. I find these moments to be incredibly refreshing and grounding, a perfect respite from the hustle and bustle of the tech world. 

        Mountains, in particular, hold a special allure for me. I love trekking through mountainous terrains and setting up camp under the stars. There's something incredibly serene about the stillness of the night, punctuated only by the gentle crackle of a campfire and the laughter and stories shared around it. 
    
        Lakes are another favorite escape of mine, especially those nestled in the heart of mountains. I enjoy boating and fishing, finding these activities to be both exciting and soothing. There's a unique sense of contentment I experience when I am casting a line out into the tranquil water, surrounded by the majesty of mountain peaks.
    
        Beyond my love for outdoor activities, I am an internet enthusiast. I spend a considerable amount of time surfing the web, staying updated on emerging trends and technologies. This curiosity extends to my professional life, as it helps me keep a pulse on the rapidly evolving tech landscape.
    
        My family plays a pivotal role in my life. I treasure the time spent watching movies with my children and exploring the outdoors with them. I believe these experiences foster their curiosity, teach them resilience, and create priceless memories.
    
        My interests, whether it's my love for nature or technology, or my devotion to family, shape who I am. They provide me with relaxation, continual learning, and cherished experiences – elements that I consider vital in life.

        ## Contact Information

        I'm always open to discussing data science, technology, and potential collaborations. You can reach me via email at [naveenkadampally@outlook.com](mailto:naveenkadampally@outlook.com).

        ### Social Media Profiles
        [![Instagram](https://img.shields.io/badge/-Instagram-black?style=flat-square&logo=instagram)](https://www.instagram.com/naveen.ka202/)
        [![LinkedIn](https://img.shields.io/badge/-LinkedIn-black?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/naveen-kumar-kadampally/)
        [![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github)](https://github.com/nani2357)
    """)


if __name__ == "__main__":
    main()
