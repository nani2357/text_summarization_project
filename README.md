# Text_summarization_project
### Enhancing Text Summarization with Custom Datasets: Leveraging Pre-trained PEGASUS Models for Advanced Summarization and Web Application Deployment

To clone the repository, use the following command:

```bash
git clone https://github.com/nani2357/text_summarization_project.git
'''

###Introduction
In the era of information overload, the ability to distill essential insights from large volumes of text is more crucial than ever. This is where the field of text summarization comes into play. Text summarization is a branch of Natural Language Processing (NLP) that involves condensing a larger body of text into a short, coherent summary while preserving its key informational elements and overall meaning.

The primary goal of this project is to build a text summarization model that can take a lengthy piece of text and generate a concise summary, much like a human would. This has wide-ranging applications in numerous fields such as journalism, where it can be used to generate news digests, in academia for literature reviews, or in business for summarizing reports and meetings.

One popular example of a text summarization tool is Quillbot, which uses advanced NLP techniques to paraphrase and summarize text. Similarly, our project aims to create a model that can understand the context and extract the salient points from a piece of text, thereby saving the reader's time and effort.

For instance, given a long article about climate change, our model should be able to generate a summary like: "The article discusses the increasing threat of climate change, highlighting the rise in global temperatures and sea levels. It emphasizes the need for urgent action, suggesting renewable energy and sustainable practices as potential solutions."


### Objective
The primary objective of this project is to design and implement an effective text summarization model that can condense lengthy textual information into a concise and coherent summary. The aim is to retain the key points and overall context of the original text in the summary, thereby providing a quick and accurate understanding of the content without the need to read through the entire text.

In the context of this project, I am particularly interested in exploring the application of advanced Natural Language Processing (NLP) techniques and machine learning algorithms for this task. I aim to build a model that can handle a variety of text types, ranging from news articles and research papers to blog posts and book chapters.

Another key objective is to ensure that the generated summaries are not only short and informative but also grammatically correct and readable. The model should be able to generate summaries that are smooth, coherent, and can stand on their own as a comprehensive reflection of the original text.

Furthermore, I aim to evaluate the performance of my model rigorously and objectively, using established evaluation metrics in the field of NLP. This will allow me to understand the strengths and weaknesses of my model and guide future improvements.

Ultimately, the goal is to contribute to the ongoing efforts in the field of NLP to make information more accessible and manageable in the face of growing data. By creating an effective text summarization model, I hope to provide a tool that can save time and effort for anyone who needs to understand large volumes of text quickly and efficiently.

![Model Development Flow Chart](https://drive.google.com/uc?export=download&id=17Bfgh0xweO6C7B_fBOHRXT_sEiSLwOU1)

### Data Collection
For this project, I used the SAMSum Corpus, a dataset that contains approximately 16,000 messenger-like conversations along with their summaries. These conversations were created and written down by linguists fluent in English, reflecting a variety of topics, styles, and registers. The dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes.

### Data Preparation
The SAMSum Corpus is already structured and annotated, which simplifies the data preparation process. However, it's still necessary to preprocess the data for the specific requirements of the model. This includes tokenizing the text and converting it into a format that can be fed into the model.

Model Selection
For this project, I chose to use the pre-trained PEGASUS model provided by Google. PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence) is a state-of-the-art model for abstractive text summarization. The specific version of the model I used is 'google/pegasus-cnn_dailymail', which has been trained on both the C4 and HugeNews datasets.

### Model Training
The pre-trained PEGASUS model was then fine-tuned on the[SAMSum Corpus](https://huggingface.co/datasets/samsum). Fine-tuning is a process where the pre-trained model is further trained on a specific task (in this case, text summarization), allowing it to adapt to the specific characteristics of the task.

### Model Evaluation
After the model was fine-tuned, it was evaluated on a separate test set from the SAMSum Corpus. The performance of the model was measured using established evaluation metrics for text summarization, such as ROUGE and BLEU scores.

### Model Deployment
Once the model was trained and evaluated, it was deployed using AWS and GitHub Actions. AWS provides a robust platform for hosting machine learning models, while GitHub Actions allows for automated deployment processes. This ensures that any updates to the model or the code are automatically reflected in the deployed application.

![Work Flow Template Flow Chart](https://drive.google.com/uc?export=download&id=142aA8dZA6lH0gLE_-OZ9Gvw-sgVc4oet)

## Workflows
1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py



