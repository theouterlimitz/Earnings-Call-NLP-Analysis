# AI-Powered Analysis of Corporate Earnings Call Transcripts

## Project Overview

This project utilizes advanced Natural Language Processing (NLP) techniques with Transformer models to automate the analysis of dense corporate earnings call transcripts. The goal is to build a suite of tools that can extract key insights, gauge sentiment, and find specific information from long-form financial documents, tasks that typically require hours of manual work by financial analysts.

The project demonstrates a full pipeline, from curating raw text files into a structured dataset to applying multiple, state-of-the-art, pre-trained models from the Hugging Face ecosystem to perform distinct NLP tasks.

---

## Key Features & Capabilities

This analysis pipeline provides three core automated capabilities for any given earnings call transcript:

1.  **Automated Summarization:** Utilizes a "chunk and summarize" technique with the BART model to distill a multi-thousand-word transcript into a concise and useful executive summary, capturing key business metrics and outcomes.
2.  **Financial Sentiment Analysis:** Employs FinBERT, a Transformer model specifically fine-tuned on financial text, to determine if the tone of the call is positive, negative, or neutral. This can be tracked over time to identify trends in corporate sentiment.
3.  **Extractive Question Answering (Q&A):** Uses a DistilBERT-based model to find specific answers to user-asked questions (e.g., "How much did AWS grow?") directly within the full text of the transcript.

---

## Dataset

* **Source:** A collection of 19 individual earnings call transcripts for **Amazon (AMZN)**, covering the period from 2016 to 2020, sourced from a public dataset.
* **Format:** The raw data consists of individual `.txt` files, one for each quarterly earnings call.

## Methodology & Workflow

1.  **Data Curation:** A Python script was developed to process the raw `.txt` files. The script automatically loops through the directory, reads the content of each file, and parses the filename to extract metadata such as the **ticker** and **date**. This information was consolidated into a single, clean pandas DataFrame (`amazon_earnings_calls_curated.pkl`).

2.  **NLP Analysis:** The Hugging Face `transformers` library was used to apply several pre-trained models to the curated transcripts:
    * **Summarization Model:** `facebook/bart-large-cnn`
    * **Sentiment Analysis Model:** `ProsusAI/finbert`
    * **Question Answering Model:** `distilbert-base-cased-distilled-squad`

---

## Example Results (from the AMZN call on 2020-07-30)

### Generated Summary:
> "Strong top line performance was driven by increased consumer demand, led by Prime members. Worldwide streaming video hours doubled year-over-year driven largely by Prime video. We continue to invest meaningfully, including $9.4 billion in CapEx and finance leases in Q2 alone. We expect a meaningfully higher year- over-year square footage growth of approximately 50%."

### Question Answering:
* **Q:** How much did AWS grow?
* **A:** `$43 billion` (Confidence: 0.82)
<br>
* **Q:** What was the impact of COVID?
* **A:** `shipping times` (Confidence: 0.55)

---

## Repository Contents

* **`01_Data_Exploration.ipynb`**: Contains the code for loading the raw `.txt` files, parsing them, and creating the clean `amazon_earnings_calls_curated.pkl` dataset.
* **`02_NLP_Analysis.ipynb`**: The main notebook where the pre-trained Transformer models are loaded and applied for Sentiment Analysis, Summarization, and Question Answering.
* **`amazon_earnings_calls_curated.pkl`**: The clean, processed dataset used as the starting point for all NLP tasks.
* **`/images/`**: Contains any saved plots, such as the sentiment-over-time visualization.

## Tools & Libraries Used

* Python 3
* Pandas & NumPy
* Hugging Face `transformers`
* PyTorch
* Matplotlib & Seaborn
