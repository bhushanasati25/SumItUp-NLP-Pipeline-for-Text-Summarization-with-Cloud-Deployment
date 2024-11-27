
# **Text Summarization NLP Project**

An AI-powered application for generating concise summaries from conversational texts using fine-tuned transformer models like BART. This project is built for real-time summarization with a user-friendly interface and scalable deployment.

---

## **Table of Contents**
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Workflow](#workflow)
- [Sample Input/Output](#sample-inputoutput)
- [Applications](#applications)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

---

## **Overview**
The **Text Summarization NLP Project** is designed to simplify lengthy dialogues into concise summaries. Leveraging the power of transformer models, the project fine-tunes BART on the SAMSum dataset to generate high-quality summaries for use cases such as meeting notes, chat summarization, and customer support.

---

## **Key Features**
1. **Automated Summarization**:
   - Transforms lengthy dialogues into concise summaries using a fine-tuned BART model.
2. **Real-Time Processing**:
   - User-friendly web interface powered by Streamlit.
3. **Custom Fine-Tuning**:
   - Fine-tuned on the SAMSum dataset for conversational structures.
4. **Dockerized Deployment**:
   - Ensures portability and consistent performance across environments.

---

## **Project Structure**
```plaintext
Text-Summarization-NLP-Project/
├── data/                     # Dataset files
│   ├── raw/                  # Raw SAMSum dataset
│   ├── processed/            # Preprocessed data
├── models/                   # Pretrained and fine-tuned models
│   ├── pretrained/           # Pretrained models like BART
│   ├── fine_tuned/           # Checkpoints and fine-tuned weights
├── notebooks/                # Jupyter notebooks for experimentation
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── training.ipynb        # Training experiments
├── src/                      # Core source code
│   ├── data_preprocessing.py # Preprocessing scripts
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation script
│   ├── inference.py          # Inference for generating summaries
├── app/                      # Deployment application
│   ├── app.py                # Streamlit app entry point
├── logs/                     # Log files for debugging
├── docker/                   # Docker-related files
│   ├── Dockerfile            # Docker image definition
│   ├── .dockerignore         # Files to exclude from Docker image
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .env                      # Environment variables (optional)
```

---

## **Technologies Used**
- **NLP Frameworks**: Hugging Face Transformers, PyTorch.
- **Deployment Tools**: Streamlit, Docker.
- **Data Handling**: pandas, NumPy.
- **Evaluation**: Hugging Face Evaluate library, ROUGE metrics.
- **Dataset**: SAMSum dataset for fine-tuning conversational summarization.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/bhushanasati25/SumItUp-NLP-Pipeline-for-Text-Summarization-with-Cloud-Deployment.git
cd SumItUp-NLP-Pipeline-for-Text-Summarization-with-Cloud-Deployment
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Train the Model (Optional)**
If you want to fine-tune the model, run:
```bash
python src/train.py
```

### **4. Run the Streamlit App**
```bash
streamlit run app/app.py
```

### **5. Dockerized Deployment**
Build and run the Docker container:
```bash
docker build -f docker/Dockerfile -t text-summarization-app .
docker run -p 8501:8501 text-summarization-app
```

---

## **Usage**
1. Enter a dialogue in the text area provided by the Streamlit app.
2. Click **"Summarize"** to generate a concise summary.
3. View the summary in real-time, displayed in a styled format.

---

## **Workflow**
1. **Data Preprocessing**:
   - Tokenization and truncation for conversational text.
2. **Model Training**:
   - Fine-tuned BART using the SAMSum dataset.
3. **Inference**:
   - Generates summaries using the fine-tuned model.
4. **Deployment**:
   - Dockerized for portability, deployed via Streamlit.

---

## **Sample Input/Output**
**Input Dialogue**:
```plaintext
Alice: Hey, can you help me with my homework?
Bob: Sure, what’s the subject?
Alice: I’m struggling with calculus.
Bob: No problem, I’ll help you after lunch.
```

**Generated Summary**:
```plaintext
Bob agrees to help Alice with her calculus homework after lunch.
```

---

## **Applications**
- **Customer Support**: Summarizing conversations between agents and customers.
- **Corporate Use**: Generating meeting notes and action points.
- **Healthcare**: Summarizing doctor-patient interactions.
- **Education**: Summarizing classroom discussions or collaborative sessions.

---

## **Future Improvements**
1. Multi-language support for global applications.
2. Customizable summary lengths (short, medium, long).
3. Integration with speech-to-text for real-time summarization.

---

## **Contributors**
- **Bhushan Asati** 

Feel free to contribute to this project! Fork the repository and submit a pull request for review.

---
