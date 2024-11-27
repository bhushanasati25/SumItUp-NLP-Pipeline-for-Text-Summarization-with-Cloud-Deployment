import sys
import os
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("models/fine_tuned/checkpoint-5523")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def summarize_text(dialogue):
    """
    Generates a summary for the given dialogue.
    """
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=64, min_length=10, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app
st.title("✨ Text Summarization Tool ✨")
st.write("Provide a dialogue below, and the tool will generate a concise summary!")

# Input field
dialogue_input = st.text_area("Enter Dialogue", placeholder="Type your dialogue here...")

if st.button("🔍 Summarize"):
    if dialogue_input.strip():
        summary = summarize_text(dialogue_input)

        # Stylish summary output
        st.markdown("### 📜 **Generated Summary**")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f3ec78, #af4261);
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                font-size: 16px;
                color: white;
                font-family: Arial, sans-serif;
            ">
                {summary}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("❗ Please enter some dialogue to summarize.")
