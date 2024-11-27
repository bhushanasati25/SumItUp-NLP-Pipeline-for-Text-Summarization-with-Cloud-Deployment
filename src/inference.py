from transformers import BartTokenizer, BartForConditionalGeneration

# Correct the path to the checkpoint
model = BartForConditionalGeneration.from_pretrained("models/fine_tuned/checkpoint-5523")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def summarize_text(dialogue):
    """
    Generates a summary for the given dialogue.
    """
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=64, min_length=10, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
