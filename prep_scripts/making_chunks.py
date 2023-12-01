import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the tokenizer and summarization model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Specify the folder path containing text files
folder_path_input = "../docs_dump"
folder_path_output = "../docs_chunk"
os.makedirs(folder_path_output, exist_ok=True)

# Function to perform summarization
def summarize_text(init_text, max_tokens=512):
    # Tokenize and truncate the input text
    inputs = tokenizer.encode("summarize: " + init_text, return_tensors="pt", max_length=max_tokens, truncation=True)

    # Generate the summary
    summary_ids = summarization_model.generate(inputs, max_length=max_tokens, min_length=max_tokens // 2,
                                               length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Loop through each text file in the folder
for filename in os.listdir(folder_path_input):
    file_path = os.path.join(folder_path_input, filename)

    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Check if summarization is needed
    if len(tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))) > 512:
        # Summarize the text
        summarized_text = summarize_text(text)

        # Save the summarized text to a new file
        summarized_filename = f"{filename.split('.')[0]}_summarized.txt"
        print(summarized_filename)
        summarized_filepath = os.path.join(folder_path_output, summarized_filename)
        with open(summarized_filepath, "w", encoding="utf-8") as summarized_file:
            summarized_file.write(summarized_text)

        print(f"File '{filename}' has been summarized.")
    else:
        final_filepath = os.path.join(folder_path_output, filename)
        with open(final_filepath, "w", encoding="utf-8") as final_file:
            final_file.write(text)
        print(f"File '{filename}' does not need summarization.")
        print(f"File '{filename}' saved to the output folder.")
