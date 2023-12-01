import spacy


def chunk_text(text):
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text using spaCy
    doc = nlp(text)

    # Extract sentences as chunks
    chunks = [sent.text for sent in doc.sents]

    return chunks


# Example text
example_text = """
In a galaxy far, far away, there was a rebellion against the tyrannical Empire.
Led by a young farm boy named Luke Skywalker, the rebels fought for freedom and justice.
"""

# Call the function to break text into chunks
resulting_chunks = chunk_text(example_text)

# Print the resulting chunks
for i, chunk in enumerate(resulting_chunks, 1):
    print(f"Chunk {i}:", chunk)




import os
from transformers import AutoTokenizer

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Specify the folder path containing text files
folder_path = "/files"

# Loop through each text file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    # Check if the number of tokens is greater than 512
    if len(tokens) > 512:
        # Chunk the text into smaller chunks
        chunk_size = 512
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Save each chunk to a new file
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{filename}_chunk_{i + 1}.txt"
            chunk_filepath = os.path.join(folder_path, chunk_filename)
            with open(chunk_filepath, "w", encoding="utf-8") as chunk_file:
                chunk_file.write(chunk)

        print(f"File '{filename}' has been chunked into {len(chunks)} smaller chunks.")
    else:
        print(f"File '{filename}' does not need to be chunked.")




import os
from transformers import AutoTokenizer

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Specify the folder path containing text files
folder_path = "/files"

# Function to chunk text based on sentence boundaries
def chunk_text(text, max_tokens=512):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    chunks = []
    current_chunk = []

    for token in tokens:
        if len(current_chunk) + len(tokenizer.tokenize(token)) <= max_tokens:
            current_chunk.append(token)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [token]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Loop through each text file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Chunk the text into smaller parts based on sentence boundaries
    chunks = chunk_text(text)

    # Save each chunk to a new file
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{filename}_chunk_{i + 1}.txt"
        chunk_filepath = os.path.join(folder_path, chunk_filename)
        with open(chunk_filepath, "w", encoding="utf-8") as chunk_file:
            chunk_file.write(chunk)

    print(f"File '{filename}' has been chunked into {len(chunks)} smaller chunks.")
