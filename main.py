import csv
import sys
from transformers import pipeline

def load_model():
    """
    Load a small open-source text-to-text generation model using Hugging Face transformers.
    Uses 'google/flan-t5-small' as it's efficient, offline, and good for instructional tasks like descriptions.
    """
    print("Loading model...")
    generator = pipeline('text2text-generation', model='google/flan-t5-small')
    return generator

def read_headers(filename):
    """
    Read the first row (headers) from the CSV file.
    Assumes the file exists and has at least one row.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return [h.strip() for h in headers] 

def generate_descriptions(generator, headers):
    """
    Generate a short descriptive text for each header using the loaded model.
    Prompts the model for a concise description.
    """
    print("Generating descriptions...")
    descriptions = []
    for header in headers:
        prompt = f"Describe what the CSV column '{header}' likely means in a short phrase."
        result = generator(prompt, max_new_tokens=15, do_sample=False)  # Deterministic, short output
        desc = result[0]['generated_text'].strip()
        line = f"{header} → {desc}"
        descriptions.append(line)
        print(line)  
    return descriptions

def write_output(descriptions, output_file='output.txt'):
    """
    Write the descriptions to a text file with UTF-8 encoding to handle special characters.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in descriptions:
            f.write(line + '\n')
    print(f"Output written to {output_file}")

def main():
    """
    Main function to orchestrate the script.
    Expects CSV file path as command-line argument.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        print("Example: python script.py input.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        generator = load_model()
        headers = read_headers(filename)
        descriptions = generate_descriptions(generator, headers)
        write_output(descriptions)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()