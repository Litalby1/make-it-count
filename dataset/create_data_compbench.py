import argparse
import pandas as pd
import random
import json
import inflect
import re

# Seed the random number generator for reproducibility
random.seed(123456)

pattern = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s(\w+)')


# Function to find number words and objects in sentences
def find_nummod(sentences):
    results = []
    for sentence in sentences:
        match = pattern.search(sentence)
        if match:
            # Extract the number word and the object noun
            nummod = [match.group(1), match.group(2)]
            results.append(nummod)
    return results


def parse_sentence(sentence):
    nummods = find_nummod([sentence])
    parts = nummods[0]
    number_word = parts[0]
    object_plural = parts[1]
    object = p.singular_noun(object_plural)
    int_number = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                  'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                  }.get(number_word, 0)

    return {
        "prompt": sentence,
        "object": object,
        "object_plural": object_plural,
        "object_id": 24,  # Example ID, change as necessary
        "scene": "",
        "number": number_word,
        "int_number": int_number,
        "seed": random.randint(100000, 999999)  # Random seed for each entry
    }


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output_directory', type=str, default='dataset',
                    help='The path to the dataset directory')
parser.add_argument('--compbench_csv', type=str, default='./dataset/T2I numbers_compbench_sentences.csv',
                    help='The path to the CSV file containing the sentences from the competition benchmark.')

if __name__ == "__main__":
    p = inflect.engine()
    args = parser.parse_args()
    directory = args.output_directory
    csv_file_path = args.compbench_csv

    df = pd.read_csv(csv_file_path)

    # Use a set to collect sentences to ensure uniqueness
    sentences_set = set()

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if row[df.columns[-1]] == 'Yes':  # Check if the last column's value is 'Yes'
            sentences_set.add(row[df.columns[0]])  # Add the sentence from the first column to the set
    unique_sentences = list(sentences_set)

    # Randomly select 200 sentences if there are enough unique sentences
    selected_sentences = random.sample(unique_sentences, 200) if len(unique_sentences) >= 200 else unique_sentences

    # Apply the parsing function to each selected sentence
    json_output = [parse_sentence(sentence) for sentence in selected_sentences]


    # Save the results to a JSON file
    with open(f'{directory}/output_data_compbench.json', 'w') as f:
        json.dump(json_output, f, indent=4)

    print("JSON file has been saved with the structured data.")

