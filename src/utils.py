import os
import requests
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document


def get_embedding_model():
    '''Returns the embedding model using HuggingFaceBgeEmbeddings with GPU support.'''
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}
    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},  # Change 'cpu' to 'cuda' for GPU
        encode_kwargs=encode_kwargs
    )  
    return model_norm


def load_vectordb(save_path: str):
    '''Loads the vector database from the specified path.'''
    model_norm = get_embedding_model()
    vectordb = FAISS.load_local(save_path, model_norm, allow_dangerous_deserialization=True)
    return vectordb


def get_test_case_code(test_case_name, search_folder):
    """
    Fetches the code of the specified test case by searching for a .py file 
    in the given folder and its subfolders.
    
    Args:
        test_case_name (str): The name of the test case to search for (without the `.py` extension).
        search_folder (str): The root folder where the .py files are located.
        
    Returns:
        str: The content of the test case file if found, or an error message if not found.
    """
    formatted_test_case_name = test_case_name + ".py"
    
    for root, _, files in os.walk(search_folder):
        for file in files:
            if file.lower() == formatted_test_case_name.lower():
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as test_case_file:
                        content = test_case_file.read()
                        return content
                except FileNotFoundError:
                    return f"Test case file not found: {file_path}"
    
    return f"Test case {test_case_name} not found in the folder: {search_folder}"


    # Function to ensure the results directory exists
def ensure_results_directory_exists(directory):
    results_directory = os.path.join('Results', directory)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

def save_output_to_file(response_text, query_name, prompt_text, model, folder_path):
    """
    Save the output to a file with a specified structure.

    Args:
        response_text (str): The generated response text.
        query_name (str): The name of the query.
        prompt_text (str): The prompt used for generating the response.
        model (str): The name of the model used for generation.
        folder_path (str): The base folder where the output will be saved.
    """
    # Generate a timestamp
    timestamp = datetime.now().strftime("%H-%M-%S")
    
    # Define the structure components
    today_directory = datetime.now().strftime("%Y-%m-%d")
    full_directory = os.path.join(folder_path, today_directory, model)
    
    # Ensure all directories exist
    os.makedirs(full_directory, exist_ok=True)
    
    # Define the output file name and path
    output_file_name = f"{query_name}_{timestamp}.txt"
    output_file = os.path.join(full_directory, output_file_name)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"Model: {model}\n")
        file.write(f"Query Name: {query_name}\n\n")
        file.write("Prompt Text:\n")
        file.write(prompt_text)
        file.write("\n\n" + '=' * 100 + "\n\n")
        file.write(response_text)
    
    print(f"Response saved to {output_file}")


def evaluate_code_with_llm(prompt_file_path, query_name, model="phi3_64000:latest", url="http://localhost:11434/api/generate", output_folder_path='/home/ml/Bhushan - Thales/Results/'):

    try:
        # Read the prompt file
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()
        
        # Append scoring guide to the prompt
        complete_prompt = f"{prompt_text}"
        
        # Prepare API data
        data = {
            "model": model,
            "prompt": complete_prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        
        # Send API request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Extract the response
        response_json = response.json()
        response_text = response_json["response"]
        
        # Save the output
        save_output_to_file(response_text, query_name, complete_prompt, model, output_folder_path)
    
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_queries(file_path):
    """
    Extracts queries from a text file where queries are separated by two empty lines.
    
    Args:
        file_path (str): Path to the text file containing the queries.
        
    Returns:
        list: A list of queries.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into queries using two consecutive newlines as the delimiter
    queries = [query.strip() for query in content.split('\n\n\n') if query.strip()]
    
    return queries


def get_review(query, review_folder):

# Extract the query name
    query_name = ([line for line in query.splitlines() if line.strip().startswith("Name:")][0].split(':')[-1]).strip()

# Find the review file
    review_file = None
    for file_name in os.listdir(review_folder):
        if file_name.startswith(query_name) and file_name.endswith(".txt"):
            review_file = os.path.join(review_folder, file_name)
            break

    if not review_file:
        raise FileNotFoundError(f"Review file not found for query name: {query_name}")

    # Extract the specific content
    with open(review_file, 'r', encoding='utf-8') as rev_file:
        rev_content = rev_file.read()

    content_below_marker = ""
    marker = "=" * 100
    if marker in rev_content:
        content_below_marker = rev_content.split(marker, 1)[1].strip()
    else:
        raise ValueError(f"Marker not found in the review file for query name: {query_name}")
    
    return content_below_marker