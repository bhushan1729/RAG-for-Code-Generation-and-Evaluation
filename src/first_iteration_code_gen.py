from langchain_community.vectorstores import FAISS
from utils import load_vectordb, extract_queries, load_vectordb, get_test_case_code, evaluate_code_with_llm


def write_to_file(query: str, initial_prompt, vectordb: FAISS, searech_folder: str, output_file_path: str):
    '''Retrieves details of documents including summaries, steps, and code from the vector database and saves to a file.'''
    
    # Retrieve the top 3 most relevant documents
    docs_with_scores = vectordb.similarity_search_with_score(query, k=2)

    # Open the file for writing
    with open(output_file_path, 'w') as file:
        file.write(f"{initial_prompt}\n\n")

        # Initialize the counter for test case numbering
        counter = 1

        for doc, _ in docs_with_scores:
            first_line = doc.page_content.split('\n')[0]
            if first_line.startswith("Name:"):
                test_case_name = first_line.split(":")[1].strip()
                file.write(f"**Test Case {counter}: Generate a Python script based on the provided testing steps to verify test mention in the summary.**\n\n")
                file.write(doc.page_content)

                # Fetch and write the code
                code = get_test_case_code(test_case_name, searech_folder)
                file.write(f"\n\nPython Code {counter}:\n###--Start of Python Code--\n{code}\n###--End of Python Code--\n\n")
                
                # Increment the counter for the next test case
                counter += 1
        
        file.write(f'###Test Case to be Automated: Generate a Python script based on the provided testing steps to verify test mention in the summary.\n')
        file.write(query)
        file.write('<|assistant|>')
        

initial_prompt_for_iter1 = '''
<|user|>Generate a complete Python code to automate the following test case to be automated based on the pattern in the given Test Case examples.
Write each and every function of your code completely with exactly as it is done in Test Case examples and do not pass any function.
Do not directly use the functions and methods provided in the context. 
Instead, understand how they are used in the test cases and write all the required functions and methods explicitly to complete the steps in the test case.
Some steps are not explicitly mentioned in the test case. You need to analyze the pattern and write code to handle the implicit steps.
Add meaningful comments at the start of each step in test case.
Use Chain of Thought while generating code for test case to be automated.
Adheres to PEP 8 coding guidelines and uses snake_case for variable, function, and method names.
\n'''

queries = extract_queries(file_path="data\query_test_cases.txt")
for query in queries:

    initial_prompt = initial_prompt_for_iter1
    faiss_path = "vector_database/faiss_index"
    vectordb = load_vectordb(faiss_path)
    search_folder = "data\test_cases_code"  # Folder with test case code paths
    temp_file = "temp.txt"
    output_folder_path = r'path\to\output\folder'

    query_name, generated_file_path = write_to_file(query, initial_prompt, vectordb, search_folder, temp_file)

    # Example Usage
    prompt_file_path = temp_file
    query_name = query_name
    generated_file_path = generated_file_path
    evaluate_code_with_llm(prompt_file_path, query_name, model='llama3.1_64k:latest', output_folder_path=output_folder_path)