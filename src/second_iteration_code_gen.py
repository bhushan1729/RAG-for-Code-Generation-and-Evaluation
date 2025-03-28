import os
from utils import load_vectordb, extract_queries, load_vectordb, get_test_case_code, evaluate_code_with_llm



def write_to_file(query, initial_prompt, vectordb, generated_folder, review_folder, output_file_path):
    """
    Write formatted content to a file with data extracted from specific folders.
    
    Args:
        query (str): The query text containing the name line.
        initial_prompt (str): The initial prompt to be included in the file.
        query_folder (str): Path to the folder containing queries.
        generated_folder (str): Path to the folder containing generated code files.
        output_file_path (str): The path to the output text file.
    """
    # Extract the query name
    query_name = ([line for line in query.splitlines() if line.strip().startswith("Name:")][0].split(':')[-1]).strip()


    # Find the generated code file
    generated_file = None
    for file_name in os.listdir(generated_folder):
        if file_name.startswith(query_name) and file_name.endswith(".txt"):
            generated_file = os.path.join(generated_folder, file_name)
            break

    if not generated_file:
        raise FileNotFoundError(f"Generated file not found for query name: {query_name}")


    # Extract the generated code
    with open(generated_file, 'r', encoding='utf-8') as gen_file:
        gen_content = gen_file.read()
    
    generated_code = ""
    start_marker = "```python"
    end_marker = "```"
    if start_marker in gen_content and end_marker in gen_content:
        generated_code = gen_content.split(start_marker)[1].split(end_marker)[0].strip()
    else:
        raise ValueError("Generated code markers not found in the generated file.")
    

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


    # Retrieve the top 1 most relevant documents
    docs_with_scores = vectordb.similarity_search_with_score(query, k=1)

    # Write all the content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{initial_prompt}\n\n")
        
        # Initialize the counter for test case numbering
        counter = 1

        for doc, _ in docs_with_scores:
            first_line = doc.page_content.split('\n')[0]
            if first_line.startswith("Name:"):
                test_case_name = first_line.split(":")[1].strip()
                output_file.write(f"###Similar Test Case {counter}: Generate a Python script based on the provided testing steps to verify test mention in the summary.\n\n")
                output_file.write(doc.page_content)

                # Fetch and write the code
                code = get_test_case_code(test_case_name, search_folder)
                output_file.write(f"\n\nSimilar Test Case Python Code {counter}:\n{code}\n\n\n")

                
                # Increment the counter for the next test case
                counter += 1
        output_file.write(f"###Test Case to be Automated: {query}\n\n")
        output_file.write(f"Code Generated by LLM: \n```\n{generated_code}\n```\n\n\n")
        output_file.write(f'###Comments and Issues: \n{content_below_marker}<|end|>\n')
        output_file.write('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    print(f"File written successfully at {output_file_path}")
    return query_name, generated_file



initial_prompt_for_iter2 = '''
<|start_header_id|>user<|end_header_id|>

###Inputs:
In the 'Similar Test Case' section, you'll find a test case with its steps and code for reference. Below that, there is Python code generated by a model based on the given test steps. This code was reviewed using LLM as a judge, and the feedback, including score, comments and issues, is listed in the 'Comments and Issues' section.

###Task:
Your task is to update and improve the code using the similar test case, provided test steps, and the LLM as a judge's feedback. While making changes, keep the overall structure and correct parts of the original code the same. Only update sections with clear differences or those related to fixing the identified issues. The updated code should be fully accurate and optimized to achieve a higher score when reviewed again by the LLM as a judge.

**Adheres to PEP 8 coding guidelines and uses snake_case for variable, function, and method names. Use chain of thoughts while generating the outputs.**
'''

queries = extract_queries(file_path="data\query_test_cases.txt")
for query in queries:

    initial_prompt = initial_prompt_for_iter2
    generated_folder = r"path\to\first\iteration\generated\code\folder"
    temp_file = "temp.txt"
    faiss_path = "vector_database/faiss_index"
    search_folder = "data\test_cases_code"  # Folder with test case code paths 
    output_folder_path = r'path\to\output\folder'
    vectordb = load_vectordb(faiss_path)

    query_name, generated_file_path = write_to_file(query, initial_prompt, vectordb, generated_folder, output_folder_path, temp_file)


    # Example Usage
    
    query_name = query_name
    generated_file_path = generated_file_path
    evaluate_code_with_llm(temp_file, query_name)