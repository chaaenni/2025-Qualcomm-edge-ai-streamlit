import os
import pandas as pd
from tqdm import tqdm

from db.create_db import process_pdfs_from_dataframe, load_pdf_databases, save_pdf_databases
# from db.model import get_QA_output, get_guide_line_output, get_case_search_output
from db.model import get_LLM_output

import warnings
import unicodedata

warnings.filterwarnings("ignore")

### DB ###

#Load if it exists, otherwise build and save a new one
base_directory = "/NAS/internship/JCY/2025-summer/develop/emerGen/" #change to your own directory path
df = pd.read_csv(base_directory + "data/full_data.csv")
db_filename = os.path.join(base_directory, 'pdf_databases.pkl')
if os.path.exists(db_filename):
    pdf_databases = load_pdf_databases(db_filename)
else:
    pdf_databases = process_pdfs_from_dataframe(df, base_directory)
    save_pdf_databases(pdf_databases, db_filename)


### Inference ###
def process_output(category, input_data, task):
    global pdf_databases
    print(pdf_databases)

    #Retrieve relevant information from the given category
    retriever = pdf_databases[category]
    context = retriever.invoke(input_data)
    print(context)
    #Pass context and input to the QA model and return output
    response = get_LLM_output(task, context, input_data)
    return response

#for caseSearch task type
def retrieve(category, input_data, task):
    global pdf_databases
    
    retriever = pdf_databases[category]
    context = retriever.invoke(input_data)
    print(context)
    
    return context

# Streaming version of process_output
def process_output_streaming(category, input_data, task, chunk_size=10, delay=0.01, timeout_seconds=60, context_limit=5):
    """
    Streaming version of process_output that yields chunks of the response.
    
    Args:
        category: Category for retrieval
        input_data: User input text
        task: Task type ("QA", "GuideLine", "caseSearch")
        chunk_size: Number of characters per chunk (default: 10)
        delay: Delay between chunks in seconds (default: 0.01)
        timeout_seconds: Timeout for the operation (default: 60)
        context_limit: Limit for context documents (default: 5)
    
    Yields:
        str: Chunks of the response text
    """
    import time
    
    global pdf_databases
    print(pdf_databases)

    # Retrieve relevant information from the given category
    retriever = pdf_databases[category]
    context = retriever.invoke(input_data)
    print(context)
    
    # Limit context if needed (assuming context_limit applies to number of documents/chunks)
    # Note: This is a simplified implementation. You may need to adjust based on your retriever's structure
    if context_limit and isinstance(context, list):
        context = context[:context_limit]
    
    # Pass context and input to the QA model and get full response
    # Note: get_LLM_output doesn't support streaming natively, so we simulate it
    response = get_LLM_output(task, context, input_data)
    
    # Simulate streaming by yielding chunks
    # In a real implementation, you would want to modify get_LLM_output to support streaming
    for i in range(0, len(response), chunk_size):
        chunk = response[i:i + chunk_size]
        if chunk:
            yield chunk
            time.sleep(delay)
    




