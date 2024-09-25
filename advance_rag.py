# import streamlit as st

# from langchain_community.llms import Ollama
# from langchain import  PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import JSONLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import  HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate

# # Constants
# DEFAULT_SYSTEM_PROMPT = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """.strip()

# SYSTEM_PROMPT = """
# Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# """

# # Functions
# def load_json(file_path):
#     with open(file_path) as f:
#         return json.load(f)

# def initialize_loader(file_path):
#     loader = TextLoader(
#         file_path=file_path)
#     return loader.load()

# def split_documents(data, chunk_size=1024, chunk_overlap=64):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_documents(data)

# def create_embeddings():
#     model_name="hkunlp/instructor-xl"
#     encode_kwargs = {'normalize_embeddings': False}
#     hf = HuggingFaceEmbeddings(
#     model_name=model_name,
#     encode_kwargs=encode_kwargs
#     )
#     return hf

# def create_vectorstore(texts, embeddings, persist_directory="db"):
#     return Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

# def generate_prompt(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
#     return f"""
# [INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# {prompt} [/INST]
# """.strip()

# def setup_llm(model_name="llama3"):
#     llm = Ollama(model=model_name)
#     return llm

# def setup_retrieval_qa_chain(llm, retriever, system_prompt=SYSTEM_PROMPT):
#     template = generate_prompt(
#     """
#     {context}

#     Question: {question}
#     """,
#         system_prompt=system_prompt,
#     )

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )

# def setup_document_processing():
#     st.title('Langchain Chatbot with Document Processing')
#     uploaded_file = st.file_uploader("Upload a file", type=["txt"])

#     if uploaded_file:
#         file_path = uploaded_file.name
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Processing steps
#         data = initialize_loader(file_path)
#         texts = split_documents(data)
#         embeddings = create_embeddings()
#         vectorstore = create_vectorstore(texts, embeddings)

#         retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#         # LLM and pipeline setup
#         llm = setup_llm()
#         qa_chain = setup_retrieval_qa_chain(llm, retriever)

#         input_query = st.text_input("Ask your question!")
        
#         # Query processing
#         if input_query:
#             result = qa_chain(input_query)
#             result_str = result.get('result', '')
#             st.write(result_str)

# def setup_general_chat():
#     st.title('Langchain Chatbot With LLAMA2 model')
#     input_text = st.text_input("Ask your question!")

#     # Initialize the Ollama model
#     llm = Ollama(model="llama3")

#     # Define a prompt template for the chatbot
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Please respond to the questions"),
#             ("user", "Question: {question}")
#         ]
#     )

#     # Create a chain that combines the prompt and the Ollama model
#     chain = prompt | llm

#     # Function to get the chatbot response
#     def get_response(question):
#         return chain.invoke({"question": question})

#     # Invoke the chain with the input text and display the output
#     if input_text:
#         response = get_response(input_text)
#         st.write(response)

# def main():
#     option = st.sidebar.selectbox(
#         'Select Chatbot',
#         ('Document Processing Chatbot', 'General Chatbot')
#     )

#     if option == 'Document Processing Chatbot':
#         setup_document_processing()
#     elif option == 'General Chatbot':
#         setup_general_chat()

# if __name__ == "__main__":
#     main()






#### Gemma Rag Agent ######




# import fitz  # PyMuPDF
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama
# from langchain_community.llms import Ollama
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from llama_index.embeddings.ollama import OllamaEmbedding

# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text += page.get_text("text")
#     return text

# # # Create embeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True, base_url="http://localhost:11434")

# db = Chroma(persist_directory="./db-hormozi",
#             embedding_function=embeddings)

# # # Create retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}
# )

# # # Create Ollama language model - Gemma 2
# local_llm = 'gemma2:2b'

# llm = ChatOllama(model=local_llm,
#                  keep_alive="3h", 
#                  max_tokens=512,  
#                  temperature=0)

# # Create prompt template
# template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
# Please write in full sentences with correct spelling and punctuation. If it makes sense use lists. \
# If the context doesn't contain the answer, just respond that you are unable to find an answer. \

# CONTEXT: {context}

# QUESTION: {question}

# <end_of_turn>
# <start_of_turn>model\n
# ANSWER:"""
# prompt = ChatPromptTemplate.from_template(template)

# # Create the RAG chain using LCEL with prompt printing and streaming output
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# # Function to ask questions
# def ask_question(context, question):
#     print("Answer:\n\n", end=" ", flush=True)
#     inputs = {"context": context, "question": question}
#     for chunk in rag_chain.stream(inputs):
#         print(chunk.content, end="", flush=True)
#     print("\n")

# # Example usage
# if __name__ == "__main__":
#     # Provide your PDF file path here
#     pdf_file_path = "C:/Users/partab.rai/Downloads/rag/GOT-OCR-2.0-paper.pdf"
#     context_text = extract_text_from_pdf(pdf_file_path)

#     while True:
#         user_question = input("Ask a question (or type 'quit' to exit): ")
#         if user_question.lower() == 'quit':
#             break
#         ask_question(context_text, user_question)

























from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = UnstructuredFileLoader("C:/Users/partab.rai/Downloads/rag/GOT-OCR-2.0-paper.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)

llm = Ollama(model="llama3")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

question = "Can you please summarize the document"
result = chain.invoke({"query": question})

print(result['result'])






# # pip install langchain-chroma













