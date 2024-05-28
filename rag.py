import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

# Constants
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

SYSTEM_PROMPT = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

# Functions
def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def initialize_loader(file_path):
    loader = TextLoader(
        file_path=file_path)
    return loader.load()

def split_documents(data, chunk_size=1024, chunk_overlap=64):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_embeddings():
    model_name="hkunlp/instructor-xl"
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
    return hf

def create_vectorstore(texts, embeddings, persist_directory="db"):
    return Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

def generate_prompt(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()

def setup_llm(model_name="llama3"):
    llm = Ollama(model=model_name)
    return llm

def setup_retrieval_qa_chain(llm, retriever, system_prompt=SYSTEM_PROMPT):
    template = generate_prompt(
    """
    {context}

    Question: {question}
    """,
        system_prompt=system_prompt,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

def setup_document_processing():
    st.title('Langchain Chatbot with Document Processing')
    uploaded_file = st.file_uploader("Upload a file", type=["txt"])

    if uploaded_file:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Processing steps
        data = initialize_loader(file_path)
        texts = split_documents(data)
        embeddings = create_embeddings()
        vectorstore = create_vectorstore(texts, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # LLM and pipeline setup
        llm = setup_llm()
        qa_chain = setup_retrieval_qa_chain(llm, retriever)

        input_query = st.text_input("Ask your question!")
        
        # Query processing
        if input_query:
            result = qa_chain(input_query)
            result_str = result.get('result', '')
            st.write(result_str)

def setup_general_chat():
    st.title('Langchain Chatbot With LLAMA2 model')
    input_text = st.text_input("Ask your question!")

    # Initialize the Ollama model
    llm = Ollama(model="llama3")

    # Define a prompt template for the chatbot
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the questions"),
            ("user", "Question: {question}")
        ]
    )

    # Create a chain that combines the prompt and the Ollama model
    chain = prompt | llm

    # Function to get the chatbot response
    def get_response(question):
        return chain.invoke({"question": question})

    # Invoke the chain with the input text and display the output
    if input_text:
        response = get_response(input_text)
        st.write(response)

def main():
    option = st.sidebar.selectbox(
        'Select Chatbot',
        ('Document Processing Chatbot', 'General Chatbot')
    )

    if option == 'Document Processing Chatbot':
        setup_document_processing()
    elif option == 'General Chatbot':
        setup_general_chat()

if __name__ == "__main__":
    main()
