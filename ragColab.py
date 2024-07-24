import gradio as gr
import ollama
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.document_loaders import CSVLoader

# csv read filepath
csv_filepath = "dataset.csv"

# url read filepath
url_filepath = "www.blablabla.it"

# pdf filepath
pdf_filepath = "dataset.pdf"

# model for inference
llm_model = "llama3"

# model for embeddings
embeddings_model = "nomic-embed-text"

# function to read context from a pdf
def read_from_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    return pages

# function to read context from url given
def read_from_url(filepath):
    loader = WebBaseLoader(filepath)
    docs = loader.load()
    return docs

# function to read csv
def read_from_csv(filepath):
    # Split the loaded documents into chunks
    loader = CSVLoader(file_path=csv_filepath)
    docs = loader.load_and_split()
    return docs

# Create Ollama embeddings
embeddings = OllamaEmbeddings(model= embeddings_model)

# create documents end vectore store
documents = read_from_csv(csv_filepath)
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

# define the model
ollama = Ollama(model=llm_model)

# Define the function to call the Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama(formatted_prompt)
    return response


# Define the RAG setup
retriever = vectorstore.as_retriever()

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)


# Define the Gradio interface
def get_important_facts(question):
    return rag_chain(question)

# Create a Gradio app interface
iface = gr.Interface(
  fn=get_important_facts,
  inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
  outputs="text",
  title="RAG with Llama3",
  description="Ask questions about the proveded context",
)

# Launch the Gradio app
iface.launch()




