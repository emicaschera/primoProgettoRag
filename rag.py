from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
import streamlit as st

csv_filepath = "recipes.csv"
recipe_number_to_load = 2
llm_model = "llama2"


@st.cache_resource()
def process_and_load_csv(filepath):
    df = pd.read_csv(filepath)
    df_filtered = df[['RecipeInstructions']]
    df_filtered.to_csv("recipes_filtered.csv", index=False)
    loader = CSVLoader(file_path='recipes_filtered.csv', source_column="RecipeInstructions")
    data = loader.load_and_split()

    recipe5 = data[:recipe_number_to_load]

    clean_recipe5 = []

    for doc in recipe5:
        clean_recipe5.append(doc.page_content.removeprefix("RecipeInstructions: c(")
                             .removesuffix(")")
                             .replace('"', '')
                             .replace('.', '')
                             .lower())

    processed_documents = [Document(page_content=content) for content in clean_recipe5]

    return processed_documents


embeddings_model = OllamaEmbeddings(model=llm_model)


@st.cache_resource()
def initialize_vectorstore(_documents_to_store):
    return DocArrayInMemorySearch.from_documents(_documents_to_store, embedding=embeddings_model)


documents = process_and_load_csv(csv_filepath)

vectorestore = initialize_vectorstore(documents)

retriever = vectorestore.as_retriever()

parser = StrOutputParser()

model = Ollama(model=llm_model)

prompt_template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(prompt_template)
prompt.format(context="Here is some context", question="Here is a question")

question = "how long do i need to grill the kebab?"

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)


st.title("Food AI assistant")
user_input = st.text_input("Hi user! Ask me something about food recipe!", "")

if st.button("Submit"):
    try:
        response = rag_chain.invoke({'question': user_input})
        #user_question = user_input.removeprefix("Hi user! Ask me something about food recipe!")

        #st.write(text)
        st.write( response)



    except Exception as e:
        st.write(f"An error occurred: {e}")




