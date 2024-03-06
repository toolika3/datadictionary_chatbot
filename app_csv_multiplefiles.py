import pandas as pd  # to create a dataframe for displaying all models from Bedrock
import boto3  # AWS SDK for Python(integrate AWS services for Python)

import streamlit as st  # user interface for the chatbot
from streamlit_chat import message # functions from streamlit_chat for chatapp like interface
from langchain.llms.bedrock import Bedrock # For accessing LLMs in Bedrock using Langchain framework ( used for chaining steps required for developing chatbots)
from langchain.text_splitter import RecursiveCharacterTextSplitter # Langchain function to split large text based on specific chunk size 
from streamlit_extras.add_vertical_space import add_vertical_space # functions from streamlit_extras for adding vertical space in the container
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores import FAISS  # Vectordb library for efficient similarity search and clustering of dense vectors
from langchain.chains import ConversationalRetrievalChain  # Langchain fuction to create chatbot that can remember past interactions
from langchain.document_loaders.csv_loader import CSVLoader  # langchain fuction to load CSV files
from langchain.document_loaders import DirectoryLoader  #langchain function to load CDV files from the Directory

# Global variables to access and save files on local system
DB_FAISS_PATH = 'vectorstore/db_faiss'
#csvfile_path = '../bedrock_test/csvstore/'
csvfile_path = '../bedrock_test/dsetstore/'

# Initialise boto session
boto_session = boto3.Session()
aws_region = boto_session.region_name

# static client,added here as an example, not used anywhere
br_client = boto_session.client(service_name='bedrock', region_name=aws_region)

# runtime client to call Bedrock models
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

# List FM models and test connection
fm = br_client.list_foundation_models()['modelSummaries']
pd_df = pd.DataFrame(fm)
pd_df.head()
#pd_df.columns
#pd_df.modelName.unique()


# Sidebar contents for Streamlit Application
st.sidebar.title('📁💬 DSET ChatBot')
st.sidebar.markdown('''
    ## About
    This is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [AWS Bedrock](https://aws.amazon.com/bedrock/) LLM models

    ''')
add_vertical_space(3)
st.sidebar.write('Data Products Team')

# upload multiple CSV files by referencing the directory/folder where all files are stored

loader = DirectoryLoader(csvfile_path, glob='**/*.csv', loader_cls=CSVLoader)
data = loader.load()

# Dividing the documents into smaller chunks that fits within the context window of the LLM. Claude -v2 has 200K(~150K words) context window to process
# Fine-Tune (1) - We would want to see how changing the chunk_size and chunk_overlap effects the LLms response. Goal is to optimally chunk the documents that can be processed conviniently by the model and reaches closer to the context window. This will help with having lesser chunk and more data available to the LLM to responding on questions asked

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
            )

chunks = text_splitter.transform_documents(data) # chunked data ready to be vector embedded for storage
embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',
                               client=br_runtime) # create embedding object that uses AWS Bedrock API and calls Titan embedding model that creates vector representation of the chunks
vectordb = FAISS.from_documents(chunks, embedding=embeddings) # the embedinngs are created on text chunks and stored in Facebook AI similarity search vector db. FAISS provides faster data retrival and access
vectordb.save_local(DB_FAISS_PATH) # saved the vector embeddings locally. IN FUTURE - should change to store in db (vectors should be created only when new data needs chunking) 

# Bedrock llm object for Claude model using bedrock runtime client. Default inference parameters are overwritten to generate chatbot response fitiing the usecase
# Temperature - determines the amount of variation in model's response. It modulates the probability masss fuction for next token. Lower value leads to deterministic response, higher value encourages randomness.
# Top_P - The percent of most likely candidates that the model considers for next toen. If top_p<0.1, then the model selects top 10% of the probability distribution of tokens that could be next in sequence
# Top_K - The # of most likely candidates that the model considers for next token. If top_k=400, then model selects 400 most probable tokens that could be next in sequence
# mak_tokens_to_sample - To configure max number of tokens to before stopping
# Claude's parameter info for default, min, max is below
# Temperature 0.5,0,4096
# top_p 1,0,1
# top_k 250,0,500
# max_tokens_to_sample 200,0,4096
# Fine-Tune (2) - We would want to see how fine tuning parameters effects the LLms response. Goal is to optimally fine tune the parameter that enhances the correctness and clearity in response 
llm = Bedrock(model_id='anthropic.claude-v2:1',
              model_kwargs={"temperature": 0.2, "top_p": 0.5, "top_k": 400, "max_tokens_to_sample": 900},
              client=br_runtime)

# ConversationalRetrievalChain allows to use memory and chain the chat history alongwith the embeddings retrived from the vectorstore that similar to the question asked in the chatbot. These two are sent to LLMs for processing and generating final answer
# vectorsdb object's provides flexibilty to change the # of embeddings that can be fetched before passing the information to theLLM model
chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectordb.as_retriever(search_kwargs={"k":30}))

# # Fine-Tune (3) - We would want to see if we can fetch vector embedding for asked questions based on a scored threshold percent in place of #. This will be more reliable
#chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectordb.as_retriever(search_kwargs={"score_threshold":0.3}))


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask Questions"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

#Container for chat history
response_container = st.container()
# container for user text
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Ask questions here", key='input')
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = conversational_chat(user_input)
        # store output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))