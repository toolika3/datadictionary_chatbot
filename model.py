import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrock
from langchain.chains import ConversationalRetrievalChain  # Langchain fuction to create chatbot that can remember past interactions
import streamlit as st
from streamlit_chat import message


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Initialise boto session
boto_session = boto3.Session()
aws_region = boto_session.region_name
# runtime client to call Bedrock models
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

#Loading the model
def load_llm():
    llm = ChatBedrock(
                      model_id='anthropic.claude-3-sonnet-20240229-v1:0',
                      model_kwargs={"temperature": 0.2, "max_tokens": 1200},
                      client=br_runtime
        )
    return llm


custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Chat History: {chat_history}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question', 'chat_history'])
    return prompt


embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
llm = load_llm()
prompt = set_custom_prompt()

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                 chain_type='stuff',
                                                 retriever=db.as_retriever(search_kwargs={'k':20}),
                                                 combine_docs_chain_kwargs={'prompt': prompt},
                                                 return_source_documents=False,
                                                 output_key='answer'
                                                )

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'> LLM Chatbot</h1>", unsafe_allow_html=True)

    def conversational_chat(user_input):
        result = qa_chain({"question": user_input, "chat_history": st.session_state['history']})
        st.session_state['history'].append((user_input, result["answer"]))
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


if __name__ == "__main__":
    main()

