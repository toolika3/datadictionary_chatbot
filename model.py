import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # Langchain fuction to create chatbot that can remember past interactions
import streamlit as st
from streamlit_chat import message


def main():
    st.markdown("<h1 style='text-align: center; color: blue;'> LLM Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
    DB_FAISS_PATH = 'vectorstore/db_faiss'

   # Initialise boto session
    boto_session = boto3.Session()
    aws_region = boto_session.region_name
    # runtime client to call Bedrock models
    br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

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

    #Retrieval QA Chain
    def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 20}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': prompt}
                                               )
        return qa_chain

    def conv_retrieval_chain(llm, prompt, db, memory):
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                         chain_type='stuff',
                                                         retriever=db.as_retriever(search_kwargs={'k':20}),
                                                         memory=memory,
                                                         combine_docs_chain_kwargs={'prompt': prompt},
                                                         return_source_documents=False,
                                                         #verbose=True,
                                                         condense_question_prompt=prompt,
                                                         get_chat_history=lambda h:h,
                                                         output_key='answer'
                                                        )
        return qa_chain

    #Loading the model
    def load_llm():
        llm = ChatBedrock(
                      model_id='anthropic.claude-3-sonnet-20240229-v1:0',
                      model_kwargs={"temperature": 0.2, "max_tokens": 1200},
                      client=br_runtime
        )
        return llm

    #QA Model Function
    def qa_bot():
        embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question"
        )
        #qa = retrieval_qa_chain(llm, qa_prompt, db)
        qa = conv_retrieval_chain(llm, qa_prompt, db, memory)
        return qa
    
    # Display conversation history using Streamlit messages
    def display_conversation(history):
        for i in range(len(history["generated"])):
            message(history["past"][i], is_user=True, key=str(i) + "_user")
            message(history["generated"][i],key=str(i))

    user_input = st.text_input("Query:", placeholder="Ask questions here", key='input')
    st.write(user_input)
    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        result = qa_bot()
        answer = result({'question': user_input})
        st.write(answer)
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    main()

