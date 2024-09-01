import streamlit as st
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Setup
st.set_page_config(page_title="Colega", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the language model
llm = ChatOpenAI(model_name='ft:gpt-4o-mini-2024-07-18:personal::9vW8AJQu', temperature=1, )

# Common sentences and predefined responses
common_sentences = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "adiós", "chao", "muchas gracias", "gracias", 
    "gracias!", "para qué estás entrenado?", "¿cómo estás?", "¿quién te creó?", "¿cuál es tu nombre?", 
    "¿qué puedes hacer?", "¿eres un robot?", "¿cómo me puedes ayudar?", "¿trabajas las 24 horas?"
]

def standard_answer(message_body):
    response_text = ""
    if message_body in ["hola", "buenos días", "buenas tardes", "buenas noches"]:
        response_text = "¡Hola! Hablas con Colega, tu asistente virtual 🤖 ¿En qué puedo ayudarte hoy? 🤓"
    elif message_body in ["adiós", "chao"]:
        response_text = "¡Hasta luego! Espero haberte ayudado. 👋 👋 👋 👋"
    elif message_body in ["gracias", "muchas gracias", "gracias!"]:
        response_text = "¡Con gusto! Si tienes otra consulta, estoy aquí para ayudarte 😊"
    elif message_body in ["para qué estás entrenado?"]:
        response_text = "Estoy entrenado para brindarte información sobre nuestra empresa, productos y ayudarte en el proceso de ventas para que tengas mejores resultados. También puedo responder una gran variedad de consultas 💡"
    elif message_body in ["¿cómo estás?"]:
        response_text = "¡Estoy funcionando al 100%! 🤖 ¿En qué puedo ayudarte?"
    elif message_body in ["¿quién te creó?"]:
        response_text = "Fui creado por un equipo de expertos en tecnología y automatización para ayudarte en todo lo que necesites 📚🔧"
    elif message_body in ["¿cuál es tu nombre?"]:
        response_text = "Mi nombre es Colega 🤖, tu asistente virtual siempre disponible para ayudarte."
    elif message_body in ["¿qué puedes hacer?"]:
        response_text = "Puedo ayudarte a encontrar información sobre nuestros productos, gestionar procesos de ventas y responder preguntas frecuentes sobre nuestra empresa 🌟"
    elif message_body in ["¿eres un robot?"]:
        response_text = "Sí, soy un robot asistente virtual diseñado para ayudarte con información y consultas 👾"
    elif message_body in ["¿cómo me puedes ayudar?"]:
        response_text = "Puedo proporcionarte información relevante, ayudarte en procesos de ventas y responder preguntas sobre nuestros productos y servicios 🔍"
    elif message_body in ["¿trabajas las 24 horas?"]:
        response_text = "¡Así es! Estoy disponible las 24 horas del día, los 7 días de la semana, siempre listo para ayudarte 💪"
    
    return response_text

# Initialize and configure the retriever for CSV files
@st.cache_resource()
def retriever_func(file_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    return retriever
retriever= retriever_func('987987.csv')
# Main chat application
def chat():
    st.write("# Preguntale a Colega")
    
    # Initialize session state for messages if not already initialized
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    user_message = st.chat_input("Escribe tu mensaje aquí...")

    if user_message:
        if user_message in common_sentences:
            final_answer = standard_answer(user_message)
        else:
            # Retrieve context and get response from model
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Prepare the message sequence for the model using the appropriate message classes
            messages = [
                SystemMessage(content=f"Context: {context_text}"),
                HumanMessage(content=user_message)
            ]
            response = llm(messages)
            final_answer = response.content  # Access the content directly
        
        # Add both the user message and the assistant's response to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    # Display the entire conversation history
    if st.session_state["messages"]:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Sidebar functionalities
def sidebar():
    st.sidebar.title("Opciones")
    
    # Restart button
    if st.sidebar.button("Reiniciar conversación"):
        st.session_state.clear()
        st.experimental_rerun()
    
    # Display CSV content
    if st.sidebar.checkbox("Mostrar base de datos (CSV)"):
        st.sidebar.write("### Contenido del CSV:")
        df = pd.read_csv('987987.csv')  # Load CSV content directly
        st.sidebar.write(df)

# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1> 🧠 Colega 🧠 </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️Tu asistente digital</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    # Sidebar
    sidebar()

    # Main chat interface
    chat()

if __name__ == "__main__":
    main()
