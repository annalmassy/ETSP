import streamlit as st

def process_chat_message(user_input, vectordb, client):
    """
    Process user messages and generate responses using the chatbot.
    
    Args:
        user_input (str): The user's message
        vectordb: The vector database for document search
        client: The OpenAI client instance
    """
    if vectordb:
        # Search in Chroma database
        search_results = vectordb.similarity_search(user_input, k=3)
        
        # Construct RAG context with source references
        context = ""
        for result in search_results:
            page_content = result.page_content
            filename = result.metadata.get("filename", "unknown document")
            page = result.metadata.get("page", "unknown page")
            context += f"\n{page_content}\n[Source: {filename}, Page: {page}]\n"
        
        # Create message placeholder for streaming
        message_placeholder = st.empty()
        full_response = ""
        
        # Generate streaming response with RAG context
        for response in client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an educational AI assistant specializing in NLP. Base your responses on the provided context and cite sources when possible."},
                {"role": "assistant", "content": f"Context from documents: {context}"},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            ],
            stream=True,
        ):
            if response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        return full_response
    
    return "I'm sorry, but I don't have access to the document database at the moment."