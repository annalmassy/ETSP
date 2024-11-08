import streamlit as st
from rouge import Rouge

test_questions_glossary = [
    "How would you define accent?",
    "How would you define classifier?",
    "How would you define nucleus? ",
    "How would you define phoneme?",
    "How would you define treebank?",
]

correct_answers_glossary = {
    "How would you define accent?": "Pronunciation, especially that associated with a particular regional or social group.",
    "How would you define classifier?" : "A morpheme that is used to grammatically individuate mass nouns or specify a subclass of nouns in a language, often on the basis of some semantic property of the noun. In the Mandarin Chinese examples wuˇ-ge rén ‘five persons’ and nèi-zha¯ng zhîé ‘that (sheet of) paper,’ the suffixes -ge and -zhang are classifiers which specify units of people and paper, respectively.",
    "How would you define nucleus? ": "The sonority peak of a syllable, usually the vowel.",
    "How would you define phoneme?" : "One of the contrastive sounds of a language; a label for a group of sounds that are perceived by the speaker to be the same.",
    "How would you define treebank?" : "A corpus of sentences in a language that has been analyzed into parse trees.",
}

def compute_rouge_scores(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference, avg=True)
    return scores

def conduct_tests(vectordb, use_rag, prompt_template, client):
    for question in test_questions_glossary:
        # Display each question (you can print this to console or log if needed)
        st.write(f"Question: {question}")

        # Add the question to the session history as if it were asked by the user
        st.session_state["chat_history"].append({"role": "user", "content": question})
        
        # Using RAG to search in Chroma database
        search_results = vectordb.similarity_search(question, k=3)

        # Constructing RAG context with source references
        pdf_extract = ""
        for result in search_results:
            page_content = result.page_content
            filename = result.metadata.get("filename", "unknown document")
            page = result.metadata.get("page", "unknown page")
            pdf_extract += f"{page_content} [Source: {filename}, Page: {page}]\n\n"

        # Create a context-based prompt for RAG model
        prompt_with_rag = [
            {"role": "system", "content": prompt_template},
            {"role": "assistant", "content": pdf_extract},
            {"role": "user", "content": question}
        ]

        # Generate the response using RAG model
        response_rag = []
        for chunk in client.chat.completions.create(
            model="gpt-4o", messages=prompt_with_rag, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text:
                response_rag.append(text)
        result_rag = "".join(response_rag).strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": result_rag})

        
        # Standard GPT-4o Mini without RAG
        prompt_basic = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": question}
        ]
        response_gpt3 = []
        for chunk in client.chat.completions.create(
            model="gpt-4o", messages=prompt_basic, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text:
                response_gpt3.append(text)
        result_gpt3 = "".join(response_gpt3).strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": result_gpt3})

        # Get the correct answer for the current question
        correct_answer = correct_answers_glossary.get(question)

        # Compute ROUGE scores for RAG and GPT-4o responses
        if result_rag:
            rouge_scores_rag = compute_rouge_scores(correct_answer, result_rag)
        else:
            rouge_scores_rag = {"rouge-1": {"f": 0, "p": 0, "r": 0}}  # No RAG answer to compare

        rouge_scores_no_rag = compute_rouge_scores(correct_answer, result_gpt3)

        # Show the results to the user
        st.write(f"Answer (with RAG): {result_rag if result_rag else 'N/A'}")
        st.write(f"Answer (without RAG): {result_gpt3}")
        st.write(f"Correct Answer: {correct_answer}")
        st.write(f"ROUGE scores (with RAG): {rouge_scores_rag}")
        st.write(f"ROUGE scores (without RAG): {rouge_scores_no_rag}")

