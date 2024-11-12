import streamlit as st
import json
from io import BytesIO, StringIO
import base64
import csv
import genanki
import random
from pdf_generator import generate_pdf
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from datetime import datetime

def generate_study_materials(vectordb, topic: str, client) -> dict:
    """Generate structured study materials using OpenAI's chat completions."""
    # Get relevant context from vector database
    context = ""
    if vectordb:
        search_results = vectordb.similarity_search(topic, k=3)
        for result in search_results:
            context += f"\nFrom {result.metadata.get('filename', 'document')}, Page {result.metadata.get('page', 'unknown')}:\n"
            context += f"{result.page_content}\n"

    messages = [
        {
            "role": "system",
            "content": "You are an expert educational content creator and professor. Your primary task is to analyze the provided reference material and create comprehensive study materials based on it. Focus heavily on incorporating and expanding upon the concepts found in the reference documents."
        },
        {
            "role": "user",
            "content": f"""Based primarily on the following reference material, create detailed study materials about {topic}.

            Reference Material:
            {context}

            Important: Your response should be heavily based on and aligned with the concepts and information found in the reference material above. Expand upon these concepts while maintaining accuracy and relevance to the source material.

            Generate a clear, concise title for these study materials that accurately reflects the content.

            Structure the study guide to include content for the following sections:
            1. Overview and Introduction
            2. Core Concepts and Fundamentals
            3. Technical Details and Methodology
            4. Practical Applications
            5. Challenges and Limitations
            6. Future Directions and Trends

            Also include 12 flashcards and 4 exercises with detailed solutions."""
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "study_materials_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "study_guide": {
                                "type": "object",
                                "properties": {
                                    "overview": {"type": "string"},
                                    "core_concepts": {"type": "string"},
                                    "technical_details": {"type": "string"},
                                    "practical_applications": {"type": "string"},
                                    "challenges": {"type": "string"},
                                    "future_directions": {"type": "string"}
                                },
                                "required": ["overview", "core_concepts", "technical_details", 
                                            "practical_applications", "challenges", "future_directions"],
                                "additionalProperties": False
                            },
                            "flashcards": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "front": {"type": "string"},
                                        "back": {"type": "string"}
                                    },
                                    "required": ["front", "back"],
                                    "additionalProperties": False
                                }
                            },
                            "exercises": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "solution": {"type": "string"}
                                    },
                                    "required": ["question", "solution"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["title", "study_guide", "flashcards", "exercises"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        generated_content = json.loads(response.choices[0].message.content)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return {
            'title': f"{topic} Study Guide",
            'subtitle': f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            'study_guide': generated_content['study_guide'],
            'flashcards': generated_content['flashcards'][:12],
            'exercises': generated_content['exercises'][:4]
        }
    except Exception as e:
        st.error(f"Error generating study materials: {e}")
        return None

def generate_downloads(content):
    """Generate and display download buttons for study materials"""
    pdf = generate_pdf(content)
    st.success("Materials generated successfully!")
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_filename = f"NLP_material_{current_time}"
    
    st.markdown("### 📥 Download Materials")
    st.markdown(get_download_link(pdf, f"{base_filename}.pdf", "📄 Download PDF Study Guide"), 
               unsafe_allow_html=True)
    
    if content['flashcards']:
        st.markdown("### 📱 Flashcard Exports")
        
        quizlet_text = generate_quizlet_format(content['flashcards'])
        st.markdown(get_download_link(quizlet_text, f"{base_filename}_quizlet.txt", "📚 Download for Quizlet Import"), 
                   unsafe_allow_html=True)
        
        try:
            anki_deck = generate_anki_deck(content['flashcards'], content['title'])
            st.markdown(get_download_link(anki_deck, f"{base_filename}_anki.apkg", "🎴 Download Anki Deck"), 
                       unsafe_allow_html=True)
        except Exception as e:
            st.warning("Anki deck generation failed. You can still use the other export options.")
        
        csv_content = generate_csv_format(content['flashcards'])
        st.markdown(get_download_link(csv_content, f"{base_filename}_flashcards.csv", "📊 Download CSV"), 
                   unsafe_allow_html=True)
    
    display_preview(content)

def generate_quizlet_format(flashcards):
    """Generate a tab-separated format suitable for Quizlet import"""
    output = StringIO()
    for card in flashcards:
        front = card['front'].replace('\n', ' ').replace('\t', ' ')
        back = card['back'].replace('\n', ' ').replace('\t', ' ')
        output.write(f"{front}\t{back}\n")
    return output.getvalue()

def generate_anki_deck(flashcards, deck_title):
    """Generate an Anki deck from flashcards"""
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'Simple Model',
        fields=[{'name': 'Question'}, {'name': 'Answer'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
        }]
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_title)
    
    for card in flashcards:
        note = genanki.Note(
            model=model,
            fields=[card['front'], card['back']]
        )
        deck.add_note(note)
    
    package = genanki.Package(deck)
    temp_file = BytesIO()
    package.write_to_file(temp_file)
    return temp_file.getvalue()

def generate_csv_format(flashcards):
    """Generate a CSV format of the flashcards"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Front', 'Back'])
    for card in flashcards:
        writer.writerow([card['front'], card['back']])
    return output.getvalue()

def get_download_link(file_content, filename, display_text):
    """Generate an HTML download link for file content"""
    b64 = base64.b64encode(file_content.encode() if isinstance(file_content, str) else file_content).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="download-button">{display_text}</a>'

def display_preview(content):
    """Display preview of generated content in Streamlit UI"""
    with st.expander("👀 Preview Generated Content"):
        # Display title and subtitle
        st.title(content['title'])
        st.write(content['subtitle'])
        
        # Display study guide sections
        st.header("Study Guide")
        
        # Overview and Introduction
        st.subheader("Overview and Introduction")
        st.write(content['study_guide']['overview'])
        
        # Core Concepts
        st.subheader("Core Concepts and Fundamentals")
        st.write(content['study_guide']['core_concepts'])
        
        # Technical Details
        st.subheader("Technical Details and Methodology")
        st.write(content['study_guide']['technical_details'])
        
        # Practical Applications
        st.subheader("Practical Applications")
        st.write(content['study_guide']['practical_applications'])
        
        # Challenges
        st.subheader("Challenges and Limitations")
        st.write(content['study_guide']['challenges'])
        
        # Future Directions
        st.subheader("Future Directions and Trends")
        st.write(content['study_guide']['future_directions'])
        
        # Display flashcards
        if content.get('flashcards'):
            st.header("Flashcards")
            for i, card in enumerate(content['flashcards'], 1):
                st.subheader(f"Card {i}")
                st.write("**Question:**")
                st.write(card['front'])
                st.write("**Answer:**")
                st.write(card['back'])
        
        # Display exercises
        if content.get('exercises'):
            st.header("Practice Exercises")
            for i, exercise in enumerate(content['exercises'], 1):
                st.subheader(f"Exercise {i}")
                st.write("**Question:**")
                st.write(exercise['question'])
                st.write("**Solution:**")
                st.write(exercise['solution'])