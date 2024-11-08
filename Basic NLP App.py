import streamlit as st
from textblob import TextBlob
import spacy
import gensim
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from streamlit_extras.mention import mention
from streamlit_extras.stylable_container import stylable_container

# Summarization Function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    return ' '.join(summary_list)

# Token and Lemma Analysis
@st.cache_data
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    return [{"Token": token.text, "Lemma": token.lemma_} for token in docx]

# Entity Extraction
@st.cache_data
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    return [{"Entity": entity.text, "Label": entity.label_} for entity in docx.ents]

# UI Layout and Style
def main():
    st.title("✨ Ultimate NLP Application")
    st.subheader("Empowering Natural Language Processing for All")
    mention(label="By Pranav Pawar", url="https://github.com/prnvpwr2612", icon="👨‍💻")

    # Adding Section Styles
    with stylable_container(key="container", css_styles="""
            .stButton button {background-color: #4CAF50; color: white; border-radius: 8px;}
            .stMarkdown p {font-size: 1.1rem; color: #333;}
            """):

        # Summarization Section
        st.markdown("## Text Summarization")
        st.markdown("Generate a concise summary of your text:")
        message = st.text_area("Input Text", "Type or paste text here...")

        summary_option = st.radio("Choose a Summarizer:", ["Sumy", "Gensim"])
        if st.button("Summarize"):
            summary_result = sumy_summarizer(message) if summary_option == "Sumy" else summarize(message)
            st.write(summary_result)

        # Sentiment Analysis
        st.markdown("## Sentiment Analysis")
        message = st.text_area("Input for Sentiment Analysis", "Type or paste text here...")
        if st.button("Analyze Sentiment"):
            blob = TextBlob(message)
            sentiment = blob.sentiment
            st.metric("Polarity", sentiment.polarity)
            st.metric("Subjectivity", sentiment.subjectivity)

        # Entity Recognition
        st.markdown("## Named Entity Recognition (NER)")
        message = st.text_area("Input for NER", "Type or paste text here...")
        if st.button("Extract Entities"):
            entities = entity_analyzer(message)
            st.json(entities)

        # Tokenization & Lemmatization
        st.markdown("## Tokenization and Lemmatization")
        message = st.text_area("Input for Tokenization", "Type or paste text here...")
        if st.button("Tokenize"):
            tokens = text_analyzer(message)
            st.json(tokens)

    # Sidebar Info
    st.sidebar.markdown("## About")
    st.sidebar.info("This app performs essential NLP tasks using **Streamlit** and **SpaCy**, designed to be user-friendly and interactive.")

if __name__ == "__main__":
    main()