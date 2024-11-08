import collections
import math
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model, load_model

# Set page config at the very beginning
st.set_page_config(
    page_title="Question/Sentence Classifier",
    page_icon="📝",
    layout="wide"
)

# Constants
MAX_SEQ_LENGTH = 22
MODEL_PATH = 'question_classifier_model'
DATA_DIR = 'data'
URL = 'http://cogcomp.org/Data/QA/QC/'
SEED = 54321

if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

def set_example():
    st.session_state.question_input = "What is the capital of France?"

def clear_text():
    st.session_state.question_input = ""


def download_and_process_data():
    """Download and process the question classification dataset"""
    def download_data(dir_name, filename, expected_bytes):
        os.makedirs(dir_name, exist_ok=True)
        filepath = os.path.join(dir_name, filename)
        if not os.path.exists(filepath):
            filepath, _ = urlretrieve(URL + filename, filepath)
        
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            raise Exception(f'Failed to verify {filepath}')
        return filepath

    def read_data(filename):
        questions, categories = [], []
        with open(filename, 'r', encoding='latin-1') as f:
            for row in f:
                row_str = row.split(":")
                cat, question = row_str[0], row_str[1]
                questions.append(' '.join(question.split()[1:]).lower().strip())
                categories.append(cat)
        return questions, categories

    # Download and read data
    train_file = download_data(DATA_DIR, 'train_5500.label', 335858)
    train_questions, train_categories = read_data(train_file)
    
    return train_questions, train_categories

def create_and_train_model(train_questions, train_categories):
    """Create and train the CNN model"""
    # Create and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_questions)
    n_vocab = len(tokenizer.index_word) + 1

    # Convert categories to numeric labels
    unique_cats = np.unique(train_categories)
    labels_map = dict(zip(unique_cats, np.arange(len(unique_cats))))
    numeric_labels = [labels_map[cat] for cat in train_categories]
    
    # Prepare sequences
    sequences = tokenizer.texts_to_sequences(train_questions)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

    # Create model
    word_inputs = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embedding = layers.Embedding(input_dim=n_vocab, output_dim=64)(word_inputs)
    
    # Convolution layers
    conv1 = layers.Conv1D(100, kernel_size=3, padding='same', activation='relu')(embedding)
    conv2 = layers.Conv1D(100, kernel_size=4, padding='same', activation='relu')(embedding)
    conv3 = layers.Conv1D(100, kernel_size=5, padding='same', activation='relu')(embedding)
    
    conv_out = layers.Concatenate(axis=-1)([conv1, conv2, conv3])
    pool_out = layers.MaxPool1D(pool_size=MAX_SEQ_LENGTH)(conv_out)
    flatten = layers.Flatten()(pool_out)
    output = layers.Dense(len(unique_cats), activation='softmax')(flatten)

    model = Model(inputs=word_inputs, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(padded_sequences, np.array(numeric_labels), epochs=10, batch_size=128, validation_split=0.1)
    
    # Save model and tokenizer
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(os.path.join(MODEL_PATH, 'model.h5'))
    np.save(os.path.join(MODEL_PATH, 'labels_map.npy'), labels_map)
    
    return model, tokenizer, labels_map

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and prepare tokenizer"""
    if not os.path.exists(os.path.join(MODEL_PATH, 'model.h5')):
        train_questions, train_categories = download_and_process_data()
        model, tokenizer, labels_map = create_and_train_model(train_questions, train_categories)
    else:
        model = load_model(os.path.join(MODEL_PATH, 'model.h5'))
        labels_map = np.load(os.path.join(MODEL_PATH, 'labels_map.npy'), allow_pickle=True).item()
        
        # Recreate tokenizer from training data
        train_questions, _ = download_and_process_data()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_questions)
    
    return model, tokenizer, labels_map

def preprocess_text(text, tokenizer):
    """Preprocess input text for prediction"""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    return padded_sequences

def predict(text, model, tokenizer):
    """Make prediction on input text"""
    processed_text = preprocess_text(text, tokenizer)
    probabilities = model.predict(processed_text)[0]
    # Convert float32 to native Python float
    return [float(p) for p in probabilities]

def main():
    # Load model and tokenizer
    model, tokenizer, labels_map = load_model_and_tokenizer()
    reverse_labels_map = {v: k for k, v in labels_map.items()}
    
    # Streamlit UI
    with st.sidebar:
        st.title("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        st.markdown("---")
        st.caption(f"Current time: {datetime.now().strftime('%H:%M:%S')}")

    st.title("📝 Question/Sentence Classifier")
    st.markdown("Classify your questions and sentences into different categories using AI")

    # Text input
    question_input = st.text_area(
        "Enter your text:",
        height=100,
        placeholder="Type your question or sentence here...",
        key="question_input"
    )

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("🧹 Clear", on_click=clear_text)
    with col2:
        st.button("📚 Example", on_click=set_example)
    with col3:
        classify_button = st.button("🔍 Classify")

    # Classification logic
    if classify_button and question_input.strip():
        st.markdown("---")
        st.markdown("### 🎯 Classification Results")
        
        with st.spinner("Analyzing text..."):
            try:
                probabilities = predict(question_input, model, tokenizer)
                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.info("#### Prediction")
                    st.metric(
                        "Text Type",
                        reverse_labels_map[predicted_class],
                        f"{confidence:.1%} confidence"
                    )
                
                with col2:
                    st.info("#### Confidence")
                    st.progress(float(confidence))
                    if confidence < confidence_threshold:
                        st.warning("⚠️ Low confidence prediction!")
                
                # Show all probabilities
                st.markdown("#### Detailed Analysis")
                cols = st.columns(len(labels_map))
                for idx, (label, class_idx) in enumerate(labels_map.items()):
                    with cols[idx]:
                        prob = float(probabilities[class_idx])
                        st.metric(
                            label,
                            f"{prob:.1%}",
                            delta="highest" if class_idx == predicted_class else None
                        )

            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

    # Tips section
    if not question_input.strip():
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Enter a complete, well-formed question or sentence
        - Text should be in English
        - Try different types of input (questions, statements, commands, etc.)
        - Check the confidence score for reliability
        """)

if __name__ == "__main__":
    main()