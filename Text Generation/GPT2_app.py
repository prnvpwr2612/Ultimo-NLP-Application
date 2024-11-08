import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="GPT-2 Text Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with removed white backgrounds
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            font-size: 18px;
            font-weight: 500;
            color: white;
            background: linear-gradient(to right, #4776E6, #8E54E9);
            border: none;
        }
        .stTextArea>div>div>textarea {
            border-radius: 10px;
            border: 2px solid #4776E6;
        }
        .output-container {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: transparent;
            border: 2px solid #e9ecef;
            margin: 1rem 0;
        }
        .parameter-label {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .gradient-border {
            background: linear-gradient(to right, #4776E6, #8E54E9);
            padding: 2px;
            border-radius: 10px;
            margin-bottom: 1em;
        }
        .section-header {
            color: #1e1e1e;
            padding: 0.5em;
            border-radius: 8px;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name="gpt2"):
    """Load GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_text(prompt, model, tokenizer, max_length, temperature, top_k, num_beams=None, num_return_sequences=1):
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    
    # Generate text based on the generation method
    if num_beams:
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=temperature
        )
    else:
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return_sequences
        )
    
    # Decode the generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; 
                   background: linear-gradient(to right, #4776E6, #8E54E9);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 3em;
                   padding: 1em 0;'>
            ✨ GPT-2 Text Generator
        </h1>
    """, unsafe_allow_html=True)

    # Load model
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["gpt2", "gpt2-medium", "gpt2-large"],
        help="Larger models may take longer to load and generate"
    )

    # Load the selected model
    with st.spinner(f"Loading {model_option} model..."):
        model, tokenizer = load_model(model_option)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div class='gradient-border'>
                <h3 class='section-header'>Enter Your Prompt</h3>
            </div>
        """, unsafe_allow_html=True)
        
        prompt = st.text_area(
            "",
            height=150,
            placeholder="Type your prompt here...",
            key="prompt_input"
        )

    with col2:
        st.markdown("""
            <div class='gradient-border'>
                <h3 class='section-header'>Generation Stats</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='padding: 1em;'>
                <p>🤖 Model: {model_option}</p>
                <p>🕒 Current Time: {get_current_time()}</p>
                <p>📝 Prompt Length: {len(prompt) if prompt else 0} characters</p>
            </div>
        """, unsafe_allow_html=True)

    # Generation parameters
    st.sidebar.markdown("""
        <div class='gradient-border'>
            <h2 class='section-header'>⚙️ Generation Parameters</h2>
        </div>
    """, unsafe_allow_html=True)

    generation_method = st.sidebar.radio(
        "Generation Method",
        ["Beam Search", "Top-K Sampling"]
    )

    max_length = st.sidebar.slider("Maximum Length", 50, 1000, 200, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    
    if generation_method == "Beam Search":
        num_beams = st.sidebar.slider("Number of Beams", 1, 20, 5, 1)
        num_sequences = st.sidebar.slider("Number of Sequences", 1, 5, 1, 1)
        top_k = None
    else:
        num_beams = None
        num_sequences = st.sidebar.slider("Number of Sequences", 1, 5, 1, 1)
        top_k = st.sidebar.slider("Top K", 1, 100, 50, 1)

    if st.button("✨ Generate Text"):
        if prompt:
            try:
                with st.spinner("🌟 Generating magical text..."):
                    generated_texts = generate_text(
                        prompt,
                        model,
                        tokenizer,
                        max_length,
                        temperature,
                        top_k,
                        num_beams,
                        num_sequences
                    )
                    
                    for i, text in enumerate(generated_texts, 1):
                        st.markdown(f"""
                            <div class='gradient-border'>
                                <h3 class='section-header'>✨ Generated Text {i}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class='output-container'>
                                {text}
                            </div>
                        """, unsafe_allow_html=True)
                
                st.success("✅ Generation completed successfully!")
                
            except Exception as e:
                st.error(f"🚫 An error occurred: {str(e)}")
        else:
            st.warning("🤔 Please enter a prompt to generate text!")

    # Footer
    st.markdown("""
        <div style='margin-top: 3em; text-align: center; padding: 1em;'>
            <p style='color: #666;'>
                Powered by GPT-2 | Built with Streamlit ❤️
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()