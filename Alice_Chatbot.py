import nltk
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Preprocess text
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words



# Load the text file
@st.cache_data
def load_text():
    try:
        file_path = "alice_in_wonderland.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')
    except FileNotFoundError:
        st.error(
            "Text file not found. Please ensure file is in the same directory"
        )
        return ""


# Prepare corpus
@st.cache_resource
def prepare_corpus(text):
    sentences = sent_tokenize(text)
    return [preprocess(sentence) for sentence in sentences]


# Calculate Jaccard similarity
def jaccard_similarity(query, sentence):
    query_set = set(query)
    sentence_set = set(sentence)
    if len(query_set.union(sentence_set)) == 0:
        return 0
    return len(query_set.intersection(sentence_set)) / len(query_set.union(sentence_set))


# Find the most relevant sentence
def get_most_relevant_sentence(query, corpus, original_sentences):
    query = preprocess(query)
    max_similarity = 0
    best_sentence = "I couldn't find a relevant answer."
    for i, sentence in enumerate(corpus):
        similarity = jaccard_similarity(query, sentence)
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = original_sentences[i]
    return best_sentence


# Main Function to streamlit APP
def main():
    st.title("Wonderland's Chatbot")
    st.write("Hello! Ask me anything related to Alice in Wonderland")

    with st.expander("Click me for suggestions"):
        st.write("""
        1. Who does Alice meet first in Wonderland?
        2. What is the significance of the bottle labeled 'Drink Me'?
        3. How does Alice enter Wonderland?
        4. What is the Queen of Hearts known for?
        5. What game does the Queen of Hearts play with Alice?
        6. What advice does the Caterpillar give Alice?
        7. Why did Alice follow the White Rabbit?
        """)

    text = load_text()
    if text:
        corpus = prepare_corpus(text)
        original_sentences = sent_tokenize(text)

        user_input = st.text_input("Enter your question:")

        if st.button("Submit"):
            if user_input.strip():
                response = get_most_relevant_sentence(user_input, corpus, original_sentences)
                st.write(f"Chatbot: {response}")
            else:
                st.write("Please enter a Question")



# Run te app
if __name__ == "__main__":
    main()
