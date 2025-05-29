import json
import nltk
import random
import string
import warnings
import streamlit as st
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# Read the JSON data
img = Image.open("logo.png")
new_size = (150, 150)
img = img.resize(new_size)
st.image(img)

st.title("EnrollEase Chatbot")
st.caption("🚀 Laguna State Polytechnic University - Los Baños Enrollment Companion")

st.markdown("""
Welcome to the EnrollEase Chatbot! 🚀 This is your companion for enrollment at Laguna State Polytechnic University - Los Baños. 

You can ask me questions like:
- What are the courses offered in your college?
- What is the college timing?
- What is your contact number?
- Where is the college located?

Just type your question in the chat box and I'll do my best to assist you, but first, enter your student information to access this chatbot properly. 
""")

st.sidebar.header('Student Information')

name = st.sidebar.text_input('Name', key="a")
age = st.sidebar.number_input("Age", min_value=0, max_value=100)
course = st.sidebar.selectbox('Course', ['College of Computer Studies', "College of Fisheries",
                                         "College of Business and Administration"])

# Add a date input for the user's birthdate
min_birthdate = date(1950, 1, 1)  # Set the minimum birthdate year to 1950
max_birthdate = date.today()  # Set the maximum birthday to today's date
birthdate = st.sidebar.date_input('Birthdate', min_value=min_birthdate, max_value=max_birthdate)

# Check if the user has entered their information
if not name or not age or not course or not birthdate:
    st.sidebar.warning("Please change the year first before changing the month and day.")
    st.sidebar.error("Please enter your name, age, course, and birthdate to use this chatbot.")
    st.stop()

# Calculate the user's age based on their birthdate
current_date = datetime.now()
birthdate_year, birthdate_month, birthdate_day = birthdate.year, birthdate.month, birthdate.day

# Subtract the birth year from the current year
calculated_age = current_date.year - birthdate_year

# Subtract one if the birthday hasn't happened yet this year
if current_date.month < birthdate_month or (current_date.month == birthdate_month and current_date.day < birthdate_day):
    calculated_age -= 1

# Check if the user's age is 17 or above
if age < 17:
    st.sidebar.error("You must be 17 years or older to use this chatbot.")
    st.stop()

# Check if the calculated age matches the age the user inputted
if age != calculated_age:
    st.sidebar.error("The entered age does not match with the birthdate.")
    st.stop()

# If all checks pass, display a success message
st.sidebar.success("Your details have been entered correctly. You can now use the chatbot!")

with open('intents.json', 'r', errors='ignore') as f:
    data = json.load(f)

# Preprocess the data
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Process intents and responses
intents = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        intents.append(pattern)
        responses[pattern] = intent['responses'][0]  # Assuming there's only one response per pattern

# Create TF-IDF vectors
tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
tfidf.fit_transform(intents)

# Define greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hello there", "yo", "oy", "hello lspu")
GREETING_RESPONSES = ["hi", "hey", "im here!", "hi there", "hello", "i am glad! you are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Define response function
def response(user_response):
    user_tfidf = tfidf.transform([user_response])
    similarities = cosine_similarity(user_tfidf, tfidf.transform(intents))
    most_similar_index = similarities.argsort()[0][-1]
    chatbot_response = responses.get(intents[most_similar_index])  # Retrieve the corresponding response
    if chatbot_response is None:
        chatbot_response = "I'm sorry, I didn't understand that. Could you please rephrase your question?"
    return chatbot_response



# Define a list of suggested questions
suggested_questions = [
    "Hi",
    "How are you?",
    "Is anyone there?",
    "What are the courses offered in your college?",
    "What is the college timing?",
    "What is your contact number?",
    "Where is the college located?",
]

try:
    # Chat loop
    flag = True

    with st.chat_message("assistant"):
        st.write(f"Hello👋 {name}! How will I assist you today?")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create buttons for each suggested question
    for question in suggested_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            # Get the assistant's response
            assistant_response = response(question)
            # Only add the assistant's response to the messages if it's not None
            if assistant_response is not None:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    user_response = st.chat_input("Write your message here.", key="c")
    if user_response:
        if user_response != 'bye':
            st.session_state.messages.append({"role": "user", "content": user_response})
            # Get the assistant's response
            assistant_response = response(user_response)
            # Only add the assistant's response to the messages if it's not None
            if assistant_response is not None:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            flag = False
            print("ROBO: Bye! take care..")

    # Display the chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
