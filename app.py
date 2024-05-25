import os
import requests
import base64
import spacy_streamlit
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import subprocess

# Función para descargar el modelo de spaCy si no está disponible
def download_spacy_model():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Intentar cargar el modelo spaCy, si falla descargarlo
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    download_spacy_model()
    nlp = spacy.load('en_core_web_sm')

# Obtener secretos de Streamlit
client_id = st.secrets['SPOTIFY_CLIENT_ID']
client_secret = st.secrets['SPOTIFY_CLIENT_SECRET']
HF_TOKEN = st.secrets['HF_TOKEN']

def get_spotify_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    auth_data = {'grant_type': 'client_credentials'}
    headers = {'Authorization': f'Basic {auth_header}'}
    response = requests.post(auth_url, headers=headers, data=auth_data)
    return response.json().get('access_token')

spotify_token = get_spotify_token(client_id, client_secret)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

st.title("Spotify Chatbot")
st.write("Este es un chatbot que utiliza la API de Spotify.")

query = st.text_input("Introduce tu consulta:")

if query:
    doc = nlp(query)
    st.write(f"Tokens: {[token.text for token in doc]}")

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

keywords = ["buscar", "canción", "track", "escuchar", "reproducir"]

def is_song_search(user_input):
    doc = nlp(user_input.lower())
    for token in doc:
        if token.lemma_ in keywords:
            return True
    return False

def extract_song_name(user_input):
    doc = nlp(user_input.lower())
    song_name = []
    for token in doc:
        if token.lemma_ not in keywords:
            song_name.append(token.text)
    return " ".join(song_name).strip()

def get_track_info(track_name, spotify_token):
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {spotify_token}"}
    params = {"q": track_name, "type": "track", "limit": 1}
    response = requests.get(search_url, headers=headers, params=params)
    track_info = response.json()
    if track_info['tracks']['items']:
        track = track_info['tracks']['items'][0]
        return f"Track: {track['name']} by {track['artists'][0]['name']}\nURL: {track['external_urls']['spotify']}"
    else:
        return "No se encontró la canción."

def chatbot_response(user_input, spotify_token):
    if is_song_search(user_input):
        track_name = extract_song_name(user_input)
        return get_track_info(track_name, spotify_token)
    else:
        return generate_response(user_input)

if query:
    response = chatbot_response(query, spotify_token)
    st.write(response)
