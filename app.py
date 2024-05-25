import os
import requests
import base64
import streamlit as st
from dotenv import load_dotenv
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

# Modelo de SpaCy
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")


load_dotenv()

client_id = st.secrets['SPOTIFY_CLIENT_ID']
client_secret = st.secrets['SPOTIFY_CLIENT_SECRET']
HT_TOKEN = st.secrets['HT_TOKEN']

# Función para obtener el token de Spotify
def get_spotify_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    auth_data = {
        'grant_type': 'client_credentials'
    }
    headers = {
        'Authorization': f'Basic {auth_header}'
    }
    response = requests.post(auth_url, headers=headers, data=auth_data)
    return response.json().get('access_token')


spotify_token = get_spotify_token(client_id, client_secret)
print(f"Spotify Token: {spotify_token}")

# Cargar el modelo y el tokenizer de Hugging Face
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Función para generar respuestas
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

# Lista de palabras clave relacionadas con la búsqueda de canciones
keywords = ["buscar", "canción", "track", "escuchar", "reproducir"]

# Función para detectar si el input del usuario es una solicitud de búsqueda de canción
def is_song_search(user_input):
    doc = nlp(user_input.lower())
    for token in doc:
        if token.lemma_ in keywords:
            return True
    return False

# Extraer el nombre de la canción del input del usuario
def extract_song_name(user_input):
    doc = nlp(user_input.lower())
    song_name = []
    for token in doc:
        if token.lemma_ not in keywords:
            song_name.append(token.text)
    return " ".join(song_name).strip()

# Función para obtener información de una canción de Spotify
def get_track_info(track_name, spotify_token):
    search_url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {spotify_token}"
    }
    params = {
        "q": track_name,
        "type": "track",
        "limit": 1
    }
    response = requests.get(search_url, headers=headers, params=params)
    track_info = response.json()
    if track_info['tracks']['items']:
        track = track_info['tracks']['items'][0]
        return f"Track: {track['name']} by {track['artists'][0]['name']}\nURL: {track['external_urls']['spotify']}"
    else:
        return "No se encontró la canción."

# Función de respuesta del chatbot
def chatbot_response(user_input, spotify_token):
    if is_song_search(user_input):
        track_name = extract_song_name(user_input)
        return get_track_info(track_name, spotify_token)
    else:
        return generate_response(user_input)

# Ejemplos de uso
user_inputs = [
    "Quiero escuchar Shape of You",
    "Puedes buscar la canción Despacito?",
    "Reproducir Hotel California",
    "Hola, ¿cómo estás?",
    "Buscar track Bohemian Rhapsody"
]

for user_input in user_inputs:
    response = chatbot_response(user_input, spotify_token)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
