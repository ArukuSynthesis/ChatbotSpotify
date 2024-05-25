import streamlit as st
import requests
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

# Función para obtener información de una canción en Spotify
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
        return f"Track: {track['name']} by {track['artists'][0]['name']}\n[URL]({track['external_urls']['spotify']})"
    else:
        return "No se encontró la canción."

# Función para generar respuestas
def generate_response(input_text):
    # Cargar el modelo y el tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Liberar memoria
    del model
    torch.cuda.empty_cache()
    
    return response_text

# Función para responder al usuario
def chatbot_response(user_input, spotify_token):
    if "buscar canción" in user_input:
        track_name = user_input.split("buscar canción")[-1].strip()
        return get_track_info(track_name, spotify_token)
    else:
        return generate_response(user_input)

# Interfaz de usuario con Streamlit
st.title("Chatbot con Spotify y DialoGPT")

client_id = st.secrets['SPOTIFY_CLIENT_ID']
client_secret = st.secrets['SPOTIFY_CLIENT_SECRET']
HF_TOKEN = st.secrets['HF_TOKEN']
spotify_token = get_spotify_token(client_id, client_secret)

# Inicializar el historial de chat en la sesión de Streamlit
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Función para manejar la entrada del usuario
def handle_user_input():
    user_input = st.session_state.user_input
    st.session_state.user_input = ''  # Limpiar la entrada del usuario
    response = chatbot_response(user_input, spotify_token)
    st.session_state.chat_history.append(('Usuario', user_input))
    st.session_state.chat_history.append(('Chatbot', response))

# Entrada del usuario
st.text_input("Escribe tu mensaje:", key='user_input', on_change=handle_user_input)

# Mostrar el historial de chat
for speaker, message in st.session_state.chat_history:
    if speaker == 'Usuario':
        st.text_area(label="Usuario", value=message, height=50, max_chars=None, key=None)
    else:
        st.markdown(f"**Chatbot:** {message}")