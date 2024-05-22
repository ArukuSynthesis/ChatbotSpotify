import os
import pandas as pd
import joblib
import requests
import gdown
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_swagger_ui import get_swaggerui_blueprint
from flasgger import Swagger, swag_from
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, Application, ApplicationBuilder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import asyncio
import threading

# Configuración del token de Telegram
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]

# Crear la carpeta data si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# Descargar el archivo CSV
url = 'https://drive.google.com/u/1/uc?id=18jTl6d0-7plusVePqU3y7CoFmvyZfG2S&export=download'
output = 'data/tracks_features.csv'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)

# Cargar el modelo lineal
model_path = 'models/linearmodel.pkl'
model = joblib.load(model_path)

# Cargar el dataset
csv_path = 'data/tracks_features.csv'
data = pd.read_csv(csv_path)

# Renombrar columnas
data.rename(columns={'id': 'track_id', 'name': 'track_name', 'album': 'album', 'artists': 'artist'}, inplace=True)
data['artist'] = data['artist'].str.replace(r"[\[\]']", "", regex=True)

# Columnas no numéricas
non_numeric_columns = ['track_id', 'track_name', 'album', 'artist']
numeric_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo']

class RecommendTrack(Resource):
    @swag_from({
        'responses': {
            200: {
                'description': 'Recommended tracks based on the provided track name',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'artist': {'type': 'string'},
                            'track_id': {'type': 'string'},
                            'track_name': {'type': 'string'},
                            'album': {'type': 'string'}
                        }
                    }
                }
            }
        }
    })
    def get(self):
        """
        Recommends tracks based on the provided track name
        ---
        parameters:
          - name: track_name
            in: query
            type: string
            required: true
            description: Name of the track to base recommendations on
        responses:
          200:
            description: A list of recommended tracks
        """
        track_name = request.args.get('track_name')
        if not track_name:
            return {'error': 'track_name is required'}, 400

        filtered_data = data[data['track_name'].str.contains(track_name, case=False, na=False)].head(1)
        if filtered_data.empty:
            return {'error': 'No tracks found'}, 404

        track_features = filtered_data[numeric_columns].values
        all_features = data[numeric_columns].values

        similarities = cosine_similarity(track_features, all_features)
        data['similarity'] = similarities[0]

        recommendations = data.sort_values(by='similarity', ascending=False).head(6)
        recommendations = recommendations.iloc[1:]  # Excluir la canción de entrada

        response = recommendations[non_numeric_columns].to_dict(orient='records')
        return jsonify(response)

class SearchTrack(Resource):
    @swag_from({
        'responses': {
            200: {
                'description': 'Search for tracks based on the provided track name',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'artist': {'type': 'string'},
                            'track_id': {'type': 'string'},
                            'track_name': {'type': 'string'},
                            'album': {'type': 'string'}
                        }
                    }
                }
            }
        }
    })
    def get(self):
        """
        Search for tracks based on the provided track name
        ---
        parameters:
          - name: track_name
            in: query
            type: string
            required: true
            description: Name of the track to search for
        responses:
          200:
            description: A list of tracks matching the search criteria
        """
        track_name = request.args.get('track_name')
        if not track_name:
            return {'error': 'track_name is required'}, 400

        filtered_data = data[data['track_name'].str.contains(track_name, case=False, na=False)].head(5)
        if filtered_data.empty:
            return {'error': 'No tracks found'}, 404

        response = filtered_data[non_numeric_columns].to_dict(orient='records')
        return jsonify(response)

api.add_resource(RecommendTrack, '/recommend_track')
api.add_resource(SearchTrack, '/search_track')

# Configuración del bot de Telegram
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('Hola! Usa /search <nombre de la canción> para buscar una canción o /recommend <nombre de la canción> para obtener recomendaciones.')

async def search(update: Update, context: CallbackContext):
    if len(context.args) == 0:
        await update.message.reply_text('Por favor proporciona el nombre de la canción después de /search')
        return

    track_name = ' '.join(context.args)
    response = requests.get(f'http://localhost:5000/search_track?track_name={track_name}').json()
    if 'error' in response:
        await update.message.reply_text(response['error'])
    else:
        message = ''
        for track in response:
            message += f"Artista: {track['artist']}\nÁlbum: {track['album']}\nCanción: {track['track_name']}\n\n"
        await update.message.reply_text(message)

async def recommend(update: Update, context: CallbackContext):
    if len(context.args) == 0:
        await update.message.reply_text('Por favor proporciona el nombre de la canción después de /recommend')
        return

    track_name = ' '.join(context.args)
    response = requests.get(f'http://localhost:5000/recommend_track?track_name={track_name}').json()
    if 'error' in response:
        await update.message.reply_text(response['error'])
    else:
        message = ''
        for track in response:
            message += f"Artista: {track['artist']}\nÁlbum: {track['album']}\nCanción: {track['track_name']}\n\n"
        await update.message.reply_text(message)

def run_telegram_bot():
    app_telegram = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app_telegram.add_handler(CommandHandler("start", start))
    app_telegram.add_handler(CommandHandler("search", search))
    app_telegram.add_handler(CommandHandler("recommend", recommend))

    asyncio.set_event_loop(asyncio.new_event_loop())
    app_telegram.run_polling()

if __name__ == '__main__':
    import threading

    def run_flask():
        app.run(debug=True, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    telegram_thread = threading.Thread(target=run_telegram_bot)
    telegram_thread.start()
