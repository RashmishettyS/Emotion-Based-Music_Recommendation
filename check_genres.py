import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = 'your_spotify_client_id'
client_secret = 'your_spotify_client_secret'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Fetch and print available genre seeds
available_genres = sp.recommendation_genre_seeds()
print("Available Genre Seeds:", available_genres)
