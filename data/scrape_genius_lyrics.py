#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import os
import csv
from dotenv import load_dotenv
import time
load_dotenv()

GENIUS_API_TOKEN = os.environ.get('genius_token')

base_url = "https://api.genius.com"
headers = {'Authorization': 'Bearer '+ GENIUS_API_TOKEN}

def lyrics_from_song_api_path(song_api_path):
  time.sleep(1)
  song_url = base_url + song_api_path
  response = requests.get(song_url, headers=headers)
  json = response.json()
  path = json["response"]["song"]["path"]
  #gotta go regular html scraping... come on Genius
  page_url = "https://genius.com" + path
  page = requests.get(page_url)
  html = BeautifulSoup(page.text, "html.parser")
  #remove script tags that they put in the middle of the lyrics
  [h.extract() for h in html('script')]
  #at least Genius is nice and has a tag called 'lyrics'!
  lyrics = html.find("div", class_="lyrics").get_text() #updated css where the lyrics are based in HTML
  return lyrics

def pull_songs_for_artist(artist, genre):
  """Pulls songs for an artist from genius, and places the result in
     a directory named after that artist.

  Arguments:
    artist {string} -- Artist name, must represent the artist's name on genius
    genre {string} -- String representing genre (e.g pop/rap)
  """
  has_results = True
  page_count = 1
  while has_results and page_count < 5:
    has_results = False
    search_url = base_url + "/search"
    data = {'q': artist, 'page': page_count}
    page_count +=1
    response = requests.get(search_url, data=data, headers=headers)
    json = response.json()
    song_info = None
    for hit in json["response"]["hits"]:
      has_results = True
      if hit["result"]["primary_artist"]["name"] == artist:
        song_info = hit
        title = song_info['result']['title']
        song_api_path = song_info['result']['api_path']
        filename = title.replace(" ", "").replace("/","").lower()
        artist_folder = artist.replace(" ", "").replace(",","").lower()
        target_dir = "structure/" + genre + "/" + artist_folder + "/"
        if not os.path.exists(target_dir):
          os.makedirs(target_dir)
        with open(target_dir + filename, 'w') as f:
          f.write(lyrics_from_song_api_path(song_api_path))
    page_count += 1
  print("COMPLETED PULL FOR " + str(artist))

def pull_artists_from_file(filename):
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
      artist = row[0]
      genre = row[1]
      pull_songs_for_artist(artist, genre)

if __name__ == "__main__":
  pull_artists_from_file('artists.csv')
