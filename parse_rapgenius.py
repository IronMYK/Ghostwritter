
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import argparse
import requests
import os

artist_url = "http://genius.com/artists/Alpha-wann"

def scrap_song_lyrics(artist_url):
    artist_name = artist_url.split("/")[-1]
    folder_path = os.path.join('data', artist_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    response = requests.get(artist_url)
    soup = BeautifulSoup(response.text, "html.parser")
    album_links = [a['href'] for a in soup.find_all('a', {'class':'vertical_album_card'})]
    song_links = []

    for album_link in album_links:
        response = requests.get(album_link)
        soup = BeautifulSoup(response.text, "html.parser")
        song_links.extend([a['href'] for a in soup.find_all('a', {'class': 'u-display_block'})])


    for song_link in song_links:
        response = requests.get(song_link)
        song_name = song_link.split('/')[-1]
        data_path = os.path.join('data', artist_name, song_name)
        print(song_name)
        if not os.path.exists(data_path):
            soup = BeautifulSoup(response.text, "html.parser")
            lyrics = soup.find('div', {'class':'lyrics'})
            if lyrics:
                with open(data_path, 'w', encoding='utf-8') as f:
                    f.write(lyrics.getText())
                    print('file_written')
            else:
                print("Error")
        else:
            print('Already scrapped')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist_url", type=str, help='Genius artist page url')
    args = parser.parse_args()
    scrap_song_lyrics(args.artist_url)
