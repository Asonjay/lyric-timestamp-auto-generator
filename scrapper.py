# Credit: https://zhuanlan.zhihu.com/p/32715324

from email import header
import requests
from bs4 import BeautifulSoup
import json
import re
import urllib


def get_html(url):
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE',
        'Refer':'http://music.163.com',
        'Host':'music.163.com'
    }
    try:
        response = requests.get(url, headers=headers)
        html = response.text
        return html
    except:
        print('request error')
        pass

    
def get_singer_info(html):
    soup = BeautifulSoup(html, 'lxml')
    links = soup.find('ul', class_='f-hide').find_all('a')
    song_IDs = []
    song_names = []
    for link in links:
        ID = link.get('href').split('=')[-1]
        name = re.sub(r'[^a-zA-Z0-9@$&:()]', ' ', link.get_text())
        song_IDs.append(ID)
        song_names.append(name)
    return zip(song_names, song_IDs)


def get_lyric(song_id):
    url = 'http://music.163.com/api/song/lyric?id=' + str(song_id) + '&lv=1&kv=1&tv=-1'
    html = get_html(url)
    json_obj = json.loads(html)
    lyric = json_obj['lrc']['lyric']
    #regex = re.compile(r'\[.*\]')
    #unsync_lyric = re.sub(regex, '', lyric).strip()
    return lyric


def write_lyric(song_name, lyric):
    print('### Writing lyric: {}'.format(song_name))
    with open('lyrics_raw\\{}.txt'.format(song_name), 'a', encoding='utf-8') as f:
        f.write(lyric)

def download_song(song_id, song_name):
    print('### Downloading song: {}'.format(song_name))
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE',
        'Refer':'http://music.163.com',
        'Host':'music.163.com'
    }
    url = 'http://music.163.com/song/media/outer/url?id=' + str(song_id) + '.mp3'
    location = requests.head(url, headers=headers, allow_redirects=False).headers['location']
    print(location)
    
    
if __name__ == '__main__':
    #singer_id = input('Input singer ID: (TS: 44266)')
    # Johnny Cash: 35347
    url = 'http://music.163.com/artist?id={}'.format(35347)
    html = get_html(url)
    singer_infos = get_singer_info(html)
    # Get lyrics
    for info in singer_infos:
        lyric = get_lyric(info[1])
        write_lyric(info[0], lyric)
    # Download songs
    # for info in singer_infos:
    #     download_song(info[1], info[0])
        