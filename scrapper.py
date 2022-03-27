# Credit: https://zhuanlan.zhihu.com/p/32715324

import requests
from bs4 import BeautifulSoup
import json
import re

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
    links = soup.find('ul', class_='f_hide').find_all('a')
    song_IDs = []
    song_names = []
    for link in links:
        ID = link.get('href').split('=')[-1]
        name = link.get_text()
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
    print('### Writing song: {}'.format(song_name))
    with open('{}.txt'.format(song_name), 'a', encoding='utf-8') as f:
        f.write(lyric)

if __name__ == '__main__':
    singer_id = input('Input singer ID: (TS: 44266)')
    url = 'http://music.163.com/artist?id={}'.format(singer_id)
    print
    html = get_html(url)
    singer_infos = get_singer_info(html)
    for info in singer_infos:
        lyric = get_lyric(info[1])
        write_lyric(info[0], lyric)
    
        