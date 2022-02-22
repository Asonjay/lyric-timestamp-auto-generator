#!coding:utf-8
import requests
import json
import re
import sys

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE',
    'Refer':'http://music.163.com',
    'Host':'music.163.com'
}

url = 'http://music.163.com/api/song/lyric?id='+str(1921503119)+'&lv=1&kv=1&tv=-1'
res = requests.get(url,headers=headers)
lyric = res.text

json_obj = json.loads(lyric)

lyric = json_obj['lrc']['lyric']

#sys.stdout = open("out.txt", "w")
#print(lyric)
#sys.stdout.close()
#lyric = re.sub(r'[\d:.[\]]','', lyric)

#

#encoding issue
with open('out.txt', 'w', encoding='utf-8') as f:
    print(lyric, file=f) 


