
import pandas as pd
import numpy as np
from openai import OpenAI


import re
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from itertools import permutations, combinations
from datetime import datetime
import json


import networkx as nx


import os
import random
import base64
import requests
from zhipuai import ZhipuAI
from tqdm import tqdm

default_model = 'glm-4v'

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

def gpt_request(user_msg, client, base64_image, model=default_model, temperature=0.2):
    if not user_msg:
        return None
    try:
        response = client.chat.completions.create(
        model=model,  
        messages= user_msg
    )
        print(response.choices[0].message.content)
        return response
    except Exception as e:
        print(e)
        return None  
    
def get_image(root_directory):
  file_paths = []
  for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
      file_path = os.path.join(dirpath, filename)
      file_paths.append(file_path)

  return file_paths

def get_number(code):
   tmp = []
   for s in code:
      tmp.append(int(s))

   if len(tmp) < 10:
      tmp.extend(['t']*(10-len(tmp)))

   return tmp


      
      

         

dir = ''
file_paths = get_image(dir)

api_key = ""
client = ZhipuAI(api_key = api_key)

system_msg = 'You are a helpful assistant to recognize and score the attributes in the facial images that I provide.\
    Please note that your response can only be selected from the content in parentheses, \
      and you don\'t need any extra words between the different fill-in-the-blank answers,\
          just a space to separate them.'

user_msg = [{"role": "system", "content": system_msg},
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "In this face picture, you think \
                the youth is _(A score of 0 indicates the oldest, 5 indicates the youngest, and the degree of age gradually deepens from 5 to 0), \
                the gender is _ (A score of 0 indicates that female characteristics are the most prominent, and 5 indicates that male characteristics are the most prominent, and from 0 to 5 female characteristics gradually weaken Male characteristics gradually increase), \
                the hairline is _ (A score of 0 indicates the lowest hairline, 5 indicates the highest hairline, and the hairline gradually increases from 0 to 5), \
                the makeup is _ (A score of 0 indicates the lightest makeup, 5 indicates the thickest makeup, and from 0 to 5 the makeup looks thicker and thicker), \
                the facial obesity is _ (A score of 0 indicates that it looks the thinnest, and 5 indicates that it looks the fattest with makeup, and the degree of obesity gradually deepens from 0 to 5), \
                and the eye bags are _ (A score of 0 indicates that the eye bags are the lightest, 5 means the deepest eye bags, and from 0 to 5, the eye bags change from shallow to dark).\
                The visibility of the mustache is _ (0 for no mustache, 5 for very obvious mustache, and the visibility of the mustache gradually deepens from 1 to 5), \
                The degree of beard visibility as _ (5 for no beard, 0 for very obvious beard, and gradually decreasing from 1 to 5 for beard visibility), \
                and the visibility of the hat as _ (0 for no hat, 5 for very obvious hat, and the visibility of the hat gradually increases from 1 to 5). \
                The degree of baldness is obvious _ (0 for no baldness, 5 for complete baldness, and the degree of visualization from 0 to 5 for baldness progressively deepens), \
                The degree of gray hair is _ (0 means that the hair does not appear grayish, 5 means that the grayish white of the hair is obvious, and the grayish white degree gradually deepens from 0 to 5), \
                The degree of blonde hair is _ (0 means that the hair is not showing blonde, 5 means that the blonde of the hair is obvious, and the blonde degree is gradually deepened from 0 to 5)\
                The degree of lipstick visibility is _ (0 means no lipstick, 5 means the lipstick is very obvious, and the degree of lipstick visibility increases from 0 to 5)\
                The degree of smile as _: (0 for no smile, 5 for smile very obviously, and the smile is gradually deepened from 1 to 5);\
                The degree of sideburns visibility is _ (0 means no sideburns, 5 means the sideburns is very obvious, and the degree of sideburns visibility increases from 0 to 5)\
                The degree of bangs as _ (0 for no bangs, 5 for the obvious bangs, and the degree of distinctiveness of bangs gradually deepens from 1 to 5)\
                The concentration of the eyebrows is _ (0 means the eyebrows are very light, 5 means the eyebrows are very thick, and the concentration of the eyebrows increases from 0-5)\
                  Please fill in the blank with only a single number for the aforementioned 25 facial attributes. Note each attribute should be considered independently. Furthermore, do not answer with a serial number at the beginning of each line, just a space separating them."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": ""
          }
        }
      ]
    }
  ]

ans = []
# ans.append()
flag = True
file = open('/root/compare/response/glmv1.txt', 'a', encoding='utf-8')

for i, path in tqdm(enumerate(file_paths[10:1000])):
   base64_image = encode_image(path)
   if flag:
    user_msg[1]["content"][1]["image_url"]["url"] = base64_image
   else:
    user_msg[0]["content"][1]["image_url"]["url"] = base64_image
   
   response = gpt_request(user_msg, client, base64_image)
   if isinstance(response,Exception) :
     file.write('\n')
     continue
   try:
    text = response.choices[0].message.content
   except:
      file.write('\n')
      continue
   cleaned_text = re.sub('[^\d]', ' ', text)
   cleaned_texts = cleaned_text.split()
   perans = get_number(cleaned_texts)
   
   perans.insert(0,path)
   file.write(str(perans) + '\n')
   print(perans)
   ans.append(perans)
   if flag:
      del user_msg[0]
      flag = False
   

file.close()