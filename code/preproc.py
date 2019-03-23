import pandas as pd
import numpy as np
import string
import re

def remove_non_ascii_1(text):
    return ''.join([i if ((ord(i) < 128) | (ord(i) == 241)) else '' for i in text])

def test_re(s):  
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', s)



#df = pd.read_csv("/home/grosati/Documents/ScrapTango/Repo/Data/letras_final.csv")

df = pd.read_csv("E:/PEN/Datasets_ML/ScrapTango/tango_scrap/Data/data_merged_raw", sep=";")

letras = df.letra_cons
letras.dropna(axis=0, inplace=True)
letras = letras.values

#df = df[(df['compositor'] == 'enrique cadicamo') | 
#    (df['compositor'] == 'homero manzi') | 
#    (df['compositor'] == 'celedonio flores')]
    

letras = [i.replace('|', '\n') for i in letras]
letras = [i.replace('à', 'a') for i in letras]
letras = [i.replace('ã', 'a') for i in letras]
letras = [i.replace('ä', 'a') for i in letras]
letras = [i.replace('è', 'e') for i in letras]
letras = [i.replace('ê', 'e') for i in letras]
letras = [i.replace('ë', 'e') for i in letras]
letras = [i.replace('ì', 'i') for i in letras]
letras = [i.replace('ò', 'o') for i in letras]
letras = [i.replace('ô', 'o') for i in letras]
letras = [i.replace('õ', 'o') for i in letras]
letras = [i.replace('ö', 'o') for i in letras]
letras = [i.replace('ù', 'u') for i in letras]
letras = [i.replace('ü', 'u') for i in letras]

#def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

letras = [test_re(i) for i in letras]
letras = [remove_non_ascii_1(i) for i in letras]


l = open('V2letras.txt', 'w')

for item in letras:
  l.write("%s\n\n" % item)

l.close()
