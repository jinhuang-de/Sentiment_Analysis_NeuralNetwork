#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame as df
import spacy
import os
import re
import sys
from subprocess import call
from string import punctuation
from tqdm import tqdm
import html


# In[27]:


# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# exceptions = ['against', 'up', 'down', 'ain', 'aren', 'out', 'off', 'over', 'no', 'nor', 'not', 'too', 't', "don't", "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'on', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'very', 's', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']


# In[2]:


#corpus = pd.read_csv("./data/sentiment140_original.csv", encoding = "ISO-8859-1")
#data = corpus.iloc [:, [0,5]]
#data.columns = ['label', 'text']


# In[8]:


class Data():
    def __init__(self, df: str, model: str = 'en_core_web_sm'):
        # catch error if spaCy model is not available, installs it and restarts the script
        corpus = pd.read_csv(df, encoding="ISO-8859-1")
        self.df = corpus.iloc [:, [0,5]].sample(frac=1) #shuffle
        self.df.columns = ['label', 'text']
        self.fp = f'./data/{df.split("/")[-1].replace("data", "processed")}'
            
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f'spaCy model:\t{model} is not installed.\nInstalling now...')
            call(['python3', '-m', 'spacy', 'download', model])  # for terminal
            print('Restarting script...')
            os.execl(sys.executable, sys.executable, sys.argv[0])

        spacy.info()
        print(f'Dataframe being processed:\t{df.split("/")[-1]}\nExcerpt:\n{self.df.head()}\n')
    
    def remove_urls(self, text: str) -> str:
        """Removes urls from string
        
        Arguments:
            str: text
        
        Returns:
            str -- text with urls removed
        """
        url_pattern = re.compile('(\w+:\/\/\S+)|(www\.[^\s]*[\s]*)')
        text = re.sub(url_pattern, ' ', text)
        return text
    
    def remove_hashtags_and_mentions(self, text: str) -> str:
        """Removes hashtags and mentions from text
        
        Arguments:
            str: text
        
        Returns:
            str -- text with hashtags and mentions removed
        """
        hashtag_pattern = re.compile('[#%&§$-+]+[A-Za-z0-9]*\s*', flags=re.IGNORECASE)
        mention_pattern = re.compile('@[A-Za-z0-9_\-#%&§$\+]*\s*', flags=re.IGNORECASE)
        text = re.sub(hashtag_pattern, ' ', text)
        text = re.sub(mention_pattern, ' ', text)
        return text
    
    def remove_nonascii(self, text: str) -> str:
        """Removes all non-ASCII characters from text
        
        Arguments:
            str: text
        
        Returns:
            str -- text with emojis removed

        """
        return text.encode('ascii', 'ignore').decode('ascii')
    
    
    def remove_punctuation_and_whitespaces(self, text: str) -> str:
        """Removes punctuation and whitespaces from a text
        
        Arguments:
            str -- text
        
        Returns:
            str -- text with all punctuation and multiple whitespaces removed
        """
        whitespaces_pattern = re.compile(r'\s+')
        punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
        
        for punct in punctuation:
            text = text.replace(punct, ' ')

        return re.sub(whitespaces_pattern, ' ', text).strip()
    
    
    def remove_digits(self, text: str) -> str:
        """Removes numbers from a tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with all numbers removed
        """

        return re.sub("\d+", " ", text)
    
    def escape_html(self, text: str) -> str:
        
        s = html.escape( """& < " ' >""" ).split()   # s = ["&amp;", "&lt;", "&quot;", "&#x27;", "&gt;"]
        
        for e in s:
            text = text.replace(e, ' ').strip()
        return text


    def remove_stopwords(self, text: str) -> str:
        """Removes stopwords from a text    
        Arguments:
            text {str} -- tweet-text
        
        Returns:
            str -- txt with all stopwords removed
        """
    
        text_list = text.split(" ")

        for word in text.split(" "):
            for sw in stop_words:
                if word == sw:
                    text_list.remove(word)
        text = " ".join(text_list)
        
        return text
    

    def lemmatize(self, text: str) -> str:
        """Lemmatizes tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- space seperated lemma
        """
        return ' '.join([
            token.lemma_ for token in self.nlp(text)
            if token.lemma_ != '-PRON-'
        ])
    
    
    def preprocess(self,
                   lowercase: bool = True,
                   sentence_length: int = 4):
        """Complete preprocessing pipeline of underlying dataframe
        
        Keyword Arguments:
            lowercase {bool} -- lowercasing text (default: {True})
            sentence_length {int} -- cutoff threshold (default: {4})
        """

        tqdm.pandas(desc='Processing data', ncols=100)
        
        if lowercase:
            self.df['text'] = self.df['text'].str.lower()
        
        self.df['text'] = self.df['text'].astype(str)
        self.df['text'] = self.df['text'].progress_apply(self.remove_nonascii)
        self.df['text'] = self.df['text'].progress_apply(self.remove_urls)
        self.df['text'] = self.df['text'].progress_apply(self.remove_hashtags_and_mentions)
        self.df['text'] = self.df['text'].progress_apply(self.remove_digits)
        self.df['text'] = self.df['text'].progress_apply(self.remove_stopwords)
        self.df['text'] = self.df['text'].progress_apply(self.remove_punctuation_and_whitespaces)
        #self.df['text'] = self.df['text'].progress_apply(self.pos)
        self.df['text'] = self.df['text'].progress_apply(self.lemmatize)
        self.df['text'] = self.df['text'].progress_apply(self.escape_html)
        self.df['text'] = self.df['text'].progress_apply(self.remove_punctuation_and_whitespaces)
        
        #select sentences with more than 4 words
        self.df = self.df[self.df['text'].str.split().str.len().ge(4)]
        
    def main(self):
        """Preprocessing and saving processed .csv table
        """
        self.preprocess(lowercase=True)
        self.df.to_csv(self.fp, sep='\t', encoding='utf-8', index=False)
        #self.to_json()
        print(f'\nData frame written to {self.fp}')
       


# In[43]:


def split_data(file):
    df = pd.read_csv(file, sep='\t')
    train = df.sample(frac=0.7)
    
    test_and_valid = df.loc[~df.index.isin(train.index)]
      
    test = test_and_valid.sample(frac=0.5)
    valid = test_and_valid.loc[~test_and_valid.index.isin(test.index)]
       
    train.to_csv(r'./data/train.csv', sep='\t', encoding='utf-8', index=False)
    test.to_csv(r'./data/test.csv', sep='\t', encoding='utf-8', index=False)
    valid.to_csv(r'./data/valid.csv', sep='\t', encoding='utf-8', index=False)

    
    #print(len(train), "training data", len(test), "test data", len(valid), "valid data")
    #print(test)


# In[255]:


if __name__ == "__main__":
    preprocess = Data("./data/sentiment140_data.csv")
    preprocess.main()
    split_data('./data/sentiment140_preprocessed.csv')


# In[41]:


#text = "@tatiana_k nope they didn't have it 'wha' "
#hashtag_pattern = re.compile('[#%&§$-+]+[A-Za-z0-9]*\s*', flags=re.IGNORECASE)###
#mention_pattern = re.compile('@[A-Za-z0-9_\-#%&§$\+]*\s*', flags=re.IGNORECASE)
#mention_pattern = re.compile('@[A-Za-z0-9]*\s*', flags=re.IGNORECASE)
#url_pattern = '(\w+:\/\/\S+)|www\.[^\s]*[\s]*'
#punctuation_pattern =  re.compile('\'\.+\'', flags=re.IGNORECASE)
#text = re.sub(url_pattern, '', text)
#text = re.sub(hashtag_pattern, '', text)
#text = re.sub(mention_pattern, '', text)
#negative_pattern = re.compile('n\'t', flags=re.IGNORECASE)
#text = re.sub(punctuation_pattern, '', text)
#text = re.sub(negative_pattern, ' not', text)
#idx = re.match(punctuation_pattern, text).pos
#print(text)
#|(\'\s+\')

#punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
        
#for punct in punctuation:
#    text = text.replace(punct, ' ')
#text

