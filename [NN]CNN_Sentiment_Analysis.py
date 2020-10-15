#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import numpy as np
import spacy
import random
import time


# In[135]:


SEED = 666

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# binary class: dtype = torch.float
TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.Field(unk_token=None,use_vocab=False, sequential=False, dtype = torch.float)

fields = [('label', LABEL), ('text', TEXT)] 


# In[137]:


# Using torchtext with custom courpus 'Sentiment140'
train_data, validation_data, test_data = data.TabularDataset.splits( path = 'data',
                                                                train = 'train.csv',
                                                                validation = 'valid.csv',
                                                                test = 'test.csv',
                                                                format = 'tsv',
                                                                fields = fields,
                                                                skip_header = True
                                                                )


# In[ ]:


# EMBEDDING 
# Load the pre-trained word embeddings
MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)


# In[ ]:


# ITERATORS
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, validation_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device,
    sort = False)


# In[ ]:


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # CONVOLUTIONAL LAYERS
        # out_channels: the number of filters
        # kernel_size: the size of the filters = [n x emb_dim]
        self.convolutional_layers = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (filter_size, embedding_dim)) 
                                    for filter_size in filter_sizes
                                    ])
        
        self.output = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, text):
         
        #text = [batch size, sent len]
        
        # EMBEDDING
        x = self.embedding(text) # => [batch size, sentence length, emb dim]
        
        x = x.unsqueeze(1)# => [batch size, 1, sentense length, emb dim]
        
        # CONVOLUTION
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convolutional_layers] # => [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        # MAX POOLING
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x] # => [batch size, n_filters]
        
        #CONCATENATE
        x = self.dropout(torch.cat(x, dim = 1)) # => [batch size, n_filters * len(filter_sizes)]
            
        return self.output(x)


# In[ ]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100 # EMBEDDING_DIM must be equal to that of the pre-trained GloVe vectors loaded earlier
N_FILTERS = 100
FILTER_SIZES = [2, 3, 4]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM,
            EMBEDDING_DIM,
            N_FILTERS,
            FILTER_SIZES,
            OUTPUT_DIM,
            DROPOUT,
            PAD_IDX)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors


# In[ ]:


model.embedding.weight.data.copy_(pretrained_embeddings)


# In[ ]:


# row
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# let initial weights of our unknown and padding tokens remains zero
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[ ]:


model.embedding.weight.data.copy_(pretrained_embeddings)


# In[56]:


# TRAIN THE MODEL
# the algorithm to update the parameters of the module
optimizer = optim.Adam(model.parameters())


# In[57]:


# LOSS
criterion = nn.BCEWithLogitsLoss() 

model = model.to(device)
criterion = criterion.to(device)


# In[42]:


def binary_accuracy(preds, y):
    """
    Calculate how many predictions match the gold labels and average it across the batch.
    
    Return: The accuracy per batch
    """
    
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# In[47]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()  # 'grad': gradient calculated by the criterion
        
        # batch.text: batch of sentences
        predictions = model(batch.text).squeeze(1)  
        
        #batch.label, with the loss being averaged over all examples in the batch
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        # update the parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    # len(iterator): the number of batches in the iterator
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[53]:


# do not update the parameters when evaluating
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()  # turns off dropout and batch normalization
    
    with torch.no_grad(): # No gradients are calculated on PyTorch operations
    
        for a_batch in iterator:

            predictions = model(a_batch.text).squeeze(1)
            
            loss = criterion(predictions, a_batch.label)
            
            acc = binary_accuracy(predictions, a_batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[54]:


def epoch_time(start, end):
    """
    Training time for each epoch.
    """
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


# TRAIN
N_EPOCHS = 50

best_validation_loss = float('inf')

for epoch in range(N_EPOCHS):

    start = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    validation_loss, validation_acc = evaluate(model, validation_iterator, criterion)
    
    end = time.time()

    epoch_mins, epoch_secs = epoch_time(start, end)
    
    # At each epoch, if the validation loss is the best so far,
    # it save the parameters of the model and then after training has 
    # finished we'll use that model on the test set.
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), 'CNN-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_acc*100:.2f}%')


# In[ ]:


model.load_state_dict(torch.load('CNN-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# In[ ]:


nlp = spacy.load('en')

def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))###

    return prediction.item()

print(predict_sentiment(model, "so disappointed"))
#0

print(predict_sentiment(model, "so excited but now so disappointed"))
# 0

print(predict_sentiment(model, "just find out twin will not even write back heartbroken"))
# 0

print(predict_sentiment(model, "phone break now using stupid nokia phone ughh miss advance phone"))
# 0

print(predict_sentiment(model, "just sad"))
# 0

print(predict_sentiment(model, "tragedy disaster new week"))
# 0

print(predict_sentiment(model, "enjoy beautiful morning here phoenix too bad out yet"))
# 4

print(predict_sentiment(model, "wake up feeling rested refreshed today about time"))
# 4

print(predict_sentiment(model, "get home hour ago eat lunch watch tv now listening kelly clarkson no exam tomorrow yay"))
# 4

print(predict_sentiment(model, "oooo haha just wake up ready eat delicious breakfast prepare go afternoon watch movie"))
# 4

print(predict_sentiment(model, "just wake up have no school best feeling ever"))
# 4

print(predict_sentiment(model, "glad ur doing well"))
# 4

