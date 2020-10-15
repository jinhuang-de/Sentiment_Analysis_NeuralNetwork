#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
import numpy as np
import spacy
import random
import time


# In[68]:


SEED = 666
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# include_lengths = True: tell the RNN how long the actual sequences are, for the
#                         using of packed padded sequences
TEXT = data.Field(tokenize = 'spacy', include_lengths = True) 

# if binary class: dtype = torch.float
LABEL = data.Field(unk_token=None,use_vocab=False, sequential=False, dtype = torch.float)

fields = [('label', LABEL), ('text', TEXT)]


# In[ ]:


# Using torchtext with custom dataset(sentiment 140)
train_data, validation_data, test_data = data.TabularDataset.splits( path = 'data',
                                                                train = 'train.csv',
                                                                validation = 'valid.csv',
                                                                test = 'test.csv',
                                                                format = 'tsv',
                                                                fields = fields,
                                                                skip_header = True
                                                                )


# In[ ]:


# PRE-TRAINED EMBEDDING
max_vocab_size = 25000

TEXT.build_vocab(train_data, 
                 max_size = max_vocab_size,
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)


# In[ ]:


# ITERATOR
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, validation_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)
# for packed padded sequences all of the tensors within a batch need to be sorted by their lengths


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        # padding_idx = pad_idx: the embedding for the pad token will remain at zero(initialized)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # Returns:'output', ('final hidden state', 'final cell state')
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers,  # adding additional layers
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # LSTM: dropout on the connections between hidden states in one layer to hidden states in the next layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        text_lengths: passing the 'lengths of our sentences' for using packed padded sequences
        """
        
        #text = [sent len, batch size]
        
        #EMBEDDING
        x =self.embedding(text)
        
        #DROP OUT
        x = self.dropout(x) # =>[sent len, batch size, emb dim]
        
        #PACK SEQUENCE
        # To only process the non-padded elements of the sentence
        # Transform it from a sentence to a tensor.
        
        x = nn.utils.rnn.pack_padded_sequence(x, text_lengths)

        x, (h, cell) = self.lstm(x)
        
        # unpack sequence (not nessesary here)
        # output over padding tokens are zero tensors
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(x) # => [sent len, batch size, hid dim * num directions]        
        
        # DROPOUT
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        h = self.dropout(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)) # => [batch size, hid dim * num directions]

        return self.fc(h)


# In[ ]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100 # EMBEDDING_DIM = dimension of the pre-trained vectors
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = LSTM(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors


# In[ ]:


model.embedding.weight.data.copy_(pretrained_embeddings)


# In[ ]:


# row
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# make the initial weights of our unknown and padding tokens remain zero
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[56]:


# TRAIN THE MODEL
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


# iterates over all examples, one batch at a time
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad() 
        
        # 'include_lengths = True', the batch.text is now a tuple
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1) #[text, text_lengths, 1] => [text, text_lengths]
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step() # update the parameters
        
        # across the epoch
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    # len(iterator): how many batches in the iterator
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[53]:


# don't update the parameters when evaluating
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()  # turns off dropout and batch normalization
    
    with torch.no_grad(): # No gradients are calculated
    
        for batch in iterator:
            
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

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
N_EPOCHS = 25

best_validation_loss = float('inf')

for epoch in range(N_EPOCHS):

    start = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    validation_loss, validation_acc = evaluate(model, validation_iterator, criterion)
    
    end = time.time()

    epoch_mins, epoch_secs = epoch_time(start, end)
    
    # At each epoch, if the validation loss is the best so far,
    # save the parameters of the model and then after training has 
    # finished we'll use that model on the test set.
    
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), 'BiLSTM-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_acc*100:.2f}%')


# In[ ]:


model.load_state_dict(torch.load('BiLSTM-model.pt'))

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
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

print(predict_sentiment(model, "so disappointed"))
#0

print(predict_sentiment(model, "so excited but now so disappointed."))
# 0

print(predict_sentiment(model, "just find out twin will not even write back heartbroken"))
# 0

print(predict_sentiment(model, "phone break now using stupid nokia phone ughh miss advance phone"))
# 0

print(predict_sentiment(model, "just sad"))
# 0

print(predict_sentiment(model, "tragedy disaster new week "))
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

