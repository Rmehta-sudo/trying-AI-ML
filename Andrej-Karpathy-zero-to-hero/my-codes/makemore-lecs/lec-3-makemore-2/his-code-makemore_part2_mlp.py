#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read in all the words
words = open('names.txt', 'r').read().splitlines()
words[:8]


# In[ ]:


len(words)


# In[ ]:


# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)


# In[ ]:


# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words:
  
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
  
X = torch.tensor(X)
Y = torch.tensor(Y)


# In[ ]:


X.shape, X.dtype, Y.shape, Y.dtype


# In[769]:


# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# In[ ]:


C = torch.randn((27, 2))


# In[ ]:


emb = C[X]
emb.shape


# In[ ]:


W1 = torch.randn((6, 100))
b1 = torch.randn(100)


# In[ ]:


h = torch.tanh(emb.view(-1, 6) @ W1 + b1)


# In[ ]:


h


# In[ ]:


h.shape


# In[ ]:


W2 = torch.randn((100, 27))
b2 = torch.randn(27)


# In[ ]:


logits = h @ W2 + b2


# In[ ]:


logits.shape


# In[ ]:


counts = logits.exp()


# In[ ]:


prob = counts / counts.sum(1, keepdims=True)


# In[ ]:


prob.shape


# In[ ]:


loss = -prob[torch.arange(32), Y].log().mean()
loss


# In[ ]:


# ------------ now made respectable :) ---------------


# In[780]:


Xtr.shape, Ytr.shape # dataset


# In[790]:


g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]


# In[791]:


sum(p.nelement() for p in parameters) # number of parameters in total


# In[792]:


for p in parameters:
  p.requires_grad = True


# In[793]:


lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre


# In[794]:


lri = []
lossi = []
stepi = []


# In[795]:


for i in range(200000):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))
  
  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

#print(loss.item())


# In[796]:


plt.plot(stepi, lossi)


# In[797]:


emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
loss


# In[798]:


emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
loss


# In[710]:


# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')


# In[ ]:


# training split, dev/validation split, test split
# 80%, 10%, 10%


# In[805]:


context = [0] * block_size
C[torch.tensor([context])].shape


# In[820]:


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))


# In[ ]:




