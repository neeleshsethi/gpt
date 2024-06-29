import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------


torch.manual_seed(1337)

#read the text file 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#get unique chars in text
chars = list(set(text))
vocab_size = len(chars)

#create mapping of chars into int
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch  for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c]  for c in s ]
decode = lambda l : ''.join([ itos[i] for i in l])

#train and test split

encoded_txt = encode(text) # enconde text
data = torch.tensor(encoded_txt, dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
   data = train_data if split == 'train' else val_data
   ix = torch.randint(0, len(data)-block_size, (batch_size,))
   listofxtensors = [data[i:i+block_size] for i in ix]
   x = torch.stack(listofxtensors)
   listofytensors = [ data[i+1:i+block_size+1] for i in ix]
   y = torch.stack(listofytensors)
   x = x.to(device)
   y = y.to(device)
   return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        tril  = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril',tril)
        self.dropout = nn.Dropout(dropout)
      

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5       #(B, T, 16) * (B * 16 * T) -> B, T, T
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei , dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)       # B,T, 16 
        out = wei @ v        # B,t,t   B,t,16  -> b, t,16
        return out



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        head_list =  [Head(head_size) for _ in range(num_heads)]
        self.heads = nn.ModuleList(head_list)
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_itr = [h(x) for h in self.heads]
        out = torch.cat(head_itr, dim =-1 )
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,n_emb,n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa  = MultiHeadAttention(head_size,n_head)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x




class BiagramLanguageModel(nn.Module):
    def __init__(self):
         super().__init__()
         self.token_embedding_table = torch.nn.Embedding(vocab_size, n_emb)
         no_of_blocks =  [ Block(n_emb, n_head=n_head) for _ in range(n_layer)]
         self.blocks = nn.Sequential(*no_of_blocks)
         self.ln_f = nn.LayerNorm(n_emb)
         self.lm_head = nn.Linear(n_emb, vocab_size)
         self.position_embedding_table = torch.nn.Embedding(block_size, n_emb)
        


    def forward(self, idx, targets=None ):
         B, T = idx.shape
     
         tok_emb = self.token_embedding_table(idx)  # Batch, Token , token_emb
         pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
         self.sa_head = Head(n_emb)
         x = tok_emb + pos_emb # B, T, C
         x = self.blocks(x) # B, T, C

         logits = self.lm_head(x)    #  (B, T, vocab_size)
        
         if targets is None:
             loss = None 
         else:
             B, T,C  = logits.shape
             logits = logits.view(B * T, C)
             targets = targets.view(B*T)
             loss  = F.cross_entropy(logits, targets)
         return logits, loss  

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # B * T * C
            logits = logits[:, -1, :]#   B * C
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1) #B * 1
            idx = torch.cat((idx, idx_next), dim =1 )
        return idx
    
model = BiagramLanguageModel()
m = model.to(device=device)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter} train loss {losses['train']}, val losses {losses['val']} ")

    xb, yb = get_batch('train')

    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
    




context = torch.zeros((1,1), dtype=torch.long, device=device)
out_tokenzied = m.generate(context, max_new_tokens=200)
out_tokenzied = out_tokenzied[0].tolist()
decoded_out = decode(out_tokenzied)
print(decoded_out)






