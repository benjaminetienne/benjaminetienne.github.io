---
slug: "A Complete Guide to Write your own Transformers"
date: 2024-02-24
categories:
  - ai 
---

An end-to-end implementation of a Pytorch Transformer, in which we will cover key concepts such as self-attention, encoders, decoders, and much more.

<!-- more -->

## Writing our own

When I decided to dig deeper into Transformer architectures, I often felt frustrated when reading or watching tutorials online as I felt they always missed something :

 - Official tutorials from Tensorflow or Pytorch used their own APIs, thus staying high-level and forcing me to have to go in their codebase to see what was under the hood. Very time-consuming and not always easy to read 1000s of lines of code.
 - Other tutorials with custom code I found (links at the end of the article) often oversimplified use cases and didn’t tackle concepts such as masking of variable-length sequence batch handling.

I therefore decided to write my own Transformer to make sure I understood the concepts and be able to use it with any dataset.

During this article, we will therefore follow a methodical approach in which we will implement a transformer layer by layer and block by block.

There are obviously a lot of different implementations as well as high-level APIs from Pytorch or Tensorflow already available off the shelf, with — I am sure — better performance than the model we will build.

> “Ok, but why not use the TF/Pytorch implementations then” ?

The purpose of this article is educational, and I have no pretention in beating Pytorch or Tensorflow implementations. I do believe that the theory and the code behind transformers is not straightforward, that is why I hope that going through this step-by-step tutorial will allow you to have a better grasp over these concepts and feel more comfortable when building your own code later.

Another reasons to build your own transformer from scratch is that it will allow you to fully understand how to use the above APIs. If we look at the Pytorch implementation of the `forward()` method of the Transformer class, you will see a lot of obscure keywords like :

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*nnPBQWTUmGmbpuMmW8nXOw.png" />
</p>

If you are already familiar with these keywords, then you can happily skip this article.

Otherwise, this article will walk you through each of these keywords with the underlying concepts.

## A very short introduction to Transformers
If you already heard about ChatGPT or Gemini, then you already met a transformer before. Actually, the “T” of ChatGPT stands for Transformer.

The architecture was first coined in 2017 by Google researchers in the “Attention is All you need” paper. It is quite revolutionary as previous models used to do sequence-to-sequence learning (machine translation, speech-to-text, etc…) relied on RNNs which were computationnally expensive in the sense they had to process sequences step by step, whereas Transformers only need to look once at the whole sequence, moving the time complexity from O(n) to O(1).

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Ml-AVbcrZoPJ0Ta5ARcAAw.png" />
</p>

Applications of transformers are quite large in the domain of NLP, and include language translation, question answering, document summarization, text generation, etc.

The overall architecture of a transformer is as below:

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:910/format:webp/1*8yA78jYVHbsCREC9obYFVQ.png" />
</p>

## Multi-head attention
The first block we will implement is actually the most important part of a Transformer, and is called the Multi-head Attention. Let’s see where it sits in the overall architecture

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:856/format:webp/1*aibMCFA6wQXegBExGocaGQ.png" />
</p>

Attention is a mechanism which is actually not specific to transformers, and which was already used in RNN sequence-to-sequence models.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*khArrNUzPx2Nqw2aSVmBWQ.png" />
</p>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*UUJL7jyDJMHNF4bCXvGJ3A.png" />
</p>

```python
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        """
        input_dim: Dimensionality of the input.
        num_heads: The number of attention heads to split the input into.
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num heads"
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Value part
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Key part
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Query part
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False) # the output layer
        
        
    def check_sdpa_inputs(self, x):
        assert x.size(1) == self.num_heads, f"Expected size of x to be ({-1, self.num_heads, -1, self.hidden_dim // self.num_heads}), got {x.size()}"
        assert x.size(3) == self.hidden_dim // self.num_heads
        
        
    def scaled_dot_product_attention(
            self, 
            query, 
            key, 
            value, 
            attention_mask=None, 
            key_padding_mask=None):
        """
        query : tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads)
        key : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        value : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
        
    
        """
        self.check_sdpa_inputs(query)
        self.check_sdpa_inputs(key)
        self.check_sdpa_inputs(value)
        
        
        d_k = query.size(-1)
        tgt_len, src_len = query.size(-2), key.size(-2)

        
        # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
        
        # Attention mask here
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.size() == (tgt_len, src_len)
                attention_mask = attention_mask.unsqueeze(0)
                logits = logits + attention_mask
            else:
                raise ValueError(f"Attention mask size {attention_mask.size()}")
        
                
        # Key mask here
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # Broadcast over batch size, num heads
            logits = logits + key_padding_mask
        
        
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, value) # (batch_size, num_heads, sequence_length, hidden_dim)
        
        return output, attention

    
    def split_into_heads(self, x, num_heads):
        batch_size, seq_length, hidden_dim = x.size()
        x = x.view(batch_size, seq_length, num_heads, hidden_dim // num_heads)
        
        return x.transpose(1, 2) # Final dim will be (batch_size, num_heads, seq_length, , hidden_dim // num_heads)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, head_hidden_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * head_hidden_dim)
        
    
    def forward(
            self, 
            q, 
            k, 
            v, 
            attention_mask=None, 
            key_padding_mask=None):
        """
        q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
        k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
       
        """
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, self.num_heads)
        k = self.split_into_heads(k, self.num_heads)
        v = self.split_into_heads(v, self.num_heads)
        
        # attn_values, attn_weights = self.multihead_attn(q, k, v, attn_mask=attention_mask)
        attn_values, attn_weights  = self.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        grouped = self.combine_heads(attn_values)
        output = self.Wo(grouped)
        
        self.attention_weigths = attn_weights
        
        return output
```

We need to explain a few concepts here.

### 1) Queries, Keys and Values.

> The query is the information you are trying to match, The key and values are the stored information.

Think of that as using a dictionary : whenever using a Python dictionary, if your query doesn’t match the dictionary keys, you won’t be returned anything. But what if we want our dictionary to return a blend of information which are quite close ? Like if we had :

```python
d = {"panther": 1, "bear": 10, "dog":3}
d["wolf"] = 0.2*d["panther"] + 0.7*d["dog"] + 0.1*d["bear"]
```

This is basically what attention is about : looking at different parts of your data, and blend them to obtain a synthesis as an answer to your query.

The relevant part of the code is this one, where we compute the attention weights between the query and the keys

```python
logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # we compute the weights of attention
```

And this one, where we apply the normalized weights to the values :

```python
attention = torch.softmax(logits, dim=-1)
output = torch.matmul(attention, value) # (batch_size, num_heads, sequence_length, hidden_dim)
```

### 2) Attention masking and padding

When attending to parts of a sequential input, we do not want to include useless or forbidden information.

Useless information is for example padding: padding symbols, used to align all sequences in a batch to the same sequence size, should be ignored by our model. We will come back to that in the last section

Forbidden information is a bit more complex. When being trained, a model learns to encode the input sequence, and align targets to the inputs. However, as the inference process involves looking at previously emitted tokens to predict the next one (think of text generation in ChatGPT), we need to apply the same rules during training.

This is why we apply a _causal mask_ to ensure that the targets, at each time step, can only see information from the past. Here is the corresponding section where the mask is applied (computing the mask is covered at the end)

```python
if attention_mask is not None:
    if attention_mask.dim() == 2:
        assert attention_mask.size() == (tgt_len, src_len)
        attention_mask = attention_mask.unsqueeze(0)
        logits = logits + attention_mask
```

#### Positional Encoding

It corresponds to the following part of the Transformer:

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:968/format:webp/1*CtvBBaPEMKMRK6insxd1xQ.png" />
</p>


When receiving and treating an input, a transformer has no sense of order as it looks at the sequence as a whole, in opposition to what RNNs do. We therefore need to add a hint of temporal order so that the transformer can learn dependencies.

The specific details of how positional encoding works is out of scope for this article, but feel free to read the original paper to understand.

```python
# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
```

#### Encoders

We are getting close to having a full encoder working ! The encoder is the left part of the Transformer.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1198/format:webp/1*bsLsL-HA2f8rcxbZK7zo-w.png" />
</p>


We will add a small part to our code, which is the Feed Forward part :

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

Putting the pieces together, we get an Encoder module !

```python
class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_padding_mask=None):
        assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))
        
        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)
       
        return output
```

As shown in the diagram, the Encoder actually contains N Encoder blocks or layers, as well as an Embedding layer for our inputs. Let’s therefore create an Encoder by adding the Embedding, the Positional Encoding and the Encoder blocks:

```python
class Encoder(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            n_dim: int, 
            dropout: float, 
            n_encoder_blocks: int,
            n_heads: int):
        
        super(Encoder, self).__init__()
        self.n_dim = n_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=n_dim
        )
        self.positional_encoding = PositionalEncoding(
            d_model=n_dim, 
            dropout=dropout
        )    
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)
        ])
        
        
    def forward(self, x, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x=x, src_padding_mask=padding_mask)
        return x

```

#### Decoders

The decoder part is the part on the left and requires a bit more crafting.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1198/format:webp/1*4LBcSxxvX5uSwslKzkZYXg.png" />
</p>

There is something called Masked Multi-Head Attention. Remember what we said before about causal mask ? Well this happens here. We will use the attention_mask parameter of our Multi-head attention module to represent this (more details about how we compute the mask at the end) :

```python
# Stuff before

self.self_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
masked_att_output = self.self_attention(
    q=tgt, 
    k=tgt, 
    v=tgt, 
    attention_mask=tgt_mask, <-- HERE IS THE CAUSAL MASK
    key_padding_mask=tgt_padding_mask)

# Stuff after
```

The second attention is called _cross-attention_. It will uses the decoder’s query to match with the encoder’s key & values ! Beware : they can have different lengths during training, so it is usually a good practice to define clearly the expected shapes of inputs as follows :

```python
def scaled_dot_product_attention(
            self, 
            query, 
            key, 
            value, 
            attention_mask=None, 
            key_padding_mask=None):
        """
        query : tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads)
        key : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        value : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
    
        """
```

And here is the part where we use the encoder’s output, called memory, with our decoder input :

```python
# Stuff before
self.cross_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
cross_att_output = self.cross_attention(
        q=x1, 
        k=memory, 
        v=memory, 
        attention_mask=None,  <-- NO CAUSAL MASK HERE
        key_padding_mask=memory_padding_mask) <-- WE NEED TO USE THE PADDING OF THE SOURCE
# Stuff after
```

Putting the pieces together, we end up with this for the Decoder :


```python
class DecoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(DecoderBlock, self).__init__()
        
        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.self_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        
        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.cross_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm2 = nn.LayerNorm(n_dim)
        
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm3 = nn.LayerNorm(n_dim)
        # self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)
        
        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)
        
        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)

        
        return output

class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        n_dim: int, 
        dropout: float, 
        n_decoder_blocks: int,
        n_heads: int):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=n_dim,
            padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(
            d_model=n_dim, 
            dropout=dropout
        )
          
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_dim, dropout, n_heads) for _ in range(n_decoder_blocks)
        ])
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for block in self.decoder_blocks:
            x = block(
                x, 
                memory, 
                tgt_mask=tgt_mask, 
                tgt_padding_mask=tgt_padding_mask, 
                memory_padding_mask=memory_padding_mask)
        return x
```

Padding & Masking
Remember the Multi-head attention section where we mentionned excluding certain parts of the inputs when doing attention.

During training, we consider batches of inputs and targets, wherein each instance may have a variable length. Consider the following example where we batch 4 words : banana, watermelon, pear, blueberry. In order to process them as a single batch, we need to align all words to the length of the longest word (watermelon). We will therefore add an extra token, PAD, to each word so they all end up with the same length as watermelon.

In the below picture, the upper table represents the raw data, the lower table the encoded version:

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qyCxYMom3TimQY2VGNFGPQ.png" />
</p>

In our case, we want to exclude padding indices from the attention weights being calculated. We can therefore compute a mask as follows, both for source and target data :

```python
padding_mask = (x == PAD_IDX)
```

What about causal masks now ? Well if we want, at each time step, that the model can attend only steps in the past, this means that for each time step T, the model can only attend to each step t for t in 1…T. It is a double for loop, we can therefore use a matrix to compute that :

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gIU1WTNJle6N0tw6P-C4OA.png" />
</p>

```python
def generate_square_subsequent_mask(size: int):
      """Generate a triangular (size, size) mask. From PyTorch docs."""
      mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
      mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
      return mask
```

## Case study : a Word-Reverse Transformer

Let’s now build our Transformer by bringing parts together !

In our use case, we will use a very simple dataset to showcase how Transformers actually learn.

> “But why use a Transformer to reverse words ? I already know how to do that in Python with word[::-1] !”

The objective here is to see whether the Transformer attention mechanism works. What we expect is to see attention weights to move from right to left when given an input sequence. If so, this means our Transformer has learned a very simple grammar, which is just reading from right to left, and could generalize to more complex grammars when doing real-life language translation.

Let’s first begin with our custom Transformer class :

```python
import torch
import torch.nn as nn
import math

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        
        for k, v in kwargs.items():
            print(f" * {k}={v}")
        
        self.vocab_size = kwargs.get('vocab_size')
        self.model_dim = kwargs.get('model_dim')
        self.dropout = kwargs.get('dropout')
        self.n_encoder_layers = kwargs.get('n_encoder_layers')
        self.n_decoder_layers = kwargs.get('n_decoder_layers')
        self.n_heads = kwargs.get('n_heads')
        self.batch_size = kwargs.get('batch_size')
        self.PAD_IDX = kwargs.get('pad_idx', 0)

        self.encoder = Encoder(
            self.vocab_size, self.model_dim, self.dropout, self.n_encoder_layers, self.n_heads)
        self.decoder = Decoder(
            self.vocab_size, self.model_dim, self.dropout, self.n_decoder_layers, self.n_heads)
        self.fc = nn.Linear(self.model_dim, self.vocab_size)
        

    @staticmethod    
    def generate_square_subsequent_mask(size: int):
            """Generate a triangular (size, size) mask. From PyTorch docs."""
            mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask


    def encode(
            self, 
            x: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, S) with elements in (0, C) where C is num_classes
        Output
            (B, S, E) embedding
        """

        mask = (x == self.PAD_IDX).float()
        encoder_padding_mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # (B, S, E)
        encoder_output = self.encoder(
            x, 
            padding_mask=encoder_padding_mask
        )  
        
        return encoder_output, encoder_padding_mask
    
    
    def decode(
            self, 
            tgt: torch.Tensor, 
            memory: torch.Tensor, 
            memory_padding_mask=None
        ) -> torch.Tensor:
        """
        B = Batch size
        S = Source sequence length
        L = Target sequence length
        E = Model dimension
        
        Input
            encoded_x: (B, S, E)
            y: (B, L) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        
        mask = (tgt == self.PAD_IDX).float()
        tgt_padding_mask = mask.masked_fill(mask == 1, float('-inf'))

        decoder_output = self.decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)), 
            tgt_padding_mask=tgt_padding_mask, 
            memory_padding_mask=memory_padding_mask,
        )  
        output = self.fc(decoder_output)  # shape (B, L, C)
        return output

        
        
    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        
        # Encoder output shape (B, S, E)
        encoder_output, encoder_padding_mask = self.encode(x)  

        # Decoder output shape (B, L, C)
        decoder_output = self.decode(
            tgt=y, 
            memory=encoder_output, 
            memory_padding_mask=encoder_padding_mask
        )  
        
        return decoder_output
```


