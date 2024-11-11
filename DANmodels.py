import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import WordEmbeddings
from collections import Counter, defaultdict


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, train=True):
        
        self.word_embeddings = word_embeddings
        self.data = []

        with open(infile, 'r') as f:
            for line in f:
                if line.strip():
                    label, sentence = line.split(maxsplit=1)
                    sentence_words = sentence.strip().split()
                    self.data.append((sentence_words, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sentence_words, label = self.data[idx]
        word_indices = [self.word_embeddings.get_embedding(word) for word in sentence_words]
        indices_tensor = torch.tensor(word_indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return indices_tensor, label_tensor
        
        



class DAN(nn.Module):
    def __init__(self, word_embeddings, hidden_size, num_classes, dropout):
        super(DAN,self).__init__()
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.embedding_len = word_embeddings.vectors.shape[1]
        self.fc1 = nn.Linear(self.embedding_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
       
        embedded = self.embedding(x) 
        sentence_embedding = embedded.mean(dim=1)
        x = self.dropout(F.relu(self.fc1(sentence_embedding)))
        x = self.fc2(x)
        return self.log_softmax(x)

class DANWG(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout):
        super(DANWG,self).__init__()      
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x) 
        sentence_embedding = embedded.mean(dim=1)
        x = self.dropout(F.relu(self.fc1(sentence_embedding)))
        x = self.fc2(x)
        return self.log_softmax(x)

class BPE(Dataset): 

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = None
        self.vocab_to_idx = {}
        self.merges = set()


    def get_vocab(self, corpus):
        words = corpus.split()
        return Counter([''.join(list(word)) for word in words])


    

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word) 
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair not in self.merges:  
                    pairs[pair] += freq
        return pairs



    def merge_vocab(self, pair, vocab):
        bigram = ''.join(pair)
        new_vocab = {}
        
        for word, freq in vocab.items():
            # Split word into parts
            parts = list(word)
            new_parts = []
            i = 0
            
            # Process parts sequentially
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(bigram)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            
            new_word = ''.join(new_parts)
            new_vocab[new_word] = freq
            
        return new_vocab

    



    def build_bpe_vocab(self, corpus):
        self.vocab = self.get_vocab(corpus)
        print(f"Initial vocab size: {len(self.vocab)}")
        
        # Initialize character vocabulary
        unique_chars = set(''.join(self.vocab.keys()))
        self.vocab_to_idx = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        #print(self.vocab_to_idx)
        #print(len(self.vocab_to_idx))
        num_merges = 0
        #print(self.vocab_size - len(self.vocab_to_idx))
        max_merges = min(self.vocab_size - len(self.vocab_to_idx) - 2, 1000)
        
        while len(self.vocab_to_idx) < self.vocab_size - 2 and num_merges < max_merges:
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair, best_freq = max(pairs.items(), key=lambda x: (x[1], x[0]))
            
            
            # If we've already merged this pair or it only occurs once, skip it
            if best_pair in self.merges or best_freq <= 1:
                break
                
            # Perform the merge
            self.vocab = self.merge_vocab(best_pair, self.vocab)
            new_token = ''.join(best_pair)
            
            # Add to vocabulary and record the merge
            if new_token not in self.vocab_to_idx:
                self.vocab_to_idx[new_token] = len(self.vocab_to_idx)
                self.merges.add(best_pair)
            
            num_merges += 1
        
        # Add special tokens
        self.vocab_to_idx['<unk>'] = len(self.vocab_to_idx)
        self.vocab_to_idx['<pad>'] = len(self.vocab_to_idx)
        
        print(f"Final vocab size: {len(self.vocab_to_idx)}")
        print(f"Number of merges performed: {num_merges}")
        return self.vocab
    
    


    
    def tokenize(self, text):
        words = text.split()
        final_tokens = []
        
        for word in words:
            current_word = list(word)  # Split word into characters
            
            # Try to apply merges iteratively
            i = 0
            while i < len(current_word) - 1:
                pair = (current_word[i], current_word[i + 1])
                combined = ''.join(pair)
                
                if combined in self.vocab_to_idx:
                    current_word[i] = combined
                    del current_word[i + 1]
                else:
                    i += 1
            
            final_tokens.extend(current_word)
        
        # Convert to indices, using <unk> for unknown tokens
        return final_tokens
    

   
        

    
class SentimentDataBPE(Dataset):


    def __init__(self, infile, bpe = None, is_training = True, vocab_size = 5000):
        self.sentences, self.labels = self.load_sentences(infile)
        if is_training:
            self.bpe = BPE(vocab_size)
            self.subword_vocab = self.bpe.build_bpe_vocab(' '.join(self.sentences))
        else:
            # Use provided BPE object for validation/test
            assert bpe is not None, "BPE object must be provided for non-training data"
            self.bpe = bpe
        self.tokenized_sentences = [self.bpe.tokenize(sent) for sent in self.sentences]
        self.vocab_to_idx = self.bpe.vocab_to_idx

    def load_sentences(self, infile):
        sentences = []
        labels = []
        with open(infile, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('\t', 1)
                label = int(parts[0])
                sentence = parts[1].strip()
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
        return sentences, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokenized_sentences[idx]
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<unk>']) for token in tokens]
        max_len = 100
        if len(indices) < max_len:
            indices = indices + [self.vocab_to_idx['<pad>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def get_vocab(self):
        return self.vocab_to_idx

   


    





