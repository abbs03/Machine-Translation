import regex as re
import pickle

en_pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
ta_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[\p{L}\u0B80-\u0BFF]++| ?\p{N}{1,2}+| ?\p{N}(?!\p{N})| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s|(?<=[a-zA-Z])(?=[\u0B80-\u0BFF])|(?<=[\u0B80-\u0BFF])(?=[a-zA-Z])""")


def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(tokens, pair, new_id):
    new_tokens = []
    i = 0
    while i <len(tokens):

        if i <len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(new_id)
            i+=1
        else:
            new_tokens.append(tokens[i])
        i+=1
    return new_tokens



class Tokenizer():

    def __init__(self,lang=None):
        
        
        self.compiled_pattern = ta_pat if lang == 'ta' else en_pat
        self.lang = lang
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        
        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in range(num_merges):
            
            stats = {}
            for chunk_ids in ids:
                
                get_stats(chunk_ids, stats)
            
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            
            idx = 256 + i
            
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        
        ids = list(text_bytes)
        while len(ids) >= 2:
            
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged anymore
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        
        text_chunks = re.findall(self.compiled_pattern, text)
        
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special=None):

        special = None
        if isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
    
        if not special:        
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        
        ids = []
        for part in special_chunks:
            if part in special:
                
                ids.append(special[part])
            else:
                
                ids.extend(self.encode_ordinary(part))
        return ids
    
    def get_vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)
         
    
    def save_params(self, path):
        with open(path + f'{self.lang}_tokenizer.pkl', "wb") as file:
            pickle.dump([self.merges, self.vocab], file)
    
    def load_params(self, path):

        with open(path, "rb") as file:
            self.merges, self.vocab = pickle.load(file)
        return 

            
            