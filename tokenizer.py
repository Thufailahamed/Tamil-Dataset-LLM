import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing

special_token_dict = {"unknown_token": "[UNK]",
                      "pad_token": "[PAD]", 
                      "start_token": "[BOS]",
                      "end_token": "[EOS]"}

class TamilTokenizer:
    def __init__(self, path_to_vocab, truncate=False, max_length=512):
        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.prepare_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.special_tokens_dict = {"[UNK]": self.tokenizer.token_to_id("[UNK]"),
                                    "[PAD]": self.tokenizer.token_to_id("[PAD]"),
                                    "[BOS]": self.tokenizer.token_to_id("[BOS]"),
                                    "[EOS]": self.tokenizer.token_to_id("[EOS]")}

        self.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
                ("[BOS]", self.tokenizer.token_to_id("[BOS]"))
            ]
        )
        
        self.truncate = truncate
        if self.truncate:
            self.max_len = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)

    def prepare_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer
    
    def encode(self,input):
        def _parse_process_tokenized(tokenized):
            if self.truncate:
                tokenized.truncate(self.max_len, direction="right")
            tokenized = self.post_processor.process(tokenized)
            return tokenized.ids
        
        if isinstance(input, str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _parse_process_tokenized(tokenized)
            
        
        elif isinstance(input, (list, tuple)):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_parse_process_tokenized(t) for t in tokenized]
        
        return tokenized

    def decode(self, input, skip_special_tokens=True):

        if isinstance(input, list):
            
            if all(isinstance(item, list) for item in input):
                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
            
        return decoded

if __name__=="__main__":
    path_to_vocab = "path_to_your_vocab_file"
    tok = TamilTokenizer("trained_tokenizer/tamil_wp.json")
    sentence = "Hello World"
    enc = tok.encode(sentence)
    print(tok.decode(enc))