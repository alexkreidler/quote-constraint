from typing import Iterable
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator.filters.base import ExLlamaV2Filter
from torch import Tensor

def traverse_trie(tokenizer: ExLlamaV2Tokenizer, sequence, last_leaf=False, offset=0):
    """Returns a set of tokens in the char_trie that continue along the sequence.
last_leaf returns only a set with the last leaf, which is the deepest in the trie, aka the longest word possible, could make inference more efficient for the quote application
Example:

```
sequence = 'I just wanted' # Input
# Output of calling traverse_trie on sequence[0:], sequence[1:] and so on
['I', 'I']
[' ', ' ju', ' ', ' just', ' j']
['j', 'ju', 'j', 'just']
[' ', ' w', ' ', ' want', ' wa', ' wanted']
```
    """
    # todo use a list to keep traversal order if clients want
    tokens = []
    w = tokenizer.get_char_trie()
    leaf = []
    for c in sequence[offset:]:
        if c in w.children:
            w = w.children[c]
        else:
            break
        leaf = w.leaf
        tokens += leaf
        # print(debug_tokens(tokenizer, leaf))
    
    return leaf if last_leaf else tokens

def debug_tokens(tokenizer: ExLlamaV2Tokenizer, tokens: Iterable[int]):
    id_to_piece = tokenizer.get_id_to_piece_list()
    return [id_to_piece[tok] for tok in tokens]

QUOTES = ['“', '”', '"']
# , "'"]
ALWAYS_ALLOWED = QUOTES
# + ["..."]

global num_gens 

num_gens= 0
# 28739, 28835, 1101, 28838}
class ExLlamaV2QuoteFilter(ExLlamaV2Filter):
    # TODO: Try with beam search or other sampling method so the model doesn't overshoot the quote and go into filler words.
    def __init__(self, model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer, source_document: str):
        super().__init__(model, tokenizer)
        self.source_document = source_document
        self.state = 'NORMAL_MODE'
        self.current_position = -1  # -1 means any token from the source document is allowed
        self.quote_length = 0

        p2id = self.tokenizer.get_piece_to_id_dict()
        self.allowed_tokens = [p2id[s] for s in ALWAYS_ALLOWED]
        print("allowed", self.allowed_tokens)
        if len(self.allowed_tokens) != len(ALWAYS_ALLOWED):
            print("warning")
        self.disallowed = self.get_disallowed_tokens()

    def get_tok(self, str):
        return traverse_trie(self.tokenizer, str, last_leaf=True)

    def get_disallowed_tokens(self):
        disallowed = set()
        keys_list = []
        for q in QUOTES:
            p2id = self.tokenizer.get_piece_to_id_dict()
            for key, value in p2id.items():
                if q in key and q != key:
                    disallowed.add(value)
                    keys_list.append(key)
            disallowed |= set(traverse_trie(self.tokenizer, q))
        return set()

    def begin(self, prefix_str=""):
        self.state = 'NORMAL_MODE'
        self.current_position = -1

    def feed(self, token: int):
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        piece = id_to_piece[token]
        print(piece, end="")

        if self.current_position >= len(self.source_document):
            print(f"Warning: current pos {self.current_position} is longer than source document")

        if any([q in piece for q in QUOTES]):
            if self.state == 'NEXT_QUOTE_TOKEN':
                # closing
                self.state = 'NORMAL_MODE'
                self.current_position = -1
                self.quote_length = 0
            else:
                self.state = 'FIRST_QUOTE_TOKEN'
                self.current_position = 0
        elif piece == '...':
            self.state = 'AFTER_ELLIPSIS_TOKEN'
        elif self.state == 'NEXT_QUOTE_TOKEN':
            if piece != self.source_document[self.current_position:self.current_position + len(piece)]:
                print(f'Warning: Token "{piece}" does not match source document at indices {self.current_position}-{self.current_position + len(piece)}')
            self.current_position += len(piece)
            self.quote_length += len(piece)
        elif self.state == 'AFTER_ELLIPSIS_TOKEN' or self.state == 'FIRST_QUOTE_TOKEN':
            self.current_position = self.source_document.find(piece, self.current_position) + len(piece)
            if self.current_position == -1:
                print(f'Warning: Token "{piece}" not found in source document')
            self.state = 'NEXT_QUOTE_TOKEN'

    def next(self):
        if self.state == 'NORMAL_MODE' or self.current_position == -1:
            return set(range(self.tokenizer.get_vocab_size())) - set(self.disallowed), set()

        if self.current_position >= len(self.source_document):
            return set(), set()

        pass_tokens = set()
        # self.allowed_tokens)

        if self.state == 'FIRST_QUOTE_TOKEN' or self.state == 'AFTER_ELLIPSIS_TOKEN':
            # if num_gens > 5:
            #     raise ValueError("Current position is too far ahead")
            
            words = self.source_document[self.current_position:].split(" ")
            # encoded: Tensor = self.tokenizer.encode(self.source_document[self.current_position:])
            # doctoks = encoded.tolist()[0]
            
            # Even this only gets the first token of each word, that's fine cause we don't want a quote to start in the middle of a word.
            # The NEXT_QUOTE_TOKEN state will handle following the rest of the word/transcript
            doctoks = [traverse_trie(self.tokenizer, word) for word in words]
            flat_list = [ x for xs in doctoks for x in xs ]
            # print(debug_tokens(self.tokenizer, flat_list))
            # print(debug_tokens(self.tokenizer, doctoks))
            pass_tokens |= set(flat_list)
            # num_gens += 1
        elif self.state == 'NEXT_QUOTE_TOKEN':
            rem_str = self.source_document[self.current_position:]
            # print("QUOTEMODE", rem_str[:200])
            pass_tokens |= set(self.get_tok(rem_str))
            if self.quote_length > 20:
                # print("Allowing close quotes")
                pass_tokens |= set(self.allowed_tokens)


        return pass_tokens, set()