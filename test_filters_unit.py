from exllamav2 import *
from exllamav2.generator import *
import time
from exllamav2.generator.filters.prefix import ExLlamaV2PrefixFilter
from quote import debug_tokens, traverse_trie

config = ExLlamaV2Config("/mnt/data/textgenmodels/LoneStriker_OpenHermes-2-Mistral-7B-4.0bpw-h6-exl2/")
config.max_seq_len = 4096
tokenizer = ExLlamaV2Tokenizer(config)


def test_traverse_trie():
    out = traverse_trie(tokenizer, "I just wanted to ask about some of the recent initiatives that you guys are working on for economic development", offset=1)
    print(debug_tokens(tokenizer, out))
    print(debug_tokens(tokenizer, set([2324, 28739])))

test_traverse_trie()