from exllamav2 import *
from exllamav2.generator import *
import time
from exllamav2.generator.filters.prefix import ExLlamaV2PrefixFilter
from quote import ExLlamaV2QuoteFilter
print("Loading model...")

config = ExLlamaV2Config("/mnt/data/textgenmodels/LoneStriker_OpenHermes-2-Mistral-7B-4.0bpw-h6-exl2/")
# print(config.__dict__)
config.max_seq_len = 4096
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.set_stop_conditions([tokenizer.eos_token_id])
gen_settings = ExLlamaV2Sampler.Settings()

max_new_tokens = 1000

generator.warmup()
time_begin = time.time()

source_doc = "You're familiar with American, the American Rescue Plan Act of 2021 and the ability to kind of infuse federal funds into local economies and so the town was awarded ARPA funds and our economic development department create some incentive programs through that funding. So just to note that funding source is drying up. We are doing our job and spending it. Once we've fully allocated those funds, you know, we're not identifying funding for continuing these programs into the future. Sorry, can I ask a clarifying question? Are you looking for information on the incentive guidelines, like the agreements we enter with raw properties and well data, or are you looking for grant information? A little bit of both. Okay. So, yeah. So, I just want to clarify, Katie's talking about our grant program. Yes. Okay, great. So, I'll make that up. Mm-hmm. Yeah, I do think I saw some of the web pages for that. Yeah, so we typically use the word incentive to talk about like this agreements we enter with like a private developer or a coming business to downtown. For example, WellDOT is one of the more recent ones we did, and that was focused on job creation. And so they came to downtown and they're- because it's solely based on how many new jobs they create. So they're projected to create 400 jobs by 2026. And they get a grant, essentially, for the program that they have these many new jobs. And that's how much that they get per job. Got it. Are all those jobs going to be in the same? I've walked by their office on Franklin. Is it all going to be just in that building? Or are they expanding? Yeah. So they also have the location of 501 West. So we're kind of right next door that's currently empty, but they're actively working on it right now. They're in the middle of doing kind of their permitting process, but they were kind of delayed when they first got to Chapel Hill because of the pandemic. Some people weren't coming back to work, but they are kind of pushing forward with mostly, if not all, in-person workers, which is really great. "
prompt = f"""<|im_start|>system
You are a helpful AI assistant
<|im_start|>user
{source_doc}

Write an 800-word AP-style newspaper article based on the interview above. Use quotes from the interview correctly attributed. Write 1-2 sentences per paragraph, use a newsy lede and a surprising closer.
<|im_start|>assistant
"""
print(prompt)

# prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, "{")
gen_settings.filters.append(ExLlamaV2QuoteFilter(model, tokenizer, source_doc))


output = generator.generate_simple(prompt, gen_settings, max_new_tokens, seed = 1234)

time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
