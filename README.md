# quote-constraint

This is an exllamav2 filter that makes a model quote properly from a provided source document. It works best if your model is fine-tuned to use quotes in response to certain kinds of prompts.

When the filter sees an open quote, it enters quote mode, which requires the model to pick tokens from the source document (first any token and then only tokens subsequent to the first one) until it sees a closing quote. It also supports using an ellipsis to skip and then continue the quote later in the source document.

## Usage

```python
# setup your model and tokenizer, then do this
gen_settings = ExLlamaV2Sampler.Settings()
gen_settings.filters.append(ExLlamaV2QuoteFilter(model, tokenizer, source_doc))
```

See `test_filters.py` for a more complete example
