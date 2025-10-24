# Optional resources

This directory hosts optional runtime assets such as cached tokenizer files.
The download helper in `scripts/download_optional_assets.py` will populate the
`gpt2_tokenizer/` subdirectory when GPT-2 resources are fetched from Hugging
Face.  The folder remains empty in source control so CI and unit tests are not
forced to download large models.
