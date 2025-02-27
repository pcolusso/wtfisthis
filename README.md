# wtfisthis

Exploring using sentence-tranformers for code search.

## TODO

- Try to static link libtorch, we can run fine via Cargo's build script finding it for us, but the binary itself cannot.


## Ideas
 - Using a pretty arbitrary model, are there better ones for code? Seems to work decently.
 - UI is pretty bad, results should be presented in a nicer way.


## Lessons
 - Polars is pretty nice, I wish it would play better with soa_derive.
 - Parquet seems fast enough for the small-ish amount of embeddings we're dealing with, about a meg or two.
 - Tree-sitter's rust bindings saved this project's idea, the original tutorial recommended using an LSP. https://huggingface.co/learn/cookbook/en/code_search
