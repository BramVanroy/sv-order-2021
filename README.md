Subject-verb order experiments
==============================

To use our scripts, clone this repository and then install the required libraries with

```shell
pip install -r requirements.txt
````

All relevant scripts have a `help` section, which you can call with the `-h` option, for instance

```shell
python add_frequencies_to_df.py -h
```

Models
------
We make use of the recent (December 2021) SOTA models by spaCy. Specifically the `nl_udv25_dutchalpino_trf` model, 
in part described [here](https://explosion.ai/blog/ud-benchmarks-v3-2).

Before using our scripts, you should install it with the following command (or install from the requirements file):

```shell
python -m pip install https://huggingface.co/explosion/nl_udv25_dutchalpino_trf/resolve/main/nl_udv25_dutchalpino_trf-any-py3-none-any.whl
```

Data
----
In our research, we calculated frequencies on the SONAR corpus and limited ourselves to components that were
written-to-be-read and published (WRP-). However, we excluded the WRPEA component, which contains data from discussion
forums. Its data is riddled with non-standard, colloquial, slang, internet language text, which not only falls outside
of the scope of our research objectives, but also makes the job of the parser very difficult
(and results unpredictable).

Sentences shorter than three words (e.g. enumerations like "1 .") or longer than 32 words were excluded. The latter 
restriction for computational feasibility.

Citation
--------

```bibtex
@ARTICLE{De_Sutter2023-si,
  title     = "Is linguistic decision-making constrained by the same cognitive
               factors in student and in professional translation?",
  author    = "De Sutter, Gert and Lefer, Marie-Aude and Vanroy, Bram",
  journal   = "Int. J. Learn. Corpus Res.",
  publisher = "John Benjamins Publishing Company",
  volume    =  9,
  number    =  1,
  pages     = "60--95",
  month     =  apr,
  year      =  2023,
  language  = "en"
}
```
