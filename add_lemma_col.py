"""Adds lemmas based on the 'finite verb forms' column to the 'finite verb lemma' column. NOTE however that these
should be manually verified. The context is so small that mistakes are likely. As context the 'main verb lemmas'
column is used if available."""

from os import PathLike
from typing import Union

import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame


def add_lemma_to_df(xlsx_f: Union[str, PathLike], outfile: Union[str, PathLike] = "frequencies_lemma.xlsx"):
    df: DataFrame = pd.read_excel(xlsx_f)
    # Not disabling tagger/parser which may be used by lemmatizer
    nlp = spacy.load("nl_udv25_dutchalpino_trf", exclude=["senter", "sentencizer", "ner", "textcat"])

    def lemmatize(r):
        if not pd.isna(r['finite verb forms']):
            # Try to provide a bit more context to the parser than just one word, if available
            if pd.isna(r['main verb lemmas']):
                doc = nlp(f"{r['finite verb forms']}")
            else:
                doc = nlp(f"{r['finite verb forms']} {r['main verb lemmas']}")
            return doc[0].lemma_

        return np.nan

    df["finite verb lemma"] = df.apply(lemmatize, axis=1)

    df.to_excel(outfile, index=False, na_rep="")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)

    cparser.add_argument("xlsx_f", help="Excel file that contains the words whose values we need to supplement.")
    cparser.add_argument("-o", "--outfile", default="frequencies_lemma.xlsx",
                         help="Path of the output file. If not given, writes to frequencies_lemma.xlsx.")

    cargs = cparser.parse_args()
    add_lemma_to_df(cargs.xlsx_f, cargs.outfile)
