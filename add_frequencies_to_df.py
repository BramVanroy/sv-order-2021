from dataclasses import dataclass, field
from os import PathLike

import pandas as pd

from typing import Dict, Union

from pandas import DataFrame


@dataclass
class AddFreqToDv:
    dep_token_c: Dict = field(default_factory=dict)
    dep_lemma_c: Dict = field(default_factory=dict)
    verb_token_c: Dict = field(default_factory=dict)
    verb_lemma_c: Dict = field(default_factory=dict)
    n_tokens_processed: int = field(default=0)
    n_sents_processed: int = field(default=0)

    def get_word_as_subj_freq(self, word, kind="tokens"):
        d = self.dep_token_c if kind == "tokens" else self.dep_lemma_c
        as_subj = d["nsubj"][word]
        total = sum([c[word] for c in d.values()])
        return as_subj, total - as_subj

    def get_n(self, dep=None, unique=False, kind="tokens"):
        """If dep is None, gets the total number of tokens or lemmas. If dep is given, then return the total number of
         tokens or lemmas for that given category.
        If unique is false, then counts for lemmas and tokens are identical (as it is equal to the total number
        of words)."""
        d = self.dep_token_c if kind == "tokens" else self.dep_lemma_c
        if unique:
            # Only count unique tokens _per dep category_. So if a token occurs twice as verb and thrice as noun
            # we count it once as verb and once as noun!
            if dep is None:
                return sum([len(c.keys()) for c in d.values()])
            else:
                return len(d[dep].keys())
        else:
            # Not unique. So if a token occurs twice as a verb, we count it as such!
            if dep is None:
                return sum([sum(c.values()) for c in d.values()])
            else:
                return sum(d[dep].values())

    def add_freq_to_df(self, xlsx_f: Union[str, PathLike], outfile: Union[str, PathLike] = "frequencies.xlsx"):
        df: DataFrame = pd.read_excel(xlsx_f)

        for row_idx, row in df.iterrows():
            fin_verb, main_verb, head_subj_lemma, fin_verb_lemma = row

            if not pd.isna(fin_verb):
                df.at[row_idx, "1A_pre_tok_finverb"] = self.verb_token_c["pre"][fin_verb.strip().lower()]
                df.at[row_idx, "1B_post_tok_finverb"] = self.verb_token_c["post"][fin_verb.strip().lower()]

            if not pd.isna(fin_verb_lemma):
                df.at[row_idx, "2A_pre_lem_finverb"] = self.verb_lemma_c["pre"][fin_verb_lemma.strip().lower()]
                df.at[row_idx, "2B_post_lem_finverb"] = self.verb_lemma_c["post"][fin_verb_lemma.strip().lower()]

            if not pd.isna(main_verb):
                df.at[row_idx, "3A_pre_lem_mainverb"] = self.verb_lemma_c["pre"][main_verb.strip().lower()]
                df.at[row_idx, "3B_post_lem_mainverb"] = self.verb_lemma_c["post"][main_verb.strip().lower()]

            if not pd.isna(head_subj_lemma):
                n_as_subj, n_not_subj = self.get_word_as_subj_freq(head_subj_lemma.strip().lower(), kind="lemma")
                df.at[row_idx, "4A_lemma_subj"] = n_as_subj
                df.at[row_idx, "4B_lemma_not_subj"] = n_not_subj

        df.to_excel(outfile, index=False)
        print(df.head(5))

        n_subjs = self.get_n(dep="nsubj")
        n_subjs_uniq = self.get_n(dep="nsubj", unique=True)
        n_tokens_uniq = self.get_n(unique=True)

        print(f"PROCESSED SENTENCES for freqs: {self.n_sents_processed:,}")
        print(f"PROCESSED TOKENS for freqs: {self.n_tokens_processed:,}")

        print(f"N SUBJECTS: {n_subjs:,}")
        print(f"N NON-SUBJECTS: {self.n_tokens_processed-n_subjs:,}")

        print(f"PROCESSED UNIQUE TOKENS for freqs: {n_tokens_uniq:,}")
        print(f"N UNIQUE SUBJECTS: {n_subjs_uniq:,}")
        print(f"N UNIQUE NON-SUBJECTS: {n_tokens_uniq-n_subjs_uniq:,}")



if __name__ == "__main__":
    import argparse
    import pickle

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("pickle_f", help="Path to previously created, pickled Extractor file that contains the needed frequencies.")
    cparser.add_argument("xlsx_f", help="Excel file that contains the words whose values we need to supplement.")
    cparser.add_argument("-o", "--outfile", default="frequencies.xlsx",
                         help="Path of the output file. If not given, writes to frequencies.xlsx.")

    cargs = cparser.parse_args()
    with open(cargs.pickle_f, "rb") as fhin:
        freq_d = pickle.load(fhin)

    adder = AddFreqToDv(**freq_d)
    adder.add_freq_to_df(cargs.xlsx_f, cargs.outfile)
