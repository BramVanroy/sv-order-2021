"""
Data used: SONAR, excluding the written-to-be-spoken components starting with WS-
"""
from collections import Counter
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
import pickle
from typing import Dict, Union

import ftfy
import pandas as pd
import spacy
from spacy import Language
from spacy.tokens import Doc
from spacy.util import minibatch
import torch
from tqdm import tqdm

using_gpu = spacy.prefer_gpu()

print("USING GPU? (spaCy)", using_gpu)
print("USING GPU? (torch)", torch.cuda.is_available())


@dataclass
class Extractor:
    indir: Union[str, PathLike] = field(default=None)
    ext: str = field(default="")
    outfile: str = field(default="extractor.pckl")
    batch_size: int = 64
    verbose: bool = False

    dep_token_c: Dict = field(default_factory=dict, init=False)
    dep_lemma_c: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.files = list(Path(self.indir).glob(f"*{self.ext}"))
        self.verb_token_c = {"pre": Counter(), "post": Counter()}
        self.verb_lemma_c = {"pre": Counter(), "post": Counter()}

        self.calculate_statistics()

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

    @staticmethod
    def lines(pfin):
        lines = []
        for line in pfin.read_text(encoding="utf-8").splitlines(keepends=False):
            line = ftfy.fix_text(line)

            # Skip short sequences
            if len(line.split(" ")) < 3:
                continue

            lines.append(line)

        return lines

    def is_pre_or_post_subjs(self, doc: Doc):
        # Here we know that there is a pv in the doc, but in some cases there is more than one. For instance:
        # Bovendien blijft de particuliere huursector achter bij de overige eigendomscategorieën , het energiebesparingspotentieel is in die sector het grootst .
        for t in doc:
            if "ww|pv" in t.tag_.lower():
                # Get subjects that are on the "same level" of the finite verb
                subjs = sorted([t for t in t.head.children if t.dep_.lower() == "nsubj"], key=lambda x: x.i)

                if not subjs:
                    if self.verbose:
                        print("NO SUBJ", doc, t, sep="--")
                    return None, None, None

                first_subj = subjs[0]

                yield "pre" if first_subj.i < t.i else "post", t, first_subj

    def calculate_statistics(self):
        # Disable sentence segmentation
        # Install with
        # python -m pip install https://huggingface.co/explosion/nl_udv25_dutchalpino_trf/resolve/main/nl_udv25_dutchalpino_trf-any-py3-none-any.whl
        nlp = spacy.load("nl_udv25_dutchalpino_trf", exclude=["senter", "sentencizer"])
        nlp.add_pipe("disable_sbd", before="parser")

        for pfin in tqdm(self.files, desc="File"):
            lines = self.lines(pfin)
            for batch in tqdm(minibatch(lines, size=self.batch_size), total=len(lines)//self.batch_size, leave=False, desc="Batch"):
                docs = nlp.pipe(batch)
                for doc in docs:
                    if self.verbose:
                        print(doc)

                    # Must contain finite verb (pv; persoonsvorm)
                    if not any("ww|pv" in t.tag_.lower() for t in doc):
                        continue

                    # Must contain a subject (we later check if the subjects and pv's are related)
                    if not any(t.dep_.lower() == "nsubj" for t in doc):
                        continue

                    for token in doc:
                        if token.dep_.lower() not in self.dep_token_c:
                            self.dep_token_c[token.dep_.lower()] = Counter()
                        self.dep_token_c[token.dep_.lower()][token.text.lower()] += 1

                        if token.dep_.lower() not in self.dep_lemma_c:
                            self.dep_lemma_c[token.dep_.lower()] = Counter()
                        self.dep_lemma_c[token.dep_.lower()][token.lemma_.lower()] += 1

                    # An approximation of sentences that contain "non main clauses", i.e. "complex clauses"
                    # As taken from the UD documentation: https://universaldependencies.org/u/overview/complex-syntax.html
                    if not any(t.dep_.lower() in ("csubj", "xcomp", "ccomp", "advcl", "acl", "conj", "acl:relcl") for
                                t in doc):
                        # So this part is only done for main clauses:
                        for subj_position, pv, subj in self.is_pre_or_post_subjs(doc):
                            if subj_position is None:
                                continue

                            self.verb_token_c[subj_position][pv.text.lower()] += 1
                            self.verb_lemma_c[subj_position][pv.lemma_.lower()] += 1
                            # put these results into counters depending on pre/post

                            if self.verbose:
                                print(subj_position, pv, subj)

                    if self.verbose:
                        print()

    def get_word_as_subj_freq(self, word):
        as_subj = self.dep_lemma_c["nsubj"][word]
        total = sum([c[word] for c in self.dep_lemma_c.values()])
        return as_subj, total-as_subj

    def add_freq_to_df(self, xlsx: Union[str, PathLike], outfile: Union[str, PathLike] = "frequencies.txt"):
        df = pd.read_excel(xlsx)

        for row_idx, row in df.iterrows():
            fin_verb, main_verb, head_subj_lemma = row

            if not pd.isna(fin_verb):
                df.at[row_idx, "1A_pre_tok_finverb"] = self.verb_token_c["pre"][fin_verb.lower()]
                df.at[row_idx, "1B_post_tok_finverb"] = self.verb_token_c["post"][fin_verb.lower()]

                df.at[row_idx, "2A_pre_lem_finverb"] = self.verb_lemma_c["pre"][fin_verb.lower()]
                df.at[row_idx, "2B_post_lem_finverb"] = self.verb_lemma_c["post"][fin_verb.lower()]

            if not pd.isna(main_verb):
                df.at[row_idx, "3A_pre_lem_mainverb"] = self.verb_lemma_c["pre"][main_verb.lower()]
                df.at[row_idx, "3B_post_lem_mainverb"] = self.verb_lemma_c["post"][main_verb.lower()]

            if not pd.isna(head_subj_lemma):
                n_as_subj, n_not_subj = self.get_word_as_subj_freq(head_subj_lemma)
                df.at[row_idx, "4A_tok_subj"] = n_as_subj
                df.at[row_idx, "4B_tok_not_subj"] = n_not_subj

        print(df.head(10))

    def save(self):
        with open(self.outfile, "wb") as fhout:
            pickle.dump(self, fhout)


@Language.factory("disable_sbd")
class SpacyDisableSentenceSegmentation:
    """Disables spaCy's dependency-based sentence boundary detection. In addition, senter and sentencizer components
    need to be disabled as well."""

    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        for token in doc:
            token.is_sent_start = False
        return doc


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    csubparsers = cparser.add_subparsers(dest="subparser_name")

    cextract = csubparsers.add_parser("extract")
    cextract.add_argument("indir", help="Path to input directory with text files. These files constitute the corpus that will be parsed and used to calculate frequencies")
    cextract.add_argument("-e", "--ext", default="",
                         help="Only process files with this extension (must include a dot).")
    cextract.add_argument("-o", "--outfile", default="extractor.pckl",
                         help="Path of the output file to save the final object and all frequencies to. If not given, writes to extractor.pckl.")
    cextract.add_argument("-b", "--batch_size", default=64, type=int, help="Mini-batch size to process at a time. Larger = faster but may lead to out of memory issues.")
    cextract.add_argument("-v", "--verbose", action="store_true", help="Print information of matching tokens.")

    ccollect = csubparsers.add_parser("collect")
    ccollect.add_argument("pickle_f", help="Path to previously created, pickled Extractor file that contains the needed frequencies.")
    ccollect.add_argument("xlsx", help="Excel file that contains the words whose values we need to supplement.")
    ccollect.add_argument("-o", "--outfile", default="frequencies.txt",
                         help="Path of the output file, must end with .txt. If not given, writes to frequencies.txt.")

    cargs = vars(cparser.parse_args())
    cparser_name = cargs.pop("subparser_name")
    if cparser_name == "extract":
        extractor = Extractor(**cargs)
        extractor.save()
    else:
        with open(cargs["pickle_f"], "rb") as fhin:
            extractor = pickle.load(fhin)

        extractor.add_freq_to_df(cargs["xlsx"], cargs["outfile"])
