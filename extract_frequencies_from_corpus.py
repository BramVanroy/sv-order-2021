"""
Data used: SONAR, only written-to-be-read, published (WRP-) and NOT WRPEA, which contains forum posts (bad spelling, hard to parse)
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue
from os import PathLike
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Union
import warnings

import ftfy
import spacy
import torch
from spacy import Language, Vocab
from spacy.tokens import Doc
from spacy.util import minibatch
from thinc.api import require_gpu
from thinc.backends import use_pytorch_for_gpu_memory
from tqdm import tqdm

# torch will complain if CUDA is not available in autocast
# see: https://github.com/pytorch/pytorch/issues/67598
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

TORCH_CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class FrequencyExtractor:
    indir: Union[str, PathLike] = field(default=None, compare=False)
    ext: str = field(default="", compare=False)
    batch_size: int = field(default=64, compare=False)
    n_workers: int = field(default=None, compare=False)
    verbose: bool = field(default=False, compare=False)
    is_tokenized: bool = field(default=False, compare=False)
    n_gpus: int = field(default=0, compare=False)
    min: int = field(default=0, compare=False)
    max: int = field(default=0, compare=False)

    dep_token_c: Dict = field(default_factory=dict, init=False)
    dep_lemma_c: Dict = field(default_factory=dict, init=False)
    verb_token_c: Dict = field(default_factory=dict, init=False)
    verb_lemma_c: Dict = field(default_factory=dict, init=False)
    n_tokens_processed: int = field(default=0, init=False)
    n_sents_processed: int = field(default=0, init=False)

    results_q: Optional[Queue] = field(default=None, init=False, repr=False, compare=False)
    work_q: Optional[Queue] = field(default=None, init=False, repr=False, compare=False)
    no_cuda: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.files = list(Path(self.indir).glob(f"*{self.ext}"))
        self.verb_token_c = {"pre": Counter(), "post": Counter()}
        self.verb_lemma_c = {"pre": Counter(), "post": Counter()}

        if (not CUPY_AVAILABLE or not TORCH_CUDA_AVAILABLE) and self.n_gpus > 0:
            logging.warning(f"CUDA requested, but environment does not support it! Disabling...\n"
                            f"\t- CUPY AVAILABLE: {CUPY_AVAILABLE}\n\t- TORCH CUDA AVAILABLE: {TORCH_CUDA_AVAILABLE}")
            self.n_gpus = 0

        # If n_workers not set, set it to 1 or n_gpus, whichever is highest
        if self.n_workers is None:
            self.n_workers = max(1, self.n_gpus)

        if self.n_gpus > self.n_workers:
            raise ValueError("'n_gpus must be equal to or less than 'n_workers'")

        self.no_cuda = self.n_gpus < 1
        self.process_dir()

    def filter_lines(self, ls):
        lines = []
        for line in ls:
            line = ftfy.fix_text(line)
            seq_len = len(line.split(" "))
            # Skip short/long sequences
            # If max not set (0), only check min (which is 0 by default)
            if not self.max and self.min <= seq_len:
                lines.append(line)
            # If max is set, test both (min is always set to 0 as default so no problem)
            elif self.min <= seq_len <= self.max:
                lines.append(line)

        return lines

    def is_pre_or_post_subjs(self, doc: Doc):
        # Here we know that there is a pv in the doc, but in some cases there is more than one - even in two main clauses. For instance:
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

    def _reader(self):
        for pfin in tqdm(self.files, desc="File"):
            lines = pfin.read_text(encoding="utf-8").splitlines(keepends=False)
            for batch in tqdm(minibatch(lines, self.batch_size), total=len(lines)//self.batch_size, desc="Batch"):
                yield batch

    def reader(self):
        for batch in self._reader():
            self.work_q.put(batch)

        for _ in range(self.n_workers):
            self.work_q.put("done")

    def process_dir(self):
        if self.n_workers > 1:
            self.calculate_statistics_mp()
        else:
            self.calculate_statistics_sp()

        # `self` should now contain all the relevant data so we can now call `save` if we want

    def calculate_statistics_sp(self):
        self.set_cuda(0)
        nlp = self.load_nlp()

        for batch in self._reader():
            self.process_batch(batch, nlp, 0)

    def calculate_statistics_mp(self):
        with Manager() as manager:
            self.results_q = manager.Queue(maxsize=self.n_workers)
            self.work_q = manager.Queue(maxsize=self.n_workers * 10)

            # Create a reader and a writer process
            reader_proc = Process(target=self.reader)
            # The reader starts filling up the work_q
            reader_proc.start()

            gpu_map = {i: (i % self.n_gpus) if self.n_gpus > 0 else 0 for i in range(self.n_workers)}
            jobs = []
            for rank in range(self.n_workers):
                proc = Process(target=self._calculate_statistics_mp, args=(rank, gpu_map[rank]))
                proc.start()
                jobs.append(proc)

            # Collect results from queue, which the processes add the results queue when they are done
            results = [self.results_q.get() for _ in range(self.n_workers)]

            for job in jobs:
                job.join()
                job.terminate()

            reader_proc.join()
            reader_proc.terminate()
            self.merge_result_dicts(results)

    def _calculate_statistics_mp(self, rank, gpuid):
        self.set_cuda(gpuid)
        nlp = self.load_nlp()

        while True:
            # Get work from the working queue
            batch = self.work_q.get()
            if batch == "done":
                break

            self.process_batch(batch, nlp, rank)

        results = {"dep_token_c": self.dep_token_c,
                   "dep_lemma_c": self.dep_lemma_c,
                   "verb_token_c": self.verb_token_c,
                   "verb_lemma_c": self.verb_lemma_c,
                   "n_tokens_processed": self.n_tokens_processed,
                   "n_sents_processed": self.n_sents_processed}

        self.results_q.put(results)

    def process_batch(self, batch, nlp, rank=0):
        batch = self.filter_lines(batch)
        docs = nlp.pipe(batch, batch_size=self.batch_size)
        for doc in docs:
            if self.verbose and rank == 0:
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

                    if self.verbose and rank == 0:
                        print(subj_position, pv, subj)

            self.n_tokens_processed += len(doc)
            self.n_sents_processed += 1

            if self.verbose and rank == 0:
                print()

    def merge_result_dicts(self, results):
        # Have to give sum() an empty counter as a start item, otherwise will complain that it cannot
        # sum default start value 0 (int) with counters
        verb_token_c = {"pre": sum([d["verb_token_c"]["pre"] for d in results], Counter()),
                        "post": sum([d["verb_token_c"]["post"] for d in results], Counter())}
        verb_lemma_c = {"pre": sum([d["verb_lemma_c"]["pre"] for d in results], Counter()),
                        "post": sum([d["verb_lemma_c"]["post"] for d in results], Counter())}

        # For tiny datasets, it might be that they do not all have the same tags. So get all of them and then iterate.
        dep_tags = sorted({k for d in results for k in d["dep_token_c"].keys()})
        dep_token_c = defaultdict(dict)
        dep_lemma_c = defaultdict(dict)
        for tag in dep_tags:
            dep_token_c[tag] = sum([d["dep_token_c"][tag] if tag in d["dep_token_c"] else Counter() for d in results],
                                   Counter())
            dep_lemma_c[tag] = sum([d["dep_lemma_c"][tag] if tag in d["dep_token_c"] else Counter() for d in results],
                                   Counter())

        self.dep_token_c = dict(dep_token_c)
        self.dep_lemma_c = dict(dep_lemma_c)
        self.verb_token_c = verb_token_c
        self.verb_lemma_c = verb_lemma_c

        self.n_tokens_processed = sum([d["n_tokens_processed"] for d in results])
        self.n_sents_processed = sum([d["n_sents_processed"] for d in results])

    def load_nlp(self):
        # Install with
        # python -m pip install https://huggingface.co/explosion/nl_udv25_dutchalpino_trf/resolve/main/nl_udv25_dutchalpino_trf-any-py3-none-any.whl
        if self.is_tokenized:
            # Exclude experimental_char_ner_tokenizer and replace pretokenizer by white-space tokenizer
            # See discussion: https://github.com/explosion/spaCy/discussions/9949
            nlp = spacy.load("nl_udv25_dutchalpino_trf",
                             exclude=["experimental_char_ner_tokenizer", "senter", "sentencizer", "ner", "textcat"])
            nlp.tokenizer = PretokenizedTokenizer(nlp.vocab)
        else:
            # Do not exclude ner as it is used by experimental_char_ner_tokenizer
            nlp = spacy.load("nl_udv25_dutchalpino_trf", exclude=["senter", "sentencizer", "textcat"])

        nlp.add_pipe("disable_sbd", before="parser")

        return nlp

    def set_cuda(self, gpuid):
        if not self.no_cuda:
            cupy.cuda.Device(gpuid).use()
            use_pytorch_for_gpu_memory()
            require_gpu(gpuid)

    def save(self, outfile: Union[str, PathLike] = "extractor.pckl"):
        results = {"dep_token_c": self.dep_token_c,
                   "dep_lemma_c": self.dep_lemma_c,
                   "verb_token_c": self.verb_token_c,
                   "verb_lemma_c": self.verb_lemma_c,
                   "n_tokens_processed": self.n_tokens_processed,
                   "n_sents_processed": self.n_sents_processed}

        with open(outfile, "wb") as fhout:
            pickle.dump(results, fhout)


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


class PretokenizedTokenizer:
    """Custom tokenizer to be used in spaCy when the text is already pretokenized."""
    def __init__(self, vocab: Vocab):
        """Initialize tokenizer with a given vocab
        :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
        """
        self.vocab = vocab

    def __call__(self, inp: Union[List[str], str]) -> Doc:
        """Call the tokenizer on input `inp`.
        :param inp: either a string to be split on whitespace, or a list of tokens
        :return: the created Doc object
        """
        words = inp.split()
        spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
        return Doc(self.vocab, words=words, spaces=spaces)


if __name__ == "__main__":
    # This is required, see https://datascience.stackexchange.com/a/98195/32669
    # And must be inside if name==main, see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
    mp.set_start_method("spawn", force=True)

    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("indir", help="Path to input directory with text files. These files constitute the corpus"
                                       " that will be parsed and used to calculate frequencies")
    cparser.add_argument("-e", "--ext", default="",
                         help="Only process files with this extension (must include a dot).")
    cparser.add_argument("-o", "--outfile", default="extractor.pckl",
                         help="Path of the output file to save the final object and all frequencies to. If not given,"
                              "  writes to extractor.pckl.")
    cparser.add_argument("-b", "--batch_size", default=64, type=int,
                         help="Mini-batch size to process at a time.Larger = faster but may lead to out"
                              " of memory issues.")
    cparser.add_argument("-n", "--n_workers", default=None, type=int,
                         help="Number of workers to use. If not given, it will be set to 'n_gpus'. If CUDA is not"
                              " available (or n_gpus=0), it will be set to 1. Note that if n_workers > n_gpus,"
                              " multiple processes will use the same GPU - which may lead to out-of-memory issues")
    cparser.add_argument("--n_gpus", type=int, default=0,
                         help="The number of GPUs to use. If n_workers is not specified, will set it to 'n_gpus'. "
                              "If n_workers is specified and it is larger than n_gpus, multiple process will use the"
                              " same GPU. To disable CUDA, set to 0.")
    cparser.add_argument("--is_tokenized", action="store_true", default=False,
                         help="Whether the input data is pretokenized.")
    cparser.add_argument("--min", type=int, default=0,
                         help="Lines with less than 'min' whitespace-separated tokens will not be processed.")
    cparser.add_argument("--max", type=int, default=0,
                         help="Lines with more than 'max' whitespace-separated tokens will not be processed. Set to 0 to disable")
    cparser.add_argument("-v", "--verbose", action="store_true", default=False,
                         help="Print information of matching tokens.")

    cargs = vars(cparser.parse_args())
    coutfile = cargs.pop("outfile")
    extractor = FrequencyExtractor(**cargs)
    extractor.save(coutfile)
