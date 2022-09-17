"""Extract sentences from a TMX file."""

from os import PathLike
from pathlib import Path
from typing import Generator, Optional, Union

from xml import etree
from xml.etree import ElementTree


def process_file(pfin: Path, src_lang: str, tgt_lang: str, split: bool) -> Optional[Generator]:
    nsmap = {"xml": "http://www.w3.org/XML/1998/namespace"}
    src_texts = []
    tgt_texts = []

    try:
        tree: ElementTree = ElementTree.parse(str(pfin))
    except ElementTree.ParseError:
        # Occurs when error parsing
        return None

    # Only select those TUs that have a prop-element. First TU seems like noise - exclude it this way
    tus = tree.findall(".//tu")

    for tu_id, tu in enumerate(tus, 1):
        src_text = tu.find(f"./tuv[@xml:lang='{src_lang}']/seg", namespaces=nsmap).text or ""
        tgt_text = tu.find(f"./tuv[@xml:lang='{tgt_lang}']/seg", namespaces=nsmap).text or ""

        if not src_text and not tgt_text:
            continue
        src_texts.append(src_text)
        tgt_texts.append(tgt_text)

    data = zip(src_texts, tgt_texts)
    if split:
        with pfin.with_suffix(f".{src_lang}").open("w", encoding="utf-8") as fhsrc, \
                pfin.with_suffix(f".{tgt_lang}").open("w", encoding="utf-8") as fhtgt:
            for src, tgt in data:
                fhsrc.write(f"{src}\n")
                fhtgt.write(f"{tgt}\n")
    else:
        with pfin.with_suffix(".txt").open("w", encoding="utf-8") as fh:
            for src, tgt in data:
                fh.write(f"{src}\n{tgt}\n\n")


def process(indir: Union[str, PathLike], src_lang: str, tgt_lang: str, split: bool = False):
    """Process all files with a given extension in a given input directory
    :param indir: the input dir
    :param src_lang: the source language as language code
    :param tgt_lang: the target language as language code
    :param split: whether or not to save the text as separate files for src/tgt or not
    """

    for pfin in Path(indir).glob("*.tmx"):
        process_file(pfin, src_lang, tgt_lang, split)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("indir", help="Path to input directory with TMX files.")
    cparser.add_argument("src_lang",
                         help="Source language code in 'tuv xml:lang' attribute of the source segment")
    cparser.add_argument("tgt_lang",
                         help="Target language code in 'tuv xml:lang' attribute of the target segment")
    cparser.add_argument("--split", action="store_true",
                         help="By default, only one file is created with the sentence pairs separated by a new line."
                              " With this flag enabled, two files are created per input file - one for source,"
                              " one for target.")
    cargs = cparser.parse_args()
    process(cargs.indir, cargs.src_lang, cargs.tgt_lang, cargs.split)
