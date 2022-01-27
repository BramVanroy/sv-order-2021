"""Get relevant data from TMX/XML and save as tab-separated file. Meant to be run on TMX files and extracting
linguistic features for both French (src) and Dutch (tgt). It was used to extract linguistic features from a
pre-parsed XML/TMX corpus.
"""

from os import PathLike
from pathlib import Path
from typing import Generator, Optional, Union

import lxml
from lxml import etree
from lxml.etree import ElementTree, Element
from pandas import DataFrame


def get_feats(token_xml: Element) -> str:
    """Get lemma, pos, xpos, ufeats of a given element
    :param token_xml: token XML element
    :return: the properties, joined as a string
    """
    all_feats = [token_xml.find(el).text for el in ("lemma", "upos", "xpos", "ufeats")]
    all_feats = [f if f else "_" for f in all_feats]
    return "-".join(all_feats)


def process_file(pfin: Union[str, PathLike], text_id: int) -> Optional[Generator]:
    """For a given XML/TMX file, yield for each segment the file name without .prep, text ID, segment ID,
    src and tgt texts and the source and target linguistic features
    :param pfin: input file
    :param text_id: this text's ID
    :return: a generator yielding the required information, or None when the file is not valid XML
    """
    nsmap = {"xml": "http://www.w3.org/XML/1998/namespace"}
    try:
        tree: ElementTree = etree.parse(str(pfin))
        # Only select those TUs that have a prop-element. First TU seems like noise - exclude it this way
        tus = tree.findall("//tu[prop]")

        for tu_id, tu in enumerate(tus, 1):
            src = tu.find("./tuv[@xml:lang='fr']/seg", namespaces=nsmap).text
            tgt = tu.find("./tuv[@xml:lang='nl']/seg", namespaces=nsmap).text
            src_feats = " ".join([get_feats(token_xml)
                                  for token_xml in tu.findall("./tuv[@xml:lang='fr']/stanza//token",
                                                              namespaces=nsmap)])
            tgt_feats = " ".join([get_feats(token_xml)
                                  for token_xml in tu.findall("./tuv[@xml:lang='nl']/stanza//token",
                                                              namespaces=nsmap)])
            yield pfin.stem.replace(".prep", ""), text_id, tu_id, src, tgt, src_feats, tgt_feats
    except lxml.etree.XMLSyntaxError:
        # Occurs when error parsing
        return None


def process(indir: Union[str, PathLike], outfile: Union[str, PathLike, None] = None, ext: str = ".tmx"):
    """Process all files with a given extension in a given input directory and write the results as a tab-separated
    file to an outputfile
    :param indir: the input dir
    :param outfile: the output file to write results to. Writes to "data/lingfeats_output.txt" if not given
    :param ext: only process files with this extension
    """
    if outfile is not None and not outfile.endswith(".txt"):
        raise ValueError("'outfile' must end with .txt")
    pdin = Path(indir).resolve()
    pfout = Path(outfile).resolve() if outfile else Path("data/lingfeats_output.txt")
    pfout.mkdir(parents=True, exist_ok=True)
    files = list(pdin.glob(f"*{ext}"))

    data = [tpl for file_id, pfin in enumerate(files, 1) for tpl in process_file(pfin, file_id) if tpl]
    df = DataFrame(data, columns=["file_name", "text_id", "seg_id", "src", "tgt", "src_feats", "tgt_feats"])
    df.to_csv(pfout, encoding="utf-8", sep="\t", index=False)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("indir", help="Path to input directory with XML/TMX files.")
    cparser.add_argument("-o", "--outfile",
                         help="Path of the output file, must end with .txt. If not given, writes to lingfeats_output.txt.")
    cparser.add_argument("-e", "--ext", default=".tmx",
                         help="Only process files with this extension.")

    cargs = cparser.parse_args()
    process(cargs.indir, cargs.outfile, cargs.ext)
