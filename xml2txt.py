"""Get relevant data from TMX/XML and save as tab-separated file. Meant to be run on TMX files and extracting
linguistic features for both French (src) and Dutch (tgt).
"""

from os import PathLike
from pathlib import Path
from typing import Union

import lxml
from lxml import etree
from lxml.etree import ElementTree, Element
from pandas import DataFrame


def get_feats(token_xml: Element) -> str:
    all_feats = [token_xml.find(el).text for el in ("lemma", "upos", "xpos", "ufeats")]
    all_feats = [f if f else "_" for f in all_feats]
    return "-".join(all_feats)


def process_file(pfin: Union[str, PathLike], text_id: int):
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
