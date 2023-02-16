# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os
import re
from typing import Dict, List, Tuple

import datasets
from bioc import biocxml

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses
from bigbio.utils.parsing import get_texts_and_offsets_from_bioc_ann

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@Article{Wei2015,
author={Wei, Chih-Hsuan and Kao, Hung-Yu and Lu, Zhiyong},
title={GNormPlus: An Integrative Approach for Tagging Genes, Gene Families, and Protein Domains},
journal={BioMed Research International},
year={2015},
month={Aug},
day={25},
publisher={Hindawi Publishing Corporation},
volume={2015},
pages={918710},
issn={2314-6133},
doi={10.1155/2015/918710},
url={https://doi.org/10.1155/2015/918710}
}
"""

_DATASETNAME = "gnormplus"
_DISPLAYNAME = "GNormPlus"

_DESCRIPTION = """\
We re-annotated two existing gene corpora. The BioCreative II GN corpus is a widely used data set for benchmarking GN
tools and includes document-level annotations for a total of 543 articles (281 in its training set; and 262 in test).
The Citation GIA Test Collection was recently created for gene indexing at the NLM and includes 151 PubMed abstracts
with both mention-level and document-level annotations. They are selected because both have a focus on human genes.
For both corpora, we added annotations of gene families and protein domains. For the BioCreative GN corpus, we also
added mention-level gene annotations. As a result, in our new corpus, there are a total of 694 PubMed articles.
PubTator was used as our annotation tool along with BioC formats.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusCorpus.zip"
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class GnormplusDataset(datasets.GeneratorBasedBuilder):
    """Dataset loader for GNormPlus corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="gnormplus_source",
            version=SOURCE_VERSION,
            description="gnormplus source schema",
            schema="source",
            subset_id="gnormplus",
        ),
        BigBioConfig(
            name="gnormplus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="gnormplus BigBio schema",
            schema="bigbio_kb",
            subset_id="gnormplus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "gnormplus_source"

    _re_tax_id = re.compile(r"(?P<db_id>\d+)\([tT]ax:(?P<tax_id>\d+)\)")

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "passages": [
                        {
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "location": {
                                "offset": datasets.Value("int64"),
                                "length": datasets.Value("int64"),
                            },
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                    "tax_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise NotImplementedError(self.config.schema)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": [
                        os.path.join(data_dir, "GNormPlusCorpus/BC2GNtrain.BioC.xml"),
                        # This sub-part of the corpus is part of the GIA Test Collection, however in
                        # the paper they used it only for training their models. So we also add it to the
                        # training split.
                        os.path.join(data_dir, "GNormPlusCorpus/NLMIAT.BioC.xml"),
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": [
                        os.path.join(data_dir, "GNormPlusCorpus/BC2GNtest.BioC.xml"),
                    ]
                },
            ),
        ]

    def _parse_bioc_entity(
        self, uid, bioc_ann, db_id_key="NCBIGene", insert_tax_id=False
    ):
        offsets, texts = get_texts_and_offsets_from_bioc_ann(bioc_ann)
        _type = bioc_ann.infons["type"]

        # parse db ids
        normalized = []
        if _type in bioc_ann.infons:
            for _id in bioc_ann.infons[_type].split(","):
                match = self._re_tax_id.match(_id)
                if match:
                    _id = match.group("db_id")

                n = {"db_name": db_id_key, "db_id": _id}
                if insert_tax_id:
                    n["tax_id"] = match.group("tax_id") if match else None

                normalized.append(n)
        return {
            "id": uid,
            "offsets": offsets,
            "text": texts,
            "type": _type,
            "normalized": normalized,
        }

    def _generate_examples(self, filepaths) -> Tuple[int, Dict]:
        uid = map(str, itertools.count(start=0, step=1))

        for filepath in filepaths:
            with open(filepath, "r") as fp:
                collection = biocxml.load(fp)

                for _, document in enumerate(collection.documents):
                    idx = next(uid)
                    text = " ".join([passage.text for passage in document.passages])

                    insert_tax = self.config.schema == "source"
                    entities = [
                        self._parse_bioc_entity(
                            next(uid), entity, insert_tax_id=insert_tax
                        )
                        for passage in document.passages
                        for entity in passage.annotations
                    ]

                    # Some of the entities have a off-by-one error. Correct these annotations!
                    self.adjust_entity_offsets(text, entities)

                    if self.config.schema == "source":
                        features = {
                            "doc_id": document.id,
                            "passages": [
                                {
                                    "text": passage.text,
                                    "type": passage.infons["type"],
                                    "location": {
                                        "offset": passage.offset,
                                        "length": passage.total_span.length,
                                    },
                                }
                                for passage in document.passages
                            ],
                            "entities": entities,
                        }

                        yield idx, features
                    elif self.config.schema == "bigbio_kb":
                        # passage offsets/lengths do not connect, recalculate them for this schema.
                        passage_spans = []
                        start = 0
                        for passage in document.passages:
                            end = start + len(passage.text)
                            passage_spans.append((start, end))
                            start = end + 1

                        features = {
                            "id": next(uid),
                            "document_id": document.id,
                            "passages": [
                                {
                                    "id": next(uid),
                                    "type": passage.infons["type"],
                                    "text": [passage.text],
                                    "offsets": [span],
                                }
                                for passage, span in zip(
                                    document.passages, passage_spans
                                )
                            ],
                            "entities": entities,
                            "events": [],
                            "coreferences": [],
                            "relations": [],
                        }

                        yield idx, features
                    else:
                        raise NotImplementedError(self.config.schema)

    def adjust_entity_offsets(self, text: str, entities: List[Dict]):
        for entity in entities:
            start, end = entity["offsets"][0]
            entity_mention = entity["text"][0]
            if not text[start:end] == entity_mention:
                if text[start - 1 : end - 1] == entity_mention:
                    entity["offsets"] = [(start - 1, end - 1)]
                elif text[start : end - 1] == entity_mention:
                    entity["offsets"] = [(start, end - 1)]
