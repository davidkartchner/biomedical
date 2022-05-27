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

"""
A dataset loader for the Plant-Disease dataset.

The corpus of plant-disease relation annotated plants and diseases and their relation to PubMed abstract.
It contains annotations for NCBI Taxonomy ID and MEDIC ID for plant and disease mention respectively.
In total, it has 1,307 relations from 199 abstracts, where the numbers of annotated plants and diseases were 1,403
and 1,758, respectively.
"""

import itertools
from pathlib import Path
from typing import Dict, Iterable
import os
import datasets

from bigbio.utils import parsing
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{kim2019corpus,
  title={A corpus of plant--disease relations in the biomedical domain},
  author={Kim, Baeksoo and Choi, Wonjun and Lee, Hyunju},
  journal={PLoS One},
  volume={14},
  number={8},
  pages={e0221582},
  year={2019},
  publisher={Public Library of Science San Francisco, CA USA}
}
"""

_DATASETNAME = "plant_disease"

_DESCRIPTION = """\
The corpus of plant-disease relation annotated plants and diseases and their relation to PubMed abstract.
It contains annotations for NCBI Taxonomy ID and MEDIC ID for plant and disease mention respectively.
In total, it has 1,307 relations from 199 abstracts, where the numbers of annotated plants and diseases were 1,403
and 1,758, respectively.
"""

_HOMEPAGE = "http://gcancer.org/pdr/"

_LICENSE = ""

_URLS = {
    'plant_disease_source': "http://gcancer.org/pdr/Plant-Disease_Corpus.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class PlantDisease(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="plant_disease_source",
            version=SOURCE_VERSION,
            description="Plant-disease source schema",
            schema="source",
            subset_id="plant_disease",
        ),
        BigBioConfig(
            name="plant_disease_bigbio_kb",
            version=SOURCE_VERSION,
            description="plant_disease BigBio schema",
            schema="bigbio_kb",
            subset_id="plant_disease",
        ),
    ]

    DEFAULT_CONFIG_NAME = "plant_disease_source"

    _ROLE_MAPPING = {
        "Disease": "Disease",
        "Plant": "Plant",
        "Cause_of_disease": "Cause_of_disease",
    }

    def _info(self):
        """
        Provide information about Plant-Disease:
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.

        """
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ],
                    "relations": [  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                        }
                    ],
                },
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # This is not applicable for MLEE.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLS[self.config.name]
        data_dirs = dl_manager.download(my_urls)
        # ensure that data_dirs is always a list of string paths
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": itertools.chain(
                        *[dl_manager.iter_archive(data_dir) for data_dir in data_dirs]
                    ),
                },
            ),
        ]

    def _standardize_arguments_roles(self, kb_example: Dict) -> Dict:

        for event in kb_example["events"]:
            for argument in event["arguments"]:
                role = argument["role"]
                argument["role"] = self._ROLE_MAPPING.get(role, role)

        return kb_example

    def _generate_examples(self, data_files: Iterable[Path]):
        """
        Yield one `(guid, example)` pair per abstract in plant_disease.
        The contents of `example` will depend on the chosen configuration.
        """
        if self.config.schema == "source":
            for guid, (filename, txt_file) in enumerate(list(data_files)):

                if not filename.endswith('.ann'):
                    continue

                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, txt_file in enumerate(list(data_files)):

                if not txt_file.name.endswith('.ann2'):
                    continue

                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file)
                )
                example = self._standardize_arguments_roles(example)
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


# TODO: remove
if __name__ == "__main__":
    datasets.load_dataset(__file__)
