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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

The Bio-ID track focuses on entity tagging and
ID assignment to selected bioentity types, with the aim of
facilitating downstream article curation both at the preand post-publication stages. The task is to annotate text
from figure legends with the entity types and IDs for taxon
(organism), gene, protein, miRNA, small molecules,
cellular components, cell types and cell lines, tissues and
organs. The dataset contains 17,883 (train + test) annotated figure panel captions
with 133,003 entity annotations linked to 10 different source ontologies.

[bigbio_schema_name] = kb
"""

import os
from typing import List, Tuple, Dict
import glob
import json

import datasets
from bioc import biocxml
from collections import defaultdict

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses
from bigbio.utils.parsing import get_texts_and_offsets_from_bioc_ann

_LOCAL = False

_CITATION = """\
"@inproceedings{arighi2017bc6id,
  title={Bio-ID track overview},
  author={Arighi, Cecilia and Hirschman, Lynette and Lemberger, Thomas and Bayer, Samuel and Liechti, Robin and Comeau, Donald and Wu, Cathy},
  booktitle={Proc. BioCreative Workshop},
  volume={482},
  pages={376},
  year={2017}
}"
"""

_DATASETNAME = "bc6id"

_DESCRIPTION = """\
This dataset is a named entity recognition and normalization NLP text dataset.
It contains a total of 133,003 gene/protein, miRNA, small molecule, cellular component, 
cell type, cell line, tissue/organ, and organism/species annotations from 17,883 figure panel 
captions taken from 4,812 figures in 766 full-text biomedical research articles.  
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-1/"

_LICENSE = Licenses.PUBLIC_DOMAIN_MARK_1p0

_URLS = {
    "bc6id":[
        "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz",
        "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining.tar.gz",
    ]
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]

_SOURCE_VERSION = "1.2.0"

_BIGBIO_VERSION = "1.0.0"


class Bc6idDataset(datasets.GeneratorBasedBuilder):
    """BioCreative VI BioID Dataset"""

    DEFAULT_CONFIG_NAME = "bc6id_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc6id_source",
            version=SOURCE_VERSION,
            description="bc6id source schema",
            schema="source",
            subset_id="bc6id",
            # data_dir='/Users/david/Downloads/BioIDtraining_2/'
        ),
        BigBioConfig(
            name="bc6id_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bc6id BigBio schema",
            schema="bigbio_kb",
            subset_id="bc6id",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bc6id_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # Variation of BioC format
            features = datasets.Features(
                {
                    "passages": [
                        {
                            "document_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "offsets": [[datasets.Value("int32")]],
                                    "text": [datasets.Value("string")],
                                    "type": datasets.Value("string"),
                                    "normalized": [
                                        {
                                            "db_name": datasets.Value("string"),
                                            "db_id": datasets.Value("string"),
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        train_dir, test_dir = dl_manager.download_and_extract(urls)

        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        # if self.config.data_dir is None:
        #     raise ValueError(
        #         "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
        #     )
        # else:
        #     data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        train_dir, "BioIDtraining_2/caption_bioc/*.xml"
                        # data_dir,  "caption_bioc/*.xml"
                    ),
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     # TODO: Update when I have the path to the test directory
            #     gen_kwargs={
            #         "filepath": os.path.join(
            #             test_dir, "caption_bioc/*.xml"
            #         ),
            #         "split": "test",
            #     },
            # ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split):
        if self.config.schema == "source":
            for i, data in self._generate_source_examples(filepath, split):
                yield i, data

        elif self.config.schema == "bigbio_kb":
            for i, data in self._generate_bigbio_kb_examples(filepath, split):
                yield i, data

    def _generate_source_examples(self, filepath, split):
        '''
        Generate examples for source schema
        '''
        for file in glob.glob(filepath):
            reader = biocxml.BioCXMLDocumentReader(str(file))
    
            for uid, xdoc in enumerate(reader):
                # doc_text = self._get_document_text(xdoc)

                passages = []
                
                for passage in xdoc.passages:
                    # Sometimes passage type is missing
                    p_type = ''
                    if 'type' in passage.infons:
                        p_type = passage.infons["type"]
                    passages.append({
                            "document_id": xdoc.id,
                            # "type": passage.infons["type"],
                            "text": passage.text,
                            "entities": [
                                x for span in passage.annotations
                                for x in self._get_bioc_entity(span, xdoc.id.split()[0])
                            ],
                        })

                yield uid, {
                    "passages": passages
                }

    def _generate_bigbio_kb_examples(self, filepath, split):
        '''
        Generate examples for BigBio KB schema
        '''
        uid = 0 # global unique id
        doc_id = 0
        failure_count = 0
        for file in glob.glob(filepath):
            reader = biocxml.BioCXMLDocumentReader(str(file))

            for doc in reader:
                pmid = doc.id.split()[0]
            
                data = {
                            "id": uid,
                            "document_id": doc.id,
                            "passages": [],
                            "entities": [],
                            "relations": [],
                            "events": [],
                            "coreferences": [],
                        }
                uid += 1

                char_start = 0
                # passages must not overlap and spans must cover the entire document
                print(len(doc.passages))
                for i, passage in enumerate(doc.passages):
                    offsets = [[char_start, char_start + len(passage.text)]]
                    
                    p_type = ''
                    if 'type' in passage.infons:
                        p_type = passage.infons["type"]
                    data["passages"].append(
                        {
                            "id": uid,
                            "type": p_type,
                            "text": [passage.text],
                            "offsets": offsets,
                        }
                    )
                    uid += 1
                
                    all_text = ' '.join([text for x in data['passages'] for text in x['text']])
                
                # entities
                # for passage in doc.passages:
                    
                    for span in passage.annotations:
                        ents = self._get_bioc_entity(span, pmid)
                        for ent in ents:
                            ent["id"] = uid  # override BioC default id
                            ent['offsets'] = [[x[0] + char_start, x[1] + char_start] for x in ent['offsets']]
                            uid += 1
                        data["entities"].extend(ents)
                        

                        # # Quality Control
                        # start, end = ents[0]['offsets'][0]
                        # ent_text = ' '.join(ent['text'])

                    #     if all_text[start:end] != ent_text:
                    #         if i > 0:
                    #             print("Passage Number:", i)
                    #             print('Span in doc:', all_text[start:end])
                    #             print('Entity span:', ent['text'])
                    #             failure_count += 1
                    #     # assert all_text[start:end] == ent_text

                    # char_start = char_start + len(passage.text) + 1


                yield doc_id, data
                doc_id += 1
                # print(failure_count)


    # def _get_bioc_source_entity(self, ann, pmid):
    #     # TODO: Finish
    #     offsets, text = get_texts_and_offsets_from_bioc_ann(ann)
    #     id_ = ann.infons["sourcedata_article_annot_id"]

    def _get_bioc_entity(self, ann, pmid):
        offsets, text = get_texts_and_offsets_from_bioc_ann(ann)
        id_ = ann.infons["sourcedata_article_annot_id"]

        normalized = defaultdict(list)
        for x in ann.infons["type"].split("|"):

            db_name = x.split(":")[0]
            db_id = x.split(":")[-1]
            if x.startswith("Uniprot"):
                e_type = "protein"
            elif x.startswith("NCBI gene"):
                e_type = "gene"
                db_name = "ncbigene"
            elif x.startswith("Rfam"):
                e_type = "mirna"
            elif x.startswith("CHEBI"):
                e_type = "molecule"
            elif x.startswith("PubChem"):
                e_type = "molecule"
            elif x.startswith("GO"):
                e_type = "subcellular"
            elif x.startswith("CVCL"):
                e_type = "cell"
                db_name = "cell"
            elif x.startswith("CL"):
                e_type = "cell"
            elif x.startswith("Uberon"):
                e_type = "tissue"
            elif x.startswith("NCBI taxon"):
                e_type = "organism"
                db_name = "ncbitaxon"
            elif x.startswith("BAO"):
                e_type = "bioassay"
            elif x.startswith("Corum"):
                e_type = "complex"

            # Handle all typed entities that don't have a CUI
            else:
                e_type = x.split(":")[0]
                db_name = "CUI-less"
                assert len(normalized[e_type]) == 0
                normalized[e_type] = []

            normalized[e_type].append({"db_name": db_name, "db_id": db_id})

        # Create an entity for each unique type annotation
        entity = [
            {
                "id": pmid + ":" + id_,
                "offsets": [list(x) for x in offsets],
                "text": text,
                "type": e_type,
                "normalized": normalized_ents,
            }
            for e_type, normalized_ents in normalized.items()
        ]

        return entity

    



# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python bc6id.py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    data = datasets.load_dataset(__file__, 'bc6id_bigbio_kb')
    print("finished loading")
    print(len([x for x in data['train']]))
    # for i, dat in data['train']:
    #     if i < 10:
    #         print(dat)
