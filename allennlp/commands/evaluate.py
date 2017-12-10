"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive-file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation-data-file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any
import argparse
import logging
import pandas as pd
import tqdm
import os
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables
from allennlp.commands.config import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset')

        archive_file = subparser.add_mutually_exclusive_group(required=True)
        archive_file.add_argument('--archive-file', type=str, help='path to an archived trained model')
        archive_file.add_argument('--archive_file', type=str, help=argparse.SUPPRESS)

        evaluation_data_file = subparser.add_mutually_exclusive_group(required=True)
        evaluation_data_file.add_argument('--evaluation-data-file',
                                          type=str,
                                          help='path to the file containing the evaluation data')
        evaluation_data_file.add_argument('--evaluation_data_file',
                                          type=str,
                                          help=argparse.SUPPRESS)

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')
        cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)
        subparser.add_argument('--c1', action='store_true', default=False)
        return subparser


def evaluate(model: Model,
             dataset: Dataset,
             iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    model.eval()

    generator = iterator(dataset, num_epochs=1)
    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset))
    output = pd.DataFrame()
    inverse_label_map = {}

    for raw_batch, batch in generator_tqdm:
        raw_fields = [x.fields for x in raw_batch.instances]
        parsed_fields = []   
        inverse_label_map = {}
        for item in raw_fields:
            real_premise = " ".join([x.text for x in item['real_premise'].tokens]).replace("@@NULL@@", '')
            hypothesis = " ".join([x.text for x in item['hypothesis'].tokens]).replace("@@NULL@@", '')
            hypothesis_binary_parse = item["metadata_hypothesis_binary_parse"].metadata
            premise_binary_parse = item["metadata_premise_binary_parse"].metadata
            hypothesis_parse = item["metadata_hypothesis_parse"].metadata
            premise_parse = item["metadata_premise_parse"].metadata
            if item.get('metadata_pair_id') is not None:
                pair_id = item['metadata_pair_id'].metadata
            else:
                pair_id = None
            if item.get('metadata_genre') is not None:
                genre = item['metadata_genre'].metadata
            else:
                genre = None
            label = item['label'].label
            if item['label']._label_id not in inverse_label_map:
                inverse_label_map.update({item['label']._label_id: item['label'].label})
            parsed_fields.append({"sentence1": real_premise,
                                  "sentence2": hypothesis,
                                  "gold_label": label,
                                  "genre": genre,
                                  "pairID": pair_id,
                                  "sentence1_parse": premise_parse,
                                  "sentence2_parse": hypothesis_parse,
                                  "sentence1_binary_parse": premise_binary_parse,
                                  "sentence2_binary_parse": hypothesis_binary_parse})

        parsed_fields = pd.DataFrame(parsed_fields)
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        tensor_batch.pop('metadata_hypothesis_binary_parse', None)
        tensor_batch.pop('metadata_premise_binary_parse', None)
        tensor_batch.pop('metadata_genre', None)
        tensor_batch.pop('metadata_pair_id', None)
        tensor_batch.pop('metadata_premise_parse', None)
        tensor_batch.pop('metadata_hypothesis_parse', None)
        tensor_batch.pop('real_premise', None)
        bo = model.forward(**tensor_batch)
        metrics = model.get_metrics()
        description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description)
        batch_output = pd.DataFrame()
        batch_output['prediction_label'] = bo['label_logits'].data.numpy().argmax(axis=1)
        batch_output['prediction_score'] = bo['label_probs'].data.numpy().max(axis=1)
        batch_output['prediction_label'] = batch_output.prediction_label.apply(lambda x: inverse_label_map[x])
        parsed_output = pd.concat([parsed_fields, batch_output], axis=1)
        output = pd.concat([output, parsed_output], axis=0)
    hard_subset = output.loc[(output.gold_label != output.prediction_label)
                             | (output.prediction_score <= 0.4)]
    easy_subset = output.loc[(output.gold_label == output.prediction_label)
                             & (output.prediction_score > 0.4)]
    return model.get_metrics(), hard_subset, easy_subset


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path, args.c1)
    dataset.index_instances(model.vocab)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics, hard_subset, easy_subset = evaluate(model, dataset, iterator, args.cuda_device)
    
    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)
    if args.c1 and DATASETS[dataset].get("easy") is not None:
        dataset = args.evaluation_data_file.split('/')[-1][:-6]
        hard_subset.to_json(DATASETS[dataset]["hard"], lines=True, orient='records')
        easy_subset.to_json(DATASETS[dataset]["easy"], lines=True, orient='records')
    return metrics
