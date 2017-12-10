DATA_DIR=test_run

python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/matched_dev_easy.json
python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/matched_dev_hard.json

python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/mismatched_dev_easy.json
python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/mismatched_dev_hard.json

python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/matched_dev_full.json
python -m allennlp.run evaluate --archive-file ./logs/allenlp_dam_mnli_matched/model.tar.gz --evaluation-data-file ./$DATA_DIR/mismatched_dev_full.json
