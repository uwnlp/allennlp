DATA_DIR = "/home/sg01/allennlp/final_data"
MODEL_DIR = "/home/sg01/allennlp/final_logs"

MATCHED_TEST = {
	"original": "/home/sg01/real_nli/data/multinli_1.0/multinli_0.9_test_matched_unlabeled.jsonl", 
	"prediction_csv": "{}/matched_predictions.csv".format(DATA_DIR)
}

MISMATCHED_TEST = {
	"original": "/home/sg01/real_nli/data/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl", 
	"prediction_csv": "{}/mismatched_predictions.csv".format(DATA_DIR)
}


MATCHED_DEV = {
	"original": "/home/sg01/real_nli/data/multinli_0.9/multinli_0.9_dev_matched.jsonl", 
	"easy": "{}/matched_dev_easy.jsonl".format(DATA_DIR),
	"hard": "{}/matched_dev_hard.jsonl".format(DATA_DIR),
}

MISMATCHED_DEV = {
	"original": "/home/sg01/real_nli/data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl", 
	"easy": "{}/mismatched_dev_easy.jsonl".format(DATA_DIR),
	"hard": "{}/mismatched_dev_hard.jsonl".format(DATA_DIR),
}

MNLI_TRAIN = {
	"original": "/home/sg01/real_nli/data/multinli_0.9/multinli_0.9_train.jsonl", 
}

SNLI_TRAIN = {
	"original": "/home/sg01/real_nli/data/snli_1.0/snli_1.0_train.jsonl", 
}

MNLI_TRAIN_HALF_1 = {
	"original": "{}/mnli_train_half_1.jsonl".format(DATA_DIR),
	"easy": "{}/mnli_train_half_1_easy.jsonl".format(DATA_DIR),
	"hard": "{}/mnli_train_half_1_hard.jsonl".format(DATA_DIR),
}

MNLI_TRAIN_HALF_2 = {
	"original": "{}/mnli_train_half_2.jsonl".format(DATA_DIR),
	"easy": "{}/mnli_train_half_2_easy.jsonl".format(DATA_DIR),
	"hard": "{}/mnli_train_half_2_hard.jsonl".format(DATA_DIR),
}


SNLI_TRAIN_HALF_1 = {
	"original": "{}/snli_train_half_1.jsonl".format(DATA_DIR),
	"easy": "{}/snli_train_half_1_easy.jsonl".format(DATA_DIR),
	"hard": "{}/snli_train_half_1_hard.jsonl".format(DATA_DIR),
}

SNLI_TRAIN_HALF_2 = {
	"original": "{}/snli_train_half_2.jsonl".format(DATA_DIR),
	"easy": "{}/snli_train_half_2_easy.jsonl".format(DATA_DIR),
	"hard": "{}/snli_train_half_2_hard.jsonl".format(DATA_DIR),
}

SNLI_DEV = {
	"original": "/home/sg01/real_nli/data/snli_1.0/snli_1.0_dev.jsonl", 
	"easy": "{}/snli_dev_easy.jsonl".format(DATA_DIR),
	"hard": "{}/snli_dev_hard.jsonl".format(DATA_DIR),
}


SNLI_TEST = {
	"original": "/home/sg01/real_nli/data/snli_1.0/snli_1.0_test.jsonl", 
	"easy": "{}/snli_test_easy.jsonl".format(DATA_DIR),
	"hard": "{}/snli_test_hard.jsonl".format(DATA_DIR),
}

DATASETS = {"multinli_0.9_test_matched_unlabeled": MATCHED_TEST,
		    "multinli_0.9_test_mismatched_unlabeled": MISMATCHED_TEST,
		    "multinli_0.9_dev_matched": MATCHED_DEV,
		    "multinli_0.9_dev_mismatched": MISMATCHED_DEV,
		    "multinli_0.9_train": MNLI_TRAIN,
		    "snli_1.0_test": SNLI_TEST,
		    "snli_1.0_dev": SNLI_DEV,
		    "snli_1.0_train": SNLI_TRAIN,
		    "mnli_train_half_1": MNLI_TRAIN_HALF_1,
		    "mnli_train_half_2": MNLI_TRAIN_HALF_2,
		    "snli_train_half_1": SNLI_TRAIN_HALF_1,
		    "snli_train_half_2": SNLI_TRAIN_HALF_2
		    }



C1_MNLI = {
	"model_name": "c1-mnli",
	"archive_file": "{}/c1-mnli/model.tar.gz".format(MODEL_DIR),
	"evaluation": [MATCHED_DEV,
				   MATCHED_TEST,
				   MISMATCHED_DEV,
				   MISMATCHED_TEST]
}

C1_SNLI = {
	"model_name": "c1-snli",
	"archive_file": "{}/c1-snli/model.tar.gz".format(MODEL_DIR),
	"evaluation": [SNLI_DEV,
				   SNLI_TEST]
}

C1_MNLI_HALF_1 = {
	"model_name": "c1-mnli-half-1",
	"archive_file": "{}/c1-mnli-half-1/model.tar.gz".format(MODEL_DIR),
	"evaluation": [MNLI_TRAIN_HALF_2]
}

C1_MNLI_HALF_2 = {
	"model_name": "c1-mnli-half-2",
	"archive_file": "{}/c1-mnli-half-2/model.tar.gz".format(MODEL_DIR),
	"evaluation": [MNLI_TRAIN_HALF_1]
}

C1_SNLI_HALF_1 = {
	"model_name": "c1-snli-half-1",
	"archive_file": "{}/c1-snli-half-1/model.tar.gz".format(MODEL_DIR),
	"evaluation": [SNLI_TRAIN_HALF_2]
}

C1_SNLI_HALF_2 = {
	"model_name": "c1-snli-half-2",
	"archive_file": "{}/c1-snli-half-2/model.tar.gz".format(MODEL_DIR),
	"evaluation": [SNLI_TRAIN_HALF_1]
}

MODELS = {"c1-mnli": C1_MNLI,
		  "c1-snli": C1_SNLI,
		  "c1-mnli-half-1": C1_MNLI_HALF_1,
		  "c1-mnli-half-2": C1_MNLI_HALF_2,
		  "c1-snli-half-1": C1_SNLI_HALF_1,
		  "c1-snli-half-2": C1_SNLI_HALF_2}