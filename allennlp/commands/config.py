DATA_DIR = "fresh_directory"


MATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_matched_unlabeled.jsonl", 
	"prediction_csv": "./{}/matched_predictions.csv".format(DATA_DIR)
}

MISMATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl", 
	"prediction_csv": "./{}/mismatched_predictions.csv".format(DATA_DIR)
}


MATCHED_DEV = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_dev_matched.jsonl", 
	"full": "./{}/matched_dev_full.json".format(DATA_DIR),
	"easy": "./{}/matched_dev_easy.json".format(DATA_DIR),
	"hard": "./{}/matched_dev_hard.json".format(DATA_DIR),
}

MISMATCHED_DEV = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl", 
	"full": "./{}/mismatched_dev_full.json".format(DATA_DIR),
	"easy": "./{}/mismatched_dev_easy.json".format(DATA_DIR),
	"hard": "./{}/mismatched_dev_hard.json".format(DATA_DIR),
}

MNLI_TRAIN = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_train.jsonl", 
	"full": "./{}/mnli_train_full.json".format(DATA_DIR),
}

SNLI_TRAIN = {
	"original": "~/real_nli/data/snli_1.0/snli_1.0_train.jsonl", 
	"full": "./{}/snli_train_full.json".format(DATA_DIR),
}

MNLI_TRAIN_HALF_1 = {
	"full": "./{}/mnli_train_half_1.json".format(DATA_DIR),
	"easy": "./{}/mnli_train_half_1_easy.json".format(DATA_DIR),
	"hard": "./{}/mnli_train_half_1_hard.json".format(DATA_DIR),
}

MNLI_TRAIN_HALF_2 = {
	"full": "./{}/mnli_train_half_2.json".format(DATA_DIR),
	"easy": "./{}/mnli_train_half_2_easy.json".format(DATA_DIR),
	"hard": "./{}/mnli_train_half_2_hard.json".format(DATA_DIR),
}


SNLI_TRAIN_HALF_1 = {
	"full": "./{}/snli_train_half_1.json".format(DATA_DIR),
	"easy": "./{}/snli_train_half_1_easy.json".format(DATA_DIR),
	"hard": "./{}/snli_train_half_1_hard.json".format(DATA_DIR),
}

SNLI_TRAIN_HALF_2 = {
	"full": "./{}/snli_train_half_2.json".format(DATA_DIR),
	"easy": "./{}/snli_train_half_2_easy.json".format(DATA_DIR),
	"hard": "./{}/snli_train_half_2_hard.json".format(DATA_DIR),
}


SNLI_TEST = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_test.jsonl", 
	"full": "./{}/snli_test_full.json".format(DATA_DIR),
	"easy": "./{}/snli_test_easy.json".format(DATA_DIR),
	"hard": "./{}/snli_test_hard.json".format(DATA_DIR),
}

DATASETS = {"multinli_0.9_test_matched_unlabeled": MATCHED_TEST,
		    "multinli_0.9_test_mismatched_unlabeled": MISMATCHED_TEST,
		    "multinli_0.9_dev_matched": MATCHED_DEV,
		    "multinli_0.9_dev_mismatched": MISMATCHED_DEV,
		    "multinli_0.9_train": MNLI_TRAIN,
		    "snli_1.0_test": SNLI_TEST,
		    "snli_1.0_train": SNLI_TRAIN,
		    "mnli_train_half_1": MNLI_TRAIN_HALF_1,
		    "mnli_train_half_2": MNLI_TRAIN_HALF_2,
		    "snli_train_half_1": SNLI_TRAIN_HALF_1,
		    "snli_train_half_2": SNLI_TRAIN_HALF_2
		    }



MNLI_C1 = {"archive_file": "./final_logs/c1-mnli/model.tar.gz"}
MODELS = {"c1-mnli": MNLI_C1}