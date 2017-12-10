DATA_DIR = "test_run"


MATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_matched_unlabeled.jsonl", 
}

MISMATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl", 
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

DATASETS = {"matched_test": MATCHED_TEST,
		    "mismatched_test": MISMATCHED_TEST,
		    "matched_dev": MATCHED_DEV,
		    "mismatched_dev": MISMATCHED_DEV,
		    "mnli_train": MNLI_TRAIN}

MNLI_C1 = {"archive_file": "./final_logs/c1-mnli/model.tar.gz"}
MODELS = {"c1-mnli": MNLI_C1}