DATA_DIR = "test_run"


MATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_matched_unlabeled.jsonl", 
	"tampered": "./{}/matched_test_full_tampered.json".format(DATA_DIR),
	"untampered": "./{}/matched_test_full_untampered.json".format(DATA_DIR),
	"easy_tampered": "./{}/matched_test_easy_tampered.json".format(DATA_DIR),
	"hard_tampered": "./{}/matched_test_hard_tampered.json".format(DATA_DIR),
	"easy_untampered": "./{}/matched_test_easy_untampered.json".format(DATA_DIR),
	"hard_untampered": "./{}/matched_test_hard_untampered.json".format(DATA_DIR)
}

MISMATCHED_TEST = {
	"original": "~/real_nli/data/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl", 
	"tampered": "./{}/mismatched_test_full_tampered.json".format(DATA_DIR),
	"untampered": "./{}/mismatched_test_full_untampered.json".format(DATA_DIR),
	"easy_tampered": "./{}/mismatched_test_easy_tampered.json".format(DATA_DIR),
	"hard_tampered": "./{}/mismatched_test_hard_tampered.json".format(DATA_DIR),
	"easy_untampered": "./{}/mismatched_test_easy_untampered.json".format(DATA_DIR),
	"hard_untampered": "./{}/mismatched_test_hard_untampered.json".format(DATA_DIR)
}


MATCHED_DEV = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_dev_matched.jsonl", 
	"full_tampered": "./{}/matched_dev_full_tampered.json".format(DATA_DIR),
	"full_untampered": "./{}/matched_dev_full_untampered.json".format(DATA_DIR),
	"easy_tampered": "./{}/matched_dev_easy_tampered.json".format(DATA_DIR),
	"hard_tampered": "./{}/matched_dev_hard_tampered.json".format(DATA_DIR),
	"easy_untampered": "./{}/matched_dev_easy_untampered.json".format(DATA_DIR),
	"hard_untampered": "./{}/matched_dev_hard_untampered.json".format(DATA_DIR)
}

MISMATCHED_DEV = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl", 
	"full_tampered": "./{}/mismatched_dev_full_tampered.json".format(DATA_DIR),
	"full_untampered": "./{}/mismatched_dev_full_untampered.json".format(DATA_DIR),
	"easy_tampered": "./{}/mismatched_dev_easy_tampered.json".format(DATA_DIR),
	"hard_tampered": "./{}/mismatched_dev_hard_tampered.json".format(DATA_DIR),
	"easy_untampered": "./{}/mismatched_dev_easy_untampered.json".format(DATA_DIR),
	"hard_untampered": "./{}/mismatched_dev_hard_untampered.json".format(DATA_DIR)
}

MNLI_TRAIN = {
	"original": "~/real_nli/data/multinli_0.9/multinli_0.9_train.jsonl", 
	"full_tampered": "./{}/mnli_train_full_tampered.json".format(DATA_DIR),
	"full_untampered": "./{}/mnli_train_full_untampered.json".format(DATA_DIR),
	"easy_tampered": "./{}/mnli_train_easy_tampered.json".format(DATA_DIR),
	"hard_tampered": "./{}/mnli_train_hard_tampered.json".format(DATA_DIR),
	"easy_untampered": "./{}/mnli_train_easy_untampered.json".format(DATA_DIR),
	"hard_untampered": "./{}/mnli_train_hard_untampered.json".format(DATA_DIR)
}

DATASETS = {"matched_test": MATCHED_TEST,
		    "mismatched_test": MISMATCHED_TEST,
		    "matched_dev": MATCHED_DEV,
		    "mismatched_dev": MISMATCHED_DEV,
		    "mnli_train": MNLI_TRAIN}

MNLI_C1 = {"archive_file": "./final_logs/c1-mnli/model.tar.gz"}
MODELS = {"c1_mnli": MNLI_C1}