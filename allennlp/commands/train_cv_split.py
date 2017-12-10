import pandas as pd
from allennlp.commands.config import *


if __name__ == '__main__':
	df = pd.read_json(DATASETS["multinli_0.9_train"]['original'], lines=True)
	df = df.sample(frac=1)
	half_size = int(df.shape[0]/2)
	half_1 = df.head(n=half_size)
	half_2 = df.tail(n=half_size)
	half_1.to_json(os.path.join(DATA_DIR, "mnli_train_half_1.json"), lines=True, orient='records')
	half_2.to_json(os.path.join(DATA_DIR, "mnli_train_half_2.json"), lines=True, orient='records')
