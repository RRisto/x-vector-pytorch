from x_vectors.trainer import Trainer, Args
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import plda


trainer=Trainer.load_model('save_model/model_args.txt','save_model/best_check_point_9_0.9911133199930191')
trainer.args.use_gpu=False
trainer.device='cpu'

df=pd.read_csv('meta_et_ru_fi/testing.txt', sep=' ', header=None)
df.columns=['file','class']

eval_logits, eval_x_vecs=trainer.predict(df.file.tolist())
print(eval_logits)