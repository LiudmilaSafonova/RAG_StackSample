import pandas as pd
import os
from download import path

questions = pd.read_csv(os.path.join(path, "Questions.csv"), encoding='latin1')
answers = pd.read_csv(os.path.join(path, "Answers.csv"), encoding='latin1')
tags = pd.read_csv(os.path.join(path, "Tags.csv"), encoding='latin1')

print(questions.head())
