import pandas as pd
from tqdm import tqdm
from to_df import questions, answers, tags
from preprocessdf import preprocess

tqdm.pandas()


for table in [(questions, "questions"), (answers, "answers"), (tags, "tags")]:
    print("-" * 30, table[1].upper(), ' ', table[0].info(), sep='\n')

questions['ProcessedBody'] = questions['Body'].apply(preprocess)
print('questions!!!')
answers['ProcessedBody'] = answers['Body'].apply(preprocess)
merged = questions.merge(answers, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))

print(merged.head())
print(merged.info())
merged = merged[['ProcessedBody_answer', 'ProcessedBody_question']]

merged.to_json('PreprocessStackSample.json', orient='records', lines=True, force_ascii=False)
print('success!')
