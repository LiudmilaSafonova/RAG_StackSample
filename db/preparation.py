import pandas as pd

df = pd.read_csv('../PreprocessStackSample.csv')

# dict: question-answer
question_answers = {}
for _, row in df.iterrows():
    qid = row['Id_question']
    if qid not in question_answers:
        question_answers[qid] = {
            'question': row['Body_question'],
            'processed_question': row['ProcessedBody_question'],
            'answers': []
        }
    question_answers[qid]['answers'].append({
        'id': row['Id_answer'],
        'body': row['Body_answer'],
        'processed_body': row['ProcessedBody_answer'],
        'score': row['Score_answer'],
        'owner': row['OwnerUserId_answer'],
        'date': row['CreationDate_answer']
    })

for qid in question_answers:
    question_answers[qid]['answers'].sort(key=lambda x: x['score'], reverse=True)