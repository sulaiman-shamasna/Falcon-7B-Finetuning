import json
import pandas as pd


with open('data/train-v2.0.json', 'r') as file:
    squad_data = json.load(file)

questions = []
answers = []

total_questions = sum(len(paragraph['qas']) for article in squad_data['data'] for paragraph in article['paragraphs'])

target_questions = int(0.0001 * total_questions)
collected_questions = 0

for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if collected_questions >= target_questions:
                break
            question = qa['question']
            if 'answers' in qa and len(qa['answers']) > 0:
                answer = qa['answers'][0]['text']
                questions.append(question)
                answers.append(answer)
                collected_questions += 1
        if collected_questions >= target_questions:
            break
    if collected_questions >= target_questions:
        break

df_squad = pd.DataFrame({
    'question': questions,
    'answer': answers
})

print(df_squad.head())
print(len(df_squad))

# df_faq = pd.DataFrame({
#     'question': ['What is AI?', 'How to train a model?'],
#     'answer': ['AI is the simulation of human intelligence in machines.', 'Training a model involves feeding it data and adjusting its parameters.']
# })

# print(df_faq.head())
