import ujson as json
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dataset split
data = json.load(open("../data/final_b.json", encoding="utf8"))

random.shuffle(data)

train_data = data[:int((len(data)+1)*.80)]
val_data = data[int((len(data)+1)*.80):int((len(data)+1)*.90)]
test_data = data[int((len(data)+1)*.90):]

with open('../data/splits/train.json', 'w') as outfile:
    json.dump(train_data, outfile)
    
with open('../data/splits/val.json', 'w') as outfile:
    json.dump(val_data, outfile)
    
with open('../data/splits/test.json', 'w') as outfile:
    json.dump(test_data, outfile)

# Number of instances
train_data = json.load(open("../data/splits/train.json", encoding="utf8"))
print( 'Train: ' + str(len(train_data)) )

val_data = json.load(open("../data/splits/val.json", encoding="utf8"))
print( 'Val: ' + str(len(val_data)) )

test_data = json.load(open("../data/splits/test.json", encoding="utf8"))
print( 'Test: ' + str(len(test_data)) )

# Context and question lengths
data = json.load(open("../data/splits/test.json", encoding="utf8"))

Avg_candidates = 0
Max_candidates = 0
Min_candidates = 9000

Avg_context_len = 0
Max_context_len = 0
Min_context_len = 9000

Avg_question_len = 0
Max_question_len = 0
Min_question_len = 9000

for ins in data:
    # ---- Context
    context_tokens = ins["passage"].split()
    
    Avg_context_len += len(context_tokens)
    
    if len(context_tokens) > Max_context_len:
        Max_context_len = len(context_tokens)
        
    if len(context_tokens) < Min_context_len:
        Min_context_len = len(context_tokens)
        
    # ---- Question
    question_tokens = ins["question"].split()
    
    Avg_question_len += len(question_tokens)
    
    if len(question_tokens) > Max_question_len:
        Max_question_len = len(question_tokens)
        
    if len(question_tokens) < Min_question_len:
        Min_question_len = len(question_tokens)
        
    # ---- candidates
    Avg_candidates += len(ins["entities_list"])
    
    if len(ins["entities_list"]) > Max_candidates:
        Max_candidates = len(ins["entities_list"])
        
    if len(ins["entities_list"]) < Min_candidates:
        Min_candidates = len(ins["entities_list"])

print('--------------- Candidates -----------------')
print('Avg candidates: ' + str(Avg_candidates / len(data)) )
print('Max candidates: ' + str(Max_candidates))
print('Min candidates: ' + str(Min_candidates))

print('--------------- Context -----------------')
print('Avg context: ' + str(Avg_context_len / len(data)) )
print('Max context: ' + str(Max_context_len))
print('Min context: ' + str(Min_context_len))

print('--------------- Question -----------------')
print('Avg question: ' + str(Avg_question_len / len(data)) )
print('Max question: ' + str(Max_question_len))
print('Min question: ' + str(Min_question_len))

#----------------- Distribution of the number of candidate entities
data = json.load(open("../data/final_b.json", encoding="utf8"))

nbr_candidate_entities = []

for ins in data:
    nbr_candidate_entities.append( len(ins['entities_list']) )

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.hist(nbr_candidate_entities, bins='auto')
plt.gca().set(title='Number of candidate entities distribution', ylabel='Frequency')

#----------------- Distribution of context length
data = json.load(open("../data/final_b.json", encoding="utf8"))

context_length = []

for ins in data:
    words = ins['passage'].split()
    context_length.append( len(words) )

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.hist(context_length, bins='auto')
plt.gca().set(title='Distribution of context length', ylabel='Frequency');

#----------------- Distribution of question length
data = json.load(open("../data/final_b.json", encoding="utf8"))

question_length = []

for ins in data:
    words = ins['question'].split()
    question_length.append( len(words) )

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.hist(question_length, bins='auto')
plt.gca().set(title='Distribution of question length', ylabel='Frequency');

#----------------- Distribution of biomedical entities by UMLS semantic group 
data = json.load(open("../data/final_b_with_semantic_groups.json", encoding="utf8"))

candidate_entities = {}

for ins in data:
    for entity in ins['entities_list']:
        for sem_group in entity['semantic_groups']:
            if sem_group in candidate_entities:
                candidate_entities[sem_group] += 1
            else:
                candidate_entities[sem_group] = 0
    
candidate_entities = dict(sorted(candidate_entities.items())) # Sort

print(candidate_entities)

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.bar(candidate_entities.keys(), candidate_entities.values())
plt.gca().set(title='Distribution of biomedical entities by UMLS semantic group (all dataset)', ylabel='Frequency')

#----------------- Distribution of answers by UMLS semantic group 
data = json.load(open("../data/final_b_with_semantic_groups.json", encoding="utf8"))

answers = {}

for ins in data:
    for entity in ins['entities_list']:
        key = list(entity)[0]
        
        if key == ins['answer']:
            for sem_group in entity['semantic_groups']:
                if sem_group in answers:
                    answers[sem_group] += 1
                else:
                    answers[sem_group] = 0
    
answers = dict(sorted(answers.items())) # Sort

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.bar(answers.keys(), answers.values())
plt.gca().set(title='Distribution of answers by UMLS semantic group (all dataset)', ylabel='Frequency')
