import ujson as json
import sys
import re
import util
import random

#------------ Baseline 1 (Random) ------------
data = json.load(open("../data/splits/test.json", encoding="utf8"))

cpt = 0

for ins in data:
    matches = re.findall("@entity[0-9]+", ins['passage'])

    if random.choice(matches) == ins['answer']:
        cpt += 1
        
print("Acc: " + str( ( cpt / len(data) ) * 100 ))

#------------ Baseline 2 (most frequent) ------------
data = json.load(open("../data/splits/test.json", encoding="utf8"))

cpt = 0

for ins in data:
    matches = re.findall("@entity[0-9]+", ins['passage'])

    if util.most_frequent(matches) == ins['answer']:
        cpt += 1
        
print("Acc: " + str( ( cpt / len(data) ) * 100 ))

#------------ Baseline 3 (first) ------------
data = json.load(open("../data/splits/test.json", encoding="utf8"))

cpt = 0

for ins in data:
    matches = re.findall("@entity[0-9]+", ins['passage'])

    if matches[0] == ins['answer']:
        cpt += 1
        
print("Acc: " + str( ( cpt / len(data) ) * 100 ))

#------------ Baseline 4 (last) ------------
data = json.load(open("../data/splits/train.json", encoding="utf8"))

cpt = 0

for ins in data:
    matches = re.findall("@entity[0-9]+", ins['passage'])

    if matches[len(matches) - 1] == ins['answer']:
        cpt += 1
        
print("Acc: " + str( ( cpt / len(data) ) * 100 ))

#------------ Baseline 5 ------------
data = json.load(open("../data/splits/test.json", encoding="utf8"))

cpt = 0

for ins in data:
    words = ins['question'].split()
    
    for index, word in enumerate(words):
        if '@placeholder' in word:
            break
    
    if index > 0:
        n_gram_prev = words[index - 1] + " " + words[index]
    else:
        n_gram_prev = words[index]
        
    if index + 1 < len(words):
        n_gram_next = words[index] + " " + words[index + 1]
    else:
        n_gram_next = words[index]
    
    most_occ_entity = ''
    total_occ = 0
    
    for entity in ins['entities_list']:
        key = list(entity)

        new_n_gram_prev = n_gram_prev.replace('@placeholder', key[0])
        new_n_gram_next = n_gram_next.replace('@placeholder', key[0])

        n_gram_prev_occ = ins['passage'].count(new_n_gram_prev)
        n_gram_next_occ = ins['passage'].count(new_n_gram_next)
        
        total = n_gram_prev_occ + n_gram_next_occ

        if total > total_occ:
            total_occ = total
            most_occ_entity = key[0]
            
    if most_occ_entity == ins['answer']:
        cpt += 1
        
print("Acc: " + str( ( cpt / len(data) ) * 100 ))

# ------------------ Human performance -------------------
data = json.load(open("../data/splits/test.json", encoding="utf8"))

random.shuffle(data)

samples = data[:30]

with open('../data/splits/human_test/samples_with_answers.json', 'w') as outfile:
    json.dump(samples, outfile)
    
for sample in samples:
    sample['answer'] = ''
    
    for entity in sample['entities_list']:
        key = list(entity)
        entity[key[0]] = ''
    
with open('../data/splits/human_test/samples_without_answers.json', 'w') as outfile:
    json.dump(samples, outfile)
