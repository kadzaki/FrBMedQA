import ujson as json
import re
import sys

jsonOut = []

file = open("../data/out.json", "r", encoding="utf8")
data = json.load(file)

# Entity IDs generation strategy (a or b)
id_generation_strategy = 'b'

entities_list_strategy_b = {}

id_index = 0

for p in data:
    entities = []
    entities_list = []
    
    if id_generation_strategy == 'a':
        id_index = 0
    
    s = str( p["passage"] )
    
    # Annotate
    for annotation in p["annotations"]:
        # Discard entities already treated
        if annotation["prefLabel"] in entities:
            continue
        
        entities.append(annotation["prefLabel"])
        
        if id_generation_strategy == 'a':
            entity_token_name = "@entity" + str(id_index)
            id_index += 1
        else:
            if annotation["prefLabel"] in entities_list_strategy_b:
                entity_token_name = entities_list_strategy_b[annotation["prefLabel"]]
            else:
                entity_token_name = "@entity" + str(id_index)
                entities_list_strategy_b[annotation["prefLabel"]] = entity_token_name
                id_index += 1
        
        # Replace named entity with @entity (case insensitive)
        pattern = re.compile(r'\S*' + annotation["prefLabel"] + '\S*', re.IGNORECASE)
        s = pattern.sub(entity_token_name, s)
        
        entities_list.append( { entity_token_name: annotation["prefLabel"] } )
            
    if len(entities_list) > 1:
        # Determine question and answer
        sentences = s.split(". ")
        
        # Split to have more samples, in the same time to normalize the length of the passage with the average, and acomodate to BERT size limit
        if len(sentences) > 1:
            if len(sentences) >= 6:
                for i in range(0, len(sentences), 3):
                    for sent in sentences[i:i + 3]:
                        rest_of_text = s.replace(sent, "")
                        matches = re.findall("@entity[0-9]+", sent)
                        question = ''
                
                        for matche in matches:
                            if rest_of_text.lower().find(matche) != -1:
                                question = sent.replace(matche, "@placeholder", 1).strip()
                
                                # Bug fix
                                if question.find('@placeholder') == -1:
                                    continue
                                
                                if rest_of_text.startswith('. '):
                                    rest_of_text = rest_of_text[2:]
                                        
                                if len(rest_of_text.split()) <= 18 or len(question.split()) < 5:
                                    continue
                                    
                                # Generate entry
                                jsonOut.append( { "passage": rest_of_text, "question": question, "entities_list": entities_list, "answer": matche } )
            
                        # The question must not exists in another context (otherwise the model will see the correct answer during training)
                        if question != '':
                            break
            else:
                for sent in sentences:
                    rest_of_text = s.replace(sent, "")
                    matches = re.findall("@entity[0-9]+", sent)
                    question = ''
            
                    for matche in matches:
                        if rest_of_text.lower().find(matche) != -1:
                            question = sent.replace(matche, "@placeholder", 1).strip()
            
                            # Bug fix
                            if question.find('@placeholder') == -1:
                                continue
                            
                            if rest_of_text.startswith('. '):
                                rest_of_text = rest_of_text[2:]
                                    
                            if len(rest_of_text.split()) <= 18 or len(question.split()) < 5:
                                continue
                            
                            # Generate entry
                            jsonOut.append( { "passage": rest_of_text, "question": question, "entities_list": entities_list, "answer": matche } )
        
                    # The question must not exists in another context (otherwise the model will see the correct answer during training)
                    if question != '':
                        break

with open('../data/final_' + id_generation_strategy + '_toto.json', 'w', encoding="utf8") as outfile:
    json.dump(jsonOut, outfile)
