import ujson as json
import pandas as pd
import re
import AnnotatorWS
import urllib.request, urllib.error, urllib.parse
import sys
import time
import math

REST_URL = "http://services.bioportal.lirmm.fr"

stopWords = ['Maladie', 'Temps', 'Patients', 'Patient', 'Adulte', 'Population', 'Nature', 'Recherche']
semantic_groups_to_discard = [ 'GEOG', 'ACTI', 'OBJC', 'OCCU', 'ORGA', 'CONC', 'LIVB' ] # More infos https://www.sciencedirect.com/science/article/pii/S1532046403001163

params = urllib.parse.urlencode( { 'ontologies': 'MSHFRE' } )

with open("../articles_v2.json", "r", encoding="utf8") as f:
    articles = json.load(f)

jsonOut = []

nbArticles = 0

for article in articles:
    spans = re.split("\.\s+\n", article) # split by paragraphes
    
    nbArticles += 1
    print( str(nbArticles) + "/" + str( len(articles) ) + " ==> " + str(math.floor(nbArticles / len(articles) * 100)) + "%" ) # Show progresss
    
    indx = 1
    
    for span in spans:
        s = re.sub("\s\n|==.+==|===.+===|< ref .+< /ref >|< ref .+/ >|\s\*\s|%|#|\[\[.+\]\]|< br / >|< gallery >.+< /gallery >", "", span) # clean up text
        s = s.strip() # trim white spaces
        
        print("    * Passage index: " + str(indx))
        indx += 1
            
        sentences = s.split(".") # split to sentences
        
        if len(sentences) >= 3:
            try:
                annotations = AnnotatorWS.get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(s) + "&" + params)
            
                annots = []
                fromToIndexes = []
    
                for an in annotations:
                    class_details = an["annotatedClass"]
                    
                    # Discard countries, cities, ...
                    if len(list(set(semantic_groups_to_discard) & set(class_details["semantic_groups"]))) > 0:
                        continue
                    
                    if class_details["prefLabel"] in stopWords:
                        continue
                    
                    for annotation in an["annotations"]:
                        # Contre chauvechaument
                        chauv = False
                        
                        for interval in fromToIndexes:
                            if annotation["from"] >= interval["from"] and annotation["to"] <= interval["to"]:
                                chauv = True
                        
                        if chauv:
                            continue
                        
                        # Discard synonyms
                        if annotation["matchType"] == "SYN":
                            continue
                        
                        fromToIndexes.append({ "from": annotation["from"], "to": annotation["to"] })

                        annot = {}
                        annot["prefLabel"] = class_details["prefLabel"]
                        annot["from"] = annotation["from"]
                        annot["to"] = annotation["to"]
                        annot["match type"] = annotation["matchType"]
                        annot['semantic_groups'] = class_details["semantic_groups"]
                        
                        annots.append(annot)
                
                if len(annots) > 1:
                    jsonOut.append( { "passage": s, "annotations": annots } )
                
            except Exception as e:
                print("---------------------------------------------------")
                print(" - ERROR occured on WS call: " + str(e))
                print("Text: " + s)
                print("---------------------------------------------------")
    
    # Write to file every 100 articles
    if nbArticles % 100 == 0:
        with open('../data/temp.json', 'w') as outfile:
            json.dump(jsonOut, outfile)
    
    #if nbArticles > 5:
    #    break
    
with open('../data/out.json', 'w') as outfile:
    json.dump(jsonOut, outfile)
