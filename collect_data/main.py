import xml.sax as sax
from WikiXmlHandler import WikiXmlHandler
from timeit import default_timer as timer
import json
import math

handler = WikiXmlHandler()

parser = sax.make_parser()
parser.setContentHandler(handler)

xml = open('../frwiki-20210601-pages-articles-multistream.xml', 'r', encoding="utf-8")

start = timer()

cpt = 1

for line in xml:
    if cpt % 10000 == 0:
        print("Processed: " + str( handler._article_count ) + " ==> " + str(math.floor(handler._article_count / 5012008 * 100)) + "%" ) # Show progresss
        print("Selected: " + str(len(handler._pages)))
        print('---------------------------------------------')
        
        if len(handler._pages) % 1000 == 0:
            with open('../articles_v2.ndjson', 'wt', encoding="utf-8") as fOut:
                for l in handler._pages:
                    fOut.write(json.dumps(l, ensure_ascii=False) + '\n')
    try:
        parser.feed(line)
    except StopIteration:
        break
    
    cpt += 1
    
end = timer()

xml.close()

pages = handler._pages

print(f'\nSearched through {handler._article_count} french articles.')
print(f'\nFound {len(pages)} biomedical articles in {round(end - start)} seconds.')

# Save list of found articles
with open('../articles_v2.ndjson', 'wt', encoding="utf-8") as fOut:
    for l in pages:
        fOut.write(json.dumps(l, ensure_ascii=False) + '\n')
