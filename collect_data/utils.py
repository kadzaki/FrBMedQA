import mwparserfromhell
import json
import sys

medTerms = json.load( open( '../medTerms_scrapper/data/terms.json' ) )

def process_article(title, text, template = ''):
    """Process a wikipedia article looking for template"""
    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)
    
    # Search through templates for the template
    matches = wikicode.filter_templates(matches = template)
    
    if len(matches) >= 1:
        return wikicode.strip_code()
    
    title_words = title.lower().split()
    
    if len( list( set(title_words) & set(medTerms) ) ):
        print(list( set(title_words) & set(medTerms) ))
        return wikicode.strip_code()
