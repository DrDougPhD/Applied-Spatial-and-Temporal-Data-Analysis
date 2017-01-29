import sys
from lxml import objectify

article_xml = objectify.parse(sys.argv[-1])
article_root = article_xml.getroot()
print('Title: {}'.format(article_root.TITLE))
print('-'*80)
print('Abstract:')
print(article_root.ABSTRACT)
print('-'*80)
print('Text:')
print(article_root.TEXT)