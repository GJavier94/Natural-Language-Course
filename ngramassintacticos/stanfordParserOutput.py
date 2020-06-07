import stanza
import codecs


document_0 = 'Economic news have little effect on financial markets'
document_1 = 'eat with wooden spoon'
document_2 = 'eat with metallic spoon'


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
doc_0 = nlp(document_0)
doc_1 = nlp(document_1)
doc_2 = nlp(document_2)

docs = [doc_0,doc_1,doc_2]

number_doc = 0

for doc in docs:
	cadenas = ""
	for sent in doc.sentences:
		for word in sent.words:
			cadena = ""
			if word.head > 0:
				cadena +=  word.deprel + ' (' +  sent.words[word.head-1].text + "-" + str( word.head  ) + ", " + word.text + "-" + str( word.id) +  ')'
			else:
				cadena +=  word.deprel + ' (' +  "ROOT"+ "-" + str( word.head  ) + ", " + word.text + "-" + str( word.id) +  ')'				
			cadenas += cadena+ '\n'		

	output_file = "input" + str(number_doc) + ".txt"
	file = open(output_file, "w", encoding='utf-8')	
	file.write(cadenas)
	file.close()
	number_doc += 1