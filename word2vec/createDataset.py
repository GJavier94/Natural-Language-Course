import glob

path = r'./input/bbt/transcripts/*'
all_files = glob.glob(path)


f = open("./input/bbt/datasetTBBT.csv", "w")  

for filename in all_files: 
	print(filename)   
	fp = open(filename,'r')
	doc = fp.readlines()
	for dialogue in doc:
		i = 0
		while( i != len(dialogue)):
			if(dialogue[i] == ':'): break
			i+=1
		newdial =  dialogue[i+2:]
		f.write(newdial)
f.close()

#print(doc)
#df = pd.read_csv(filename,sep=':' )
#li.append(df)
#frame = pd.concat(li, axis=0, ignore_index=True, sort = False)
