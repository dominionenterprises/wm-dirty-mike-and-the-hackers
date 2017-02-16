import json
import csv
import sys 
import string

reload(sys)
sys.setdefaultencoding('utf-8')

first = True
inputFile = sys.argv[1]
outPut = sys.argv[2]
count = 0
numLines = 1000000

def numToStr(num):
	out = ""
	if num == 1:
		out = "one"
	elif num == 2:
		out = "two"
	elif num == 3:
		out = "three"
	elif num == 4:
		out = "four"
	elif num == 5:
		out = "five"
	return out

def remPunc(punc):
	out = "".join(c for c in punc if c not in (string.punctuation))
	return out

with open(inputFile, 'r') as f:
    
    with open(outPut, 'w') as o:
        write = csv.writer(o)
        for line in f:
            data = json.loads(line)
            if first:
                header = data.keys()
                # write.writerow(header)
                # don't write header; it messes up the psql import
                first = False
                continue
            if count <= numLines:    
            	write.writerow([numToStr(data.values()[5]),remPunc(data.values()[3]),data.values()[7],data.values()[9],data.values()[0]])
            	#print(data.values())
            else:
            	sys.exit()
            count+= 1