import json
import csv
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

first = True
inputFile = sys.argv[1]
outPut = sys.argv[2]

with open(inputFile, 'r') as f:
    
    with open(outPut, 'w') as o:
        write = csv.writer(o)
        for line in f:
            data = json.loads(line)
            if first:
                header = data.keys()
                write.writerow(header)
                first = False
                
            write.writerow(data.values())