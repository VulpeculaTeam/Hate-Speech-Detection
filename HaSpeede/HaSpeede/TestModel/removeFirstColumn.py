import csv

commenti = []

with open("haspeede_FB-testClean.csv", encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter=",", quotechar='"')

    for row in rd:
    	row.pop(0)
    	commenti.append(row)

with open("haspeede_FB-testCleanPredict.csv","w",newline="",encoding="utf8") as f:  
    cw = csv.writer(f)
    cw.writerows(r for r in commenti)