import csv

def trovaParolacce(frase):
  cnt = 0
  c = 0
  spamReader = csv.reader(open('parolacce.csv', newline=''), delimiter=';', quotechar='|')
  for row in spamReader:
    if " "+row[0] in frase:
      cnt = cnt + 1

  b = str(cnt)
  return b.split()

train = []
with open("haspeede_FB-trainClean.csv", encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter=",", quotechar='"')
    for row in rd:
    	try:
    		row = row + trovaParolacce(row[1])
    		train.append(row)
    	except:
    		train.append(row)

with open("haspeede_FB-train1Clean1.csv","w",newline="",encoding="utf8") as f:  
    cw = csv.writer(f)
    cw.writerows(r+[""] for r in train)