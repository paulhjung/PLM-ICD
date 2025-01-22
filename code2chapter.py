import csv
import re
chapterNumbers = {}
with open('ICD10chapters.csv') as fn:
    docreader = csv.reader(fn)
    headerRow = next(docreader)
    for row in docreader:
        chapNum  = row[0] 
        coderange  = row[1]
        desc = row[2]
        chapterNumbers.update({chapNum:(coderange, desc)})
cn = chapterNumbers
code2chapterSection = {}
with open('ICD10sections.csv') as f:
    dreader = csv.reader(f)
    headerRow = next(dreader)
    for row in dreader:
        coderange  = row[0] 
        name  = row[1] #short description
        name0  = re.sub("\(.*?\)","",name) #remove the parenthetical range
        name1  = name0.rstrip()
        minrange = row[2]
        maxrange = row[3]
        chapter = row[4]
        if minrange == maxrange:
            code2chapterSection.update({coderange:(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
        elif maxrange == "I1A":
            for i in range(10,17):
                code2chapterSection.update({"I"+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
            code2chapterSection.update({"I1A":(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
        elif maxrange == "I5A":
            for i in range(30,53):
                code2chapterSection.update({"I"+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
            code2chapterSection.update({"I5A":(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
        elif maxrange == "J4A":
            for i in range(40,48):
                code2chapterSection.update({"J"+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
            code2chapterSection.update({"J4A":(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
        elif maxrange == "O9A":
            for i in range(94,100):
                code2chapterSection.update({"O"+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
            code2chapterSection.update({"O9A":(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
        else:
            p = int(minrange[-2:])
            q = int(maxrange[-2:])
            for i in range(p,q+1):
                if i < 10:
                    code2chapterSection.update({minrange[0]+"0"+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
                else:
                    code2chapterSection.update({minrange[0]+str(i):(chapter, cn[chapter][0], cn[chapter][1], coderange, name1)})
                

