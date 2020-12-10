import pandas as pd
import numpy as np
import pdb

xls = pd.ExcelFile("./final_problems.xls")
sheet = xls.parse()

students = [sheet['First name'][2*i]+' '+sheet['Last name'][2*i] for i in range(int(len(sheet)/2))]
responses = [list(sheet[np.logical_and(sheet['First name']==name.split()[0], sheet['Last name']==name.split()[1])]['Response']) for name in students]

d = dict(zip(students, responses))

def check_answer(indicators, response):
    count = 0
    for indicator in indicators:
        if indicator in response.lower():
            count += 1
    return count/len(indicators), len(response)

q1_indicators = ['no', 'cannot fit', 'smaller', 'perspective', 'simultaneous', 'simultaneity', 'agree']
q2_indicators = ['half', '10^17', '10^6', '10^12']

names = []
scores = []
lengths = []

for key, value in list(d.items()):
    try:
        print(value[0])
        score1, length1 = check_answer(q1_indicators, value[0])
        score2, length2 = check_answer(q2_indicators, value[1])
        names.append(key)
        scores.append((score1, score2))
        lengths.append((length1, length2))
    except:
        names.append(key)
        scores.append((0, 0))
        lengths.append((0, 0))

scores = np.array(scores)/np.max(scores) * 100
lengths = np.array(lengths)/np.max(lengths) * 100

for i, name in enumerate(names):
    print()
    print(' '*15+'      Content   Length')
    print('{}      {:.1f}%      {:.1f}%'.format(name.ljust(15), scores[i][0], lengths[i][0]))
    print('{}      {:.2f}%      {:.1f}%'.format(' '*15, scores[i][1], lengths[i][1]))

