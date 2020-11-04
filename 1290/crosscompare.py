import pandas as pd
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', '-i', type=str, help='html file for the weekly blog')
args = parser.parse_args()

df = pd.read_excel('roster.xlsx', index_col=0, header=None)
names = list(df.index[4:])

incomplete = []

with open(args.inputfile) as f:
    lines = f.readlines()
    for name in names[2:]:

        # BEGIN SEARCH
        passed = False

        # Last name match
        if any(name.split(', ')[0].lower() in line.lower() and name.split(', ')[1].split(' ')[0].lower() in line.lower() for line in lines):
            passed = True

        # Print hits
        if passed:
            print("{:<30}".format(name), 1)
        else:
            print("{:<30}".format(name), 0)
            incomplete.append(name)

print('\nINCOMPLETE')
print('==========')
a = [print(name) for name in incomplete]
