import pandas as pd
import nltk
import numpy as np
import textmining
import pdb

stemmer = nltk.stem.snowball.SnowballStemmer("english")
exclude = ['is', 'be', 'of']
parts = ['NN', 'NNS', 'VB', 'VBZ', 'VBN', 'IN', 'JJ', 'VBP', 'RB']

def read_sheet(filename):
    # Load in the gradesheet
    gradesheet = pd.ExcelFile(filename)
    dataframe = gradesheet.parse('Student Responses')

    # Split the dataframe by row wherever there's a null value
    groups = dataframe.isnull().all(axis=1)
    diff = np.diff(groups.tolist())
    inds = np.where(diff == 1)[0]

    data = dataframe[inds[-2]+2:inds[-1]+1]
    return data

def get_responses(data):
    colnames = data.iloc[0, :].tolist()
    inds = [i for i, item in enumerate(colnames) if item == 'Response']
    responses = [data[data.columns[ind]].tolist()[1:] for ind in inds]
    return responses

def tokenize(response):
    text = nltk.word_tokenize(response)
    tagged = nltk.pos_tag(text)
    nouns_verbs = [text[i] for i, item in enumerate(tagged) if item[1] in parts]
    bases = [stemmer.stem(item) for item in nouns_verbs]
    filtered = list(filter(lambda a: a.lower() not in exclude, bases))
    return filtered

def build_matrix(question):
    tdm = textmining.TermDocumentMatrix()
    for response in question:
       tdm.add_doc(' '.join(tokenize(response)))
    matrix = np.array([row for row in tdm.rows()])
    words = matrix[0]
    weights = matrix[1:].astype(np.int)
    return tdm, weights
    
#def evaluate(matrix):
    

if __name__ == '__main__':
    data = read_sheet('final__astr1210_spring_2021__regular_100_minutes_-1621011014605.xlsx')
    responses = get_responses(data)
    for question in responses:
        build_matrix(question)
