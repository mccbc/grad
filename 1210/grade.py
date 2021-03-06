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

    data = dataframe[inds[-1]+2:]
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
    weights = np.sum(matrix[1:].astype(np.int), axis=0)
    mask = weights > np.mean(weights) + np.std(weights)
    weights = weights[mask] / np.sum(weights[mask])
    return matrix.T[mask].T, words[mask], weights
    
def evaluate(matrix, weights):
    raw_scores = np.zeros(len(matrix[1:]))
    for i, row in enumerate(matrix[1:].astype(np.int)):
        wordbool = np.array(row > 0).astype(np.int)
        raw_scores[i] = np.sum(wordbool*weights)

    # Build bins
    mean = np.mean(raw_scores[raw_scores != 0])
    std = np.std(raw_scores[raw_scores != 0])
    bins = np.linspace(mean-2*std, mean, 3)
    scores = np.digitize(raw_scores, bins)
    return scores

def effort(question):
    raw_scores = np.zeros(len(question))
    for i, response in enumerate(question):
        raw_scores[i] = len(nltk.word_tokenize(response))

    # Build bins
    mean = np.mean(raw_scores[raw_scores != 0])
    std = np.std(raw_scores[raw_scores != 0])
    bins = np.linspace(3, mean-0.5*std, 2)
    scores = np.digitize(raw_scores, bins)
    return scores

if __name__ == '__main__':
    data = read_sheet('final__astr1210_spring_2021__regular_100_minutes_-1621011014605.xlsx')
    responses = get_responses(data)

    for i, question in enumerate(responses):
        print("\nQUESTION", i)
        matrix, words, weights = build_matrix(question)
        content_scores = evaluate(matrix, weights)
        effort_scores = effort(question)
        for i, response in enumerate(question):
            #print("\nCONTENT SCORE: {}".format(content_scores[i]))
            #print("EFFORT SCORE: {}".format(effort_scores[i]))
            #print(response)
            #print()
            print(content_scores[i]+effort_scores[i])
        input("Press ENTER to continue.")
