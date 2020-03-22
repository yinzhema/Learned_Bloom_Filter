"""Bloom Filters' helper class: get number of element to be input and a list of all the lines"""
class Helpers:

    """Initialization"""
    def __init__(self):
        self.title=''
        self.word_count=0
        self.words=[]

    """Read the input text, set the title, calculate the word counts, and store the list of all lines """
    def read_txt(self, title):
        wordCount=0
        file=open(title,"r")
        for line in file:
            wordCount+=len(line.split())
            self.words.append(line)
        self.word_count=wordCount
        self.title=title
