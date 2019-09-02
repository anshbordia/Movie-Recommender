# Movie Recommender System
# Author: Ansh Bordia (ansh979@gmail.com)
#------------------------------------------------------
# Importing Libraries
import json
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#------------------------------------------------------
# Helper Functions

def get_title(index):
    return movie_df[movie_df.index == index]["title"].values[0]

def get_index(title):
    return movie_df[movie_df.title == title]["index"].values[0]

# Combining the features chosen
def get_features(row):
    return row['title'] + " " + list_to_string(row['GenreList']) + " " + process(row['overview'])
    
# Getting String represenation of Genres from List format    
def list_to_string(x):
    if(len(x) == 0):
        return ""
    string = x[0]
    for i in range(1, len(x)):
        string + " " + x[i]
    return string

#---------------------------------------------------------
# Reading the dataset
# Only first 5000 rows
movie_df = pd.read_csv("movies_metadata.csv", usecols = [3,5,9,16,20], nrows = 5000)
#---------------------------------------------------------
# Preprocessing

#Adding extra columns 
movie_df['GenreList'] = np.empty((len(movie_df), 0)).tolist()
movie_df['index'] = 0

# JSON Parsing
for i in range(0, len(movie_df)):
    temp_list = []
    temp = json.loads(re.sub('\'', '\"', movie_df.iloc[i][0]))
    for genre in temp:
        temp_list.append(genre["name"])
    movie_df.at[i,'GenreList'] = temp_list
    movie_df.at[i, 'index'] = i

# Stop Word Removal and Lemmatization
lm = WordNetLemmatizer()
def process(s):
    overview = re.sub('[^a-zA-Z]', ' ', s)
    overview = overview.lower()
    overview = overview.split()
    overview = [lm.lemmatize(word, pos = "v") for word in overview if not word in set(stopwords.words('english'))]
    overview = ' '.join(overview)
    return overview

# Drop Null rows with null values
movie_df = movie_df.dropna()
#---------------------------------------------------------
# Feature Selection and Engineering
feature_set = ['overview', 'title', 'GenreList']
movie_df['features'] = movie_df.apply(get_features, axis = 1)      

# Bag of Words 
cv = CountVectorizer()
X = cv.fit_transform(movie_df['features'])

# Cosine Similarity
similarity = cosine_similarity(X)

#----------------------------------------------------------
# User Input
notOver = True
while(notOver):
    user_movie = input("Enter the movie for which you want recommendations: ")

# Generate Recommendations
    recommendations = sorted(list(enumerate(similarity[get_index(user_movie)])), key = lambda x:x[1], reverse = True)
    print("The top 3 recommendations for" + " " + user_movie + " " + "are: ")
    print(get_title(recommendations[1][0]), get_title(recommendations[2][0]), get_title(recommendations[3][0]), sep = "\n")
    decision = input("Press 1 to enter another movie, 0 to exit")
    if(int(decision) == 0):
        print("Bye")
        notOver = False 
#--------------------------End of Program----------------




