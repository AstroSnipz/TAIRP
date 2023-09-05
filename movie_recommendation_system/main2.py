# importing modules
import numpy as np
import pandas as pd  
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# data collection and PreProcessing
movies_data = pd.read_csv("movie_recommendation_system/movies.csv")
#print(movies_data.head())

# number of rows and columns in the dataframe
#print(movies_data.shape)

# selecting the relevant features for recommandation
selected_features = ['genres','keywords','tagline','cast','director']
#print(selected_features)

# replacing the null values with null string    Note-> textual data may contain lot of missing values called as 'null-values' and we need to replace this with null string.
for features in selected_features:
    movies_data[features] = movies_data[features].fillna('')

# combining all the five selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
#print(combined_features)

# converting the text data to feature vectors (i.e..numerical values)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features) 
#print(feature_vectors)
'''
o/p:-
(i, j) Values: The subsequent lines show entries in the sparse matrix in the format "(i, j) value," where:

(i, j) represents the row and column indices in the matrix.
value represents the numerical value at that position.
For example, "(0, 2432) 0.17272411194153" means that in row 0 and column 2432 of the matrix, the value is approximately 0.1727.'''

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
#print(similarity)
#print(similarity.shape)

# getting the movie name from the user
movie_name = input("enter you favorite movie name: ")

# creating a list with all the movie names given in the dataset    --> this is done bcoz we can compare it with the value given by the user.
list_of_all_titles = movies_data['title'].tolist()
#print(list_of_all_titles)

# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
#print(find_close_match)

close_match = find_close_match[0]
#print(close_match)

# finding the index of the movie with title   --> find the index value bcoz u can find the similarity score later
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
#print(index_of_the_movie)
'''
note:-  
similarity_score will contain index and the similarity score value. thats the reason we find the index of the close_match
'''

# getting the list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
#print(similarity_score)
#print(len(similarity_score))

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score,  key = lambda x:x[1], reverse=True)
#print(sorted_similar_movies)

# print the similar movies based on the index
#print("Movies suggested for you: \n")

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if(i<31):
        print(i,'.',title_from_index)
        i+=1
