#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.sparse import csr_matrix
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import streamlit as st
from tempfile import NamedTemporaryFile
import shutil
import csv
import urllib.request
import json
import textwrap
import time


import warnings
warnings.filterwarnings("ignore")


#Title
st.set_page_config(page_title = "THE BOOK RECOMMENDER SYSTEM", layout = "wide")
st.title("WELCOME TO THE BOOK RECOMMENDER SYSTEM")

#Reading Datasets
def read_datasets():
    books = pd.read_csv('/data/BX-Books.csv', sep = ';', error_bad_lines = False, encoding = 'latin-1')
    users = pd.read_csv('/data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings = pd.read_csv('/data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    return books, users, ratings

#PRE-PROCESSING
def preprocess(ratings):
    ratings = ratings[ratings['Book-Rating']!=0]
    x = ratings['User-ID'].value_counts() > 200
    y = x[x].index  #user_ids
    ratings = ratings[ratings['User-ID'].isin(y)]
    ratings = ratings.merge(books, left_on = 'ISBN', right_on = 'ISBN')
    ratings.drop(['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1, inplace=True)
    ratings = ratings.drop_duplicates(['User-ID', 'Book-Title'])
    #ratings = ratings.groupby("ISBN").filter(lambda x: len(x) > 10)
    return ratings

#RATINGS MATRIX
def rat_pivot(data):
    x = data.pivot(index='User-ID', columns='Book-Title', values = 'Book-Rating')
    return x

#FILTERING WITH LOCATION
def rat_loc(ratings_pivot):
    ratings_pivot=ratings_pivot.merge(users, left_on = 'User-ID', right_on = 'User-ID')
    ratings_pivot = ratings_pivot[ratings_pivot['Location'].str.contains("usa|canada|united kingdom|australia")]
    ratings_pivot.drop(['Location', 'Age'],axis=1, inplace = True)
    ratings_pivot.set_index('User-ID', inplace = True)
    return ratings_pivot

#This is the function that would be run everytime
def read_new():
    ratings = pd.read_csv('/data/Ratings.csv', error_bad_lines = False)
    users = pd.read_csv('/data/Users.csv', error_bad_lines = False)
    books = pd.read_csv('/data/Books.csv', error_bad_lines = False)
    ratings_pivot = pd.read_csv('/data/Ratings_Pivot.csv', index_col=0, error_bad_lines = False)
    return ratings,users,books,ratings_pivot

#KNN TO RETURN ITEMS SIMILAR TO THE GIVEN ITEM
def knn(data, k, query, indices):
    model = NearestNeighbors(metric='cosine', algorithm='brute',n_jobs=-1)
    model.fit(data)
    #query = data[np.where(indices == user)[0][0]]
    neighbors = model.kneighbors(query, n_neighbors=k)
    simItems = indices[neighbors[1]]
    sim = neighbors[0]
    return simItems


#RECOMMEND FUNCTION THAT RECOMMENDS BOOKS BASED ON PAST RECOMMENDATIONS
def recommend(data, user,books,userData):
    data1 = pd.DataFrame()
    isbns = userData.loc[userData['User-ID'] == user,'prevRec'].tolist()[0].split('\t')
    data1 = data.loc[:,data.loc[user,:]==0]
    indices = data1.columns
    recs = []
    for x in isbns[1:]:
        if x in (data1.columns.values):
            data1 = data1.drop([x], axis=1)
    for i in isbns[1:]:
        recs = recs + ((knn(csr_matrix(data1.T.values), 1, csr_matrix(data.loc[:,data.columns == i].T.values), indices))[0].tolist())
    return (np.unique(np.array(recs)))
    

#TOP 5 BOOKS TO RECOMMEND TO USERS WHEN THEY ARE NEW
def top_books(data):
    return data.mean().sort_values(ascending=False)[:5].index.values

#SYSTEM FOR THE NEW USER
def new_user():
    userID = (users['User-ID']).max() + 1
    st.write('Thank you for utilising this resource. Your user ID is %d. Please keep note of it!'%(userID))
    st.write('Please fill in the following information:')
    loc =  st.text_input('Enter your Location:').lower()
    Age =  st.text_input('Enter your Age:')
    if Age:
        st.write('To start off, please choose 3 books you like from the top down menu:')
        for j in (top_books(ratings_pivot)):
            st.write(j)
            lnk = '!['+ j + ']('+ books.loc[books['Book-Title']== j,'Image-URL-L'].values[0] + ')'
            #img = Image.open(lnk)
            st.markdown(lnk)
        bks = top_books(ratings_pivot)
        option_a = st.checkbox(bks[0])
        option_b = st.checkbox(bks[1])
        option_c = st.checkbox(bks[2])
        option_d = st.checkbox(bks[3])
        option_e = st.checkbox(bks[4])
        an = st.button('Click me when done')
        if an:
            users.loc[max(users.index.values)+1] = [int(userID), loc, float(Age), np.nan]
            choBooks = [option_a, option_b, option_c, option_d, option_e]
            rat = [7, 7, 7]
            bk=''
            for l in bks[choBooks][:3]:
                r = ratings_pivot.loc[:,l].mean()
                bk+= '\t' + l
                ratings.loc[max(ratings.index.values)+1] = [int(userID), int(r), l]
            if bk:
                users.loc[users['User-ID']==userID, 'prevRec'] = bk
                usRow = str(userID) + ',' + str(loc) + ',' + str(Age) + ',' +  bk + '\n'
                csv_add('Users', usRow)
                return userID

def get_book_info(bkName, books):
    try:
        isbn = books.loc[books['Book-Title']==bkName,'ISBN'].values[0]
        base_api_link = "https://www.googleapis.com/books/v1/volumes?q=isbn:"
        user_input = isbn.strip()

        with urllib.request.urlopen(base_api_link + user_input) as f:
            text = f.read()

        decoded_text = text.decode("utf-8")
        obj = json.loads(decoded_text) # deserializes decoded_text to a Python object
        volume_info = obj["items"][0] 
        authors = obj["items"][0]["volumeInfo"]["authors"]

        # displays title, summary, author, domain, page count and language
        st.write("\nTitle:", volume_info["volumeInfo"]["title"])
        st.write("\nSummary:\n")
        st.write(textwrap.fill(volume_info["searchInfo"]["textSnippet"], width=65))
        st.write("\nAuthor(s):", ",".join(authors))
        st.write("\nPage count:", volume_info["volumeInfo"]["pageCount"])
        st.write("\n***")
    except:
        st.write("No info!")

def first():
    #Ran this function first for preprocessing--> these need not be run everytime! The excel files can be used directly
    books, users, ratings = read_datasets()
    ratings = preprocess(ratings)
    ratings_pivot = rat_pivot(ratings)
    ratings_pivot = rat_loc(ratings_pivot)
    users['prevRec'] = np.nan
    books1 = books.loc[books['Book-Title'].isin(ratings_pivot.columns.values[1:])]
    users1 = users.loc[users['User-ID'].isin(ratings_pivot.index.values)]

    for i in users1['User-ID']:
        ratUser = ratings.loc[ratings['User-ID']==i,:]
        ratUser = ratUser.loc[ratUser['Book-Rating']>7,:].sort_values(by = 'Book-Rating', ascending = False)[:3]
        pRec=''
        for j in ratUser['Book-Title']:
            pRec = pRec + '\t' + j
            users1.loc[users['User-ID']==i, 'prevRec'] = pRec
    books1['Book-Title'] = books1['Book-Title'].str.strip()
    ratings['Book-Title'] = ratings['Book-Title'].str.strip()

    users1.to_csv('Users.csv',index=False)
    books1.to_csv('Books.csv',index=False)
    ratings.to_csv('/data/Ratings.csv',index=False)
    ratings_pivot.to_csv('Ratings_Pivot.csv')

def csv_add(data, myCsvRow):
    lnk = '/data/' + str(data) + '.csv'
    with open(lnk,'a') as fd:
        fd.write(myCsvRow)
        fd.close()

def edit_csv(recChange, userID):
    filename = '/data/Users.csv'
    tempfile = NamedTemporaryFile(mode='w', delete=False)
    fields = ['User-ID', 'Location', 'Age', 'prevRec']
    with open(filename, 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
        for row in reader:
            if row['User-ID'] == str(userID):
                row['prevRec'] = recChange
            row = {'User-ID': row['User-ID'], 'Location': row['Location'], 'Age': row['Age'], 'prevRec': row['prevRec']}
            writer.writerow(row)
    shutil.move(tempfile.name, filename)
    csvfile.close()
    tempfile.close()

#MAIN STRUCTURE
ratings, users, books, ratings_pivot = read_new()

#New user or Existing user
option = st.selectbox('Existing User or New User?' , ('Existing User', 'New User'))

if option == 'New User':
    userID = new_user()
    if userID:
        ratings_pivot = rat_pivot(ratings)
        st.write('You might like: ')
        recBks = recommend(ratings_pivot.fillna(0), int(userID), books, users)
        bkStr = ''
        for b in recBks:
            bkStr = bkStr +'\t' + b
            lnk = '!['+ b + ']('+ books.loc[books['Book-Title']== b,'Image-URL-L'].values[0] + ')'
            st.markdown(lnk)
            get_book_info(b, books)
        users.loc[users['User-ID']==int(userID),'prevRec'] = bkStr
        edit_csv(bkStr, int(userID))
        ratings_pivot.to_csv('/data/Ratings_Pivot.csv')

if option == 'Existing User':
    userID = st.text_input('Enter your User id:')
    if userID:
        if int(userID) in users['User-ID'].values:
            rat = st.checkbox('Rate the previous Recommendation')
            rec = st.checkbox('Get new Recommendations!')
            if rat:
                bks = users.loc[users['User-ID'] == int(userID),'prevRec'].tolist()[0].split('\t')
                chBooks = st.multiselect(
                    'Choose the books you would like to rate please:',
                    bks[1:],
                    bks[1])
                st.write('Please rate the chosen books below')
                m = 0
                for l in chBooks:
                    st.write(l)
                    m+=1
                    rat = st.text_input('Enter your rating:', key = m)
                    if rat:
                        ratings.loc[max(ratings.index.values)+1] = [int(userID), int(rat), str(l)]
                        ratings_pivot.loc[int(userID),l] = int(rat)
                        bkRow = str(userID) + ',' + str(rat) + ',' + str(l) + '\n'
                        csv_add('Ratings', bkRow)
                        ratings_pivot.to_csv('/data/Ratings_Pivot.csv')
            if rec:
                st.write('YOU MIGHT LIKE:')
                recBks = (recommend(ratings_pivot.fillna(0), int(userID), books, users))
                bkStr = ''
                for b in recBks:
                    bkStr = bkStr +'\t' + b
                    lnk = '!['+ b + ']('+ books.loc[books['Book-Title']== b,'Image-URL-L'].values[0] + ')'
                    st.markdown(lnk)
                    get_book_info(b, books)
                users.loc[users['User-ID']==int(userID),'prevRec'] = bkStr
                edit_csv(bkStr, int(userID))
        
        elif userID!= '':
            st.write('Username does not exist. Please try again!')







