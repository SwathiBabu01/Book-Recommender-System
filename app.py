#BOOK RECOMMENDER SYSTEM
#DSC 478 FINAL PROJECT - WEB APPLICATION
#SWATHI BABU

#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.sparse import csr_matrix
import streamlit as st
from tempfile import NamedTemporaryFile
import shutil
import csv
import urllib.request
import json
import textwrap
import time
from git import Repo
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pprint
from isbntools.app import *

import warnings
warnings.filterwarnings("ignore")


#Title of the web page 
st.set_page_config(page_title = "THE BOOK RECOMMENDER SYSTEM", layout = "wide")
st.title("WELCOME TO THE BOOK RECOMMENDER SYSTEM")

#Reading Datasets initially -- runs only for the first time I ran the app.
def read_datasets():
    books = pd.read_csv('data/BX-Books.csv', sep = ';', error_bad_lines = False, encoding = 'latin-1')
    users = pd.read_csv('data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    return books, users, ratings

#PRE-PROCESSING -- runs only for the 1st 
def preprocess(ratings):
    #Removing the ratings that are equal to 0 --> implicit ratings
    ratings = ratings[ratings['Book-Rating']!=0]
    # Extracting the ratings of users who have rated more than 200 books
    x = ratings['User-ID'].value_counts() > 200
    y = x[x].index  #user_ids
    ratings = ratings[ratings['User-ID'].isin(y)]
    #Adding book title to the ratings file as that is used as the key for books and dropping unnecessary variables from ratings dataframe
    ratings = ratings.merge(books, left_on = 'ISBN', right_on = 'ISBN')
    ratings.drop(['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1, inplace=True)
    #dropping duplicates
    ratings = ratings.drop_duplicates(['User-ID', 'Book-Title'])
    return ratings

#USERS-BOOKS RATINGS MATRIX
def rat_pivot(data):
    data = data.drop_duplicates(['User-ID', 'Book-Title'])
    x = data.pivot(index='User-ID', columns='Book-Title', values = 'Book-Rating')
    return x

#FILTERING WITH LOCATION - Extracting ratings of users in USA and Canada alone as the data is extremely large
def rat_loc(ratings_pivot):
    ratings_pivot=ratings_pivot.merge(users, left_on = 'User-ID', right_on = 'User-ID')
    ratings_pivot = ratings_pivot[ratings_pivot['Location'].str.contains("usa|canada|united kingdom|australia")]
    ratings_pivot.drop(['Location', 'Age'],axis=1, inplace = True)
    ratings_pivot.set_index('User-ID', inplace = True)
    return ratings_pivot

#KNN TO RETURN ITEMS SIMILAR TO THE GIVEN ITEM
def knn(data, k, query, indices):
    model = NearestNeighbors(metric='cosine', algorithm='brute',n_jobs=-1)
    model.fit(data)
    neighbors = model.kneighbors(query, n_neighbors=k)
    simItems = indices[neighbors[1]]
    return simItems


#RECOMMEND FUNCTION THAT RECOMMENDS BOOKS BASED ON PAST RECOMMENDATIONS
def recommend(data, user,books,userData):
    data1 = pd.DataFrame()
    isbns = userData.loc[userData['User-ID'] == user,'prevRec'].tolist()[0].split('\t') #gets the books that were recommended last time to the user
    data1 = data.loc[:,data.loc[user,:]==0] #items that the user hasn't rated yet
    indices = data1.columns #to keep track of the book names
    recs = []
    for x in isbns[1:]: #running through the prev. recommended books
        if x in (data1.columns.values): #this is to make sure the same books don't get recommended
            data1 = data1.drop([x], axis=1)
    for i in isbns[1:]:
        #this gets one book similar to each book that was recommended previously 
        recs = recs + ((knn(csr_matrix(data1.T.values), 1, csr_matrix(data.loc[:,data.columns == i].T.values), indices))[0].tolist()) 
    return (np.unique(np.array(recs))) #returns the new list of recommended books
    

#TOP 5 BOOKS TO RECOMMEND TO NEW USERS- to address the cold start problem
def top_books(data):
    return data.mean().sort_values(ascending=False)[:5].index.values

#SYSTEM FOR THE NEW USER
def new_user():
    userID = (users['User-ID']).max() + 1 #assigning the new user an user ID
    st.write('Thank you for utilising this resource. Your user ID is %d. Please keep note of it!'%(userID))
    st.write('Please fill in the following information:')
    loc =  st.text_input('Enter your Location:').lower() #asking for location
    Age =  st.text_input('Enter your Age:') #asking for age
    if Age: #after both values are entered
        st.write('To start off, please choose 3 books you like:') #Make the new user choose 3 books they like from the most popular books to address the cold start problem
        for j in (top_books(ratings_pivot)): #printing the book cover page if available
            st.write(j)
            lnk = '!['+ j + ']('+ books.loc[books['Book-Title']== j,'Image-URL-L'].values[0] + ')'
            st.markdown(lnk)
        bks = top_books(ratings_pivot)
        #options to choose the books they like
        option_a = st.checkbox(bks[0])
        option_b = st.checkbox(bks[1])
        option_c = st.checkbox(bks[2])
        option_d = st.checkbox(bks[3])
        option_e = st.checkbox(bks[4])
        an = st.button('Click me when done')
        if an: #assigning the mean rating for these books and adding it to the database
            users.loc[max(users.index.values)+1] = [int(userID), loc, float(Age), np.nan]
            #Inserting data to google sheets
            row = [int(userID), loc, int(Age), ' ']
            index = len(users)+1
            sheet2.insert_row(row,index)
            choBooks = [option_a, option_b, option_c, option_d, option_e]
            bk=''
            for l in bks[choBooks][:3]:
                r = ratings_pivot.loc[:,l].mean()
                bk+= '\t' + l
                ratings.loc[max(ratings.index.values)+1] = [int(userID), int(r), l]
                #Inserting the data into the google sheet
                row = [int(userID), int(r), l]
                index = len(ratings) +1
                sheet1.insert_row(row,index)
            if bk: #keeping track of prev recommendations (keep track of user activity)
                users.loc[users['User-ID']==int(userID), 'prevRec'] = bk
                #Inserting data to google sheet
                sheet2.update_cell(users[users['User-ID']==int(userID)].index[0]+2,4, bk)
                return userID #returns the new user ID in the end for further process

#INFORMATION ON BOOKS
def get_book_info(bkName, books): #if available prints basic info about the books pulled from google API
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

#EXECUTED FIRST - not needed anymore
def first():
    #Ran this function first for preprocessing--> these need not be run everytime! The excel files can be used directly
    books, users, ratings = read_datasets()
    ratings = preprocess(ratings)
    ratings_pivot = rat_pivot(ratings)
    ratings_pivot = rat_loc(ratings_pivot)
    #Adding prev Rec as the top 3 rated books by each user for existing users as the dataset did not their activity
    users['prevRec'] = np.nan
    books1 = books.loc[books['Book-Title'].isin(ratings_pivot.columns.values[1:])]
    users1 = users.loc[users['User-ID'].isin(ratings_pivot.index.values)]
    books1['Book-Title'] = books1['Book-Title'].str.strip()
    ratings['Book-Title'] = ratings['Book-Title'].str.strip()
    for i in users1['User-ID']:
        ratUser = ratings.loc[ratings['User-ID']==i,:]
        ratUser = ratUser.loc[ratUser['Book-Rating']>7,:].sort_values(by = 'Book-Rating', ascending = False)[:3]
        pRec=''
        for j in ratUser['Book-Title']:
            pRec = pRec + '\t' + j
            users1.loc[users['User-ID']==i, 'prevRec'] = pRec
    #Writing to csv file which is accessed everytime the web app runs
    users1.to_csv('Users.csv',index=False)
    books1.to_csv('Books.csv',index=False)
    ratings.to_csv('data/Ratings.csv',index=False)
    ratings_pivot.to_csv('Ratings_Pivot.csv')

#FUNCTIONS THAT WERE USED TO TEST THE SYSTEM - Won't be run here

#Returns the index of the similar item (ratings are extracted in the cross-validate function)
def knn_test(data, k, query):
    model = NearestNeighbors(metric='cosine', algorithm='brute',n_jobs=-1)
    model.fit(data)
    neighbors = model.kneighbors(query, n_neighbors=k)
    return neighbors[1]

#Function is the same as the one provided for the assignment with minor changes to accommodate this system
def cross_validate_user(dataMat, user, test_ratio):
    number_of_items = np.shape(dataMat)[1] #number of books
    rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0]) #books rated by user
    test_size = int(test_ratio * len(rated_items_by_user))
    test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
    withheld_items = rated_items_by_user[test_indices]
    original_user_profile = np.copy(dataMat[user])
    data1 = np.copy(dataMat)
    dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
    error_u = 0.0
    count_u = len(withheld_items)
    # Compute absolute error for user u over all test items
    for item in withheld_items:
        # Estimate rating on the withheld item
        ind = knn_test(dataMat.T, 1, data1[:,item].reshape(1, -1)) #index of the most similar book
        estimatedScore = original_user_profile[ind] #rating for the book
        error_u = error_u + abs(estimatedScore - original_user_profile[item])

    # Now restore ratings of the withheld items to the user profile
    for item in withheld_items:
        dataMat[user, item] = original_user_profile[item]

    # Return sum of absolute errors and the count of test cases for this user
    # Note that these will have to be accumulated for each user to compute MAE
    return error_u, count_u

#Calculates the MAE
def test(dataMat, test_ratio):
    errTotal = 0 #to calculate the total error
    cntTotal = 0  #to get total test cases
    for i in range(np.shape(dataMat)[0]):
        err, cnt = cross_validate_user(dataMat, i, test_ratio)#for each user, calculating error
        errTotal += err
        cntTotal += cnt
    mae = errTotal/cntTotal #calculating MAE
    return mae

#Just checking the performance if made into a classifier 
# Ratings - 1 to 5 --> Bad; 6 - 10 --> Good 
#This is only to check the performance of the system and not utilised anywhere else
def cross_validate_user_class(dataMat, user, test_ratio):
    number_of_items = np.shape(dataMat)[1] #number of books
    rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0]) #books rated by user
    test_size = int(test_ratio * len(rated_items_by_user))
    test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
    withheld_items = rated_items_by_user[test_indices]
    original_user_profile = np.copy(dataMat[user])
    data1 = np.copy(dataMat)
    dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
    crct = 0.0
    count_u = len(withheld_items)
    # Compute absolute error for user u over all test items
    for item in withheld_items:
        # Estimate rating on the withheld item
        ind = knn_test(dataMat.T, 1, data1[:,item].reshape(1, -1))
        estimatedScore = original_user_profile[ind] #rating for the book
        if estimatedScore>5 :
            rat = 'Good'
        else:
            rat = 'Bad'
        if original_user_profile[item]>5:
            orgRat = 'Good'
        else:
            orgRat = 'Bad'
        if rat == orgRat:
            crct = crct + 1
        else:
            crct = crct
        for item in withheld_items:
            dataMat[user, item] = original_user_profile[item]

    return crct/count_u

#test function for the classifier
def test_class(dataMat, test_ratio):
    acc = 0
    cnt = 0
    for i in range(np.shape(dataMat)[0]):
        acc += cross_validate_user_class(dataMat, i, test_ratio)#for each user, calculating error
        cnt+=1
    mae = acc/cnt #calculating mean accuracy
    return mae

#The test runs were done in Jupyter Notebook - Results included in the report - will not be run here
def run_test():
    p = ratings['User-ID'].value_counts()
    #extracting only a few users at a time as the dataset is huge and the kernel crashes if it run with the entire dataset
    lst = ratings['User-ID'].value_counts()[(p<1000) & (p>480)].index 
    X = ratings.loc[ratings['User-ID'].isin(lst)]
    X1 = rat_pivot(X)
    print(test(np.array(X1.fillna(0)), 0.2)[0][0]) #with-holding 20% for testing
    print(test_class(np.array(X1.fillna(0)), 0.2))
    #Running test again for different set of users
    lst = ratings['User-ID'].value_counts()[(p<500) & (p>350)].index 
    X = ratings.loc[ratings['User-ID'].isin(lst)]
    X1 = rat_pivot(X)
    print(test(np.array(X1.fillna(0)), 0.2)[0][0]) #with-holding 20% for testing
    print(test_class(np.array(X1.fillna(0)), 0.2))


#MAIN STRUCTURE
#Authorize the API
scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
    ]
file_name = 'client_key.json'
creds = ServiceAccountCredentials.from_json_keyfile_name(file_name,scope)
client = gspread.authorize(creds)


sheet1 = client.open('Ratings').sheet1
python_sheet1 = sheet1.get_all_records()
ratings = pd.DataFrame(python_sheet1)

sheet2= client.open('Users').sheet1
python_sheet2 = sheet2.get_all_records()
users = pd.DataFrame(python_sheet2)

sheet3 = client.open('Books').sheet1
python_sheet3 = sheet3.get_all_records()
books = pd.DataFrame(python_sheet3)

ratings_pivot = rat_pivot(ratings)

#New user or Existing user
option = st.selectbox('Existing User or New User?' , ('Existing User', 'New User'))

#PROCESS FOR NEW USER
if option == 'New User':
    userID = new_user() 
    if userID: #After receiving the userID of the new user, they are recommended new books and all the informaation is updated in respective dataframes
        ratings = ratings.drop_duplicates(['User-ID', 'Book-Title'])
        ratings_pivot = rat_pivot(ratings)
        try:
            ratings_pivot.drop([8.4, 253, 1984, 2001, np.inf],axis=1,inplace=True) #some cleaning
        except:
            pass
        st.write('You might like: ')
        recBks = recommend(ratings_pivot.fillna(0), int(userID), books, users)
        bkStr = ''
        for b in recBks:
            try:
                st.write(b)
                bkStr = bkStr +'\t' + b
                lnk = '!['+ b + ']('+ books.loc[books['Book-Title']== b,'Image-URL-L'].values[0] + ')'
                st.markdown(lnk)
                get_book_info(b, books)
            except IndexError:
                ind = np.where(recBks == b)[0][0]
                b = users.loc[users['User-ID'] == int(userID),'prevRec'].tolist()[0].split('\t')[ind]
                st.write(b)
                bkStr = bkStr +'\t' + b
                lnk = '!['+ b + ']('+ books.loc[books['Book-Title']== b,'Image-URL-L'].values[0] + ')'
                st.markdown(lnk)
        users.loc[users['User-ID']==int(userID),'prevRec'] = bkStr
        ind = users[users['User-ID']==int(userID)].index[0] + 2
        sheet2.update_cell(ind,4, bkStr)

#PROCESS FOR EXISTING USER
if option == 'Existing User':
    userID = st.text_input('Enter your User id:') #Asking for the user ID
    if userID: # if user id exists
        if int(userID) in users['User-ID'].values:
            #Asking if they would like to review the previous recommendations or get new recommendations
            rat = st.checkbox('Rate the previous Recommendation') 
            rec = st.checkbox('Get new Recommendations!')
            if rat: #if they would like to rate, the ratings are updated
                bks = users.loc[users['User-ID'] == int(userID),'prevRec'].tolist()[0].split('\t')
                chBooks = st.multiselect(
                    'Choose the books you would like to rate please:',
                    bks[1:],
                    bks[1])
                st.write('Please rate the chosen books below')
                m = 0
                for l in chBooks:
                    time.sleep(1)
                    st.write(l)
                    m+=1
                    rat = st.text_input('Enter your rating:', key = m)
                    if rat:
                        ratings.loc[max(ratings.index.values)+1] = [int(userID), int(rat), str(l)]
                        #Inserting data
                        row = [int(userID), int(rat), str(l)]
                        index = len(ratings.index.values) + 1
                        sheet1.insert_row(row,index)
                        ratings_pivot.loc[int(userID),l] = int(rat)
            if rec: #they receive new recommendations
                st.write('YOU MIGHT LIKE:')
                ratings_pivot = rat_pivot(ratings)
                try:
                    ratings_pivot.drop([8.4, 253, 1984, 2001, np.inf],axis=1,inplace=True) 
                except:
                    ratings_pivot = ratings_pivot
                recBks = (recommend(ratings_pivot.fillna(0), int(userID), books, users))
                bkStr = ''
                for b in recBks:
                    try:
                        st.write(b)
                        bkStr = bkStr +'\t' + b
                        lnk = '!['+ b + ']('+ books.loc[books['Book-Title']== b,'Image-URL-L'].values[0] + ')'
                        st.markdown(lnk)
                        get_book_info(b, books)
                    except IndexError:
                        st.write('.')
                users.loc[users['User-ID']==int(userID),'prevRec'] = bkStr
                #Inserting data
                sheet2.update_cell(users[users['User-ID']==int(userID)].index[0]+2,4, bkStr)
        
        elif userID!= '': #if the userID does not exist
            st.write('Username does not exist. Please try again!')