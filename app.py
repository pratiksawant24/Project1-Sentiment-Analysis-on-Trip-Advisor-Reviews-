# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:39:06 2020

@author: nithy
"""

from flask import Flask, request, jsonify, render_template
import pickle
import flask
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
input_Title = " "
input_Review = " "
cleaned_review_text = pd.DataFrame()
input_Rating = 10
input_variables = pd.DataFrame()
input_variablesX = pd.DataFrame()
input_variablesY = pd.DataFrame()
reviews = pd.read_csv("C:/Users/nithy/Desktop/Nithya 25_10_2020/Data Science/ExcelR/Project 39/reviews.csv")

def cleaning(input_Title,input_Review,input_Rating):
    import pandas as pd 
    import string # special operations on strings
       
    from nltk import pos_tag
    from nltk.corpus import stopwords
    
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
 
    
    

    reviews = pd.read_csv("C:/Users/nithy/Desktop/Nithya 25_10_2020/Data Science/ExcelR/Project 39/reviews.csv")
#    reviews = pd.DataFrame(columns=reviews.columns)
    reviews.describe()
  

    df2 = {'Title': input_Title, 'Review': input_Review, 'Rating': input_Rating} 
    print("df2:",df2)
    reviews = reviews.append(df2, ignore_index = True)
    #nltk.downloader.download('vader_lexicon') 
    # Checking Missing Values


    #print(f"Missing values: {n_null}")
    #print(f"Number of empty Reviews: {n_empty}")
    
    # Text Contents
    

    
    #Cleaning the data
    
    from nltk.corpus import wordnet
    
    def get_wordnet_pos(pos_tag):     #POS tags are used in corpus searches and in text analysis tools and algorithms
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    
    def clean_text(text):
        text = str(text)
        text = text.lower()   #Lower Text
        
        text = [word.strip(string.punctuation) for word in text.split(" ")]   #Tokenize text and remove punctuation
        
        text = [word for word in text if not any(c.isdigit() for c in word)]  #Remove words that contains numbers
        
        stop = stopwords.words('english')                                     #Remove Stopwords
        text = [x for x in text if x not in stop]
        
        text = [t for t in text if len(t)>0]                                  #Remove Empty tokens
        
        pos_tags = pos_tag(text)                                              #Pos tag Text
        
        text = [WordNetLemmatizer().lemmatize(t[0],get_wordnet_pos(t[1]))for t in pos_tags]   #Lemmatize Text
        
        text = [t for t in text if len(t)>1]                                  #Remove words with only 1 letter
        
        text = " ".join(text)                                                 #Join all
        return(text)
    
    reviews["Review_clean"] = reviews["Review"].apply(lambda x:clean_text(x))
#    print("Review :",reviews.Review)
    stop = stopwords.words('english')
    
    other_stopwords = ['Hotel','hotel','Roosevelt','roosevelt']
    
    reviews['Review_clean'] = reviews['Review_clean'].apply(lambda x: "".join(" ".join(x for x in x.split() if x not in other_stopwords)))
    
    
    # Calculate polarity
    
    from textblob import TextBlob
    reviews['polarity'] = reviews['Review_clean'].apply(lambda x: TextBlob(x).sentiment[0])
    
    sia = SentimentIntensityAnalyzer()
    
    reviews['neg'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neg'])
    reviews['neu'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neu'])
    reviews['pos'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['pos'])
    reviews['compound'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['compound'])
    
    pos_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]>0.2]
    print("pos_review:",pos_review[1])
    
    neu_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]>=-0.2]
    print("neu_review:",neu_review[1])
    neg_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]<0.2]
    print("neg_review:",neg_review[1])
    
    print("Percentage of Positive review: {}%".format(len(pos_review)*100/len(reviews['Review_clean'])))
    print("Percentage of Neutral review: {}%".format(len(neu_review)*100/len(reviews['Review_clean'])))
    print("Percentage of Negative review: {}%".format(len(neg_review)*100/len(reviews['Review_clean'])))
    
    #reviews.sort_values("pos", ascending = False)[["Review_clean", "pos"]].head(10)
    #reviews.sort_values("neg", ascending = False)[["Review_clean","neg"]].head(10)
    
    
    reviews['word_count'] = reviews['Review_clean'].apply(lambda x: len(str(x).split()))  #Word count in each review
    reviews['review_len'] = reviews['Review_clean'].astype(str).apply(len)                #Length of per review
    
    
    for x in [0, 1]:
        subset = reviews[reviews['polarity'] == x]
        
        # Draw the density plot
        if x == 0:
            label = "Good reviews"
        else:
            label = "Bad reviews"
            
    #sns.distplot(subset['compound'], hist = True, label = label)
        
    #Creating a function to encode labels: 2=positive, 1=neutral, 0=negative
    
    # def sentiment(label):
    #     if label == 5 or label == 4:
    #         return 2
    #     if label==3:
    #         return 1
    #     else:
    #         return 0
        
    def sentiment(label):
        # if label == 5:
        #     return 4 
        # if label == 4:	
        #     return 3 
        # if label==3:
        #     return 2
        # if label==2:
        #     return 1	
        # else:
        #     return 0
        if label == 5 :
           return 5
        elif label == 4:
            return 4
        elif label==3:
            return 3
        elif label == 2:
            return 2
        else:
            return 1
        
    labels=['negative','neutral','positive']
    
    #Creating the new column with the encoded labels
    reviews['label']=reviews.Rating.apply(lambda x: sentiment(x))
    # print("reviews['label']:",reviews['label'])
    reviews.Review_clean = reviews.Review_clean.fillna('x')
    # print("reviews['label']",reviews['label'])
    #A few hundred ratings had a score above 5, filtering these out
#    reviews = reviews[reviews['Rating']<=5]
    
    #A few hundred ratings had decimals, rounding each of those down to an integer
#    reviews.Rating = reviews.Rating.astype(int)
    
    #Creating a function that I will use to clean review strings
    #Function makes the string 'txt' lowercase, removes stopwords, finds the length, and pulls out only adjectives
    #Returns a list of the length, cleaned txt, and only adjective txt
    
    # print("reviews ------- polarity:",reviews['polarity'].iloc[-1])
    # moving our review cleaned text to a variable for displaying in html
    if reviews['polarity'].iloc[-1] > 0.2:
#        reviews['Rating'].iloc[-1] = 5
        reviews.at[reviews.index[-1], 'Rating'] = 5
    elif reviews['polarity'].iloc[-1] >=-0.2:
#        reviews['Rating'].iloc[-1] = 3
        reviews.at[reviews.index[-1], 'Rating'] = 3
    elif reviews['polarity'].iloc[-1] <0.2:
#        reviews['Rating'].iloc[-1] = 1
        reviews.at[reviews.index[-1], 'Rating'] = 1
    #A few hundred ratings had a score above 5, filtering these out
#    reviews = reviews[reviews['Rating']<=5]
    
    #A few hundred ratings had decimals, rounding each of those down to an integer
    reviews.Rating = reviews.Rating.astype(int)
#    print("reviews last line:",reviews[-1])
    cleaned_review_text = reviews.tail(1)
    
    #Creating a vectorizer to split the text into unigrams and bigrams
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    rt = cv.fit_transform(reviews.Title).toarray()
    reviews.Title  = rt
  #  print("transformed title:",reviews.Title)
    rc = cv.fit_transform(reviews.Review_clean).toarray()
    reviews.Review_clean  = rc
    rr = cv.fit_transform(reviews.Review).toarray()
    reviews.Review  = rr
#    reviews.cleanrev = cv.fit_transform(reviews.cleanrev).toarray()
#    reviews.adjreview = cv.fit_transform(reviews.adjreview).toarray()
    
    # feature selection
    y = ["label"]
    
    # ignore_cols = ["Review","label","cleanrev","adjreview","reviewlen"]
    ignore_cols = ["label","Review_clean","Review","Title"]
#    ignore_cols = ["label"]
    features = [c for c in reviews.columns if c not in ignore_cols]
    reviews[features]
#    print("reviews:",reviews)
    #input_variables=pd.DataFrame([reviews],dtype=float,index=['input'])
    
    input_variablesX = pd.DataFrame(reviews[features])
    input_variablesY = pd.DataFrame(reviews[y])
#    input_variables = pd.DataFrame(reviews)
#    print("input_variables:", input_variables)
    return input_variablesX,input_variablesY,cleaned_review_text,reviews
#    return input_variables
#   input_variables=pd.DataFrame([reviews],index=['input'])
#@app.route('/', methods=['GET', 'POST'])
#@app.route('/')
#def home():
#    return render_template('index.html')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def predict():
    
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
         input_Title=flask.request.form['Title']
         input_Review=flask.request.form['Review']
#         input_Rating=flask.request.form['Rating']
        # Extract the input
         
#         input_Title=TextBlob(input_Title)
#         input_Review=TextBlob(input_Review)
         
         input_variablesX,input_variablesY,cleaned_review_text,reviews = cleaning(input_Title,input_Review,input_Rating)
#         input_variablesX,input_variablesY,cleaned_review_text = cleaning(input_Title,input_Review) 
#        input_variables = cleaning(input_Title,input_Review)
#         print("reviews just before calling model:", input_variablesX)
         
         
#         input_variables=pd.DataFrame([['1.0', '2.0']],columns=['Title','Review'],dtype=float,index=['input'])
         
        # Get the model's prediction
#         print("review cleaned last line:",input_variablesX['Review_clean'][-1])

         
         prediction = model.predict(input_variablesX)    
# version 1 09/01         prediction = model.predict(reviews)
 #        print("prediction:",prediction)
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
         predicted_rating = ""
         prediction[-1] = prediction[-1].astype('int')
      #   print("prediction[-1]:",prediction[-1]) 
         if prediction[-1] == 5 or prediction[-1] == 4:
            predicted_rating = "Positive"
      #      print("polarity:",cleaned_review_text['polarity'].values[-1])
         elif prediction[-1] == 3:
            predicted_rating = "Neutral"
       #     print("polarity:",cleaned_review_text['polarity'].values[-1])
         elif prediction[-1] == 2 or prediction[-1] == 1:
            predicted_rating = "Negative" 
        #    print("polarity:",cleaned_review_text['polarity'].values[-1])
   
            
   # commenting below to check if lable can give rating also
#          if cleaned_review_text['polarity'].values[-1] >= 0.4 :
#                ratingT=5
#                predicted_rating = "Positive"
#          elif cleaned_review_text['polarity'].values[-1] >= 0.2 and cleaned_review_text['polarity'].values[-1] < 0.4 :
#               ratingT = 4
#               predicted_rating = "Positive"
#          elif cleaned_review_text['polarity'].values[-1]  >= 0.1 and cleaned_review_text['polarity'].values[-1] < 0.2 :
# #             predicted_rating = "Neutral"
#               ratingT = 3
#               predicted_rating = "Neutral"
#          elif cleaned_review_text['polarity'].values[-1] >= -0.1 and cleaned_review_text['polarity'].values[-1] < 0.1 :
#               predicted_rating = "Negative" 
#               ratingT = 2
#          else:
#               predicted_rating = "Negative" 
#               ratingT = 1  
# and including below 
         if prediction[-1] == 5 :
              ratingT=5
#             predicted_rating = "Positive"
         elif prediction[-1] == 4 :
              ratingT = 4
         elif prediction[-1] == 3 :
#             predicted_rating = "Neutral"
              ratingT = 3
         elif prediction[-1] == 2 :
#             predicted_rating = "Negative" 
              ratingT = 2
         else:
              ratingT = 1  
             
#         print("prediction[-1]:",prediction[-1]) 
#         print(predicted_rating,"predicted_rating")
         # if prediction[-1] == 4:
         #     ratingT = 5 
         # elif prediction[-1] == 3:
         #     ratingT = 4 
         # elif prediction[-1] == 2:
         #     ratingT = 3 
         # elif prediction[-1] == 1:
         #     ratingT = 2 
         # else:
         #     ratingT = 1  
#         ratingT = prediction[-1]
#         print("prediction[-1] is :",prediction[-1]) 
            
                     
         
         #return flask.render_template('index.html',original_input={'Predicted Feedback is':predicted_rating},Review_Text="'Review Text entered':input_Review",Polarity={'Polarity Calculated':round(cleaned_review_text['polarity'].values[-1],2)},CatchyWords={'Words that caught attention':cleaned_review_text['Review_clean'].values[-1]},)
                                                                                                                                                                                                                                                                                                 
#         return flask.render_template('index.html', ratingText = cleaned_review_text['Rating'].values[-1] , sentiText = predicted_rating, rvText= input_Review , polText = round(cleaned_review_text['polarity'].values[-1],2), catchyText = cleaned_review_text['Review_clean'].values[-1] )
         return flask.render_template('index.html', ratingText = ratingT , sentiText = predicted_rating, rvText= input_Review , polText = round(cleaned_review_text['polarity'].values[-1],2), catchyText = cleaned_review_text['Review_clean'].values[-1] )
#@app.route('/results',methods=['POST'])
#def results():

#    data = request.get_json(force=True)
#    prediction = model.predict([np.array(list(data.values()))])

#    output = prediction[0]
#    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
 
    