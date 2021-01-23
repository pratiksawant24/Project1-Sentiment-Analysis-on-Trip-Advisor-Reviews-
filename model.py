# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:32:12 2021

@author: nithy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 00:20:10 2020

@author: nithy
"""



def modelrun():
    import pandas as pd 
    import string # special operations on strings
            
    import pickle
    
    from nltk import pos_tag
    from nltk.corpus import stopwords
    
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
      
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
        
#    import xlrd
    
#    reviews = pd.read_excel("C:/Users/nithy/Desktop/Nithya 25_10_2020/Data Science/ExcelR/Project 39/reviews - Copy.xlsx")
    reviews = pd.read_csv("C:/Users/nithy/Desktop/Nithya 25_10_2020/Data Science/ExcelR/Project 39/reviews - Copy.csv")
    
    
    #nltk.downloader.download('vader_lexicon') 
    # Checking Missing Values
     
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
    
    stop = stopwords.words('english')
    
    other_stopwords = ['Hotel','hotel','Roosevelt','roosevelt']
    
    reviews['Review_clean'] = reviews['Review_clean'].apply(lambda x: "".join(" ".join(x for x in x.split() if x not in other_stopwords)))
    
    
    
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('wordnet')
    
    # Calculate polarity
    
    from textblob import TextBlob
    reviews['polarity'] = reviews['Review_clean'].apply(lambda x: TextBlob(x).sentiment[0])
    
    sia = SentimentIntensityAnalyzer()
    
    reviews['neg'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neg'])
    reviews['neu'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neu'])
    reviews['pos'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['pos'])
    reviews['compound'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['compound'])
    
    pos_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]>0.2]
    pos_review
    
    neu_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]>=-0.2]
    neg_review = [j for i,j in enumerate(reviews['Review_clean']) if reviews['polarity'][i]<0.2]
    
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
    
    def sentiment(label):
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
    
    reviews.Review_clean = reviews.Review_clean.fillna('x')
    
    #A few hundred ratings had a score above 5, filtering these out
#    reviews = reviews[reviews['Rating']<=5]
    
    #A few hundred ratings had decimals, rounding each of those down to an integer
    reviews.Rating = reviews.Rating.astype(int)
    
    
#     #Creating a vectorizer to split the text into unigrams and bigrams
    
    cv = CountVectorizer(max_features = 1500)
    rt = cv.fit_transform(reviews.Title).toarray()
    reviews.Title  = rt
    rc = cv.fit_transform(reviews.Review_clean).toarray()
    reviews.Review_clean  = rc
    rr = cv.fit_transform(reviews.Review).toarray()
    reviews.Review  = rr
#    reviews.cleanrev = cv.fit_transform(reviews.cleanrev).toarray()
#    reviews.adjreview = cv.fit_transform(reviews.adjreview).toarray()
    
#     # feature selection
    y = "label"
    
    ignore_cols = ["Review_clean","label","cleanrev","adjreview","reviewlen","Review","Title"]
    features = [c for c in reviews.columns if c not in ignore_cols]
    ignore_cols
    features
    
    x_train, x_test, y_train, y_test = train_test_split(reviews[features],reviews[y], test_size=0.2, random_state = 42)
    modelGNB = GaussianNB()
    modelGNB.fit(x_train,y_train)
    predGNB = modelGNB.predict(x_test)
    print("Naive Bayes Gaussian:",accuracy_score(y_test.round(), predGNB.round()))
   
    pickle.dump(modelGNB,open('model.pkl','wb'))
    

classGNB = modelrun()
