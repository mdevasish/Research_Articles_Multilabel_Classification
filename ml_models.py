import pandas as pd
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix,f1_score
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain,LabelPowerset
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import nltk
import re
import datetime
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
import joblib
stop_words = set(nltk.corpus.stopwords.words('english'))

def read_files():
    '''
    Function to read the train and test datset
    
    Returns:
        
        df   : Train data
        test : Test data
    
    '''
    df = pd.read_csv('./Train.csv')
    test = pd.read_csv('./Test.csv')
    return df,test

def generate_plots(df,col = None):
    plt.figure(figsize=(40,25))
    if col :
        df = df[df[col]==1]
    text = df.ABSTRACT.values
    cloud_research = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))
    plt.axis('off')
    if col:
        fig_name = './plots/'+col+'.jpeg'
        plt.title(col,fontsize=40)
    else:
        plt.title('Research',fontsize=40)
        fig_name = './plots/research_articles_all.jpeg'
    plt.imshow(cloud_research)
    plt.savefig(fig_name)
    
def preprocess_text(text,stem = None,lemma = 'wordlemma'):
    '''
    Function to clean the textual data
    
    Input Parameters :
        text  : Input text
        stem  : Type of stemmer
        lemma : Type of Lemmatizer
    
    Returns :
        x : Cleaned text
    '''
    text = re.sub('-',' ',text)
    text = text.lower()
    tokens = [word for word in word_tokenize(text) if len(word) > 3 and word.isalpha()] 
    if stem == 'porter':
        stemmer = PorterStemmer('english')
        tokens =  [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    elif stem == 'snowball':
        stemmer = SnowballStemmer('english')
        tokens =  [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    if lemma == 'wordlemma':
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(item) for item in tokens if (item not in stop_words)]
    x = ' '.join(tokens)
    return x
    
    
def corpus_formation(vectorizer,df,test,target,min_df = None,max_feat = None):
    '''
    Function to create features from the data
    
    Input parameters :
        
        vectorizer : Vectorizer used to derive features out of the data
        df         : Data Frame of the training dataset
        test       : Data frame of the test dataset
        max_feat   : number of features to be extracted
        target     : target columns for the dataset
    
    Returns :
        
        trn : features of the train dataset
        val : features of the validation dataset
        tst : features of the test dataset
        y   : target of the train dataset
        z   : target of the validation dataset
        
    '''
    if max_feat:
        vec = vectorizer(tokenizer = preprocess_text,max_features = max_feat)    
    if min_df:
        vec = vectorizer(tokenizer = preprocess_text,min_df = min_df)    
    combined = list(df['ABSTRACT']) + list(test['ABSTRACT'])
    vec.fit(combined)
    
    trn, val = train_test_split(df, test_size=0.2, random_state=2021)
    y = trn[target]
    z = val[target]

    trn_abs = vec.transform(trn['ABSTRACT'])
    val_abs = vec.transform(val['ABSTRACT'])
    tst_abs = vec.transform(test['ABSTRACT'])
    
    trn2 = np.hstack((trn_abs.toarray(), trn[topic_col]))
    val2 = np.hstack((val_abs.toarray(), val[topic_col]))
    tst2 = np.hstack((tst_abs.toarray(), test[topic_col]))
    
    trn = csr_matrix(trn2.astype('int16'))
    val = csr_matrix(val2.astype('int16'))
    tst = csr_matrix(tst2.astype('int16'))
    
    return trn,val,tst,y,z

def get_best_thresholds(true, preds):
    thresholds = [i/100 for i in range(100)]
    best_thresholds = []
    for idx in range(25):
        f1_scores = [f1_score(true[:, idx], (preds[:, idx] > thresh) * 1) for thresh in thresholds]
        best_thresh = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_thresh)
    return best_thresholds


class Model_Construction:
    '''    Class to construct a models and evalaute the models   '''
    def __init__(self,meta_model,model,X_train,y_train,X_val,y_val,logreg_C = None,logreg_max_iter = None, logreg_solver = None, svm_C = None, svm_kernel = None,naive_alpha = None):
        '''
        Constructor to set values before creating the model
        
        Input paramters :
            
            meta_model      : Type of multilabel classification model
            model           : Estimator object for the classification model
            X_train         : features of the train set
            y_train         : target of the train set
            X_val           : features of the validation set
            y_val           : target of the validation set
            logreg_C        : Regularization parameter of LogisticRegression
            logreg_max_iter : Number of iterationsfor LogisticRegression
            logreg_solver   : solver for LogisticRegression
            svm_C           : Regularization parameter of SVM
            svm_kerenl      : Type of kernel for SVM
            naive_alpha     : Smoothing parameter for Naive Bayes
            
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.logreg_C = logreg_C 
        self.logreg_solver = logreg_solver 
        self.svm_C = svm_C 
        self.svm_kernel =svm_kernel 
        self.naive_alpha = naive_alpha 
        self.str_model = meta_model+'_'+model
        
        if meta_model == 'ovr':
            if model == 'LogisticRegression':
                self.model = OneVsRestClassifier(LogisticRegression(C = logreg_C,max_iter = logreg_max_iter, solver = logreg_solver,n_jobs=-1))
            elif model == 'SVM':
                self.model = OneVsRestClassifier(SVC(C=svm_C,kernel = svm_kernel))
            elif model == 'Naive':
                self.model = OneVsRestClassifier(MultinomialNB(alpha = naive_alpha))
        
        elif meta_model == 'BinaryRelevance':
            if model == 'LogisticRegression':
                self.model = BinaryRelevance(LogisticRegression(C = logreg_C, max_iter = logreg_max_iter,solver = logreg_solver,n_jobs=-1))
            elif model == 'SVM':
                self.model = BinaryRelevance(SVC(C=svm_C,kernel = svm_kernel))
            elif model == 'Naive':
                self.model = BinaryRelevance(MultinomialNB(alpha = naive_alpha))
                
        elif meta_model == 'ClassifierChain':
            if model == 'LogisticRegression':
                self.model = ClassifierChain(LogisticRegression(C = logreg_C, max_iter = logreg_max_iter,solver = logreg_solver,n_jobs=-1))
            elif model == 'SVM':
                self.model = ClassifierChain(SVC(C=svm_C,kernel = svm_kernel))
            elif model == 'Naive':
                self.model = ClassifierChain(MultinomialNB(alpha = naive_alpha))
        
        elif meta_model == 'LabelPowerset':
            if model == 'LogisticRegression':
                self.model = LabelPowerset(LogisticRegression(C = logreg_C, max_iter = logreg_max_iter,solver = logreg_solver,n_jobs=-1))
            elif model == 'SVM':
                self.model = LabelPowerset(SVC(C=svm_C,kernel = svm_kernel))
            elif model == 'Naive':
                self.model = LabelPowerset(MultinomialNB(alpha = naive_alpha))
                
        elif meta_model == 'MultiOutputClassifier':
            if model == 'LogisticRegression':
                self.model = MultiOutputClassifier(LogisticRegression(C = logreg_C, max_iter = logreg_max_iter,solver = logreg_solver,n_jobs=-1))
            elif model == 'SVM':
                self.model = MultiOutputClassifier(SVC(C=svm_C,kernel = svm_kernel))
            elif model == 'Naive':
                self.model = MultiOutputClassifier(MultinomialNB(alpha = naive_alpha))
                
    def implement_model(self):   
        '''  
        Method inside the Model_construction class, used for implementing the model and return performance metrics
       
        Input Parameters :
            
            self : class attributes
            
        Returns :
            
            f1_train : f1 score on train set
            f1_val   : f1 score on validation set
            c_matrix : confusion matrix 
            
        '''
        model = self.model
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        model.fit(X_train,y_train)
        val_preds = model.predict(X_val)
        c_matrix = multilabel_confusion_matrix(y_val, val_preds)
        f1_train = f1_score(y_train, model.predict(X_train), average='micro')
        f1_val = f1_score(y_val, val_preds, average='micro')
        print('f1 score for '+ repr(model) +' on train:',f1_train)
        print('f1 score for '+ repr(model) +' on validation:',f1_val)
        joblib.dump(model,'./models'+self.str_model+'.sav')
        return f1_train,f1_val,c_matrix
    
def model_performance(meta_models,models,X_train,y_train,X_val,y_val,logreg_C = None, logreg_max_iter = None, logreg_solver = None,svm_C = None, svm_kernel = None, naive_alpha = None):
    '''
    Function to record the evaluation metrics and save the models
    
    Input parameters :
        meta_models : List of multilabel classification model
        models      : List of Estimator objects for the classification model
        X_train     : features of the train set
        y_train     : target of the train set
        X_val       : features of the validation set
        y_val       : target of the validation set
    '''
    train_list = []
    val_list = []
    model_list = []
    time_list = []
    for each in meta_models:
        for every in models:
            if every == 'LogisticRegression':
                z = Model_Construction(each,every,X_train,y_train,X_val,y_val,logreg_C = logreg_C, logreg_max_iter = logreg_max_iter, logreg_solver=logreg_solver)
                # count parameters : logreg_C = 0.05, logreg_max_iter = 300, logreg_solver='lbfgs'
            elif every == 'SVM':
                z = Model_Construction(each,every,X_train,y_train,X_val,y_val,svm_C = svm_C, svm_kernel = svm_kernel)
                # count parameters : svm_C = 0.1, svm_kernel = 'linear'
            elif every == 'Naive':
                z = Model_Construction(each,every,X_train,y_train,X_val,y_val,naive_alpha = naive_alpha)
                # count parameters : naive_alpha = 0.5
            now = datetime.datetime.now()
            train,val,matrix = z.implement_model()
            after = datetime.datetime.now()
            train_list.append(train)
            val_list.append(val)
            model_list.append(each+'_'+every)
            time_list.append(after-now)
            
    performance = pd.DataFrame(zip(model_list,train_list,val_list,time_list),columns = ['models','f1 score for train','f1 score val','time'])
    return performance


        
if __name__ == '__main__':
    df,test = read_files()
    target = ['Analysis of PDEs', 'Applications','Artificial Intelligence', 'Astrophysics of Galaxies','Computation and Language', 'Computer Vision and Pattern Recognition',
              'Cosmology and Nongalactic Astrophysics','Data Structures and Algorithms', 'Differential Geometry','Earth and Planetary Astrophysics','Fluid Dynamics',
              'Information Theory', 'Instrumentation and Methods for Astrophysics','Machine Learning', 'Materials Science','Methodology','Number Theory',
              'Optimization and Control', 'Representation Theory', 'Robotics','Social and Information Networks', 'Statistics Theory',
              'Strongly Correlated Electrons', 'Superconductivity','Systems and Control']
    topic_col = ['Computer Science','Mathematics', 'Physics','Statistics']
    
    meta_models = ['ovr','BinaryRelevance','ClassifierChain','LabelPowerset','MultiOutputClassifier']
    models = ['LogisticRegression','SVM','Naive']
    generate_plots(df)
    for each in topic_col+target:
        generate_plots(df,each)
    X_train,X_val,X_test,y_train,y_val = corpus_formation(CountVectorizer,df,test,target = target,min_df = 3)
    performance_count = model_performance(meta_models,models,X_train,y_train,X_val,y_val,logreg_C = 0.05, logreg_max_iter = 300, logreg_solver = 'lbfgs',svm_C = 0.1, svm_kernel = 'linear',naive_alpha = 0.5)
    performance_count['method'] = 'count'
    
    X_train,X_val,X_test,y_train,y_val = corpus_formation(TfidfVectorizer,df,test,target = target,min_df = 3)
    performance_tfidf = model_performance(meta_models,models,X_train,y_train,X_val,y_val,logreg_C = 5, logreg_max_iter = 300, logreg_solver = 'lbfgs',svm_C = 5, svm_kernel = 'linear',naive_alpha = 1)
    performance_tfidf['method'] = 'tfidf'
    
    performance = pd.concat([performance_count,performance_tfidf],ignore_index = True)
    performance.to_csv('./metrics/performance_after_cleaning_min_df=3.csv')
    
    