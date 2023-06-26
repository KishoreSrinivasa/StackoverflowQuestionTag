#IMPORTING LIBRARIES
import pandas as pd 
import numpy as np 

#READ THE PROCESSED DATASET
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\PROJECT-STACKOVERFLOW\\dataset\\final_question_tags.csv')
print(df.isnull().sum())
df = df.dropna(subset=['Title','Body'])
print(df.isnull().sum())

#CONVERTING TAGS COLUMN INTO LIST 
import ast
df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))

#IMPORT ALL THE NECESSARY MODEL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix, accuracy_score,multilabel_confusion_matrix,classification_report


#ASSIGNING Y AS COLUMN TAGS,X1 AS BODY COLUMN,X2 AS TITLE COLUMN
X = df['Body']
y = df['Tags']

#CONVERT Y COLUMN TO CLASSES
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(y)
multilabel.classes_
pd.DataFrame(y ,columns = multilabel.classes_)

#TF-IDF VECTORIZER
#tfidf = TfidfVectorizer(analyzer = 'word' , max_features = 10000, ngram_range = (1,3), stop_words = 'english')
vectorizer_X1 = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)



X_tfidf = vectorizer_X1.fit_transform(X)


#SPLITING THE DATASET
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size = 0.2, random_state = 0)

#INTIALING THE MODELS
sgd = SGDClassifier()
lr = LogisticRegression()
svc = LinearSVC()



#JACCAD SCORE USED TO ANALYSIS THE ACCURACY OF THE MULTILABEL CLASSIFICATION MODEL
def avg_jacard(y_true,y_pred):
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    return jacard.mean()*100

def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Jacard score: {}".format(avg_jacard(y_test, y_pred)))
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))
    print("---")

#PREPARING THE MODEL
for classifier in [sgd,lr,svc]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)
    #y_pred = multilabel.inverse_transform(y_pred)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2f}%".format(accuracy * 100))

    # Generate the classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)    

#EXPORTING THE BEST MODEL 
import joblib
joblib_file = "tagPredictor.pkl"
joblib.dump(clf, joblib_file)