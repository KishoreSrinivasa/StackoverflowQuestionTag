#IMPORTING PACKAGES REQUIRED
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
 
#LOADING THE DATASET
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\PROJECT-STACKOVERFLOW\\dataset\\final_question_tags.csv')
df = df.dropna()

#CONVERTING TAGS COLUMN INTO LIST 
import ast
df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))

#LOADING THE PREDICTION FILE
tag_pred = joblib.load('tagPredictor.pkl')

#iNTIALISATION OF THE REQUIRED METHODS
vectorizer = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)



x_tfidf = vectorizer.fit_transform(df['Body'])


multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Tags'])

#DEFINING THE FUNCTION THE TAG PREDICTION
def get_Tags(question):
    question = [question]
    question = vectorizer.transform(question)
    tags = multilabel.inverse_transform(tag_pred.predict(question))
    return tags 

out = get_Tags("what is python and how it works")
print(out)