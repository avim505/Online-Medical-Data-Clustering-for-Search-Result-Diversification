import csv
import re
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer



x = open("top-passages-york07ga1.txt")
y = open("top-passages-york07ga1.txt")
z = open("top-passages-york07ga1.txt")
topics = []
pmid =[]
rank = []
score=[]
offset = []
length = []
tagid = []	
passages= []

# if the line begins with the word Topic, extract the content from the 7th character to the second last,
# replace any single or double quotation marks with an empty space and append it to topics

for line in x:
    if line.startswith("Topic"):
        topic = line[7:-1]
        topic = re.sub("['\"]","",topic)
        topics.append(topic)

# if the line begins with "1." extract the content from the 4th character to the second last,
# removes any single or double quotes within the 'cleaned' string.
# splits the cleaned string into a list of strings using the "_" character as a separator
# append pmid, rank, score, offset, length, and tagid with the respective item from the 'list'

for line in y:
    if line.startswith(" 1."):
        cleaned = line[3:-1]
        cleaned = re.sub("['\"]","",cleaned)
        list = cleaned.split('_')
        pmid.append(list[0])
        rank.append(list[1])
        score.append(list[2])
        offset.append(list[3])
        length.append(list[4])
        tagid.append("york07ga1")

for line in z:
    if line.startswith(" 2."):
        passages.append(line)



    
# preprocessing steps (stop words, punctuation)
def stop_removal(text, stops):
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)

def preprocess(docs):
    stops = stopwords.words("english")
    final = []
    for doc in docs:
        clean_doc = stop_removal(doc, stops)
        final.append(clean_doc)
    return (final)

cleaned_passages = preprocess(passages)




vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_features=100,
                                max_df=0.8,
                                min_df=5,
                                ngram_range = (1,3),
                                stop_words = "english"
                            )
vectors = vectorizer.fit_transform(cleaned_passages)
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
all_keywords = []
for passages in denselist:
    x=0
    keywords = []
    for word in passages:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)
print (passages[0])
print (all_keywords[0])
with open("result.csv", "w", newline="") as f:
    thewriter = csv.writer(f)
    thewriter.writerow(["Topic","PassageID","Rank","Score","Offset","length","tagid","Passage"])
    for i in range(0,len(topics)):
        thewriter.writerow([topics[i],pmid[i],rank[i],score[i],offset[i],length[i],tagid[i],all_keywords[i]])



