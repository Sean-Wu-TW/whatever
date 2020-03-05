import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
import csv
import random
import matplotlib.pyplot as plt
import os

def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label
    df = pd.read_csv(input_csv_file)
    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r", encoding="utf-8")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return df, tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):
    # Store to Desktop
    # path = os.path.join(os.environ['USERPROFILE'], 'Desktop', output_csv_file)
    with open(output_csv_file, mode='w', encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])


# logistic regression class
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = np.zeros((n_in, n_out))  # initialize W 0
        self.b = np.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    @staticmethod
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:  
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
        
    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = self.softmax(np.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)
        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = self.softmax(np.dot(self.x, self.W) + self.b)

        cross_entropy = - np.mean(
            np.sum(self.y * np.log(sigmoid_activation) +
            (1 - self.y) * np.log(1 - sigmoid_activation),
                      axis=1))
        return cross_entropy

        
    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return self.softmax(np.dot(x, self.W) + self.b)

class Account():
    
    def __init__(self, S):
        # set id
        self.ID = S["tweet_id"]
        # set issue feature vector
        self.issue = [1 if ['guns', 'aca', 'lgbt', 'immig', 'isis', 'abort'].index(S["issue"]) == i else 0 for i in range(6)]
        # set author vector
        if S["author"] == 'democrat':
            self.author = [1]
        else:
            self.author = [0]
        # clean text 
        text = [re.sub(r'@\w+','',i) for i in S["text"].split()]
        text = [re.sub(r'http.?://[^\s]+[\s]?','',i) for i in text]
        text = [re.sub(r'\W+','',i) for i in text]
        text = [re.sub(r'RT','',i) for i in text]
        text = [re.sub(r'\d+','',i) for i in text]
        text = [i.lower() for i in text]
        text = [PorterStemmer().stem(i) for i in text]
        text = ['' if i in stopwords.words('english') else i for i in text]
        text = [i for i in text if i]
        text = list(dict.fromkeys(text))
        self.text = text
        # form label
        if S["label"] != 'None':
            self.label = [1 if int(S["label"])-1 == i else 0 for i in range(17)]
        else:
            self.label = None
        
    def getFeatures(self):
        ''' returns a list of features '''
        assert type(self.text) == list
        return (self.issue + self.author + self.text)
    
    def getLabels(self):
        ''' returns label '''
        return self.label
    
class TestCase():
    
    def __init__(self, S, word_model):
        self.ID = S["tweet_id"]
        self.issue = [1 if ['guns', 'aca', 'lgbt', 'immig', 'isis', 'abort'].index(S["issue"]) == i else 0 for i in range(6)]
        if S["author"] == 'democrat':
            self.author = [1]
        else:
            self.author = [0]
        text = [re.sub(r'@\w+','',i) for i in S["text"].split()]
        text = [re.sub(r'http.?://[^\s]+[\s]?','',i) for i in text]
        text = [re.sub(r'\W+','',i) for i in text]
        text = [re.sub(r'RT','',i) for i in text]
        text = [re.sub(r'\d+','',i) for i in text]
        text = [i.lower() for i in text]
        text = [PorterStemmer().stem(i) for i in text]
        text = ['' if i in stopwords.words('english') else i for i in text]
        text = [i for i in text if i]
        text = list(dict.fromkeys(text))
        self.text = list(sum([word_model.wv[i] if i in word_model.wv else np.zeros(len(word_model.wv['last'])) for i in text]))
        
    def getFeatures(self):
        assert type(self.text) == list
        return (self.issue + self.author + self.text)
    
    
def buildModel(examples, toPrint = True):
    ''' builds training model '''
    featureVecs, labels = [], []
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabels())
    classifier = LogisticRegression(np.array(featureVecs), np.array(labels), n_in=len(e.getFeatures()), n_out=len(e.getLabels()))
    return classifier


def train_classifier(classifier, lr=0.01, ep=500):
    learning_rate = lr
    for epoch in range(ep):
        classifier.train(lr=learning_rate)
        learning_rate *= 0.95
    return classifier

def buildExamples(df):
    ''' create a list of examples , returns examples and a word model'''
    examples = []
    for i in range(len(df)):
        examples.append(Account(df.iloc[i]))
    word_model = Word2Vec([i.text for i in examples], min_count=1, sg=1)
    for x in examples:
        x.text = sum([word_model.wv[i] for i in x.text])
    for e in examples:
        e.text = list(e.text)
    return examples, word_model
        
def sigmoid_derv(s):
    return s * (1 - s)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    
    def __init__(self, x, y, lr=0.5):
        self.x = x
        neurons = 128
        self.lr = lr
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backprop(self):
        loss = error(self.a3, self.y)
        # print('Loss :', loss)
        a3_delta = cross_entropy(self.a3, self.y) #w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) #w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) #w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()

def get_acc(x, y, model):
    # get NN accuracy
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100



def split80_20(examples):
    ''' splits examples, return training(80%) and testing(20%)'''
    sampleIndices = random.sample(range(len(examples)), len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet

def learning_rate_change_LR(examples, _from=0.001, _to=0.05):
    ''' sees accuracy change over learning rate for LR'''
    trainingSet, testSet = split80_20(examples)
    acc = []
    x_axis = []
    lr = _from
    while lr < _to:
        model = train_classifier(buildModel(trainingSet), lr)
        r = 0
        for i in testSet:
            if i.getLabels().index(1) == np.argmax(model.predict(np.array(i.getFeatures()))):
                r += 1
        acc.append(r/len(testSet))
        x_axis.append(lr)
        print("LR Training Accuracy: " + str(r/len(testSet)))
        lr *= 1.3
    return x_axis, acc
    
def learning_rate_change_NN(X_train, X_test, y_train, y_test):
    ''' sees accuracy change over learning rate for LR'''
    lr = _from 
    acc = []
    x_axis = []
    while lr < _to:
        model = MyNN(np.array(X_train), np.array(X_test), lr)
        epochs = 500
        for x in range(epochs):
            model.feedforward()
            model.backprop()
        ac = get_acc(np.array(y_train), np.array(y_test), model)*0.01
        acc.append(ac)
        x_axis.append(lr)
        print("NN Training accuracy:", ac)
        lr *= 1.3
    return x_axis, acc

def LR():
    # Read training data
    df, train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Logistic Regression classifier here
    '''
    examples, word_model = buildExamples(df)
    
    trainingSet, testSet = split80_20(examples)
    
    model = buildModel(trainingSet)
    model = train_classifier(model)
    
    r = 0
    for i in testSet:
        if i.getLabels().index(1) == np.argmax(model.predict(np.array(i.getFeatures()))):
            r += 1
    print("LR Training Accuracy: " + str(r/len(testSet)))
    
    
    x, y = learning_rate_change_LR(examples, _from=0.001, _to=0.005)
    # plot figure for LR
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,1,1])
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Logistic Regression')
    ax.plot(x, y)
    plt.savefig('LR', dpi=500, bbox_inches = 'tight')
    plt.show()

    
    # Read test data
    dft, test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')

    # Predict test data by learned model

    '''
    Replace the following random predictor by your prediction function.
    '''
    gg = []
    for i in range(len(dft)):
        guess = np.argmax(model.predict(TestCase(dft.iloc[i], word_model).getFeatures()))
        gg.append(guess)
    for i in range(len(gg)):
        dft.at[i, "label"] = gg[i]
    dft = dft.set_index('tweet_id')
    for i in test_tweet_id2label:
        test_tweet_id2label[i] = dft.loc[i]["label"]
    
    
#    for tweet_id in test_tweet_id2text:
#        # Get the text
#        text = test_tweet_id2text[tweet_id]
#        # Get author
#        author = test_tweet_id2author_label[tweet_id]
#        # Get issue
#        issue = test_tweet_id2issue[tweet_id]
#        
#        # Predict the label
#        label=randrange(1, 18)
#
#        # Store it in the dictionary
#        test_tweet_id2label[tweet_id]=label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')


def NN():

    # Read training data
    df, train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Neural Network classifier here
    '''
    df.index = df["tweet_id"]
    df = df.drop("tweet_id", axis=1)
    df_vec = df.copy()
    df_vec["issue"] = df_vec["issue"].apply(lambda x:list(df_vec["issue"].unique()).index(x)).apply(lambda x:[1 if x == i else 0 for i in range(len(df["issue"].unique()))])
    df_vec["author"] = df_vec["author"].apply(lambda x: {l:[i] for i,l in enumerate(df["author"].unique())}[x])

    df_vec["text"] = df_vec["text"].apply(lambda x:[re.sub(r'@\w+','',i) for i in x.split()])\
    .apply(lambda x:[re.sub(r'http.?://[^\s]+[\s]?','',i) for i in x])\
    .apply(lambda x:[re.sub(r'\W+','',i) for i in x])\
    .apply(lambda x:[re.sub(r'RT','',i) for i in x])\
    .apply(lambda x:[re.sub(r'\d+','',i) for i in x])\
    .apply(lambda x:[i.lower() for i in x])\
    .apply(lambda x:[PorterStemmer().stem(i) for i in x])\
    .apply(lambda x:['' if i in stopwords.words('english') else i for i in x ])\
    .apply(lambda x:[i for i in x if i])\
    .apply(lambda x:list(dict.fromkeys(x)))
    
    model = Word2Vec(list(df_vec["text"]), min_count=1)
    df_vec["text"] = df_vec["text"].apply(lambda x:list(sum(model.wv[i] for i in x)))
    df_vec["new"] = df_vec["issue"]+df_vec["text"]+df_vec["author"]
    df_vec["label"] = df_vec["label"].apply(lambda x:[1 if x-1 == i else 0 for i in range(17)])
    
    train = df_vec["new"]
    train_label = df_vec["label"]
    
    inp = []
    for i in train:
        inp.append(i)
    lab = []
    for j in train_label:
        lab.append(j)
    inp = np.array(inp)
    lab = np.array(lab)

    model = MyNN(np.array(inp), np.array(lab), 0.5)
    
    epochs = 1000
    for x in range(epochs):
        model.feedforward()
        model.backprop()
    
    
    def get_prec(x, y):
        prec_lab = []
        for xx,yy in zip(x, y):
            s = model.predict(xx)
            prec_lab.append(s+1)
        return prec_lab
    
    
    
    
    
    
    # Read test data
    dft, test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    
    
    dft.index = dft["tweet_id"]
    dft = dft.drop("tweet_id", axis=1)
    dft_vec = dft.copy()
    dft_vec["issue"] = dft_vec["issue"].apply(lambda x:list(dft_vec["issue"].unique()).index(x)).apply(lambda x:[1 if x == i else 0 for i in range(len(dft["issue"].unique()))])
    dft_vec["author"] = dft_vec["author"].apply(lambda x: {l:[i] for i,l in enumerate(df["author"].unique())}[x])

    dft_vec["text"] = dft_vec["text"].apply(lambda x:[re.sub(r'@\w+','',i) for i in x.split()])\
    .apply(lambda x:[re.sub(r'http.?://[^\s]+[\s]?','',i) for i in x])\
    .apply(lambda x:[re.sub(r'\W+','',i) for i in x])\
    .apply(lambda x:[re.sub(r'RT','',i) for i in x])\
    .apply(lambda x:[re.sub(r'\d+','',i) for i in x])\
    .apply(lambda x:[i.lower() for i in x])\
    .apply(lambda x:[PorterStemmer().stem(i) for i in x])\
    .apply(lambda x:['' if i in stopwords.words('english') else i for i in x ])\
    .apply(lambda x:[i for i in x if i])\
    .apply(lambda x:list(dict.fromkeys(x)))
    
    word_model = Word2Vec(list(dft_vec["text"]), min_count=1)
    dft_vec["text"] = dft_vec["text"].apply(lambda x:list(sum(word_model.wv[i] for i in x)))
    dft_vec["new"] = dft_vec["issue"]+dft_vec["text"]+dft_vec["author"]
    
    inpt = []
    for i in train:
        inpt.append(i)
    inpt = np.array(inpt)
    
    res = get_prec(np.array(inp), np.array(inpt))

    for i, j in zip(test_tweet_id2label,res):
        test_tweet_id2label[i] = j



    #  test acc
    train = df_vec["new"].head(1000).copy()
    train_label = df_vec["label"].head(1000).copy()

    inp = []
    for i in train:
        inp.append(i)
    lab = []
    for j in train_label:
        lab.append(j)
    inp = np.array(inp)
    lab = np.array(lab)
    test = df_vec["new"][1000:].copy()
    test_label = df_vec["label"][1000:].copy()

    inpt = []
    for i in test:
        inpt.append(i)
    labt = []
    for j in test_label:
        labt.append(j)
    inpt = np.array(inpt)
    labt = np.array(labt)
    X_train, X_test, y_train, y_test = inp, lab, inpt, labt

    model = MyNN(np.array(X_train), np.array(X_test))

    epochs = 1000
    for x in range(epochs):
        model.feedforward()
        model.backprop()

    print("NN Training accuracy: ", get_acc(np.array(y_train), np.array(y_test), model)*0.01)
    
    x, y = learning_rate_change_NN(X_train, X_test, y_train, y_test)
    # plot figure for LR
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,1,1])
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Multi-layer Neural Network')
    ax.plot(x, y)
    plt.savefig('NN', dpi=500, bbox_inches = 'tight')
    plt.show()

    
    # Predict test data by learned model
    # Replace the following random predictor by your prediction function

    # for tweet_id in test_tweet_id2text:
    #     # Get the text
    #     text=test_tweet_id2text[tweet_id]

    #     # Predict the label
    #     label=randrange(1, 18)

    #     # Store it in the dictionary
    #     test_tweet_id2label[tweet_id]=label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

if __name__ == '__main__':
   LR()
   NN()
