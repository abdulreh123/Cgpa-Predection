
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
data = pd.read_csv("letter-recognition.data")
le = preprocessing.LabelEncoder()

xbox = le.fit_transform(list(data["x-box"]))
ybox = le.fit_transform(list(data["y-box"]))
width = le.fit_transform(list(data["width"]))
high = le.fit_transform(list(data["high"]))
xbar  = le.fit_transform(list(data["x-bar"]))
ybar = le.fit_transform(list(data["y-bar"]))
x2bar = le.fit_transform(list(data["x2bar"]))
y2bar = le.fit_transform(list(data['y2bar']))
onpix = le.fit_transform(list(data['onpix']))
xybar = le.fit_transform(list(data['xybar']))
x2ybr = le.fit_transform(list(data["x2ybr"]))
xy2br = le.fit_transform(list(data['xy2br']))
xege = le.fit_transform(list(data['x-ege']))
xegvy = le.fit_transform(list(data['xegvy']))
yege = le.fit_transform(list(data['y-ege']))
yegvx = le.fit_transform(list(data['yegvx']))

lettr = le.fit_transform(list(data['lettr']))

predict = 'lettr'
# X = list(zip(xbox, ybox, width, high, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybr, xy2br, xege, xegvy, yege, yegvx))
X = np.array(data.drop([predict], 1))
y = list(lettr)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predicted = model.predict(x_test)
print(acc)

names = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z' ]

# for x in range(len(predicted)):
    # print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])


