
# import numpy as np
# import pandas as pd
# import sklearn
# from sklearn import linear_model
import pickle
import uvicorn
import json
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,Query

# data =pd.read_csv("studentCgpa.csv")
# predict="final"
# X = np.array(data.drop([predict], 1))
# y = np.array(data[predict])
#
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
# linear = linear_model.LinearRegression()
#
# X = np.array(data.drop(["cgpa_two","cgpa_three","cgpa_four","cgpa_five","cgpa_six", "cgpa_seven","final"], 1))
# y = np.array(data["final"])
#
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
# linear = linear_model.LinearRegression()
#
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
#
# with open("studentCgpaOne.pickle", "wb") as f:
#     pickle.dump(linear, f)
# print(linear.coef_)
# print(linear.intercept_)
#
# predictions = linear.predict(x_test)
# for x in range (len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])


# print (custom)
# plot = "cgpa_one" # Change this to G1, G2, studytime or absences to see other graphs
# plt.scatter(data[plot], data["final"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()
# print(custom)
app = FastAPI()
origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get('/')
async def index():
    pickle_in = open("studentCgpaThree.pickle", "rb")
    linear = pickle.load(pickle_in)
    customs = linear.predict([[3.1,2.8]])
    return {"result": customs[0]}


@app.get('/predict/{value}')
async def predict(name):
    y = json.loads(name)
    length = len(y)
    if length == 1:
          pickle_in = open("studentCgpaOne.pickle", "rb")
          linear = pickle.load(pickle_in)
          customs = linear.predict([y])
          return {"result": customs[0]}

    elif length == 2:
          pickle_in = open("studentCgpaTwo.pickle", "rb")
          linear = pickle.load(pickle_in)
          customs = linear.predict([y])
          return {"result": customs[0]}

    elif length == 3:
        pickle_in = open("studentCgpaThree.pickle", "rb")
        linear = pickle.load(pickle_in)
        customs = linear.predict([y])
        return {"result": customs[0]}

    elif length == 4:
        pickle_in = open("studentCgpaFour.pickle", "rb")
        linear = pickle.load(pickle_in)
        customs = linear.predict([y])
        return {"result": customs[0]}

    elif length == 5:
        pickle_in = open("studentCgpaFive.pickle", "rb")
        linear = pickle.load(pickle_in)
        customs = linear.predict([y])
        return {"result": customs[0]}
    elif length == 6:
        pickle_in = open("studentCgpaSix.pickle", "rb")
        linear = pickle.load(pickle_in)
        customs = linear.predict([y])
        return {"result": customs[0]}
    elif length == 7:
        pickle_in = open("studentCgpaSeven.pickle", "rb")
        linear = pickle.load(pickle_in)
        customs = linear.predict([y])
        return {"result": customs[0]}

if __name__ == '__main__':
	uvicorn.run(app, host="127.0.0.1", port=7002)
