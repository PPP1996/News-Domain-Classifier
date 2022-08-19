from flask import Flask,render_template,request
import pickle
import json
import numpy as np

model = pickle.load(open("model.pkl","rb"))

count_vectorizer = pickle.load(open("count_vec.pkl","rb"))

with open("columns_name.json","r") as json_file:
    col_name = json.load(json_file)
col_name_list = col_name['col_name']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods= ["GET","POST"])
def predict():
    data = request.form["text"]
    input_data = np.zeros(len(col_name_list))

    Text = ["".join(data)]
    input_data = count_vectorizer.transform(Text).toarray()

    # Text = count_vectorizer.transform('text').toarray()
    # user_count_vec = ["".join(data[Text])]
    # input_data[0] = user_count_vec

    print(input_data)
    my_prediction = model.predict(input_data)
    if my_prediction[0]== 0:
        result = "business"
    elif my_prediction[0]== 1:
        result = "entertainment"
    elif my_prediction[0]== 2:
        result = "politics"            
    elif my_prediction[0]== 3:
        result = "sport"        
    else:
        result = "tech"
    print(result)
    
    return render_template("index.html",prediction = result)

if __name__== "__main__":
    app.run(host= "0.0.0.0",port=8080, debug=True) ##### AWS Deployment host = 0.0.0.0 port= 8080 debug= False