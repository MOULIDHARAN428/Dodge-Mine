from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/instruction')
def instruction():
    return render_template('instruction.html')

@app.route('/predict',methods=['POST'])
def home():
    Sonor_1 = request.form['sonor1']
    Sonor_2 = request.form['sonor2']
    Sonor_3 = request.form['sonor3']
    Sonor_4 = request.form['sonor4']
    Sonor_5 = request.form['sonor5']
    Sonor_6 = request.form['sonor6']
    Sonor_7 = request.form['sonor7']
    Sonor_8 = request.form['sonor8']
    Sonor_9 = request.form['sonor9']
    Sonor_10 = request.form['sonor10']
    Sonor_11 = request.form['sonor11']
    Sonor_12 = request.form['sonor12']
    Sonor_13 = request.form['sonor13']
    Sonor_14 = request.form['sonor14']
    Sonor_15 = request.form['sonor15']
    Sonor_16 = request.form['sonor16']
    Sonor_17 = request.form['sonor17']
    Sonor_18 = request.form['sonor18']
    Sonor_19 = request.form['sonor19']
    Sonor_20 = request.form['sonor20']
    Sonor_21 = request.form['sonor21']
    Sonor_22 = request.form['sonor22']
    Sonor_23 = request.form['sonor23']
    Sonor_24 = request.form['sonor24']
    Sonor_25 = request.form['sonor25']
    Sonor_26 = request.form['sonor26']
    Sonor_27 = request.form['sonor27']
    Sonor_28 = request.form['sonor28']
    Sonor_29 = request.form['sonor29']
    Sonor_30 = request.form['sonor30']
    Sonor_31 = request.form['sonor31']
    Sonor_32 = request.form['sonor32']
    Sonor_33 = request.form['sonor33']
    Sonor_34 = request.form['sonor34']
    Sonor_35 = request.form['sonor35']
    Sonor_36 = request.form['sonor36']
    Sonor_37 = request.form['sonor37']
    Sonor_38 = request.form['sonor38']
    Sonor_39 = request.form['sonor39']
    Sonor_40 = request.form['sonor40']
    Sonor_41 = request.form['sonor41']
    Sonor_42 = request.form['sonor42']
    Sonor_43 = request.form['sonor43']
    Sonor_44 = request.form['sonor44']
    Sonor_45 = request.form['sonor45']
    Sonor_46 = request.form['sonor46']
    Sonor_47 = request.form['sonor47']
    Sonor_48 = request.form['sonor48']
    Sonor_49 = request.form['sonor49']
    Sonor_50 = request.form['sonor50']
    Sonor_51 = request.form['sonor51']
    Sonor_52 = request.form['sonor52']
    Sonor_53 = request.form['sonor53']
    Sonor_54 = request.form['sonor54']
    Sonor_55 = request.form['sonor55']
    Sonor_56 = request.form['sonor56']
    Sonor_57 = request.form['sonor57']
    Sonor_58 = request.form['sonor58']
    Sonor_59 = request.form['sonor59']
    Sonor_60 = request.form['sonor60']
    arr = np.array([[Sonor_1,Sonor_2,Sonor_3,Sonor_4,Sonor_5,Sonor_6,Sonor_7,Sonor_8,Sonor_9,Sonor_10,Sonor_11,Sonor_12
    ,Sonor_13,Sonor_14,Sonor_15,Sonor_16,Sonor_17,Sonor_18,Sonor_19,Sonor_20,Sonor_21,Sonor_22,Sonor_23,Sonor_24,Sonor_25
    ,Sonor_26,Sonor_27,Sonor_28,Sonor_29,Sonor_30,Sonor_31,Sonor_32,Sonor_33,Sonor_34,Sonor_35,Sonor_36,Sonor_37,Sonor_38
    ,Sonor_39,Sonor_40,Sonor_41,Sonor_42,Sonor_43,Sonor_44,Sonor_45,Sonor_46,Sonor_47,Sonor_48,Sonor_49,Sonor_50,Sonor_51
    ,Sonor_52,Sonor_53,Sonor_54,Sonor_55,Sonor_56,Sonor_57,Sonor_58,Sonor_59,Sonor_60]])
    
    input_reshape = arr.reshape(1,-1)
    predict = model.predict(input_reshape)
    
    pred = model.predict(arr)
    
    
    return render_template('results.html',data=pred)

if __name__ == '__main__':
    app.run(debug=True)
