from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':

        file=open('cropmodel.pkl','rb')
        model=pickle.load(file)

        n=float(request.form['n'])
        p = float(request.form['p'])
        k = float(request.form['k'])
        temp = float(request.form['temp'])
        hum = float(request.form['hum'])
        ph = float(request.form['ph'])
        rain = float(request.form['rain'])

        testdata=np.array([[n,p,k,temp,hum,ph,rain]])
        prediction=model.predict(testdata)
        print(prediction)
        print(f" prediction crop is :{prediction[0]}")
        msg=f"recommended crop is : {prediction[0]}"
        return render_template('index.html',res=msg)

    else:
        return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")
