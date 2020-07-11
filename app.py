from flask import Flask , render_template , request , url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/survey',methods=['POST'])
def main():
    name = request.form['name']
    email = request.form['email']
    
    return render_template('main.html',name=name,email=email)


@app.route('/predict',methods= ['POST'])
def predict():
	Fuel_Type_Diesel = 0
	Present_Price = float(request.form['Present_Price'])
	Present_Price = np.log(Present_Price)
	Kms_Driven = int(request.form['Kms_Driven'])
	Kms_Driven = np.log(Kms_Driven)
	owner= int(request.form['Owner'])
	Year = int(request.form['Year'])
	Year = 2020 - Year
	Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
	if(Fuel_Type_Petrol=='Petrol'):
		Fuel_Type_Petrol=1
		Fuel_Type_Diesel=0
	else:
		Fuel_Type_Petrol=0
		Fuel_Type_Diesel=1

	Seller_Type_Individual=request.form['Seller_Type_Individual']
	if(Seller_Type_Individual=='Individual'):
		Seller_Type_Individual=1
	else:
		Seller_Type_Individual=0	

	Transmission_Mannual=request.form['Transmission_Mannual']
	if(Transmission_Mannual=='Mannual'):
		Transmission_Mannual=1
	else:
		Transmission_Mannual=0

	prediction=model.predict([[Present_Price,Kms_Driven,owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
	output=round(prediction[0],2)

	return render_template('after.html',output=output)




	

if __name__ == '__main__':
	app.run(debug=True)
