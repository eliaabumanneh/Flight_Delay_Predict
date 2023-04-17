from flask import Flask, render_template, request, redirect, url_for
import pickle

##################################################################################################
app = Flask(__name__)

airport_list, airline_list = data_setup()

@app.route('/')
def temp():
    print("this is temp")

    return render_template('webpage.html')

@app.route('/',methods=['POST','GET'])
def index():
    print("This is index")
    if request.method == "POST":

        
        
        airport_index = int(request.form["airport"])
        airline_index = int(request.form["airline"])

        # Make prediction based on selected values
        delay, cancellation = run_pred(airport_index, airline_index)

        return render_template("index.html", airport_list=airport_list, airline_list=airline_list, delay=delay, cancellation=cancellation)
    else:
        return render_template("index.html", airport_list=airport_list, airline_list=airline_list)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5010, debug=True, threaded=True)


#old streamlit code

#html code
#env = Environment(loader=FileSystemLoader('.'))
#template = env.get_template('webpage.html')

#Streamlit Componenets

#initialising containers
#header = st.beta_container()
#input = st.beta_container()

#current_dir = os.getcwd()
#with header: 
#    st.title('Flight Delay Predict')
#    st.write('Data Science Bootcamp Capstone Project')
#    st.write('Elia Abu-Manneh')
#    st.write('April 12 2023')

#    st.write(current_dir)

#with input:
#    selected_index = st.selectbox('Select an airline:', airline_list)