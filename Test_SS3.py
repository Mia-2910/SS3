from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('C:\Users\Vu Minh Anh\.ipython\20% project IFP\Weather prediction 2.csv', 
                   skiprows = 35065,
                   skipfooter = 30,
                   names = ['Timestamp','Basel Temperature [2 m elevation corrected]','Basel Cloud Cover Total', 'Basel UV Radiation'],
                   sep = ","
                   )
# Step 1: Ensure it's a string so we can clean up any formatting like 'T'
data['Timestamp'] = data['Timestamp'].astype(str)

# Step 2: Remove 'T' if it exists
data['Timestamp'] = data['Timestamp'].str.replace('T', '', regex=False)

# Step 3: Let pandas parse it automatically (handles mixed formats)
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Step 4: Convert to Unix timestamp in seconds
data['Timestamp_numeric'] = data['Timestamp'].astype('int64') / 1e9

# Step 5: Drop the original timestamp column
data = data.drop(columns=['Timestamp'])


def data_train():
     global clf  # Make model available globally

     X = data.drop(["Basel Temperature [2 m elevation corrected]"], axis=1) 
       #X is your feature matrix (i.e., the data used to make predictions)
     y = data["Basel Temperature [2 m elevation corrected]"]
        
     X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5, 
                                                        random_state=123, 
                                                        ) 
     
  
     scaler = preprocessing.StandardScaler().fit(X_train)
     X_train_scaled = scaler.transform(X_train)
 
     #A modeling pipeline that first transforms the data using StandardScaler() 
     #and then fits a model using a random forest regressor 

     pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100,
                         random_state=123))

     #Model cant learned directly from data
     hyperparameters = { 'randomforestregressor__max_features' : ['10', '10', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1], }    

     #Solving the NaN value in x and y by replacing them with 0.2 
     X_train = np.nan_to_num(X_train, nan=0.2)
     X_test = np.nan_to_num(X_test, nan = 0.2)
     y_train = np.nan_to_num(y_train, nan=0.2)
     y_test = np.nan_to_num(y_test, nan = 0.2)

     #This will help make sure that all features are treated equally by the model.
     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train)
     X_test_scaled = scaler.transform(X_test)

     # Train RandomForest model
     clf = RandomForestRegressor(n_estimators = 10)
     clf.fit(X_train_scaled, y_train)

     # Predict and evaluate
     pred_clf = clf.predict(X_test_scaled)
     print("Random Forest R² score:", r2_score(y_test, pred_clf))
     print("Random Forest MSE:", mean_squared_error(y_test, pred_clf))
    
     joblib.dump(clf, 'SS3 weather_prediction')

def run_menu():
    print("*" *48)
    print("-" *10 + " What would you like to do? " + "-" *10)
    print("\n")
    print("1. Look up the weather on a specific day")
    print("2. Predict the weather on a specific day")
    print("3. Break")
    print("\n")

    option = input("Enter option: ")
          
    
def run_program(option):
        if option == "1":
            date = input("Enter a date (y-m-d)")
            try:
                date_obj = datetime.datetime.strptime(date, "%y-%m-%d")
                date_unix = int(date_obj.timestamp())
                print(f"Weather on {date}: {get_the_weather(date_unix)}°C\n")
            except ValueError:
                print("Invalid format")

        elif option == "2":
            predict_weather()
            
        elif option == "3":
            print("Exit")
            exit()
        else:
            print("Invalid option. Please chose again!")

def get_the_weather(date): #Returns specific temperature for a specififc day
    weather = data['Timestamp_numeric']
    temp = data['Basel Temperature [2 m elevation corrected]']
    for i in range(len(weather)):
        if weather.iloc[i] == date:
            return temp.iloc[i]
    return "No data available for this date."
        
def predict_weather():
    
    clf = joblib.load('SS3 weather_prediction')  # Load trained model
   
    print("-" * 48)
    print("Enter the details of the date you would like to predict")
    print("\n")
    year = input("Year: ")
    month = input("Month number (00): ")
    day = input("Day number (00): ")

    try:
        date_obj = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        date_unix = int(date_obj.timestamp())

        next_day_obj = date_obj + datetime.timedelta(days=1)
        next_day_unix = int(next_day_obj.timestamp())

        X = [[date_unix, next_day_unix]]
        predicted_temp = clf.predict(X)[0]

        print("\n" + "-" * 48)
        print(f"The temperature is predicted to be: {predicted_temp:.2f}°C")
        print(f"The temperature was actually: {get_the_weather(date_unix)}°C")
        print("-" * 48 + "\n")
    except ValueError:
        print("Invalid date input")


if __name__== "__main__":
    data_train()

    while True:
        option = run_menu()
        if option == "3":  
            print("Exit")
            False
            break 
        else:
            run_program(option)