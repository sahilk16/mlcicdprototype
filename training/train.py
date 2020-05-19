# Import libraries
import numpy as np 
import pandas as pd
from scipy import stats
import random, warnings
from copy import deepcopy 
import math
import pickle 

from funtions import Scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

random.seed(101)
warnings.filterwarnings("ignore")

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace


#Loading the training dataset
print("Loading training data...")
datastore = ws.get_default_datastore()
datastore_paths_train = [(datastore, 'pred_mnt/training_set.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths_train)
train = traindata.to_pandas_dataframe()
print("Columns:", train.columns) 
print("Diabetes data set dimensions : {}".format(train.shape))


#Getting the value for output from the dataset variables

torque_vector = train['Turbine_shaft_torque_kNm']
rpm_vector = train['Turbine_rpm']
torque_vector = torque_vector.values
rpm_vector = rpm_vector.values
# Torque: Knm -> Nm
torque_vector = torque_vector*1000
# Speed: RPM -> RPS (rev per sec)
rps_vector = np.divide(rpm_vector, 60) 
# Calculate power in Watt
output_power = 2*(math.pi)*rps_vector*torque_vector
# Power: Watt -> Kilowatt
output_power = np.divide(output_power, 1000) 
# Power: Kilowatt -> Megawatt
output_power = np.divide(output_power, 1000) 
output_power = pd.DataFrame(output_power)



#After analyzing chosing the values which are highly correlated to output(output_power) 
df_train = train[['Turbine_exit_pressure_bar', 'Turbine_exit_temperature_C', 'Ship_speed_knots']]
df_test = output_power

#Splitting the data for train and validation set 
X_train, X_test, y_train, y_test = train_test_split(df_train, df_test, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

#Scaling the data
print('Scale the data')

#Reshape the testdata before scaling
y_train = y_train.reshape(-1, 1)
y_test = y_train.reshape(-1,1)


X_train, y_train = Scaling(X_train, y_train)
X_test, y_test = Scaling(X_test, y_test)


#Training on a Linear Regression model
LR_model = LinearRegression()
model = LR_model.fit(data['train']['X'], data['train']['y'])
run.log_list("coefficients", model.coef_)

#Evaluating the model
print('Evaluating the model')
model_test = model.predict(data['test']['X'])
r_square = metrics.r2_score( data["test"]["y"], model_test)
MAE = metrics.mean_absolute_error(data["test"]["y"], model_test)
print("Mean Absolute Error:", mse)
run.log("mse", mse)
print('R2 error:', r_square)
run.log('r2', r_square) 


# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_turbine_output.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(model, model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()






