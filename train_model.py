import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("C:\\Users\\Suhas_2\\Downloads\\student_grade_prediction_dataset.csv")

# Features and target
X = data[['StudyHours', 'PreviousGrade']]
y = data['FinalGrade']

# Train model
model = LinearRegression()
model.fit(X, y)

# Extract coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
