# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
student_data = pd.read_csv(url, compression='zip', sep=';', header=0)

# Preprocess the data
# We will convert categorical variables using one-hot encoding and scale numeric features

# Step 1: Create the target variable
# Let's say we are predicting whether a student passed or failed based on the final grade (G3).
# We'll create a binary target: 1 if G3 >= 10 (pass), 0 if G3 < 10 (fail).
student_data['pass_fail'] = (student_data['G3'] >= 10).astype(int)

# Step 2: Drop unnecessary columns (like G3 since it's directly related to the target)
student_data = student_data.drop(columns=['G3'])

# Step 3: Separate features (X) and target (y)
X = student_data.drop(columns=['pass_fail'])
y = student_data['pass_fail']

# Step 4: Convert categorical features to one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 5: Split the data into training and test sets (optional, but common practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now you have X_train, X_test, y_train, y_test ready for logistic regression
print(X_test)