import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm

df = pd.read_csv('original_hotel_bookings.csv')

df = df.drop_duplicates()
df = df.drop(columns=['agent', 'company', 'adr'])  # drop unnecessary columns

# Keeping only US data
df = df[df['country'] == 'USA']

# Convert 'children' to integer (was float)
df['children'] = df['children'].astype(int)

# Look at the overall data stats and quality.
print(df.info())
print(df[['total_of_special_requests']].describe())
print(df.isnull().sum())

# Dummy variables for 'arrival_date_month'
df['arrival_month_dummy'] = df['arrival_date_month']
df = pd.get_dummies(df, columns=['arrival_month_dummy'], prefix='month', dtype=int)

# Dummy variables for 'deposit_type'
df['deposit_dummy'] = df['deposit_type']
df = pd.get_dummies(df, columns=['deposit_dummy'], prefix='dep', dtype=int)

# Dummy variables for 'customer_type'
df['customer_dummy'] = df['customer_type']
df = pd.get_dummies(df, columns=['customer_dummy'], prefix='cus', dtype=int)

# Dummy variables for 'hotel'
df['hotel_dummy'] = df['hotel']
df = pd.get_dummies(df, columns=['hotel_dummy'], prefix='h', dtype=int)

print(df.columns)

# Balancing data
df_sample = df[df['is_canceled'] == 0].sample(frac=.5)
df_canceled = df[df['is_canceled'] == 1]

df = pd.concat([df_sample, df_canceled])
print(df.info())

print(df['is_canceled'].mean())

# Logistic regression
# Split the data into features and target
X = df[['lead_time', 'month_April', 'month_August', 'month_December', 'month_February', 'month_January', 'month_July',
        'month_June', 'month_March', 'month_May', 'month_November', 'month_October', 'month_September',
        'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'is_repeated_guest',
        'previous_cancellations', 'cus_Contract', 'cus_Group', 'cus_Transient', 'cus_Transient-Party',
        'dep_No Deposit', 'dep_Refundable', 'adults', 'children', 'babies', 'total_of_special_requests',
        'h_City Hotel', 'h_Resort Hotel']]  # Features
Y = df['is_canceled']  # Target

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Predictions
Y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
