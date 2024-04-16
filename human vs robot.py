import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE

# Function to collect user input
def collect_user_data():
    print("Please answer the following questions on a scale of 1 to 10 (1 being least, 10 being most):")
    
    age = int(input("How old are you? "))
    happiness_level = int(input("On a scale of 1 to 10, how happy do you feel right now? "))
    anger_level = int(input("On a scale of 1 to 10, how angry do you feel right now? "))
    speech_rate = int(input("How fast do you usually speak? (words per minute) "))
    eye_contact = int(input("How often do you make eye contact during conversation? (1 being rarely, 10 being always) "))
    emotion_recognition = int(input("How good are you at recognizing emotions in others? (1 being poor, 10 being excellent) "))
    logical_thinking = int(input("How would you rate your logical thinking skills? (1 being poor, 10 being excellent) "))
    physical_movements = int(input("How would you rate your physical movements? (1 being clumsy, 10 being precise) "))
    
    return {
        'age': age,
        'happiness_level': happiness_level,
        'anger_level': anger_level,
        'speech_rate': speech_rate,
        'eye_contact': eye_contact,
        'emotion_recognition': emotion_recognition,
        'logical_thinking': logical_thinking,
        'physical_movements': physical_movements
    }

# Generate synthetic dataset
np.random.seed(42)

n = 2000
human_data = {
    'age': np.random.randint(20, 70, n),
    'happiness_level': np.random.randint(7, 10, n) + np.random.rand(n),
    'anger_level': np.random.randint(1, 3, n) + np.random.rand(n),
    'speech_rate': np.random.randint(90, 140, n) + np.random.rand(n) * 10,
    'eye_contact': np.random.randint(8, 10, n),
    'emotion_recognition': np.random.randint(8, 10, n),
    'logical_thinking': np.random.randint(7, 10, n),
    'physical_movements': np.random.randint(7, 10, n),
    'species': 'human'
}

robot_data = {
    'age': np.random.randint(2, 25, n),
    'happiness_level': np.random.randint(4, 7, n) + np.random.rand(n),
    'anger_level': np.random.randint(1, 3, n) + np.random.rand(n),
    'speech_rate': np.random.randint(140, 190, n) + np.random.rand(n) * 10,
    'eye_contact': np.random.randint(1, 5, n),
    'emotion_recognition': np.random.randint(1, 5, n),
    'logical_thinking': np.random.randint(5, 8, n),
    'physical_movements': np.random.randint(5, 8, n),
    'species': 'robot'
}

df_human = pd.DataFrame(human_data)
df_robot = pd.DataFrame(robot_data)

# Combine the dataframes
df = pd.concat([df_human, df_robot])

# Balance the classes using SMOTE
X = df.drop('species', axis=1)
y = (df['species'] == 'robot').astype(int)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(clf, 'random_forest_model_robot_or_human.pkl')

# Load the trained model
clf = joblib.load('random_forest_model_robot_or_human.pkl')

# ... [Previous code remains unchanged up to the user input section]

# ... [Previous code remains unchanged up to the user input section]

# ... [Previous code remains unchanged up to the user input section]

# ... [Previous code remains unchanged up to the user input section]

# Collect user data and predict
try:
    user_data = collect_user_data()

    # Limit user input values to a more realistic range
    for key in user_data:
        if user_data[key] < 1:
            user_data[key] = 1
        elif user_data[key] > 10:
            user_data[key] = 10
    
    # Print user input for debugging
    print("\nUser Input:")
    for key, value in user_data.items():
        print(f"{key}: {value}")

    # Create DataFrame from user input
    df_user = pd.DataFrame([user_data])
    
    # Scale user data to match the training data
    user_data_scaled = scaler.transform(df_user)
    
    # Predict
    prediction = clf.predict(user_data_scaled)
    
    if prediction[0] == 1:
        print("\nBased on your answers, it seems you are a ROBOT! 🤖")
        print("Here's a puzzle for you: What comes once in a minute, twice in a moment, but never in a thousand years?")
    else:
        print("\nBased on your answers, it seems you are a HUMAN! 👤")
        print("Here's a puzzle for you: What has keys but can't open locks?")
        
except Exception as e:
    print("An error occurred:", e)
