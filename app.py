import streamlit as st
import json
import bcrypt
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

USERS_FILE = 'users.json'

# Load users from file or create empty dict if missing
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)


# Save updated users dict to file
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


# Hash plain password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# Check hashed password vs plain password
def check_password(hashed, password):
    return bcrypt.checkpw(password.encode(), hashed.encode())


# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'username' not in st.session_state:
    st.session_state['username'] = ''


st.title("PhonePro Authentication")

users = load_users()

menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])

if menu == "Sign Up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Sign Up"):
        if new_user in users:
            st.error("Username already exists")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        elif new_user == "" or new_password == "" or new_email == "":
            st.error("Please fill out all fields")
        else:
            # Store new user with hashed password
            users[new_user] = {
                "email": new_email,
                "password": hash_password(new_password)
            }
            save_users(users)
            st.success("User created successfully! Please login.")

elif menu == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username not in users:
            st.error("User does not exist")
        elif not check_password(users[username]['password'], password):
            st.error("Incorrect Password")
        else:
            st.success(f"Welcome, {username}!")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username

# If logged in, show PhonePro app else prompt login/signup
if st.session_state['logged_in']:
    st.write(f"Logged in as: {st.session_state['username']}")

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.rerun()

    # Load label encoders and scaler
    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load datasets
    df = pd.read_csv('processed_dataset.csv')
    df_original = pd.read_csv('CleanedDataset.csv')  # Load the cleaned dataset

    features = ['price', 'brand_name', '5G_or_not', 'processor_brand', 'battery_capacity',
                'ram_capacity', 'internal_memory', 'refresh_rate', 'os', 'primary_camera_rear',
                'fast_charging']
    target = 'model'

    # Define feature weights (higher values indicate higher priority)
    weights = {
        'brand_name': 5,
        'price': 4,
        'os': 3,
        'processor_brand': 2,
        '5G_or_not': 1,
        'battery_capacity': 1,
        'ram_capacity': 1,
        'internal_memory': 1,
        'refresh_rate': 1,
        'primary_camera_rear': 1,
        'fast_charging': 1
    }

    st.title("Welcome to PhonePro")
    st.write("### Your Personal Phone Recommendation System")

    st.sidebar.header("Explore PhonePro")
    st.sidebar.write("### About PhonePro")
    st.sidebar.write("PhonePro helps you find the best phone based on your preferences.")

    st.sidebar.write("### Search Phone Details")
    search_model = st.sidebar.selectbox("Search phone model", options=df_original[target].unique(), index=0)

    if search_model:
        result = df_original[df_original[target] == search_model].iloc[0]
        st.sidebar.write(f"### Details for {search_model} :")
        st.sidebar.write(result)

    st.header("Get Phone Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        input_data = {
            'price': st.slider('Price (in INR)', min_value=0, max_value=100000, step=1000, value=15000),
            'brand_name': st.selectbox('Brand Name', options=label_encoders['brand_name'].classes_),
            '5G_or_not': st.selectbox('5G Support', options=['No', 'Yes']),
            'processor_brand': st.selectbox('Processor Brand', options=label_encoders['processor_brand'].classes_),
            'primary_camera_rear': st.slider('Primary Rear Camera (MP)', min_value=5, max_value=108, step=2, value=64),
            'fast_charging': st.slider('Charging(W)', min_value=0, max_value=120, step=5, value=25)
        }
    with col2:
        input_data.update({
            'battery_capacity': st.slider('Battery Capacity (mAh)', min_value=1000, max_value=10000, step=100, value=4500),
            'ram_capacity': st.slider('RAM Capacity (GB)', min_value=1, max_value=16, step=1, value=6),
            'internal_memory': st.slider('Internal Memory (GB)', min_value=8, max_value=1024, step=64, value=128),
            'refresh_rate': st.slider('Refresh Rate (Hz)', min_value=30, max_value=144, step=5, value=90),
            'os': st.selectbox('Operating System', options=label_encoders['os'].classes_)
        })

    # Convert 'Yes'/'No' to 1/0
    input_data['5G_or_not'] = 1 if input_data['5G_or_not'] == 'Yes' else 0

    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    for column in ['brand_name', 'processor_brand', 'os']:
        input_df[column] = label_encoders[column].transform(input_df[column])

    input_df_scaled = input_df.copy()
    input_df_scaled[['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'refresh_rate', 'primary_camera_rear', 'fast_charging']] = \
        scaler.transform(input_df[['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'refresh_rate', 'primary_camera_rear', 'fast_charging']])

    # Apply feature weights
    def apply_weights(df, weights):
        weighted_df = df.copy()
        for feature, weight in weights.items():
            if feature in weighted_df.columns:
                weighted_df[feature] = weighted_df[feature] * weight
        return weighted_df

    # Filter the dataset based on the selected brand
    filtered_df = df[df['brand_name'] == input_df['brand_name'].iloc[0]]

    if not filtered_df.empty:
        # Apply weights to features
        df_features_weighted = apply_weights(filtered_df[features], weights)
        input_features_weighted = apply_weights(input_df_scaled[features], weights)

        # Compute similarity on the weighted features
        similarity_scores = cosine_similarity(input_features_weighted, df_features_weighted)
        most_similar_index = similarity_scores.argmax()
        suggested_model = filtered_df.iloc[most_similar_index][target]
        st.write(f"### Suggested Model: {suggested_model}")

        if st.button("Show Details"):
            original_phone = df_original[df_original[target] == suggested_model].iloc[0]
            st.write("### Recommended Phone Details :")
            st.write(original_phone)
    else:
        st.write("No phones found for the selected brand.")

    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            padding: 10px;
            font-size: 12px;
            color: #888;
        }
        </style>
        <div class="footer">Made by PhonePro Team from PCCOE ❤️</div>
        """, unsafe_allow_html=True)
else:
    st.info("Please login or sign up to use PhonePro.")
