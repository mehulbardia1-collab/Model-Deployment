import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
""")

# Customer inputs

age = st.number_input("Age", min_value=18, max_value=100, value=30)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    value=1
)

passport = st.selectbox("Has Passport?", [0, 1])

owncar = st.selectbox("Owns a Car?", [0, 1])

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=30000
)

pitchsatisfactionscore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

numberoffollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=1
)

durationofpitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=10
)

# Create input DataFrame

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "ProductPitched": productpitched,
    "NumberOfFollowups": numberoffollowups,
    "DurationOfPitch": durationofpitch
}])

# Make prediction

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    result = (
        "Likely to Purchase the Wellness Tourism Package"
        if prediction == 1
        else "Unlikely to Purchase the Wellness Tourism Package"
    )

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
