import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_salary_model.pkl")

st.set_page_config(page_title="💼Employee Salary Prediction", page_icon="💰", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Welcome! Fill in the details below to estimate the monthly salary")

# Sidebar inputs
st.markdown("---")
st.subheader("👤 Employee Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=23, max_value=50, value=30)
    education = st.selectbox("Education Level", ["Bachelor's Degree", "Master's Degree", "PhD", "High School"])
    experience = st.slider("Years of Experience", min_value=0, max_value=23, value=7)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_title = st.selectbox("Job Title", [
        'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director',
        'Marketing Analyst', 'Product Manager', 'Sales Manager', 'Marketing Coordinator',
        'Senior Scientist', 'Software Developer', 'HR Manager', 'Financial Analyst',
        'Project Manager', 'Customer Service Rep', 'Operations Manager', 'Marketing Manager',
        'Senior Engineer', 'Data Entry Clerk', 'Sales Director', 'Business Analyst',
        'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager',
        'Social Media Specialist', 'Software Manager', 'Junior Developer',
        'Senior Consultant', 'Product Designer', 'CEO', 'Accountant', 'Data Scientist',
        'Marketing Specialist', 'Technical Writer', 'HR Generalist', 'Project Engineer',
        'Customer Success Rep', 'Sales Executive', 'UX Designer', 'Operations Director',
        'Network Engineer', 'Administrative Assistant', 'Strategy Consultant', 'Copywriter',
        'Account Manager', 'Director of Marketing', 'Help Desk Analyst',
        'Customer Service Manager', 'Business Intelligence Analyst', 'Event Coordinator',
        'VP of Finance', 'Graphic Designer', 'UX Researcher', 'Social Media Manager',
        'Director of Operations', 'Senior Data Scientist', 'Junior Accountant',
        'Digital Marketing Manager', 'IT Manager', 'Customer Service Representative',
        'Business Development Manager', 'Senior Financial Analyst', 'Web Developer',
        'Research Director', 'Technical Support Specialist', 'Creative Director',
        'Senior Software Engineer', 'Human Resources Director', 'Content Marketing Manager',
        'Technical Recruiter', 'Sales Representative', 'Chief Technology Officer',
        'Junior Designer', 'Financial Advisor', 'Junior Account Manager',
        'Senior Project Manager', 'Principal Scientist', 'Supply Chain Manager',
        'Senior Marketing Manager', 'Training Specialist', 'Research Scientist',
        'Junior Software Developer', 'Public Relations Manager', 'Operations Analyst',
        'Product Marketing Manager', 'Senior HR Manager', 'Junior Web Developer',
        'Senior Project Coordinator', 'Chief Data Officer', 'Digital Content Producer',
        'IT Support Specialist', 'Senior Marketing Analyst', 'Customer Success Manager',
        'Senior Graphic Designer', 'Software Project Manager', 'Supply Chain Analyst',
        'Senior Business Analyst', 'Junior Marketing Analyst', 'Office Manager',
        'Principal Engineer', 'Junior HR Generalist', 'Senior Product Manager',
        'Junior Operations Analyst', 'Senior HR Generalist', 'Sales Operations Manager',
        'Senior Software Developer', 'Junior Web Designer', 'Senior Training Specialist',
        'Senior Research Scientist', 'Junior Sales Representative', 'Junior Marketing Manager',
        'Junior Data Analyst', 'Senior Product Marketing Manager', 'Junior Business Analyst',
        'Senior Sales Manager', 'Junior Marketing Specialist', 'Junior Project Manager',
        'Senior Accountant', 'Director of Sales', 'Junior Recruiter',
        'Senior Business Development Manager', 'Senior Product Designer',
        'Junior Customer Support Specialist', 'Senior IT Support Specialist',
        'Junior Financial Analyst', 'Senior Operations Manager', 'Director of Human Resources',
        'Junior Software Engineer', 'Senior Sales Representative',
        'Director of Product Management', 'Junior Copywriter',
        'Senior Marketing Coordinator', 'Senior Human Resources Manager',
        'Junior Business Development Associate', 'Senior Account Manager',
        'Senior Researcher', 'Junior HR Coordinator', 'Director of Finance',
        'Junior Marketing Coordinator', 'Junior Data Scientist', 'Senior Operations Analyst',
        'Senior Human Resources Coordinator', 'Senior UX Designer', 'Junior Product Manager',
        'Senior Marketing Specialist', 'Senior IT Project Manager',
        'Senior Quality Assurance Analyst', 'Director of Sales and Marketing',
        'Senior Account Executive', 'Director of Business Development',
        'Junior Social Media Manager', 'Senior Human Resources Specialist',
        'Senior Data Analyst', 'Director of Human Capital', 'Junior Advertising Coordinator',
        'Junior UX Designer', 'Senior Marketing Director', 'Senior IT Consultant',
        'Senior Financial Advisor', 'Junior Business Operations Analyst',
        'Junior Social Media Specialist', 'Senior Product Development Manager',
        'Junior Operations Manager', 'Senior Software Architect', 'Junior Research Scientist',
        'Senior Financial Manager', 'Senior HR Specialist', 'Senior Data Engineer',
        'Junior Operations Coordinator', 'Director of HR', 'Senior Operations Coordinator',
        'Junior Financial Advisor', 'Director of Engineering', 'Software Engineer Manager',
        'Back end Developer', 'Senior Project Engineer', 'Full Stack Engineer',
        'Front end Developer', 'Developer', 'Front End Developer', 'Director of Data Science',
        'Human Resources Coordinator', 'Junior Sales Associate', 'Human Resources Manager',
        'Juniour HR Generalist', 'Juniour HR Coordinator', 'Digital Marketing Specialist',
        'Receptionist', 'Marketing Director', 'Social M', 'Social Media Man', 'Delivery Driver'
    ])

# Build input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience]
})

st.markdown("---")
st.write("### 🔍 Input Preview")
st.write(input_df)

if st.button("📈 Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary: ₹{int(prediction[0]):,} per month")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch CSV Prediction")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data["Predicted Salary"] = batch_preds
    st.write("✅ Batch Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="salary_predictions.csv", mime="text/csv")
