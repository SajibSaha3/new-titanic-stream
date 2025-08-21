import streamlit as st
import pickle
import numpy as np

# --- Load Model ---
model = pickle.load(open("model.pkl","rb"))

# --- Page Config ---
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

# --- Background CSS ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg");
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.7);
    color: white;
}
h1, h2, h3 {
    color: white;
    text-align: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Title ---
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Enter Passenger Details in the Sidebar")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Input Features")
    pclas = st.selectbox("Pclass ", [0, 1, 2, 3])
    sex = st.slider("Sex", 0, 1, step=1)
    age = st.slider("Age", 1.0, 100.0, step=.01)
    sibsp = st.select_slider("SibSp", [0, 1, 2, 3, 4, 5])
    parch = st.text_input("Parch", placeholder="Enter parch value 0-6")
    fare = st.slider("Fare", 1.0, 100.0, step=.01)
    embarked = st.selectbox("Embarked", [0, 1, 2])
    classs = st.select_slider("Class", [0, 1, 2])
    who = st.select_slider("Who", [0, 1, 2])
    adult_male = st.text_input("Adult Male", placeholder="Enter value 0-2")
    embark_town = st.selectbox("Embark Town", [0, 1, 2])
    alive = st.select_slider("Alive", [0, 1])
    alone = st.selectbox("Alone", [0, 1])

    submit = st.button("🚀 Predict")

# --- Prediction ---
if submit:
    try:
        feature = [[
            pclas, sex, age, sibsp, int(parch) if parch else 0, fare,
            embarked, classs, who, int(adult_male) if adult_male else 0,
            embark_town, alive, alone
        ]]
        prediction = model.predict(feature)
        result = "🌟 Survived" if prediction[0] == 1 else "💀 Died"

        # Result Card
        st.markdown(
            f"""
            <div style='text-align:center; background:rgba(0,0,0,0.7);
                        padding:20px; border-radius:15px;'>
                <h2 style='color:white;'>Prediction Result:</h2>
                <h1 style='color:gold;'>{result}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error: {e}")
