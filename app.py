import pickle as pkl
import streamlit as st
import numpy as np

model = pkl.load(open(r"Models\Model.pkl","rb"))
scaler = pkl.load(open(r"Models\Scaler.pkl","rb"))
print(scaler)


def Prediction_model(LIMIT_BAL,SEX,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6):
    SEX = 1 if SEX.lower() == "male" else 2
    feature = np.array([LIMIT_BAL,SEX,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6])
    feature_scale = scaler.transform([feature])
    model_pred = model.predict(feature_scale)
    return model_pred[0]





st.markdown(
    """
<style>

section , header {
    background: #ecf0f3 !important;
    font-family: sans-serif;
}
.stMainBlockContainer{
    max-width: 90%;
}
div[data-testid="stNumberInputContainer"] , div[data-testid="stTextInputRootElement"]{
    border: 1px solid;
}
input{
    background: #ffffff61 !important;
    }

div[data-testid="stElementContainer"]{
width: 45%;
}
div[direction="column"] > :first-child {
    width: 100% !important;
}
div[direction="column"] {
    display: flex;
    align-items: center;
    gap: 30px;
    flex-direction: row;
    align-content: center;
    flex-wrap: wrap;
    justify-content: center;
}
.stForm{
    padding: 30px 0px;
    box-shadow:
		10px 10px 10px #d1d9e6,
		-10px -10px 10px #d1d9e6;
        padding-bottom: 40px; 
}
p {
    font-size: 15px !important;
}
h1 span {
    font-size: 35px;
    color: #4a89dc;
}
span[data-testid="stHeaderActionElements"] {
    display: none;
}
div[data-testid="stHeadingWithActionElements"]{
text-align: center;
}

div[direction="column"] > :last-child , div[direction="column"] > :nth-last-child(2){
    width: 92% !important;
}

@media screen and (max-width: 600px){

h1 span {
    font-size: 30px;
}
div[data-testid="stElementContainer"] , div[direction="column"] > :last-child , div[direction="column"] > :nth-last-child(2){
width: 85% !important;
}
.stMainBlockContainer {
    max-width: 100%;
}

}
</style>
""",
    unsafe_allow_html=True,
)


with st.form("Input_form"):
    st.title("Default Payment Prediction System")
    LIMIT_BAL = st.number_input("Amount of given credit : ")
    SEX = st.text_input(" Gender : ")
    AGE = st.number_input("Age in years(like 24 etc) : ")
    PAY_0 = st.number_input("Repayment status in  Last Mounth : ")
    PAY_2 = st.number_input("Repayment status in 2nd Last Mounth : ")
    PAY_3 = st.number_input("Repayment status in 3rd Last Mounth : ")
    PAY_4 = st.number_input("Repayment status in 4th Last Mounth : ")
    PAY_5 = st.number_input("Repayment status in 5th Last Mounth : ")
    PAY_6 = st.number_input("Repayment status in 6th Last Mounth : ")
    BILL_AMT1 = st.number_input("Amount of bill statement Last Mounth : ")
    BILL_AMT2 = st.number_input("Amount of bill statement 2nd Last Mounth :")
    BILL_AMT3 = st.number_input("Amount of bill statement 3rd Last Mounth :")
    BILL_AMT4 = st.number_input("Amount of bill statement 4th Last Mounth :")
    BILL_AMT5 = st.number_input("Amount of bill statement 5th Last Mounth :")
    BILL_AMT6 = st.number_input("Amount of bill statement 6th Last Mounth :")
    PAY_AMT1 = st.number_input("Amount of previous payment ")
    PAY_AMT2 = st.number_input("Amount of previous payment 2nd Last Mounth")
    PAY_AMT3 = st.number_input("Amount of previous payment 3rd Last Mounth")
    PAY_AMT4 = st.number_input("Amount of previous payment 4th Last Mounth")
    PAY_AMT5 = st.number_input("Amount of previous payment 5th Last Mounth")
    PAY_AMT6 = st.number_input("Amount of previous payment 6th Last Mounth")
    submit_btn = st.form_submit_button("Model Prediction")

if submit_btn:
    print(LIMIT_BAL,SEX,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6)
    model_prediction  = Prediction_model(LIMIT_BAL,SEX,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6)
    model_prediction = "Person is the default option" if model_prediction == 1 else "Person is not the default"
    st.success(f"According to the RandomForestClassifier Model, ' {model_prediction} ' " )