import streamlit as st
import sklearn

st.components.v1.html(
    """
    <script>
    const inputs = document.querySelectorAll("input[type='number'], input[type='text']");
    inputs.forEach((input, index) => {
        input.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault(); // prevent the default action (form submission)
                const nextInput = inputs[index + 1];
                if (nextInput) {
                    nextInput.focus(); // move to the next input box
                }
            }
        });
    });
    </script>
    """,
    height=0, 
)
st.title("Welcome to Path Predtictor") 
st.info("This will help you to find best career ta suits you")


import pandas as pd
import numpy as np

dataset=[
    {"maths": 90, "english": 57, "science": 89, "social_studies": 53, "iq": 102, "stream": "science"},
    {"maths": 88, "english": 55, "science": 60, "social_studies": 58, "iq": 105, "stream": "commerce"},
    {"maths": 53, "english": 87, "science": 57, "social_studies": 90, "iq": 72, "stream": "arts"},

    {"maths": 85, "english": 59, "science": 86, "social_studies": 50, "iq": 108, "stream": "science"},
    {"maths": 85, "english": 52, "science": 57, "social_studies": 50, "iq": 98, "stream": "commerce"},
    {"maths": 55, "english": 86, "science": 55, "social_studies": 91, "iq": 69, "stream": "arts"},

    {"maths": 87, "english": 52, "science": 90, "social_studies": 54, "iq": 125, "stream": "science"},
    {"maths": 90, "english": 53, "science": 58, "social_studies": 55, "iq": 110, "stream": "commerce"},
    {"maths": 50, "english": 90, "science": 56, "social_studies": 87, "iq": 74, "stream": "arts"},

    {"maths": 92, "english": 55, "science": 91, "social_studies": 56, "iq": 116, "stream": "science"},
    {"maths": 92, "english": 58, "science": 54, "social_studies": 50, "iq": 97, "stream": "commerce"},
    {"maths": 51, "english": 89, "science": 58, "social_studies": 88, "iq": 73, "stream": "arts"},

    {"maths": 88, "english": 60, "science": 87, "social_studies": 52, "iq": 109, "stream": "science"},
    {"maths": 86, "english": 57, "science": 55, "social_studies": 56, "iq": 86, "stream": "commerce"},
    {"maths": 52, "english": 85, "science": 53, "social_studies": 85, "iq": 71, "stream": "arts"},

    {"maths": 91, "english": 54, "science": 88, "social_studies": 55, "iq": 120, "stream": "science"},
    {"maths": 89, "english": 50, "science": 56, "social_studies": 52, "iq": 91, "stream": "commerce"},
    {"maths": 50, "english": 87, "science": 54, "social_studies": 86, "iq": 70, "stream": "arts"},

    {"maths": 89, "english": 53, "science": 85, "social_studies": 50, "iq": 127, "stream": "science"},
    {"maths": 90, "english": 50, "science": 53, "social_studies": 57, "iq": 100, "stream": "commerce"},
    {"maths": 51, "english": 85, "science": 57, "social_studies": 90, "iq": 69, "stream": "arts"},

    {"maths": 86, "english": 58, "science": 90, "social_studies": 57, "iq": 109, "stream": "science"},
    {"maths": 91, "english": 54, "science": 55, "social_studies": 53, "iq": 80, "stream": "commerce"},
    {"maths": 54, "english": 88, "science": 59, "social_studies": 89, "iq": 70, "stream": "arts"},

    {"maths": 90, "english": 51, "science": 88, "social_studies": 52, "iq": 116, "stream": "science"},
    {"maths": 87, "english": 54, "science": 59, "social_studies": 53, "iq": 89, "stream": "commerce"},
    {"maths": 54, "english": 86, "science": 55, "social_studies": 87, "iq": 71, "stream": "arts"},

    {"maths": 93, "english": 55, "science": 92, "social_studies": 54, "iq": 117, "stream": "science"},
    {"maths": 89, "english": 52, "science": 52, "social_studies": 54, "iq": 100, "stream": "commerce"},
    {"maths": 53, "english": 89, "science": 56, "social_studies": 88, "iq": 72, "stream": "arts"}
]




df=pd.DataFrame(dataset)


X=df.drop(columns=['stream'])
Y=df['stream']

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Apply IQR method to filter out outliers
X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]





from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset_science = [
    {"math": 65, "physics": 90, "chemistry": 88, "biology": 95, "label": "NEET"},
    {"math": 94, "physics": 89, "chemistry": 92, "biology": None, "label": "JEE"},
    {"math": 58, "physics": 92, "chemistry": 90, "biology": 97, "label": "NEET"},
    {"math": 93, "physics": 89, "chemistry": 94, "biology": 80, "label": "JEE"},
    {"math": None, "physics": 91, "chemistry": 85, "biology": 96, "label": "NEET"},
    {"math": 87, "physics": 90, "chemistry": 91, "biology": 78, "label": "JEE"},
    {"math": 55, "physics": 89, "chemistry": 92, "biology": 94, "label": "NEET"},
    {"math": 88, "physics": 91, "chemistry": 85, "biology": 75, "label": "JEE"},
    {"math": 62, "physics": 92, "chemistry": 89, "biology": 95, "label": "NEET"},
    {"math": 60, "physics": 88, "chemistry": 90, "biology": 91, "label": "NEET"},
    {"math": 90, "physics": 93, "chemistry": 88, "biology": None, "label": "JEE"},
    {"math": 88, "physics": 87, "chemistry": 89, "biology": None, "label": "JEE"},
    {"math": 68, "physics": 91, "chemistry": 89, "biology": 95, "label": "NEET"},
    {"math": 57, "physics": 90, "chemistry": 92, "biology": 94, "label": "NEET"},
    {"math": 87, "physics": 92, "chemistry": 88, "biology": None, "label": "JEE"},
    {"math": 64, "physics": 90, "chemistry": 86, "biology": 97, "label": "NEET"},
    {"math": 91, "physics": 91, "chemistry": 90, "biology": None, "label": "JEE"},
    {"math": 59, "physics": 91, "chemistry": 87, "biology": 92, "label": "NEET"},
    {"math": 92, "physics": 85, "chemistry": 89, "biology": None, "label": "JEE"},
    {"math": 70, "physics": 85, "chemistry": 90, "biology": 93, "label": "NEET"},
    {"math": None, "physics": 92, "chemistry": 88, "biology": 93, "label": "NEET"},
    {"math": 66, "physics": 88, "chemistry": 85, "biology": 94, "label": "NEET"},
    {"math": 94, "physics": 90, "chemistry": 92, "biology": None, "label": "JEE"},
    {"math": 60, "physics": 90, "chemistry": 91, "biology": 92, "label": "NEET"},
    {"math": 85, "physics": 92, "chemistry": 88, "biology": 76, "label": "JEE"},
    {"math": 93, "physics": 90, "chemistry": 92, "biology": 82, "label": "JEE"},
    {"math": 64, "physics": 88, "chemistry": 90, "biology": 95, "label": "NEET"},
    {"math": 62, "physics": 91, "chemistry": 89, "biology": 95, "label": "NEET"},
    {"math": 96, "physics": 93, "chemistry": 94, "biology": None, "label": "JEE"},
    {"math": 63, "physics": 89, "chemistry": 87, "biology": 96, "label": "NEET"},
    {"math": 92, "physics": 89, "chemistry": 90, "biology": 79, "label": "JEE"},
    {"math": 58, "physics": 92, "chemistry": 86, "biology": 94, "label": "NEET"},
    {"math": 89, "physics": 94, "chemistry": 90, "biology": None, "label": "JEE"},
    {"math": 67, "physics": 93, "chemistry": 87, "biology": 94, "label": "NEET"},
    {"math": 87, "physics": 91, "chemistry": 89, "biology": None, "label": "JEE"},
    {"math": 57, "physics": 90, "chemistry": 92, "biology": 94, "label": "NEET"},
    {"math": 64, "physics": 88, "chemistry": 85, "biology": 94, "label": "NEET"},
    {"math": 58, "physics": 92, "chemistry": 88, "biology": 97, "label": "NEET"},
    {"math": 89, "physics": 90, "chemistry": 91, "biology": None, "label": "JEE"},
    {"math": 70, "physics": 85, "chemistry": 90, "biology": 93, "label": "NEET"},
    {"math": 55, "physics": 89, "chemistry": 92, "biology": 94, "label": "NEET"}
]
for entry in dataset_science:
    for key in entry:
        if entry[key] is None:
            entry[key] = 0
        if entry[key] is 'JEE':
            entry[key] = 1
        if entry[key] is 'NEET':
            entry[key] = 2
df_science = pd.DataFrame(dataset_science)


# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1_S = df_science.quantile(0.25)
Q3_S = df_science.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR_S = Q3_S - Q1_S

# Apply IQR method to filter out outliers
df_science = df_science[~((df_science < (Q1_S - 1.5 * IQR_S)) |(df_science > (Q3_S + 1.5 * IQR_S))).any(axis=1)]

X_S=df_science.drop(columns=['label'])
Y_S=df_science['label']




dataset_commerce = [
    {"math": 75, "finance": 80, "accounting": 85, "economics": 80, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 85, "finance": 90, "accounting": 92, "economics": 88, "label": "Cost Accountancy"},
    {"math": 68, "finance": 72, "accounting": 77, "economics": 70, "label": "Company Secretary (CS)"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 74, "label": "Chartered Accountancy (CA)"},
    {"math": 77, "finance": 80, "accounting": 85, "economics": 76, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 58, "finance": 63, "accounting": 70, "economics": 60, "label": "Company Secretary (CS)"},
    {"math": 78, "finance": 84, "accounting": 89, "economics": 81, "label": "Chartered Accountancy (CA)"},
    {"math": 85, "finance": 88, "accounting": 90, "economics": 85, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 75, "label": "Bachelor of Commerce (BCom)"},
    {"math": 90, "finance": 95, "accounting": 95, "economics": 90, "label": "Cost Accountancy"},
    {"math": 63, "finance": 67, "accounting": 72, "economics": 65, "label": "Chartered Accountancy (CA)"},
    {"math": 82, "finance": 87, "accounting": 91, "economics": 84, "label": "Chartered Accountancy (CA)"},
    {"math": 55, "finance": 60, "accounting": 65, "economics": 55, "label": "Company Secretary (CS)"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 75, "label": "Bachelor of Commerce (BCom)"},
    {"math": 88, "finance": 82, "accounting": 87, "economics": 80, "label": "Bachelor of Commerce (BCom)"},
    {"math": 90, "finance": 95, "accounting": 94, "economics": 89, "label": "Cost Accountancy"},
    {"math": 68, "finance": 72, "accounting": 76, "economics": 70, "label": "Bachelor of Commerce (BCom)"},
    {"math": 85, "finance": 90, "accounting": 92, "economics": 88, "label": "Cost Accountancy"},
    {"math": 77, "finance": 80, "accounting": 85, "economics": 76, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 60, "finance": 65, "accounting": 70, "economics": 60, "label": "Company Secretary (CS)"},
    {"math": 75, "finance": 80, "accounting": 85, "economics": 80, "label": "Company Secretary (CS)"},
    {"math": 78, "finance": 82, "accounting": 87, "economics": 80, "label": "Chartered Accountancy (CA)"},
    {"math": 80, "finance": 85, "accounting": 90, "economics": 85, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 73, "finance": 77, "accounting": 82, "economics": 75, "label": "Bachelor of Commerce (BCom)"},
    {"math": 76, "finance": 82, "accounting": 88, "economics": 82, "label": "Bachelor of Commerce (BCom)"},
    {"math": 85, "finance": 88, "accounting": 90, "economics": 85, "label": "BBA (Bachelor of Business Administration)"},
    {"math": 68, "finance": 72, "accounting": 77, "economics": 70, "label": "Company Secretary (CS)"},
    {"math": 90, "finance": 95, "accounting": 95, "economics": 90, "label": "Cost Accountancy"},
    {"math": 85, "finance": 90, "accounting": 92, "economics": 88, "label": "Cost Accountancy"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 74, "label": "Chartered Accountancy (CA)"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 75, "label": "Bachelor of Commerce (BCom)"},
    {"math": 85, "finance": 90, "accounting": 92, "economics": 88, "label": "Cost Accountancy"},
    {"math": 78, "finance": 84, "accounting": 89, "economics": 81, "label": "Chartered Accountancy (CA)"},
    {"math": 68, "finance": 72, "accounting": 77, "economics": 70, "label": "Company Secretary (CS)"},
    {"math": 70, "finance": 75, "accounting": 80, "economics": 75, "label": "Bachelor of Commerce (BCom)"},
    {"math": 80, "finance": 85, "accounting": 90, "economics": 85, "label": "BBA (Bachelor of Business Administration)"}
]
df_commerce = pd.DataFrame(dataset_commerce)


X_C=df_commerce.drop(columns=['label'])
Y_C=df_commerce['label']





# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1_C = X_C.quantile(0.25)
Q3_C = X_C.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR_C = Q3_C - Q1_C


# Apply IQR method to filter out outliers
non_outliers = ~((X_C < (Q1_C - 1.5 * IQR_C)) | (X_C > (Q3_C + 1.5 * IQR_C))).any(axis=1)

# Keep only the non-outlier data for both X and Y
X_C = X_C[non_outliers]
Y_C = Y_C[non_outliers]

dataset_arts = [
    {"math": 60, "design": 70, "creativity": 75, "management": 80, "communication": 75, "label": "Bachelor of Arts (BA)"},
    {"math": 65, "design": 85, "creativity": 80, "management": 70, "communication": 80, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 80, "design": 90, "creativity": 85, "management": 75, "communication": 70, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 75, "design": 95, "creativity": 80, "management": 70, "communication": 65, "label": "Bachelor of Design (B.Des)"},
    {"math": 90, "design": 80, "creativity": 85, "management": 75, "communication": 90, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 70, "design": 60, "creativity": 75, "management": 70, "communication": 80, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 55, "design": 90, "creativity": 70, "management": 80, "communication": 65, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 85, "design": 75, "creativity": 90, "management": 80, "communication": 70, "label": "Bachelor of Design (B.Des)"},
    {"math": 75, "design": 80, "creativity": 70, "management": 75, "communication": 80, "label": "Bachelor of Arts (BA)"},
    {"math": 60, "design": 85, "creativity": 90, "management": 70, "communication": 75, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 90, "design": 70, "creativity": 85, "management": 75, "communication": 80, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 65, "design": 80, "creativity": 75, "management": 85, "communication": 70, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 70, "design": 90, "creativity": 85, "management": 75, "communication": 80, "label": "Bachelor of Design (B.Des)"},
    {"math": 80, "design": 95, "creativity": 90, "management": 80, "communication": 70, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 55, "design": 70, "creativity": 80, "management": 65, "communication": 75, "label": "Bachelor of Arts (BA)"},
    {"math": 85, "design": 90, "creativity": 75, "management": 80, "communication": 65, "label": "Bachelor of Design (B.Des)"},
    {"math": 70, "design": 85, "creativity": 90, "management": 80, "communication": 75, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 75, "design": 80, "creativity": 70, "management": 75, "communication": 80, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 60, "design": 75, "creativity": 80, "management": 70, "communication": 85, "label": "Bachelor of Arts (BA)"},
    {"math": 90, "design": 85, "creativity": 90, "management": 75, "communication": 70, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 80, "design": 70, "creativity": 85, "management": 90, "communication": 80, "label": "Bachelor of Design (B.Des)"},
    {"math": 70, "design": 80, "creativity": 90, "management": 85, "communication": 75, "label": "Bachelor of Business Administration (BBA)"},

    # Expanded entries
    {"math": 55, "design": 65, "creativity": 70, "management": 60, "communication": 70, "label": "Bachelor of Arts (BA)"},
    {"math": 85, "design": 80, "creativity": 90, "management": 75, "communication": 70, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 90, "design": 95, "creativity": 85, "management": 80, "communication": 70, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 75, "design": 70, "creativity": 80, "management": 75, "communication": 85, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 60, "design": 85, "creativity": 75, "management": 70, "communication": 80, "label": "Bachelor of Design (B.Des)"},
    {"math": 65, "design": 90, "creativity": 85, "management": 75, "communication": 70, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 70, "design": 80, "creativity": 75, "management": 80, "communication": 85, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 80, "design": 75, "creativity": 90, "management": 75, "communication": 70, "label": "Bachelor of Arts (BA)"},
    {"math": 55, "design": 85, "creativity": 70, "management": 65, "communication": 75, "label": "Bachelor of Fine Arts (BFA)"},
    {"math": 90, "design": 95, "creativity": 80, "management": 85, "communication": 75, "label": "Bachelor of Design (B.Des)"},
    {"math": 80, "design": 85, "creativity": 90, "management": 75, "communication": 70, "label": "Bachelor of Fashion Design (B.FD)"},
    {"math": 70, "design": 80, "creativity": 75, "management": 70, "communication": 80, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 60, "design": 75, "creativity": 80, "management": 70, "communication": 85, "label": "Bachelor of Arts (BA)"},
    {"math": 85, "design": 90, "creativity": 75, "management": 80, "communication": 70, "label": "Bachelor of Business Administration (BBA)"},
    {"math": 75, "design": 95, "creativity": 80, "management": 70, "communication": 65, "label": "Bachelor of Design (B.Des)"},
    {"math": 70, "design": 80, "creativity": 90, "management": 75, "communication": 80, "label": "Bachelor of Fashion Design (B.FD)"}
]

df_arts=pd.DataFrame(dataset_arts)

X_A=df_arts.drop(columns=['label'])
Y_A=df_arts['label']


Q1_A = X_A.quantile(0.25)
Q3_A = X_A.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR_A = Q3_A - Q1_A

# Apply IQR method to filter out outliers
X_A = X_A[~((X_A < (Q1_A - 1.5 * IQR_A)) |(X_A> (Q3_A + 1.5 * IQR_A))).any(axis=1)]





name=st.text_input("Enter your name: ")



if 'age' not in st.session_state:
    st.session_state['age'] = 0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0


year1 = st.number_input("Please enter your age:", key="rare")


if st.button("Submit Age"):
    if 16 <= year1 <= 25:
        st.session_state['flag'] = 1
        st.session_state['age'] = year1
        st.success(f"Your valid age is: {year1}")
    else:
        st.session_state['age'] = 0
        st.info("Please enter a valid age (16-25).")

age = st.session_state['age']  

if age != 0:
    

    last_exam=st.text_input("Last exam passed is 10th or 12th :")
    if '10' in last_exam or 'ten' in last_exam:
        st.success(f"Your Class was : 10")
        last_exam=10

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        k = 10 # Number of neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        # print(f"Accuracy: {accuracy}")
        maths=st.number_input("please enter your maths marks :")
        english=st.number_input("please enter your english marks :")
        science=st.number_input("please enter your science marks :")
        social_studies=st.number_input("please enter your social studies marks :")
        iq=st.number_input("please enter your iq marks :")
        m_i=[maths,english,science,social_studies,iq]
        m_i_reshaped = np.array(m_i).reshape(1, -1)
        Y_pred=knn.predict(m_i_reshaped)
        a=Y_pred[0]
        st.markdown(f"# {a} will be best for you.")
    elif '12' in last_exam or 'twelve' in last_exam and 10 not in last_exam:
        st.success(f"Your Class was : 12")
        last_exam=12
        if last_exam==12:
           stream=st.text_input("in which stream you were in 12th (Arts/science/commerce) :")
           st.success(f"Your Stream was : {stream}")
        if 'cienc' in stream :
            X_train_S,X_test_S,Y_train_S,Y_test_S=train_test_split(X_S,Y_S,test_size=0.3,random_state=1)
            scaler = StandardScaler()
            X_train_S = scaler.fit_transform(X_train_S)
            X_test_S = scaler.transform(X_test_S)

            k = 10 # Number of neighbors
            knn = KNeighborsClassifier(n_neighbors=k)

            knn.fit(X_train_S, Y_train_S)
            Y_pred_S = knn.predict(X_test_S)
            accuracy = accuracy_score(Y_test_S, Y_pred_S)
            # print(f"Accuracy: {accuracy}")
            maths=int(st.number_input("please enter your maths marks :"))
            physics=int(st.number_input("please enter your physics marks :"))
            chemistry=int(st.number_input("please enter your chemistry marks :"))
            biology=int(st.number_input("please enter your biology marks :"))
            m_i_s=[maths,physics,chemistry,biology]
            # st.write(m_i_s)
            m_i_reshaped = np.array(m_i_s).reshape(1, -1)
            Y_pred_S=knn.predict(m_i_reshaped)
            if Y_pred_S[0]==1:
                st.markdown("# JEE will be best for you.")
            else:
                if Y_pred_S[0]!=1:
                  st.markdown("# NEET will be best for you.")

        elif 'omm'  in stream :
            X_train_C,X_test_C,Y_train_C,Y_test_C=train_test_split(X_C,Y_C,test_size=0.3,random_state=1)
            Scaler=StandardScaler()
            X_train_C=Scaler.fit_transform(X_train_C)
            X_test_C=Scaler.transform(X_test_C)
            K=3
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_train_C,Y_train_C)
            Y_pred_C=knn.predict(X_test_C)
            accuracy=accuracy_score(Y_test_C,Y_pred_C)
            print(f"Accuracy: {accuracy}")      
            # Streamlit input fields
            maths = st.number_input("Enter your maths marks:", key="maths_input")
            finance = st.number_input("Enter your finance marks:", key="finance_input")
            accounting = st.number_input("Enter your accounting marks:", key="accounting_input")
            economics = st.number_input("Enter your economics marks:", key="economics_input")

            # maths = st.number_input("Please enter your maths marks:", key="maths_input")
            # finance = st.number_input("Please enter your finance marks:", key="finance_input")
            # accounting = st.number_input("Please enter your accounting marks:", key="accounting_input")
            # economics = st.number_input("Please enter your economics marks:", key="economics_input")
            m_i_c=[maths,finance,accounting,economics]
            m_i_reshaped = np.array(m_i_c).reshape(1, -1)
            Y_pred_C=knn.predict(m_i_reshaped)
            a=Y_pred_C[0]
            st.markdown(f"# {a} will be best for you.")
        if 'rt' in stream :
            X_train_A,X_test_A,Y_train_A,Y_test_A=train_test_split(X_A,Y_A,test_size=0.4,random_state=119)
            Scaler=StandardScaler()
            X_train_A=Scaler.fit_transform(X_train_A)
            X_test_A=Scaler.transform(X_test_A)
            K=6
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_train_A,Y_train_A)
            Y_pred_A=knn.predict(X_test_A)
            accuracy=accuracy_score(Y_test_A,Y_pred_A)
            # print(f"Accuracy: {accuracy}")
            maths=int(st.number_input("please enter your maths marks :",key="Arts_maths_input"))
            design=int(st.number_input("please enter your design marks :"))
            creativity=int(st.number_input("please enter your creativity marks :"))
            management=int(st.number_input("please enter your management marks :"))
            communication=int(st.number_input("please enter your communication marks :"))
            m_i_a=[maths,design,creativity,management,communication]
            m_i_reshaped = np.array(m_i_a).reshape(1, -1)
            Y_pred_A=knn.predict(m_i_reshaped)
            a=Y_pred_A[0]
            st.markdown(f"# {a} will be best for you.")
    else:
             st.info(" Please enter a valid input")


