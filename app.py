import streamlit as st
import pandas as pd
import numpy as np
from sequence_split import split_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pickle

X_train, X_test, y_train, y_test = None, None, None, None
scaler = MinMaxScaler()
pca = PCA(n_components=2)
model = None
data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/BBTN.JK.csv')
n_steps = 3


def main():
    st.sidebar.title("Prediksi Data Time Series Bank BTN")
    menu = ["Data", "Pre processing data", "Implementasi"]
    choice = st.sidebar.selectbox("Menu", menu)
    global data
    global n_steps

    if choice == "Data":
        st.title("1.Wahyu Aril Saputra 200411100055")
        st.title("2.Billy Morgen Simbolon 200411100057")
        st.markdown("<h4>Data</h4>", unsafe_allow_html=True)
        
        st.write(data)
        if st.button("Penjelasan"):
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:justify'>Dataset Finance Bank BTN</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h3 style='text-align:justify'>Halaman Menu Sistem</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>1. Data : Menampilkan Data</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>2. Pre Processing Data : Menampilkan data yang sudah di pre processing </h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>3. Modelling : Menampilkan proses modeling data</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>4. Implementasi : Input data untuk prediksi data baru</h5>",
                unsafe_allow_html=True,
            )
        
    elif choice == "Pre processing data":
        st.title("1.Wahyu Aril Saputra 200411100055")
        st.title("2.Billy Morgen Simbolon 200411100057")
        st.markdown("<h4>Pre processing data</h4>", unsafe_allow_html=True)
        selected_column = st.selectbox("Pilih Kolom", ('Open','High', 'Low', 'Close'))
        sequence = data[selected_column].astype(float).values
        n_steps = 3
        X, y = split_sequence(sequence, n_steps)

        st.markdown("<h6>Data Setelah Di sequence</h6>", unsafe_allow_html=True)
        newFitur = pd.DataFrame(X, columns=['t-'+str(i+1) for i in range(n_steps-1, -1, -1)])
        newTarget = pd.DataFrame(y, columns=['Data Prediksi'])
        newData = pd.concat([newFitur, newTarget], axis=1)
        st.write(pd.DataFrame(newData, columns=newData.columns))


        st.markdown("<h6>Pre Processing Data</h6>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h6>Data Setelah Dinormalisasi</h6>", unsafe_allow_html=True)
            # scaler.fit(newFitur)
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(newFitur)
            st.write(pd.DataFrame(X_norm, columns=newFitur.columns))
        
        with col2:
            st.markdown("<h6>Data Setelah Dilakukan PCA</h6>", unsafe_allow_html=True)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_norm)
            st.write(pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2']))
        
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)
        st.markdown("<h3>Modelling</h3>", unsafe_allow_html=True)
        selected_model = st.selectbox("Pilih Model", ('Decision Tree', 'KNN'))
        if selected_model == "Decision Tree":
            model = DecisionTreeRegressor()
        elif selected_model == "KNN":
            model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Akurasi model:", accuracy)
        # st.write("Model di training.")

    elif choice == "Implementasi":
        st.title("1.Wahyu Aril Saputra 200411100055")
        st.title("2.Billy Morgen Simbolon 200411100057")
        st.markdown("Implementasi")
        input1 = st.number_input('Masukkan data ke-1 : ')
        input2 = st.number_input('Masukkan data ke-2 : ')
        input3 = st.number_input('Masukkan data ke-3 : ')
        input = np.array([input1, input2, input3])
        input = input.reshape(1, -1)
        scaler = MinMaxScaler()
        pca = PCA(n_components=1)

        input = scaler.fit_transform(input)
        input = pca.fit_transform(input)
        if st.button("Prediksi"):
                model_NB = pickle.load(open('model/model_NB.sav', 'rb'))
                prediksi = model_NB.predict(input)
                st.write("Prediksi harga saham : ", prediksi)
            

if __name__ == '__main__':
    main()
