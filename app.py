import streamlit as st
import pandas as pd
import numpy as np
from sequence_split import split_sequence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
# path_to_fungsi = r'A:\Matkul\Semester 7\PPW\UTS\Fungsi'
# sys.path.append("A:/Matkul/Semester 7/PPW/UTS/Fungsi")
from Cleaning import cleaning
from tokenisasi import tokenize_text
from stopword import remove_stopwords

#import data
# data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/DataPTAInformatikaLabel.csv',delimiter=';')
data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/databerita.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


def main():
    st.sidebar.title("Pengolahan Data Berita dari Detik.Com")
    menu = ["Data", "Pre processing data","Ekstraksi Fitur","Topic Modeling", "Klasifikasi Data"]
    choice = st.sidebar.selectbox("Menu", menu)
    global data

    #menampilkan Data
    if choice == "Data":
        st.title("Pengolahan Data Berita dari Detik.Com")
        st.markdown("<h4>Menampilkan Data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        with st.expander("Dataset Detik.Com"):
            st.write(data)
        with st.expander("Data Per Baris"):
            data_rows = list(data.index)
            selected_row = st.selectbox("Pilih Baris Data:", data_rows)
            st.write(data.loc[selected_row])
        with st.expander("Penjelasan Sistem"):
            st.markdown(
                "<h3 style='text-align:justify'>Halaman Menu Sidebar Sistem</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>1. Data : Menampilkan Data PTA</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>2. Pre Processing Data : Mengolah dan Menampilkan data yang sudah di pre processing </h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>3. Ektraksi Fitur : Menampilkan hasil Jumlah Term</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>5. Topic Modeling : Menampilkan hasil dari LDA topic modeling</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>6. Klasifikasi Data : Menampilkan hasil dari Klasifikasi menggunakan Naive Bayes ,SVM , Decision Tree</h5>",
                unsafe_allow_html=True,
            )
        
    #pre processing data
    elif choice == "Pre processing data":
        st.title("Pengolahan Data Berita dari Detik.Com")
        st.markdown("<h4>Pre processing data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        # Cek missing value
        with st.expander("Cek Data Kosong"):
            st.write(data.isnull().sum())
            if st.button("Hapus Data Kosong"):
                data.dropna(inplace=True)
                st.write("Data kosong telah dihapus. Jumlah Data Kosong sekarang:")
                st.write(data.isnull().sum())

        # Cleaning Data
        with st.expander("Cleaning Data"):
            st.markdown("<h6>Proses Pembersihan Teks (Isi) yang meliputi :</h6>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h7>- Tag HTML Bawaan</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- LowerCase Data</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Spasi pada teks</h7>", unsafe_allow_html=True)
            with col2:
                st.markdown("<h7>- Tanda baca dan karakter spesial</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Nomor</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Komponen Lainnya</h7>", unsafe_allow_html=True)
            if st.button("Cleaning Data"):
                data['Isi'] = data['Isi'].apply(lambda x: cleaning(x))
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h6>Data Cleaning Abstrak :</h6>",unsafe_allow_html=True)
                data.dropna(inplace=True)
                st.write(data['Isi'])

        # Tokenisasi Data
        with st.expander("Tokenisasi"):
            st.markdown("<h6>Proses Memisahkan sebuah Dokumen menjadi susunan per kata / term</h6>", unsafe_allow_html=True)
            if st.button("Tokenisasi Data"):
                data.dropna(inplace=True)
                data['Isi'] = data['Isi'].apply(lambda x: cleaning(x))
                data['Isi'] = data['Isi'].fillna('')
                data['Isi_tokens'] = data['Isi'].apply(lambda x: tokenize_text(x))
                st.write(data[['Isi','Isi_tokens']])

        # Stopword Data
        with st.expander("StopWord"):
            st.markdown("<h6>Mengubah isi dokumen sesuai dengan kamus data</h6>", unsafe_allow_html=True)
            if st.button("StopWord Data"):
                data['Isi'] = data['Isi'].apply(lambda x: cleaning(x))
                data['Isi'] = data['Isi'].fillna('')
                data['Isi_tokens'] = data['Isi'].apply(lambda x: tokenize_text(x))
                data['Isi_tokens'] = data['Isi_tokens'].apply(lambda x: remove_stopwords(x))
                data['Isi'] = data['Isi_tokens'].apply(lambda tokens: ' '.join(tokens))
                st.write(data[['Isi','Isi_tokens']])

    # Ekstraksi Fitur
    elif choice == "Ekstraksi Fitur":
        st.title("Pengolahan Data Berita dari Detik.Com")
        st.markdown("<h4>Ekstraksi Fitur</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('dataCount.csv')
        # Term Frekuensi
        with st.expander("Term Frekuensi"):
            # with open("model/count_vectorizer_model.pkl", "rb") as file:
            #     count_vectorizer = pickle.load(file)
            # st.markdown("<h6>Seberapa sering sebuah kata atau term tertentu muncul dalam sebuah dokumen </h6>", unsafe_allow_html=True)
            # X_count = count_vectorizer.transform(np.array(data['Isi'].values.astype('U')))
            # terms_count = count_vectorizer.get_feature_names_out()
            # data_countvect = pd.DataFrame(data=X_count.toarray(), columns=terms_count)
            # data_countvect['label'] = data['Label'].values
            subset_df = data.iloc[:50, :]
            st.write(subset_df)

    elif choice == "Topic Modeling":
        st.title("Pengolahan Data Berita dari Detik.Com")
        st.markdown("<h4>Topic Modeling</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/dataProcessing.csv')
        data_tfidf = pd.read_csv('dataCount.csv')
        # with open("model/lda_model.pkl", "rb") as file:
        with open("model/lda_model100.pkl", "rb") as file:
            lda_model = pickle.load(file)
        with st.expander("Update Bobot"):
            with open("model/count_vectorizer_model.pkl", "rb") as file:
                vectorizer = pickle.load(file)
            terms = vectorizer.get_feature_names_out()
            X_Count = vectorizer.fit_transform(data['Isi'].values.astype('U'))
            w1 = lda_model.transform(X_Count)
            h1 = lda_model.components_
            st.write(w1)    
        with st.expander("Proporsi topic pada dokumen"):            
            n_components = 100
            colnames = ["Topic" + str(i) for i in range(n_components)]
            docnames = ["Doc" + str(i) for i in range(len(data['Isi']))]
            df_doc_topic = pd.DataFrame(np.round(w1, 2), columns=colnames, index=docnames)
            df_doc_topic['label'] = data['Label'].values
            st.write(df_doc_topic)
        with st.expander("Proporsi topic pada Kata"):
            label = []
            for i in range(1, (lda_model.components_.shape[1] + 1)):
                masukan = data_tfidf.columns[i - 1]
                label.append(masukan)
            data_topic_word = pd.DataFrame(lda_model.components_, columns=label)
            data_topic_word.rename(index={0: "Topik 1", 1: "Topik 2", 2: "Topik 3"}, inplace=True)
            st.write(data_topic_word.transpose())

    elif choice == "Klasifikasi Data":
        st.title("Pengolahan Data Berita dari Detik.Com")
        st.markdown("<h4>Klasifikasi Data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/dataProcessing.csv')
        new_data = []
        new_text = st.text_input("Masukkan teks baru untuk diprediksi:", "")
        if new_text:
            new_data.append(new_text)
            df_baru = pd.DataFrame({'Isi': new_data})
            df_baru['Isi'] = df_baru['Isi'].apply(lambda x: cleaning(x))
            df_baru['Isi'] = df_baru['Isi'].fillna('')
            df_baru['Isi_tokens'] = df_baru['Isi'].apply(lambda x: tokenize_text(x))
            df_baru['Isi_tokens'] = df_baru['Isi_tokens'].apply(lambda x: remove_stopwords(x))
            df_baru['Isi_tokens'] = df_baru['Isi_tokens'].apply(lambda x: stem_text(' '.join(x)).split(' '))
            df_baru['Isi'] = df_baru['Isi_tokens'].apply(lambda tokens: ' '.join(tokens))
            st.markdown("<h4>Hasil Processing</h4>", unsafe_allow_html=True)
            st.write(df_baru[['Isi','Isi_tokens']])
            st.markdown("<h4>Pilih model Klasifikasi Data</h4>", unsafe_allow_html=True)
            data_count = pd.read_csv('dataCount.csv')
            data_count['labels'] = data['Label'].values
            
            # Count Data
            with open("model/count_vectorizer_model.pkl", "rb") as file:
                    count_vectorizer = pickle.load(file)
            X_count = count_vectorizer.fit_transform(data['Isi'].values.astype('U'))

            # Count Data Baru
            X_count_baru = count_vectorizer.transform(df_baru['Isi'])

            # Topic Modeling Data
            # with open("model/lda_model.pkl", "rb") as file:
            with open("model/lda_model100.pkl", "rb") as file:
                lda_model = pickle.load(file)
            w1 = lda_model.transform(X_count)
            h1 = lda_model.components_
            df_doc_topic = pd.DataFrame(np.round(w1,2))
            df_doc_topic['label'] = data['Label'].values
            
            # Topic Modeling Data
            w1_baru = lda_model.transform(X_count_baru)

            # Training dengan topic modeling
            X = df_doc_topic.drop('label', axis=1)
            y = df_doc_topic['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #Training dengan tf-idf
            X_2 = data_count.drop('labels', axis=1)
            y_2 = data_count['labels']
            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

            with st.expander("Naive Bayes"):
                with open("model/naive_bayes_model.pkl", "rb") as file:
                    naive_bayes_classifier = pickle.load(file)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                        naive_bayes_classifier.fit(X_train, y_train)
                        y_pred = naive_bayes_classifier.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        y_pred_baru = naive_bayes_classifier.predict(w1_baru)
                        st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                        st.write(y_pred_baru)
                        st.write(accuracy)
                        st.write(classification_report(y_test, y_pred))
            with st.expander("SVM"):
                with open("model/svm_model.pkl", "rb") as file:
                    Vector = pickle.load(file)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                    Vector = Vector.fit(X_train, y_train)
                    y_pred_svm = Vector.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred_svm)
                    y_pred_svm_baru = Vector.predict(w1_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_svm_baru)
                    st.write(accuracy)
                    st.write(classification_report(y_test, y_pred_svm))
            with st.expander("Decision Tree"):
                with open("model/tree_model.pkl", "rb") as file:
                    clf = pickle.load(file)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                    decision_tree = clf.fit(X_train, y_train)
                    y_pred_clf = decision_tree.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred_clf)
                    y_pred_clf_baru = decision_tree.predict(w1_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_clf_baru)
                    st.write(accuracy)
                    st.write(classification_report(y_test, y_pred_clf))

if __name__ == '__main__':
    main()
