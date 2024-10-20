import csv
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Fungsi untuk membaca data dari file CSV
def baca_data_khodam():
    khodam_data = []
    with open('khodam_dataset.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            khodam_data.append((row['kombinasi'], row['khodam'], row['arti']))
    return khodam_data

# Membaca data khodam dari file CSV
khodam_data = baca_data_khodam()

kombinasi = [item[0] for item in khodam_data]
khodams = [item[1] for item in khodam_data]
artis = [item[2] for item in khodam_data]

# Membangun model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 3), lowercase=True)),
    ('classifier', MultinomialNB())
])

# Melatih model
model.fit(kombinasi, khodams)

# Fungsi untuk mencari khodam berdasarkan nama
def cari_khodam(nama):
    nama = nama.lower()
    khodam_prediksi = model.predict([nama])
    index_prediksi = khodams.index(khodam_prediksi[0])
    return {
        'kombinasi': kombinasi[index_prediksi],
        'khodam': khodam_prediksi[0],
        'arti': artis[index_prediksi]
    }

# Antarmuka pengguna menggunakan Streamlit
st.title("Pencarian Khodam")
nama = st.text_input("Masukkan nama untuk mencari khodam")

if st.button("Cek Khodam"):
    if not nama:
        st.warning("Masukkan nama untuk mencari khodam.")
    else:
        khodam_info = cari_khodam(nama)
        st.success(f"Khodam: {khodam_info['khodam']}")
        st.info(f"Arti: {khodam_info['arti']}")
        st.write(f"Kombinasi: {khodam_info['kombinasi']}")
