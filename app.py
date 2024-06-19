import csv
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

def baca_data_khodam():
    khodam_data = []
    with open('khodam_dataset.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            khodam_data.append((row['kombinasi'], row['khodam'], row['arti']))
    return khodam_data

khodam_data = baca_data_khodam()

kombinasi = [item[0] for item in khodam_data]
khodams = [item[1] for item in khodam_data]
artis = [item[2] for item in khodam_data]

model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 3), lowercase=True)),
    ('classifier', MultinomialNB())
])

model.fit(kombinasi, khodams)

def cari_khodam(nama):
    nama = nama.lower()
    khodam_prediksi = model.predict([nama])
    index_prediksi = khodams.index(khodam_prediksi[0])
    return {
        'kombinasi': kombinasi[index_prediksi],
        'khodam': khodam_prediksi[0],
        'arti': artis[index_prediksi]
    }

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cek_khodam', methods=['POST'])
def cek_khodam():
    nama = request.form.get('nama')
    if not nama:
        return "<h1>Masukkan nama untuk mencari khodam</h1>"
    khodam_info = cari_khodam(nama)
    return render_template('hasil.html', khodam=khodam_info['khodam'], arti=khodam_info['arti'])

if __name__ == '__main__':
    app.run(debug=True)
