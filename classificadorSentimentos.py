import nltk

# Baixa todos os recursos necessários (só precisa rodar uma vez)
nltk.download('subjectivity')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation, extract_unigram_feats
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ==============================================================
# PARTE 1: NAIVE BAYES
# Treina um modelo com exemplos de frases subjetivas e objetivas
# ==============================================================

print("\n" + "=" * 60)
print("PARTE 1: NAIVE BAYES")
print("=" * 60)

# Carrega 100 frases subjetivas (opiniões) e 100 objetivas (fatos)
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs  = [(sent, 'obj')  for sent in subjectivity.sents(categories='obj')[:n_instances]]

# Divide em treino (80 frases) e teste (20 frases) de cada tipo
train_subj = subj_docs[:80]
test_subj  = subj_docs[80:]
train_obj  = obj_docs[:80]
test_obj   = obj_docs[80:]

train_docs = train_subj + train_obj
test_docs  = test_subj  + test_obj

# Extrai características (palavras relevantes) das frases de treino
sentim = SentimentAnalyzer()
all_words_neg = sentim.all_words([mark_negation(doc) for doc, label in train_docs])
unigram_feats = sentim.unigram_word_feats(all_words_neg, min_freq=4)
sentim.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# Aplica as características e treina o modelo
train_set = sentim.apply_features(train_docs)
test_set  = sentim.apply_features(test_docs)

trainer    = nltk.classify.NaiveBayesClassifier.train
classifier = sentim.train(trainer, train_set)

# Avalia o modelo
print("\nResultados da avaliação:")
for key, value in sorted(sentim.evaluate(test_set).items()):
    print(f"  {key}: {value:.4f}")

print("\nTop 10 palavras mais informativas:")
classifier.show_most_informative_features(10)


# ==============================================================
# PARTE 2: VADER
# Analisa o sentimento de frases sem precisar de treinamento
# ==============================================================

print("\n" + "=" * 60)
print("PARTE 2: VADER")
print("=" * 60)

sid = SentimentIntensityAnalyzer()

sentences = [
    "VADER is smart, handsome, and funny.",
    "VADER is smart, handsome, and funny!",
    "VADER is VERY SMART, handsome, and FUNNY.",
    "VADER is VERY SMART, handsome, and FUNNY!!!",
    "The book was good.",
    "The book was kind of good.",
    "A really bad, horrible book.",
    "At least it isn't a horrible book.",
    ":) and :D",
    "Today sux",
    "Today SUX!",
    "Today kinda sux! But I'll get by, lol",
]

for sentence in sentences:
    print(f"\nFrase: {sentence}")
    scores = sid.polarity_scores(sentence)
    for key, value in sorted(scores.items()):
        print(f"  {key}: {value}")
    compound = scores['compound']
    if compound >= 0.05:
        print("  >> Sentimento: POSITIVO")
    elif compound <= -0.05:
        print("  >> Sentimento: NEGATIVO")
    else:
        print("  >> Sentimento: NEUTRO")

print("\n" + "=" * 60)