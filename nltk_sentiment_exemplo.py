# =====================================================
# Classificação de Sentimentos com NLTK
# Reprodução do exemplo: https://www.nltk.org/howto/sentiment.html
# =====================================================

import nltk

# Baixar os dados necessários (só precisa rodar uma vez)
nltk.download('subjectivity', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =====================================================
# PARTE 1: Classificador NaiveBayes (Subjetividade)
# =====================================================

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation, extract_unigram_feats

print("=" * 60)
print("PARTE 1: NaiveBayes - Classificação de Subjetividade")
print("=" * 60)

n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs  = [(sent, 'obj')  for sent in subjectivity.sents(categories='obj')[:n_instances]]

train_subj_docs = subj_docs[:80]
test_subj_docs  = subj_docs[80:100]
train_obj_docs  = obj_docs[:80]
test_obj_docs   = obj_docs[80:100]
training_docs   = train_subj_docs + train_obj_docs
testing_docs    = test_subj_docs  + test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg   = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats   = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set     = sentim_analyzer.apply_features(testing_docs)

trainer    = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

print("\n--- Métricas do Classificador ---")
results = sentim_analyzer.evaluate(test_set)
for key, value in sorted(results.items()):
    print(f"  {key}: {value}")

# Resultado final da Parte 1
accuracy = results.get('Accuracy', 0)
print(f"\n>>> RESULTADO FINAL (Parte 1): o classificador acertou {accuracy*100:.0f}% das frases no conjunto de teste.")
print(f"    Ele consegue distinguir frases de OPINIÃO (subjetivas) de frases de FATO (objetivas).\n")


# =====================================================
# PARTE 2: VADER - Análise de Intensidade
# =====================================================

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

print("=" * 60)
print("PARTE 2: VADER - Análise de Intensidade de Sentimento")
print("=" * 60)

def classificar_sentimento(compound):
    """Converte o score composto em um veredito legível."""
    if compound >= 0.05:
        return "POSITIVO"
    elif compound <= -0.05:
        return "NEGATIVO"
    else:
        return "NEUTRO"

sentences = [
   "VADER is smart, handsome, and funny.",
   "VADER is smart, handsome, and funny!",
   "VADER is very smart, handsome, and funny.",
   "VADER is VERY SMART, handsome, and FUNNY.",
   "VADER is VERY SMART, handsome, and FUNNY!!!",
   "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",
  "The book was good.",
  "The book was kind of good.",
  "The plot was good, but the characters are uncompelling and the dialog is not great.",
  "A really bad, horrible book.",
  "At least it isn't a horrible book.",
  ":) and :D",
  "",
  "Today sux",
  "Today sux!",
  "Today SUX!",
  "Today kinda sux! But I'll get by, lol",
]

paragraph = (
   "It was one of the worst movies I've seen, despite good reviews. "
  "Unbelievably bad acting!! Poor direction. VERY poor production. "
  "The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"
)
sentences.extend(tokenize.sent_tokenize(paragraph))

tricky_sentences = [
   "Most automated sentiment analysis tools are shit.",
   "VADER sentiment analysis is the shit.",
   "Sentiment analysis has never been good.",
   "Sentiment analysis with VADER has never been this good.",
   "Warren Beatty has never been so entertaining.",
   "I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.",
   "I like to hate Michael Bay films, but I couldn't fault this one",
   "I like to hate Michael Bay films, BUT I couldn't help but fault this one",
   "It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it",
   "The movie was too good",
   "This movie was actually neither that funny, nor super witty.",
   "This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
   "Those who find ugly meanings in beautiful things are corrupt without being charming.",
   "There are slow and repetitive parts, BUT it has just enough spice to keep it interesting.",
   "The script is not fantastic, but the acting is decent and the cinematography is EXCELLENT!",
   "Roger Dodger is one of the most compelling variations on this theme.",
   "Roger Dodger is one of the least compelling variations on this theme.",
   "Roger Dodger is at least compelling as a variation on the theme.",
   "they fall in love with the product",
   "but then it breaks",
   "usually around the time the 90 day warranty expires",
   "the twin towers collapsed today",
   "However, Mr. Carter solemnly argues, his client carried out the kidnapping under orders and in the ''least offensive way possible.''"
]
sentences.extend(tricky_sentences)

# textes = [ 
#     "Experimentos com VADER são interessantes.",       
#     Todos os testes foram feitos aqui
# ] 
# sentences.extend(textes)

sid = SentimentIntensityAnalyzer()

contagem = {"POSITIVO": 0, "NEGATIVO": 0, "NEUTRO": 0}

for sentence in sentences:
    if sentence.strip() == "":
        continue
    scores   = sid.polarity_scores(sentence)
    compound = scores['compound']
    veredito = classificar_sentimento(compound)
    contagem[veredito] += 1

    print(f"\nFrase : {sentence}")
    print(f"Scores: pos={scores['pos']:.3f}  neu={scores['neu']:.3f}  neg={scores['neg']:.3f}  compound={compound:.4f}")
    print(f">>> RESULTADO: {veredito}  (score composto: {compound:.4f})")
