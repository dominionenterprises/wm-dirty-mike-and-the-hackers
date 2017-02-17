from centCom import CentCom
import cPickle

tokenizer = cPickle.load(open("review_classifier_tokenizer.pkl", "rb"))
new_classifier = CentCom('initial_model.h5')

new_classifier.tokenizer = tokenizer
sample = ['I absolutely loved this restaurant Great lighting service and food I could even bring my big fat dog']
new_classifier.predict(sample)