import smiley_predict
import sys

texts = []
while True:
    try:
        line = input()
    except EOFError:
        break
    texts.append(line.strip())
    

predicted = smiley_predict.predict_batch(texts)

print(predicted)
