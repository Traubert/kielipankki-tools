import sentiment
import sys

texts = open(sys.argv[1]).readlines()

predicted = sentiment.predixt(texts)

print(predicted)
