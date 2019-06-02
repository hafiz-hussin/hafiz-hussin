import nltk
from textblob import TextBlob


blob1 = TextBlob("CIMB encourages Malaysians to save to forward their ambitions_")
print(format(blob1.polarity))

blob2 = TextBlob("CIMB Group announces RM3.41 billion Net Profit for 9M17")
print(format(blob2.polarity))
