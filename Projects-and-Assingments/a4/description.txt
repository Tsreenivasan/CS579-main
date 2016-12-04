
-----------------------------------------------------------------------------
collect.py
-----------------------------------------------------------------------------

Streams for tweets on the top trending topic in Chicago.

Tweet limit is set at 5000 tweets, however, the program can be interupted at anytime after the collection of atleast 10 tweets to proceed onto next steps [cluster.py & classify.py]
The tweets are limited to non-retweets and the tweets should be from regular users[ Promotional tweets are omitted here]


File-Writes:: USTweets.pkl to save the tweets streamed.
-----------------------------------------------------------------------------
cluster.py
-----------------------------------------------------------------------------

for the first 15 users in the twitter data collected, we build a network by connecting the friends of the 15 users.[Network of common friends]

with the network of friends of the 15 users, we aim to form 5 clusters using the Girvan Newman algorithm, using Networkx' implementation of edge betweenness calculations as the best edge predictor.

The 5 clusters are plotted using the Matplotlib utility with different node coloring for Cluster Visibility.

Cluster Density highly depends on the demographic of the data collected.
[Graph Edge:Vertex varies by a large margin on various occasions  from 635:217 to 7560:3564 for just 15 users, Cases were the maximun common friend between users would be just 2, thus resulting in single node clusters]

the image Cluster_Pre.png is the cluster representation for the data collected when the trending topic was "#CalmMeDownIn3Words" which was very active on November 27th with over 80,000 tweets

File-Writes:: Cluster.pkl to save the cluster components detected by GIrvan Newman.
File-Writes:: Friend_dict.pkl to save the friends for the 15 users which were choosen from the raw tweets.
File-Required:: USTweets.pkl to read the tweets collected by the collect.py
File-Required:: Pre_Cluster.pkl to pre-calculated Clucters for plotting Cluster_Pre.png
-----------------------------------------------------------------------------
classify.py
-----------------------------------------------------------------------------

Tweet Sentiment classifier:

The training tweet's features were extracted using the TfidfVectorizer utlity from sklearn kit.

Using the feature matrix containing TF_IDF features where then fit into a Machine learning Model (Logistic regression here)

The Live/Test tweets are featurized again with the tf_idf featurizer and then transformed for classification.

The training data found 175725 terms to featurize and has a training accuracy of 83.1%.

The top weighted terms for the positive class as per the training data are:
[('thanks', 7.285234271008326), ('thank', 5.84250837013139), ('welcome', 5.4706131874535053), ('great', 4.8641161617356223), ('followfriday', 4.6527563254510289), ('awesome', 4.6508278201378905), ('worries', 4.3911631332831274), ('congratulations', 4.3332453129534709), ('worry', 4.2424250281994862), ('congrats', 4.2410968790007919)]

The top weighted terms for negative class as per the training data are:
[('sad', -11.681105888597008), ('miss', -8.6920005468044881), ('sucks', -8.0794400992391395), ('sorry', -7.7858151462349952), ('poor', -7.5560503009311466), ('wish', -6.8859545311639208), ('sadly', -6.4624621857529725), ('unfortunately', -6.3133918670894635), ('missed', -6.3042855705531), ('bummer', -5.50835773571667)]


Training data for the Sentiment analysis was downloaded from the Sentiment140 Site, which has various training sets for Sentiment Analysis.
Training data consists of 190000 tweets pre-classified and labeled as either Positive or Negative.
Source:: http://help.sentiment140.com/for-students/


The Training data set was limited from 60 millions to 189941 tweets for space reasons


File-Writes:: Classify.pkl to save the tweets and thier classification information.
File-Required:: USTweets.pkl to read the tweets collected by the collect.py
File-Required:: SS.csv to the training tweets and their sentiment labels for training the classifier.
-----------------------------------------------------------------------------
summarize.py
-----------------------------------------------------------------------------

Reads the 3 pickle output files to write the crucial output parameters as listed in the requirements.

Number of messages collected: 900
Number of users collected: 15
Average number of users per community:: 67.2
Tweets classified as Positive:: 1734
One tweet from the Positive class is below::
 ('#CalmMeDownIn3Words\nLove your smile', 'Positive', 0.9978497918996555, [('Negative', 0.0021502081003444973), ('Positive', 0.9978497918996555)])
Tweets classified as Negative:: 507
One tweet from the Negative class is below::
 ("#CalmMeDownIn3Words I'm so sorry", 'Negative', 0.99727515810945533, [('Negative', 0.99727515810945533), ('Positive', 0.0027248418905446918)])