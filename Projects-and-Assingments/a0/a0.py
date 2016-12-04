from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import configparser
from TwitterAPI import TwitterAPI

def get_twitter():
    config = configparser.ConfigParser()
    config.read('Twitter_tokens.cfg')
    consumer_key = config.get('twitter','consumer_key')
    consumer_secret = config.get('twitter','consumer_secret')
    access_token = config.get('twitter','access_token')
    access_token_secret = config.get('twitter','access_token_secret')
    twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    return twitter

def robust_request( resource, params, max_tries=1):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            #time.sleep(61 * 15)

def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.
    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    fdesc = open(filename, 'r')
    alist=[]
    for l in fdesc:
        alist.append(l.strip().split(","))
    return alist[0]

def get_users(twitter,screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    aa=[]
    for a  in range(len(screen_names)):
        r = twitter.request('users/lookup',{'screen_name': screen_names[a]})
        for item in r:
            aa.append(item)
    
    """
    returning a list of user dictionaries 
    """
    return aa


def print_num_friends(users):
    alist=[]
    aa1={}
    print('Friends per candidate:')
    for a  in range(len(users)):
        count=0
        for item in users[a]['friends']:
            count+=1
        aa1['Friends_Count']=count
        aa1['Name']= users[a]['screen_name']
        print( users[a]['screen_name'],count)
    
def get_friends(twitter,screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.
    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.
    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    alist=[]
    count=0
    r = twitter.request('friends/ids',{'screen_name': screen_name,'count':5000})
    for item in r:
        alist.append(item)
    alist.sort()
    return alist

def diction_list(twitter,users):
    """
    Custom function for the graphs
    """
    alist=[]
    aa1={}
    aa2={}
    for a  in range(len(users)):
        alist=[]
        count=0
        r = twitter.request('friends/ids',{'screen_name': users[a],'count':5000})
        for item in r:
            alist.append(item)
        aa1['friends']=alist
        aa1['screen_name']=users[a]
        aa2[a]=dict(aa1)
    return aa2

def add_all_friends(twitter,users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.
    Store the result in each user's dict using a new key called 'friends'.
    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    alist=[]
    aa1={}
    aa2={}
    for a  in range(len(users)):
        alist=[]
        count=0
        r = twitter.request('friends/ids',{'screen_name': users[a]['screen_name'],'count':5000})
        for item in r:
            alist.append(item)
        alist.sort()
        users[a]['friends']=alist

def count_friends_no_counter(twitter,names):
    from itertools import groupby
    alist=[]
    count=0
    for a  in range(len(names)):
        count=0
        r = twitter.request('friends/ids',{'screen_name': names[a],'count':5000})
        for item in r:
            alist.append(item)
    alist.sort()
    d = {x:alist.count(x) for x in alist}
    e=sorted(d, key=lambda x: -d[x])
    print('Most common Friends::')
    for k in e:
        count+=1
        if (d[k]>1) and count<5:
            print("{} : {}".format(k, d[k]))
    #print(abc)
    #print(aa2)

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    from itertools import groupby
    alist=[]
    count=0
    for a  in range(len(users)):
        for i in users[a]['friends']:
            alist.append(i)
    alist.sort()
    c=Counter(alist)
    return c
    
def get_Followers_Count(twitter,names):
    aa1={}
    aa2={}
    for i  in range(len(names)):
        r = twitter.request('friends/list', {'screen_name': names[i],'count':2})
        aaa=[]
        for item in r:
            aaa.append(item['screen_name'])
        for a  in range(len(aaa)):
            count=0
            r = twitter.request('followers/list',{'screen_name': aaa[a],'count':200})
            for item in r:
                count+=1
            aa1['Friend_Name']=names[i]
            aa1['Follow_Count']=count
            aa1['Name']=aaa[a]
            print(aa1)
            aa2[a]=dict(aa1)
        print(aa2)
  
      
def get_common_Friends(dct):
    from collections import defaultdict
    from itertools import groupby
    #dct = dict({0: {'Name': 'DrJillStein', 'Friends': [1, 227720229, 456794981, 3, 35175054]}, 1: {'Name': 'GovGaryJohnson', 'Friends': [1, 41272328, 581275345, 3, 184869706]}, 2: {'Name': 'HillaryClinton', 'Friends': [325886383, 802430450, 729761993461248000, 3, 34782406]}, 3: {'Name': 'realDonaldTrump', 'Friends': [720293443260456960, 2325495378, 245963716, 50769180, 22203756]}})
    groups = defaultdict(list)
    finallist=[]
    for r in range(len(dct)):
        for rs in range(len(dct)):
            prelist=[]
            count=0
            if dct[rs]['screen_name']==dct[r]['screen_name']:
                #break
                pass 
            else:
                a=set(dct[r]['friends'])&set(dct[rs]['friends'])
                if len(a)!=0:
                    prelist.append(dct[r]['screen_name'])
                    prelist.append(dct[rs]['screen_name'])
                    prelist.sort()
                    prelist.append(a)
                    finallist.append(prelist)
    finallist.sort()
    unique = []
    [unique.append(item) for item in finallist if item not in unique]
    return unique
  
      
def followed_by_hillary_and_donald(users,twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup
    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    aa1={}
    aa2={}
    for a  in range(len(users)):
        aaa=[]
        #for item in r:
         #   aaa.append(item['screen_name'])
        count=0
        alist=[]
        r = twitter.request('friends/list',{'screen_name': users[a],'count':200})
        for item in r:
            alist.append(item['screen_name'])
        alist.sort()
        aa1['Friends']=alist
        aa1['Name']=users[a]
        aa2[a]=dict(aa1)
    print('User followed by both Hillary and Donald::',set(aa2[0]['Friends']) & set(aa2[1]['Friends']))
    
def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.
    Args:
        users...The list of user dicts.
    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.
    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    from collections import defaultdict
    from itertools import groupby
    #dct = dict([('a',['1', '2', '3','4']),('b',['2', '3', '4','5']),('c',['1', '2', '3','4'])])
    groups = defaultdict(list)
    finallist=[]
    for r in range(len(users)):
        for rs in range(len(users)):
            prelist=[]
            count =0
            if users[rs]['screen_name']==users[r]['screen_name']:
                #break
                pass 
            else:
                #print(dct[r]['Name'],dct[rs]['Name'])
                newval=users[r]['friends']+users[rs]['friends']
                newval.sort()
                """
                creating a list of lists where each list has user friend count in [A,B,300] and sort the list by name
                """
                abc= [len(list(group)) for key, group in groupby(newval)]
                for value in abc:
                    if value>1:
                        count+=1
                prelist.append(users[r]['screen_name'])
                prelist.append(users[rs]['screen_name'])
                prelist.sort()
                prelist.append(count)
                finallist.append(prelist)
    finallist.sort()
    unique = []
    """
    Removing Duplicates from the overlap list
    """
    [unique.append(item) for item in finallist if item not in unique]
    unique.sort(key=lambda x:x[2],reverse=True)
    u=[]
    for i in unique:
        u.append(tuple(i))
    
    return u

def create_graph(users,friend_counts):
    import networkx as nx
    import matplotlib.pyplot  as plt
    #ls = list([['DrJillStein', 'GovGaryJohnson', {41877508, 70799371, 12, 65404941, 39184406, 14075928, 15224867, 335489061, 23865382, 182503471, 1180379185, 952086582, 33908800, 16017475, 606412878, 299364430, 108617810, 393588826, 15667291, 130613344, 43202658, 17825891, 16664681, 22190185, 45713517, 438716528, 2835886194, 64264307, 37599351, 28395645, 17291393, 1075333250, 364423298, 631851140, 5392522, 16545933, 5695632, 2467621014, 18700441, 48847011, 87818409, 807095, 14270650, 15614141, 14246088, 610533, 390764798, 2916305152, 15970567, 409456907, 199848209, 115941650, 50325797, 9300262, 210684204, 428333, 38271276, 332603696, 17637692, 14293310, 20068679, 425171276, 26657119, 222953824, 14066024, 44134773, 115485051, 15458694, 727472528, 18622869, 16589206, 322603418, 48802204, 41779621, 182106541, 1308334518, 14924233, 128346572, 19034576, 27075032, 121817564, 214653407, 1128886760, 15030766, 15485441, 6017542, 294246919, 274780681, 107190796, 793295376, 23112236, 29465136, 16116288, 18164289, 29612613, 315331161, 14957147, 15108702, 51241574, 208155240, 29330030, 19081841, 30182005, 219011705, 525419131, 15012486, 19051166, 326255267, 20017835, 35785401, 4816, 6351572, 25055963, 6904552, 19608297, 24578794, 450941680, 20552442, 742143, 18451200, 183333634, 243639047, 7684882, 554898206, 1917731, 80366372, 17017636, 16042794, 19614512, 286303032, 202924869, 17466186, 32774989, 28785486, 7848802, 1520501, 387980164, 228223882, 43019146, 17470346, 18125710, 30725017, 611986351, 21115824, 2456142770, 174338995, 18615231, 50490315, 34927577, 290180065, 455115746, 224320485, 47490022, 18172905, 17093617, 9532402, 10726392, 11856892, 48088063, 16303106, 1053236226, 15846407, 19346439, 18510860, 17568791, 1201191, 175711277, 2303751216, 95740980, 168295477, 15463486, 183403584, 15414356, 43068508, 218780765, 1178700896, 974822502, 14855279, 300893304, 2953708673, 14173315, 35218566, 30729355, 192935052, 14222486, 475294870, 14345368, 5741722, 58926235, 1276206235, 13393052, 21228708, 18754727, 20092071, 40242357, 473564344, 547114168, 119504076, 64519379, 59159771, 15717596, 1469101279, 288363743, 15463671, 21368060, 20956414, 16076032, 21669123, 82656516, 39308549, 466519303, 701725963, 624774427, 137288990, 15205668, 309822757, 16125224, 17655090, 350661938, 287413569, 17220934, 161752401, 4207961, 19721592, 10886522, 14630267, 22326665, 14529929, 18812301, 174953869, 30807438, 14128528, 904539540, 61267358, 20702626, 2218177956, 66870702, 15711668, 22093237, 16625082, 759251, 902886877, 43277789, 254117355, 197496309, 3108351, 21982720, 12506632, 22685200, 31464977, 16815644, 25314856, 4432916014, 224351791, 187426352, 41166408, 89052747, 17423948, 17006157, 14515799, 21612122, 18208354, 15502950, 28096105, 16467567, 37721714, 311248499, 21542518, 112803449, 75052666, 63813253, 208875148, 21360280, 33310366, 58945187, 1947301, 7343782, 91180720, 435146421, 82939583, 21307076, 43638469, 58709723, 5988062, 307687137, 44660449, 16244449, 9567972, 19091173, 34295531, 4509306615, 89820928, 15675138, 84162307, 20954885, 19697415, 17243913, 118245138, 54208274, 44988185, 158426909, 2282706721, 71294756, 18956073, 17112878, 24258355, 13850422, 93069110, 28012345, 14855994, 1652541, 1375289149, 18587457, 760639303, 21862217, 14511951, 112283476, 533061461, 15503210, 28049266, 115754870, 16400248, 373157754, 2542475130, 16129920, 15384449, 20608910, 429227921, 16668573, 12011422, 1532497832, 14645160, 22677427, 216776631, 2836421, 158769098, 2467791, 19568591, 2424569809, 19580890, 66768858, 136402916, 34795502, 64643056, 33669105, 37965812, 5402612, 13832182, 158414847}], ['DrJillStein', 'HillaryClinton', {15846407, 19608297, 19248106, 48269483, 43433353, 120153772, 63112528, 19397785, 40684090, 18766459}], ['DrJillStein', 'realDonaldTrump', {4121225056, 245963716, 39349894, 22203756, 108471631, 620571475, 52136185, 25429371}], ['GovGaryJohnson', 'HillaryClinton', {352548417, 46213956, 30313925, 14185317, 15846407, 50374439, 19608297, 35743863, 45671898}], ['GovGaryJohnson', 'realDonaldTrump', {15513604, 37764422, 216299334, 14839147, 14669951, 246500501, 23970102, 16031927, 41634520, 50769180, 196168350, 21619519}], ['HillaryClinton', 'realDonaldTrump', {248900032}]])
    G=nx.Graph()

    for in1 in users:
        prev1=''
        prev2=''
        prev3=[]
        count=0
        for in2 in in1:
            if (type(in2)is str):
                G.add_node(in2)
                if len(prev1)!=0:
                    if(prev2!=prev1):
                        G.add_edge(in2,prev1)
                        prev2=in2
                if(len(prev1)==0):
                    prev1=in2
            elif (type(in2)is set):
                prev3=in2
                for in3 in in2:
                    G.add_node(in3)
        if(len(prev1)!=0 and len(prev2)!=0 and len(prev3)!=0):
            for i in prev3:
                G.add_edge(i,prev1)
                G.add_edge(i,prev2)

    nx.draw(G,with_labels=True)
    print('graph with N=%d E=%d' 
              % (len(G.nodes()), len(G.edges())))#nx.draw(G,with_labels=True)
    #plt.savefig("simple_path.png")
    #plt.show() # display

def draw_network(graph, users, filename):
    import networkx as nx
    import matplotlib.pyplot  as plt
    #ls = list([['DrJillStein', 'GovGaryJohnson', {41877508, 70799371, 12, 65404941, 39184406, 14075928, 15224867, 335489061, 23865382, 182503471, 1180379185, 952086582, 33908800, 16017475, 606412878, 299364430, 108617810, 393588826, 15667291, 130613344, 43202658, 17825891, 16664681, 22190185, 45713517, 438716528, 2835886194, 64264307, 37599351, 28395645, 17291393, 1075333250, 364423298, 631851140, 5392522, 16545933, 5695632, 2467621014, 18700441, 48847011, 87818409, 807095, 14270650, 15614141, 14246088, 610533, 390764798, 2916305152, 15970567, 409456907, 199848209, 115941650, 50325797, 9300262, 210684204, 428333, 38271276, 332603696, 17637692, 14293310, 20068679, 425171276, 26657119, 222953824, 14066024, 44134773, 115485051, 15458694, 727472528, 18622869, 16589206, 322603418, 48802204, 41779621, 182106541, 1308334518, 14924233, 128346572, 19034576, 27075032, 121817564, 214653407, 1128886760, 15030766, 15485441, 6017542, 294246919, 274780681, 107190796, 793295376, 23112236, 29465136, 16116288, 18164289, 29612613, 315331161, 14957147, 15108702, 51241574, 208155240, 29330030, 19081841, 30182005, 219011705, 525419131, 15012486, 19051166, 326255267, 20017835, 35785401, 4816, 6351572, 25055963, 6904552, 19608297, 24578794, 450941680, 20552442, 742143, 18451200, 183333634, 243639047, 7684882, 554898206, 1917731, 80366372, 17017636, 16042794, 19614512, 286303032, 202924869, 17466186, 32774989, 28785486, 7848802, 1520501, 387980164, 228223882, 43019146, 17470346, 18125710, 30725017, 611986351, 21115824, 2456142770, 174338995, 18615231, 50490315, 34927577, 290180065, 455115746, 224320485, 47490022, 18172905, 17093617, 9532402, 10726392, 11856892, 48088063, 16303106, 1053236226, 15846407, 19346439, 18510860, 17568791, 1201191, 175711277, 2303751216, 95740980, 168295477, 15463486, 183403584, 15414356, 43068508, 218780765, 1178700896, 974822502, 14855279, 300893304, 2953708673, 14173315, 35218566, 30729355, 192935052, 14222486, 475294870, 14345368, 5741722, 58926235, 1276206235, 13393052, 21228708, 18754727, 20092071, 40242357, 473564344, 547114168, 119504076, 64519379, 59159771, 15717596, 1469101279, 288363743, 15463671, 21368060, 20956414, 16076032, 21669123, 82656516, 39308549, 466519303, 701725963, 624774427, 137288990, 15205668, 309822757, 16125224, 17655090, 350661938, 287413569, 17220934, 161752401, 4207961, 19721592, 10886522, 14630267, 22326665, 14529929, 18812301, 174953869, 30807438, 14128528, 904539540, 61267358, 20702626, 2218177956, 66870702, 15711668, 22093237, 16625082, 759251, 902886877, 43277789, 254117355, 197496309, 3108351, 21982720, 12506632, 22685200, 31464977, 16815644, 25314856, 4432916014, 224351791, 187426352, 41166408, 89052747, 17423948, 17006157, 14515799, 21612122, 18208354, 15502950, 28096105, 16467567, 37721714, 311248499, 21542518, 112803449, 75052666, 63813253, 208875148, 21360280, 33310366, 58945187, 1947301, 7343782, 91180720, 435146421, 82939583, 21307076, 43638469, 58709723, 5988062, 307687137, 44660449, 16244449, 9567972, 19091173, 34295531, 4509306615, 89820928, 15675138, 84162307, 20954885, 19697415, 17243913, 118245138, 54208274, 44988185, 158426909, 2282706721, 71294756, 18956073, 17112878, 24258355, 13850422, 93069110, 28012345, 14855994, 1652541, 1375289149, 18587457, 760639303, 21862217, 14511951, 112283476, 533061461, 15503210, 28049266, 115754870, 16400248, 373157754, 2542475130, 16129920, 15384449, 20608910, 429227921, 16668573, 12011422, 1532497832, 14645160, 22677427, 216776631, 2836421, 158769098, 2467791, 19568591, 2424569809, 19580890, 66768858, 136402916, 34795502, 64643056, 33669105, 37965812, 5402612, 13832182, 158414847}], ['DrJillStein', 'HillaryClinton', {15846407, 19608297, 19248106, 48269483, 43433353, 120153772, 63112528, 19397785, 40684090, 18766459}], ['DrJillStein', 'realDonaldTrump', {4121225056, 245963716, 39349894, 22203756, 108471631, 620571475, 52136185, 25429371}], ['GovGaryJohnson', 'HillaryClinton', {352548417, 46213956, 30313925, 14185317, 15846407, 50374439, 19608297, 35743863, 45671898}], ['GovGaryJohnson', 'realDonaldTrump', {15513604, 37764422, 216299334, 14839147, 14669951, 246500501, 23970102, 16031927, 41634520, 50769180, 196168350, 21619519}], ['HillaryClinton', 'realDonaldTrump', {248900032}]])
    G=nx.Graph()
    dit={}
    for in1 in users:
        prev1=''
        prev2=''
        prev3=[]
        count=0
        for in2 in in1:
            if (type(in2)is str):
                G.add_node(in2)
                if len(prev1)!=0:
                    if(prev2!=prev1):
                        G.add_edge(in2,prev1)
                        prev2=in2
                if(len(prev1)==0):
                    prev1=in2
            elif (type(in2)is set):
                prev3=in2
                for in3 in in2:
                    G.add_node(in3)
        if(len(prev1)!=0 and len(prev2)!=0 and len(prev3)!=0):
            for i in prev3:
                G.add_edge(i,prev1)
                if(prev1 not in dit.keys()):
                    dit[prev1]=list(str(i))
                elif(prev2 not in dit.keys()):
                    dit[prev2]=list(str(i))
                else:
                    dit[prev1].append(str(i))
                    dit[prev2].append(str(i))
                G.add_edge(i,prev2)
    labels = {} 
    hubs=['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    for node in G.nodes():
        if node in hubs:
            #set the node name as the key and the label as its value 
            labels[node] = node
    #set the argument 'with labels' to False so you have unlabeled graph
    pos=nx.spring_layout(G)
    #Now only add labels to the nodes you require (the hubs in my case)
    plt.clf()    
    nx.draw_networkx(G,pos,node_size=60,width=0.2,alpha=.5,with_labels=False)
    nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='black')
    plt.savefig(filename,dpi=1200)
    print('Network drawn to ::',filename)
    plt.show() # display
    

def main():
    api= get_twitter()
    print('Established Twitter connection.')
    r = api.request('followers/list', {'count':2})
    File_List=read_screen_names('candidates.txt')
    print('Read screen names: ',File_List)
    users=get_users(api,File_List)
    diction=diction_list(api,File_List)
    add_all_friends(api,users)
    print_num_friends(users)
    c=count_friends(users)
    print('Most Common friends')
    print(c.most_common(5))
    followed_by_hillary_and_donald(['HillaryClinton','realdonaldtrump'],api)
    print('Friend Overlap')
    u=friend_overlap(diction)
    print(u)
    ret_list= get_common_Friends(diction)
    create_graph(ret_list,'')
    draw_network('',ret_list,'network.png')
    
if __name__ == '__main__':
    main()
