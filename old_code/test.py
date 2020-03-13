import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from mahalanobis import Mahalanobis
#%matplotlib inline
from fastnumbers import fast_real
import math
import pandas as pd
import scipy
import numpy as np
from category_encoders import *
from scipy import spatial
from sklearn.datasets import load_boston
from category_encoders import OneHotEncoder

import sys, getopt
import pymongo
from pymongo import MongoClient
from pymongo import Connection
#from pymongo.dbref import DBRef
from pymongo.database import Database
from pprint import pprint



#CONTROL PARAMS

TEMP_STR_LEAD='temp/leads_file_out_enc3.csv'
TEMP_STR_AGENT='temp/ag_file_out_enc3.csv'
ml_inputfile='temp/predecisions.csv'

DEB_FLAG=1
#inputfileagent = ""
#inputfilelead = ""
#outputfile = ""
#DISTANCE LONG LAT
def geo_distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

# WEIGHT ENCODING
#weighted encoding based on sales of electric vehicle
gender_mapping = { "female" : 2,"company":5, "male" : 10}
lang_mapping = { "french":2, "english":5,"spanish" : 8,"portoguese" : 10 }
famstatus_mapping = { "single" : 1,"married" : 6}
profession_mapping = { "itprofessional" : 2,"taxidriver" : 20,"hotelmanager":6,"retailer":7,"biker" : 18,"bankmanager" : 13,"hotelmanager":17,"retailer":8,"architect":9,"propertymanager":10, "tourismmanager":11,"agriculturemanager":1,"none":-1    }
hobby_mapping = { "swimming" : 3,"cycling" : 10,"fitness":7,"cinema":1,"shopping" : 7,"diving" : 5,"meetupsocial":8,"investing":4,"carsocial":12,"garden":6, "none":-1 }
sport_mapping = { "Martial arts" : 7,"Watersports" : 2,"cycling":10,"dancing":5,"tennis" : 6,"motorsport" : 11,"Canoeing":9,"football":8,"basketball":4,"none":-1 }
travel_distance_mapping = { "low" : 100,"medium" :500,"high":1000}
age_group_mapping= { "young" : 20,"midage" :30,"senior":50}
lang_group_mapping= { "nonport" :0,"port_spain_group":1}
vehicle_life_group_mapping = { "Family" :3,"Mid":1,"VIP":10}
vehiclekind_mapping = { "BMWi3":1,"BMW330e":2,"Mercedes-BenzE300":3,"MitsubishiOutlander":4,"BMW530":5,"NISSAN LEAF":6,"JaguarPace":7,"Hyunday":8,"RenaultZOE":9,"TeslaModel3":10 }

# WEIGHT SIMLIARITY FOR 2 INPUT VECTORS
def cmp_all( dataSetI, dataSetII):
    
    diflang=abs(dataSetI[0]-dataSetII[0])     
    difage=abs(dataSetI[1]-dataSetII[1])        
    difgender=abs(dataSetI[2]-dataSetII[2])     
    
    if dataSetI[0] ==dataSetII[0]:
        lang_decision=1
    else:
        lang_decision=0 if (dataSetI[0]<8 or dataSetII[0]<8 ) else 0.75
          
    age_decision = 0 if difage>=30 else (30-difage)/30        
         
    if difgender>=6:
        gender_decision=0.25
    else:
        gender_decision=(10-difgender)/10
                      
    
    res=lang_decision*age_decision*gender_decision
    
    
    return res

def semantic_logic(inputfileagent, inputfilelead, outputfile):
    # INPUT LEADS
    #Open Leads
    print(inputfilelead)
    bunch = pd.read_csv(inputfilelead,dtype=str)
    df = bunch

    df['gender']=df['gender'].map(gender_mapping)
    df['lang']=df['lang'].map(lang_mapping)
    df['famstatus']=df['famstatus'].map(famstatus_mapping)
    df['profession']=df['profession'].map(profession_mapping)
    df['hobby']=df['hobby'].map(hobby_mapping)
    df['sport']=df['sport'].map(sport_mapping)
    df['travel_distance']=df['travel_distance'].map(travel_distance_mapping)
    df['age_group']=df['age_group'].map(age_group_mapping)
    df['lang_group']=df['lang_group'].map(lang_group_mapping)
    df['vehicle_life_group']=df['vehicle_life_group'].map(vehicle_life_group_mapping)
    df['vehiclekind']=df['vehiclekind'].map(vehiclekind_mapping)

    #used Mapping or One Hot encoding
    #enc = OrdinalEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
    #enc.fit(X)
    #df = enc.transform(X)


    # transform the dataset
    numeric_dataset = df
    numeric_dataset_file= pd.DataFrame(numeric_dataset)

    numeric_dataset_file['TYPE']=bunch['TYPE']
    numeric_dataset_file['userID']=bunch['userID']
    numeric_dataset_file['UserLatitude']=bunch['UserLatitude']
    numeric_dataset_file['UserLongitude']=bunch['UserLongitude']
    numeric_dataset_file['age']=bunch['age']
    numeric_dataset_file['target_id']=bunch['target_id']
    numeric_dataset_file.to_csv(TEMP_STR_LEAD, header=True, index=False)

    #Open Agents
    bunch= pd.read_csv(inputfileagent,dtype=str)
    #print ('****agents******')
    #X=bunch[['gender','famstatus','lang','profession','hobby', 'sport','travel_distance', 'age_group','lang_group']]
    #enc = OrdinalEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
    #enc.fit(X)
    #df= enc.transform(X)

    df=bunch

    df['gender']=df['gender'].map(gender_mapping)
    df['lang']=df['lang'].map(lang_mapping)
    df['famstatus']=df['famstatus'].map(famstatus_mapping)
    df['profession']=df['profession'].map(profession_mapping)
    df['hobby']=df['hobby'].map(hobby_mapping)
    df['sport']=df['sport'].map(sport_mapping)
    df['travel_distance']=df['travel_distance'].map(travel_distance_mapping)
    df['age_group']=df['age_group'].map(age_group_mapping)
    df['lang_group']=df['lang_group'].map(lang_group_mapping)


    numeric_dataset = df

    numeric_dataset_file= pd.DataFrame(numeric_dataset)
    numeric_dataset_file['TYPE']=bunch['TYPE']
    numeric_dataset_file['userID']=bunch['userID']
    numeric_dataset_file['UserLatitude']=bunch['UserLatitude']
    numeric_dataset_file['UserLongitude']=bunch['UserLongitude']
    numeric_dataset_file['age']=bunch['age']
    numeric_dataset_file['target_id']=bunch['target_id']
    numeric_dataset_file['agent_experience']=bunch['agent_experience']
    numeric_dataset_file['agent_location_group']=bunch['agent_location_group']
    numeric_dataset_file['alpha']=bunch['alpha']
    numeric_dataset_file['Exclusive']=bunch['Exclusive']

    #One Hot encoding /Ordinal Save
    numeric_dataset_file.to_csv(TEMP_STR_AGENT, header=True, index=False)

    df_leads = pd.read_csv(TEMP_STR_LEAD,dtype=str)
    df_agents = pd.read_csv(TEMP_STR_AGENT,dtype=str)

    df_leads.head()
    df_agents.head()


    if DEB_FLAG :
        #DISTANCE INTELLIGENCE
        np_df_leads=np.array(df_leads[['TYPE','userID']])
        Subdf_lead_user =df_leads[['userID']]
        Subdf_lead =df_leads[['lang','age','gender','famstatus','profession','hobby', 'sport', 'age_group', 'lang_group']]
        Subdf_lead_ag =df_agents[['userID']]
        Subdf_ag =df_agents[['lang','age','gender','famstatus','profession','hobby', 'sport', 'age_group', 'lang_group']]

        outframe= pd.DataFrame(columns=[ 'LeadID', 'ContaMediaAccount', 'Distance', 'SemDistCosine', 'SemDistHamming','SemDistCorrel','DistVIPHamming','WeightSem','Final','Final2','LeadLong','LeadLat','AgentLong','AgentLat','StoryLead','StoryAgent','VIPLeadStory','VIPAgentStory' ])
        outframe_res=outframe

        dataSetI_distance=df_leads[['UserLatitude','UserLongitude']]
        dataSetI_VIPLEAD = df_leads[['vehicle_life_group','vehicle_life_group','vehicle_life_group','vehicle_life_group']]
        dataSetII_distance=df_agents[['UserLatitude','UserLongitude']]
        dataSetII_ALPHAAG = df_agents[['agent_experience', 'agent_location_group', 'alpha', 'Exclusive']]

        for i in range (0,len(Subdf_lead.iloc[:,0])):
            dataSetI = np.array((Subdf_lead.iloc[i,:]))
            dataSetI_d =np.array(dataSetI_distance.iloc[i,0:2])
            dataSetI_u = Subdf_lead_user.iloc[i,0:1]
            dataSetI_VIPLEAD_action=np.array(dataSetI_VIPLEAD.iloc[i,0:4])


            for j in range(0,len(Subdf_ag.iloc[:,0])):
                dataSetII = np.array((Subdf_ag.iloc[j,:]))
                dataSetII_d =np.array(dataSetI_distance.iloc[j,0:2])
                dataSetII_u = Subdf_lead_ag.iloc[j,0:1]
                dataSetII_ALPHAAG_action=np.array( dataSetII_ALPHAAG.iloc[j,0:4])

                dataSetI = [float(i) for i in dataSetI]
                dataSetII = [float(i) for i in dataSetII]

                dataSetI_d = [float(i) for i in dataSetI_d]
                dataSetII_d = [float(i) for i in dataSetII_d]

                dataSetI_u = [int(i) for i in dataSetI_u]
                dataSetII_u = [int(i) for i in dataSetII_u]




                #SEMANTIC COUSINE
                result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

                #SEMANTIC HAMMING
                result3= 0.1+1 - spatial.distance.hamming(dataSetI, dataSetII)
                #GEO
                result2 = geo_distance(np.array(dataSetI_d), np.array(dataSetII_d))

                #CANBERRA
                result5=  1/(spatial.distance.canberra(dataSetI, dataSetII)+0.001)

                #WEIGHTED SEMANTIC

                result6=cmp_all(dataSetI, dataSetII)
                if result6<0.5:
                    result6=0  #dropout in case of dismatches gender, age, lang


                #VIP HAMMING
                result4= 0.1+ 1 - spatial.distance.hamming(dataSetII_ALPHAAG_action, dataSetI_VIPLEAD_action)

                final=(500-result2)*result3*math.sqrt(result6)*(0.2* result+0.3*result3+0.1*result4+0.3*result5+result6*0.3)
                final2=result3*math.sqrt(result6)*(0.2* result+0.3*result3+0.1*result4+0.3*result5+result6*0.3)+ math.sqrt(500-result2)/1000

                #print('Lead id:',np.array(dataSetI_u),'ContaMediaAccount:',np.array(dataSetII_u),'Distance:', result2, 'SemDistCosine:',result, 'SemDistHamming:',result3,'SemDistCorrel:',result5,'DistVIPHamming:',result4,'WeightSem:',result6,'Final:',final, 'Final2:',final2,'LeadLong:', np.array(dataSetI_d),  'LeadLat:','AgentLong:', 'AgentLat:',np.array(dataSetII_d)  )
                if i % 10 ==1 and j % 20 ==1 :
                    print('Lead id:',np.array(dataSetI_u),'ContaMediaAccount:',np.array(dataSetII_u),'Distance:', result2, 'Final:',final, 'Final2:',final2 )


                outframe = outframe.append({'LeadID' :int(np.array(dataSetI_u)),'ContaMediaAccount': int(np.array(dataSetII_u)), 'Distance':result2, 'SemDistCosine':result, 'SemDistHamming':result3,'SemDistCorrel':result5, 'DistVIPHamming':result4,'WeightSem': result6,'Final':final,'Final2':final2,'LeadLat':np.array(dataSetI_d[0]),'LeadLong':np.array(dataSetI_d[1]),'AgentLat':np.array(dataSetII_d[0]),'AgentLong':np.array(dataSetII_d[1]) ,  'StoryLead':dataSetI,'StoryAgent':dataSetII, 'VIPLeadStory':dataSetI_VIPLEAD_action, 'VIPAgentStory':dataSetII_ALPHAAG_action  }, ignore_index=True)
                #outframe = outframe.append({'LeadID' :int(np.array(dataSetI_u)),'ContaMediaAccount': int(np.array(dataSetII_u)), 'Distance':result2, 'SemDistCosine':result, 'SemDistHamming':result3,'SemDistCorrel':result5, 'DistVIPHamming':result4,'Final':final,'Final2':final2,'LeadLat':np.array(dataSetI_d[0]),'LeadLong':np.array(dataSetI_d[1]),'AgentLat':np.array(dataSetII_d[0]),'AgentLong':np.array(dataSetII_d[1])  }, ignore_index=True)


        outframe2=outframe
        resultfr=outframe.sort_values(by=['Final'],ascending=False).head(5*160)
        resultfr2=outframe2.sort_values(by=['Final2'],ascending=False).head(5*160)

        resultfr3=resultfr.sort_values(by=['LeadID','Final'],ascending=False)
        resultfr4=resultfr2.sort_values(by=['LeadID','Final2'],ascending=False)





        print('Saved')

        resultfr3.to_csv(outputfile, header=True, index=True)
        resultfr3.to_csv(ml_inputfile, header=True, index=True)

        #resultfr4.to_csv('out/out2.csv', header=True, index=True)
    
    
def machine_learning_logic(inputfileagent, inputfilelead, outputfile) :
    #MACHINE LEARNING LOGIC FOR SUPERVISED LEARNING ADJUSTMENT BASED ON TARGET COLUMNS AS FUNCTION FROM FEATURES, RULES
    #APPLIED FOR LEADS

    df_leads = pd.read_csv( TEMP_STR_LEAD, dtype=str)
    df_agents = pd.read_csv(TEMP_STR_AGENT, dtype=str)
    df_decisions = pd.read_csv(ml_inputfile, dtype=str)

    print (df_decisions.head())



    df=df_leads
    df = df.fillna('None')
    df = df.groupby('target_id').filter(lambda x: len(x) >= 5)


    def seqtodict(sequence, start=0):
        #transform a sequence to a set of pairs to make into a dictionary
        n = start
        for elem in sequence:
            yield elem,n
            n += 1

    def getItemDict(itemvalues):
        itemvalues = list(itemvalues.unique())
        itemdict = {}
        if 'None' not in itemvalues:
            itemvalues = ['None'] + itemvalues
        itemdict = dict( seqtodict(itemvalues))
        return itemdict


    itemdict = getItemDict(df['target_id'])

    agecode=getItemDict(df['age'])
    genderecode=getItemDict(df['gender'])
    famstatuscode=getItemDict(df['famstatus'])
    lang1code=getItemDict(df['lang'])

    professioncode=getItemDict(df['profession'])
    hobbycode=getItemDict(df['hobby'])

    sportcode=getItemDict(df['sport'])
    agegroupcode=getItemDict(df['age'])


    f11code=getItemDict(df['UserLongitude'])
    f12code=getItemDict(df['UserLatitude'])
    f21code=getItemDict(df['userID'])

    #decision assisting explainable
    f22code=getItemDict(df['age_group'])
    f23code=getItemDict(df['lang_group'])
    f24code=getItemDict(df['vehicle_life_group'])

    target_code=getItemDict(df['target_id'])


    def getCodeorNone(x,dictionary):
        output = 0
        try:
            output=dictionary[x]
        except:
            output=dictionary['None']
        return output

    agecode=getItemDict(df['age'])

    gendercode=getItemDict(df['gender'])

    famstatuscode=getItemDict(df['famstatus'])
    langcode=getItemDict(df['lang'])
    sportcode=getItemDict(df['sport'])

    df['age'] = df['age'].apply(lambda x: getCodeorNone(x,agecode))
    df['gender'] = df['gender'].apply(lambda x: getCodeorNone(x,genderecode))
    df['famstatus'] = df['famstatus'].apply(lambda x: getCodeorNone(x,famstatuscode))
    df['lang'] = df['lang'].apply(lambda x: getCodeorNone(x,langcode))
    df['sport'] = df['sport'].apply(lambda x: getCodeorNone(x,sportcode))
    df['profession'] = df['profession'].apply(lambda x: getCodeorNone(x,professioncode))
    df['hobby'] = df['hobby'].apply(lambda x: getCodeorNone(x,hobbycode))


    df['feature11-code'] = df['UserLongitude'].apply(lambda x: getCodeorNone(x,f11code))
    df['feature12-code'] = df['UserLatitude'].apply(lambda x: getCodeorNone(x,f12code))
    df['feature21-code'] = df['userID'].apply(lambda x: getCodeorNone(x,f21code))
    df['feature22-code'] = df['age_group'].apply(lambda x: getCodeorNone(x,f22code))
    df['feature23-code'] = df['lang_group'].apply(lambda x: getCodeorNone(x,f23code))
    df['feature24-code'] = df['vehicle_life_group'].apply(lambda x: getCodeorNone(x,f24code))


    # target code
    df['target_code'] = df['target_id'].apply(lambda x: getCodeorNone(x,itemdict))


    train, test = train_test_split(df, test_size=0.2, random_state=23,stratify=df['target_id'])

    featurecolumns = ['age','gender','lang',
                       'feature11-code',
                      'feature12-code',
                     'sport','famstatus',
                     'feature22-code',
                      'feature23-code',
                          'feature24-code'



              ]
    '''
    
    
               
                        
                          
     '''
    features_train = train[featurecolumns].values
    labels_train = train['target_code'].values
    features_test = test[featurecolumns].values
    labels_test = test['target_code'].values

    #print (features_train)
    #print (features_test)

    dtrain = xgb.DMatrix(features_train, label=labels_train)
    dtest = xgb.DMatrix(features_test, label=labels_test)


    # user defined evaluation function, return a pair metric_name, result
    # This function takes the top 10 predictions and checks to see if the target label is in that set.
    # The error is 1 - the fraction of rows where the label is in the top 10.
    def evalerror(preds, dtrain,topNvalue=10):
        labels = dtrain.get_label()
        vals = np.argpartition(preds,-topNvalue)[:,-topNvalue:]
        error = 1 - float(vals.size - np.count_nonzero((vals.transpose() - labels).transpose()))/len(labels)
        # return a pair metric_name, result
        return 'error', error


    num_round=300
    param = {'max_depth': 5,
        'eta': 0.2,
        'silent': 0,
        'gamma':2,
        'objective':'multi:softprob',
        'num_class':len(np.unique(labels_train))+1,
        'seed':32}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist, feval=evalerror, early_stopping_rounds=30)






def connect_db():

    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    #client = MongoClient( "towert5:towert5@cluster0 - intti.mongodb.net / test?retryWrites = true & w = majority")
    #mongodb + srv: // < username >: < password > @< cluster - url > / test?retryWrites = true & w = majority

    #client = pymongo.MongoClient("towert5:towert5@cluster0 - intti.mongodb.net/test?retryWrites=true&w=majority")
    #client = pymongo.MongoClient("mongodb://towert5:towert5@cluster0 - intti.mongodb.net/test?retryWrites=true&w=majority:27017]")
    #client = pymongo.MongoClient(db="campaign_db",alias=alias,"mongodb://towert5:towert5@cluster0 - intti.mongodb.net/test?retryWrites=true&w=majority:27017")
    #client = pymongo.MongoClient("mongodb://towert5:towert5@cluster0 - intti.mongodb.net/test?retryWrites=true&w=majority")
    #client = pymongo.MongoClient("mongodb://towert5:towert5@cluster0-intti.mongodb.net/test?retryWrites = true & w = majority")

    #client = pymongo.MongoClient("mongodb+srv://    towert5:towert5@cluster0-intti.mongodb.net/test?retryWrites=true&w=majority")
    client = pymongo.MongoClient("mongodb+srv://towert5:towert5@cluster0-intti.mongodb.net/test?retryWrites=true&w=majority")
    #client = pymongo.MongoClient("mongodb:// towert5: towert5 @ cluster0 - intti.mongodb.net / test?retryWrites = true & w = majority")
    #client = pymongo.MongoClient("mongodb:// towert5: towert5 @ cluster0 - intti.mongodb.net:27017/ test?retryWrites = true & w = majority")
    #

    #db =  client.campaign_db
    # Issue the serverStatus command and print the results
    #serverStatusResult = db.command("serverStatus")
    #print(serverStatusResult)

    #col = db.sample-collection
    #connection = Connection()
    #db = Database(connection, "campaign_db")
    # db = client.test
    # db = client.admin
    #db = client.campaign_db

    ##mongodb + srv: // towert5: towert5 @ cluster0 - intti.mongodb.net / test?retryWrites = true & w = majority
    #campaign_db
    #collections agents, leads, matches, campaigns
    # connect
    '''
    connection = Connection()
    db = Database(connection, "things")
    # clean up
    db.owners.remove()
    db.tasks.remove()
    # owners and tasks
    db.owners.insert({"name": "Jim"})
    db.tasks.insert([
        {"name": "read"},
        {"name": "sleep"}
    ])

    # update jim with tasks: reading and sleeping
    reading_task = db.tasks.find_one({"name": "read"})
    sleeping_task = db.tasks.find_one({"name": "sleep"})

    jim_update = db.owners.find_one({"name": "Jim"})
    jim_update["tasks"] = [
        DBRef(collection="tasks", id=reading_task["_id"]),
        DBRef(collection="tasks", id=sleeping_task["_id"])
    ]

    db.owners.save(jim_update)

    # get jim fresh again and display his tasks
    fresh_jim = db.owners.find_one({"name": "Jim"})
    print
    "Jim's tasks are:"
    for task in fresh_jim["tasks"]:
        print
        db.dereference(task)["name"]

    client = MongoClient()
    client = MongoClient('localhost', 27017)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.test_database
    db = client['test-database']
    collection = db.test_collection
    collection = db['test-collection']
    import datetime
    >> > post = {"author": "Mike",
                 ...         "text": "My first blog post!",
                                     ...
    "tags": ["mongodb", "python", "pymongo"],
    ...
    "date": datetime.datetime.utcnow()}
    '''

def main(argv):
   inputfileagent = sys.argv[1]
   inputfilelead = sys.argv[2]
   outputfile = sys.argv[3]



   print ('Input file 1 is "', inputfileagent )
   print('Input file 2 is "', inputfilelead)
   print ('Output file is "', outputfile)

   #connect_db()


   print('SEMANTIC Logic')
   #semantic_logic(inputfileagent, inputfilelead, outputfile)
   #print('ML Logic')
   machine_learning_logic(inputfileagent, inputfilelead, outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])

