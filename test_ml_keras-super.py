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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from IPython.core.display import display, HTML
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import heapq



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

def getCodeorNone(x,dictionary):
    output = 0
    try:
        output=dictionary[x]
    except:
        output=dictionary['None']
    return output



#CONTROL PARAMS MID FILES
TEMP_STR_LEAD='temp/lead_encoder.csv'
TEMP_STR_AGENT='temp/agent_encoder.csv'
#ml_inputfile='temp/predecision.csv'
ml_inputfile = 'temp/predecisions_manual_humansupervised.csv'

#METIC SCORE
DIST_TOP_HIGH=100
#RIGID PARAM
DISTANCE_THRESHOLD=50

#ML
Epochs=200


Num=3 #TOP FOR ML
#GENERIC
TOP_NUMBER_SEARCH=10
TOP_NUMBER_ALL_LEADS=160

DEB_FLAG=1
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
gender_mapping = { "F" : 2,"Z":5, "M" : 10}
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
agent_location_group_mapping= { "DMN" :0,"DMI":1,"DMS":2 }

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

def MLDecision_xgb(VectorDataFrame):
    # IN PROGRESS
    # load saved model
    #xgb = joblib.load('xgb.model')
    #xgb = pickle.load(open("xgb-python.model", "rb"))
    #preds2 = xgb.predict(dtest, ntree_limit=bst2.best_ntree_limit)
    #print(1 - evalerror(preds2, dtest)[1])
    return 1


def MLDecision_keras(VectorDataFrame):
    # MACHINE LEARNING LOGIC FOR SUPERVISED LEARNING ADJUSTMENT BASED ON TARGET COLUMNS AS FUNCTION FROM FEATURES, RULES
    # APPLIED FOR LEADS
    #IN PROGRESS
    from keras.models import load_model
    model1 = load_model('model512.h5')

    featurecolumns = ['LeadID', 'ContaMediaAccount', 'StoryLead-Code', 'StoryAgent-Code', 'SemDistHamming',
                      'SemDistCorrel', 'DistVIPHamming', 'WeightSem']

    #df = VectorDataFrame[featurecolumns]
    #df = pd.DataFrame(VectorDataFrame)


    df_decisions = pd.read_csv(ml_inputfile, dtype=str)
    # df_decisions = df
    df=df_decisions

    def seqtodict(sequence, start=0):
        # transform a sequence to a set of pairs to make into a dictionary
        n = start
        for elem in sequence:
            yield elem, n
            n += 1

    def getItemDict(itemvalues):
        itemvalues = list(itemvalues.unique())
        itemdict = {}
        if 'None' not in itemvalues:
            itemvalues = ['None'] + itemvalues
        itemdict = dict(seqtodict(itemvalues))
        return itemdict

    def getModel(n, num_epochs=Epochs):
        model = Sequential()
        model.add(Dense(n, activation='relu', input_dim=len(feature_columns)))
        model.add(Dropout(0.5))
        model.add(Dense(int(n / 2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nitems, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        callbacks = [EarlyStopping(monitor='acc', patience=10, verbose=0),
                     TQDMNotebookCallback()]
        model.fit(X_train, y_train,
                  epochs=num_epochs,
                  batch_size=128, verbose=1,
                  callbacks=callbacks)

        return model




    leadid = getItemDict(df['LeadID'])
    agentid = getItemDict(df['ContaMediaAccount'])

    StoryLeadCode = getItemDict(df['StoryLead'])
    StoryAgentCode = getItemDict(df['StoryAgent'])

    df['StoryLead-Code'] = df['StoryLead'].apply(lambda x: getCodeorNone(x, StoryLeadCode))
    df['StoryAgent-Code'] = df['StoryAgent'].apply(lambda x: getCodeorNone(x, StoryAgentCode))


    leadid = getItemDict(df['LeadID'])
    agentid = getItemDict(df['ContaMediaAccount'])



    udf = pd.get_dummies(df['LeadID'], prefix='f1')
    udf = pd.get_dummies(df['ContaMediaAccount'], prefix='f2')
    udf = pd.get_dummies(df['StoryLead-Code'], prefix='f3')
    udf = pd.get_dummies(df['StoryAgent-Code'], prefix='f4')
    udf = pd.get_dummies(df['SemDistHamming'], prefix='f5')
    udf = pd.get_dummies(df['SemDistCorrel'], prefix='f6')
    udf = pd.get_dummies(df['DistVIPHamming'], prefix='f7')
    udf = pd.get_dummies(df['WeightSem'], prefix='f8')

    X_test = np.asarray(udf)
    #print(X_test)
    preds = model1.predict_proba(X_test)
    score_eval = pd.DataFrame({'topN': range(1, 40)})
    #score_eval['error'] = score_eval['topN'].apply(lambda x: getScore(preds, y_test, x))
    #score_eval.plot(x='topN', y='error', kind='scatter')
    #plt.ylim(0, 1)

    # df_leads = pd.read_csv( TEMP_STR_LEAD, dtype=str)
    #df_agents = pd.read_csv(TEMP_STR_AGENT, dtype=str)






def semantic_logic(inputfileagent, inputfilelead, outputfile):
    # INPUT LEADS
    #Open Leads
    bunch = pd.read_csv(inputfilelead,dtype=str)
    df = bunch
    #print(df.head())

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
    numeric_dataset_file['UserLatitude']=bunch['latitude']
    numeric_dataset_file['UserLongitude']=bunch['longitude']
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
    df['agent_location_group'] = df['agent_location_group'].map(agent_location_group_mapping)

    numeric_dataset = df

    numeric_dataset_file= pd.DataFrame(numeric_dataset)
    #numeric_dataset_file['TYPE']=bunch['TYPE']
    numeric_dataset_file['userID']=bunch['userID']
    numeric_dataset_file['latitude']=bunch['latitude']
    numeric_dataset_file['longitude']=bunch['longitude']
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
        outframe= pd.DataFrame(columns=[ 'LeadID', 'ContaMediaAccount', 'Distance', 'SemDistCosine', 'SemDistHamming','SemDistCorrel','DistVIPHamming','WeightSem','Final','LeadLong','LeadLat','AgentLong','AgentLat','StoryLead','StoryAgent','VIPLeadStory','VIPAgentStory', 'MLDecision' ])
        final_decision_frame= pd.DataFrame(columns=[ 'LeadID', 'ContaMediaAccount', 'Distance', 'SemDistCosine', 'SemDistHamming','SemDistCorrel','DistVIPHamming','WeightSem','Final','StarRating','LeadLong','LeadLat','AgentLong','AgentLat','StoryLead','StoryAgent','VIPLeadStory','VIPAgentStory', 'MLDecision' ])

        outframe_res=outframe

        dataSetI_distance=df_leads[['latitude','longitude']]
        dataSetI_VIPLEAD = df_leads[['vehicle_life_group','vehicle_life_group','vehicle_life_group','vehicle_life_group']]
        dataSetII_distance=df_agents[['latitude','longitude']]
        dataSetII_ALPHAAG = df_agents[['agent_experience', 'agent_location_group', 'alpha', 'Exclusive']]

        for i in range (0,len(Subdf_lead.iloc[:,0])):
            dataSetI = np.array((Subdf_lead.iloc[i,:]))
            dataSetI_d =np.array(dataSetI_distance.iloc[i,0:2])

            dataSetI_u = Subdf_lead_user.iloc[i,0:1]
            dataSetI_VIPLEAD_action=np.array(dataSetI_VIPLEAD.iloc[i,0:4])


            for j in range(0,len(Subdf_ag.iloc[:,0])):
                dataSetII = np.array((Subdf_ag.iloc[j,:]))
                dataSetII_d =np.array(dataSetII_distance.iloc[j,0:2])
                dataSetII_u = Subdf_lead_ag.iloc[j,0:1]
                dataSetII_ALPHAAG_action=np.array( dataSetII_ALPHAAG.iloc[j,0:4])

                dataSetI = [float(i) for i in dataSetI]
                dataSetII = [float(i) for i in dataSetII]

                dataSetI_d = [float(i) for i in dataSetI_d]
                dataSetII_d = [float(i) for i in dataSetII_d]

                dataSetI_u = [int(i) for i in dataSetI_u]
                dataSetII_u = [int(i) for i in dataSetII_u]



                geo_sim = geo_distance(np.array(dataSetI_d), np.array(dataSetII_d))
                if geo_sim < DISTANCE_THRESHOLD :

                    #SEMANTIC COUSINE
                    sem_cousine_sim = 1 - spatial.distance.cosine(dataSetI, dataSetII)
                    if np.isnan(sem_cousine_sim) :
                        sem_cousine_sim=0
                    #print(dataSetI, dataSetII) , Nan bug
                    #print('sim')
                    #print(sem_cousine_sim)

                    #SEMANTIC HAMMING
                    sem_hamming= 0.1+1 - spatial.distance.hamming(dataSetI, dataSetII)
                    #GEO


                    #CANBERRA
                    canberra_sim=  1/(spatial.distance.canberra(dataSetI, dataSetII)+0.001)

                    #WEIGHTED SEMANTIC

                    weighted_sim=cmp_all(dataSetI, dataSetII)
                    if weighted_sim<0.5:
                        weighted_sim=0  #dropout in case of dismatches gender, age, lang

                    # Machine Learning Decision


                    #VIP HAMMING
                    viphamming_sim= 0.1+ 1 - spatial.distance.hamming(dataSetII_ALPHAAG_action, dataSetI_VIPLEAD_action)

                    final=(DIST_TOP_HIGH-geo_sim)*sem_hamming*math.sqrt(weighted_sim)*(0.2* sem_cousine_sim+0.3*sem_hamming+0.1*viphamming_sim+0.3*canberra_sim+weighted_sim*0.3)

                    if np.isnan(final) :
                        final=0

                    # %Convert to 10 star
                    #starrate =int(final/10)
                    starrate =max(min(int(final/10),10),0)

                    # In Progress ML Logic
                    #result_ML = MLDecision(int(np.array(dataSetI_u)),int(np.array(dataSetII_u)),final,geo_sim, sem_cousine_sim,sem_hamming,canberra_sim, weighted_sim,  viphamming_sim,  dataSetI, dataSetII)

                    #print('Lead id:',np.array(dataSetI_u),'ContaMediaAccount:',np.array(dataSetII_u),'Distance:', result2, 'SemDistCosine:',result, 'SemDistHamming:',result3,'SemDistCorrel:',result5,'DistVIPHamming:',result4,'WeightSem:',weighted_sim,'Final:',final, 'Final2:',final2,'LeadLong:', np.array(dataSetI_d),  'LeadLat:','AgentLong:', 'AgentLat:',np.array(dataSetII_d)  )
                    if geo_sim <DISTANCE_THRESHOLD and final>0:


                        #if i % 10 ==1 and j % 20 ==1 :
                        #    print('Lead id:',np.array(dataSetI_u),'ContaMediaAccount:',np.array(dataSetII_u),'Distance:', geo_sim, 'Final:',final, 'StarRating:',final2 )


                        result_ML =0
                        outframe = outframe.append(
                            {'LeadID': int(np.array(dataSetI_u)), 'ContaMediaAccount': int(np.array(dataSetII_u)),
                             'Distance': geo_sim, 'SemDistCosine': sem_cousine_sim, 'SemDistHamming': sem_hamming,
                             'SemDistCorrel': canberra_sim, 'DistVIPHamming': viphamming_sim, 'WeightSem': weighted_sim,
                             'Final': final, 'StarRating': starrate,  'LeadLat': np.array(dataSetI_d[0]),
                             'LeadLong': np.array(dataSetI_d[1]), 'AgentLat': np.array(dataSetII_d[0]),
                             'AgentLong': np.array(dataSetII_d[1]), 'StoryLead': dataSetI, 'StoryAgent': dataSetII,
                             'VIPLeadStory': dataSetI_VIPLEAD_action, 'VIPAgentStory': dataSetII_ALPHAAG_action,
                             'MLDecision': result_ML}, ignore_index=True)



                        #result_ML = MLDecision(outframe)
                        #outframe = outframe.append({'LeadID' :int(np.array(dataSetI_u)),'ContaMediaAccount': int(np.array(dataSetII_u)), 'Distance':result2, 'SemDistCosine':result, 'SemDistHamming':result3,'SemDistCorrel':result5, 'DistVIPHamming':result4,'Final':final,'Final2':final2,'LeadLat':np.array(dataSetI_d[0]),'LeadLong':np.array(dataSetI_d[1]),'AgentLat':np.array(dataSetII_d[0]),'AgentLong':np.array(dataSetII_d[1])  }, ignore_index=True)

        #FILTRATION TOP CANDIDATES
        outframe2=outframe
        resultfr=outframe.sort_values(by=['Final'],ascending=False).head(TOP_NUMBER_SEARCH*TOP_NUMBER_ALL_LEADS)
        itemvalues = list(resultfr['LeadID'].unique())

        for k in itemvalues:

            datafr=resultfr.loc[resultfr['LeadID'] == k]
            res=datafr.sort_values(by=['Final'],ascending=False).head(3)

            '''
            for t in res:
                outframe = outframe.append(
                    {'LeadID': res'LeadID'[], 'ContaMediaAccount': int(np.array(dataSetII_u)),
                     'Distance': geo_sim, 'SemDistCosine': sem_cousine_sim, 'SemDistHamming': sem_hamming,
                     'SemDistCorrel': canberra_sim, 'DistVIPHamming': viphamming_sim, 'WeightSem': weighted_sim,
                     'Final': final, 'LeadLat': np.array(dataSetI_d[0]),
                     'LeadLong': np.array(dataSetI_d[1]), 'AgentLat': np.array(dataSetII_d[0]),
                     'AgentLong': np.array(dataSetII_d[1]), 'StoryLead': dataSetI, 'StoryAgent': dataSetII,
                     'VIPLeadStory': dataSetI_VIPLEAD_action, 'VIPAgentStory': dataSetII_ALPHAAG_action,
                     'MLDecision': result_ML}, ignore_index=True)
            '''
            final_decision_frame=pd.concat([final_decision_frame, res])


        resultfr3=final_decision_frame.sort_values(by=['LeadID'],ascending=True)





        print('Saved')

        resultfr3.to_csv(outputfile, header=True, index=True)
        resultfr3.to_csv(ml_inputfile, header=True, index=True)

        #resultfr4.to_csv('out/out2.csv', header=True, index=True)




def getScore(preds, y_test, topNvalue=10):
    labels = np.argmax(y_test, 1)
    vals = np.argpartition(preds, -topNvalue)[:, -topNvalue:]
    score = float(vals.size - np.count_nonzero((vals.transpose() - labels).transpose())) / len(labels)

    return score


def machine_learning_logic(inputfileagent, inputfilelead, outputfile):
    # MACHINE LEARNING LOGIC FOR SUPERVISED LEARNING ADJUSTMENT BASED ON TARGET COLUMNS AS FUNCTION FROM FEATURES, RULES
    # APPLIED FOR LEADS

    df_agents = pd.read_csv(TEMP_STR_AGENT, dtype=str)
    df_decisions = pd.read_csv(ml_inputfile, dtype=str)

    print(df_decisions.head())

    df = df_decisions

    df = df.fillna('None')

    df = df.groupby('HumanSupervisionSTAR').filter(lambda x: len(x) >= 5)

    def seqtodict(sequence, start=0):
        # transform a sequence to a set of pairs to make into a dictionary
        n = start
        for elem in sequence:
            yield elem, n
            n += 1

    def getItemDict(itemvalues):
        itemvalues = list(itemvalues.unique())
        itemdict = {}
        if 'None' not in itemvalues:
            itemvalues = ['None'] + itemvalues
        itemdict = dict(seqtodict(itemvalues))
        return itemdict


    StoryLeadCode = getItemDict(df['StoryLead'])
    StoryAgentCode = getItemDict(df['StoryAgent'])


    leadid = getItemDict(df['LeadID'])
    agentid = getItemDict(df['ContaMediaAccount'])


    df['StoryLead-Code'] = df['StoryLead']
    df['StoryAgent-Code'] = df['StoryAgent']

    df['StoryLead-Code'] = df['StoryLead'].apply(lambda x: getCodeorNone(x, StoryLeadCode))
    df['StoryAgent-Code'] = df['StoryAgent'].apply(lambda x: getCodeorNone(x, StoryAgentCode))

    #Target_Criteria = (df['Final'])
    Target_Criteria = (df['HumanSupervisionSTAR'])
    itemdict = getItemDict(df['HumanSupervisionSTAR'])



    Target_Criteria_code = [float(i) for i in Target_Criteria]
    #Target_Criteria_code = [1 + int(min(float(i), 300) / 20) for i in Target_Criteria]
    Target_Criteria_code = [1 + int(float(i)*1) for i in Target_Criteria]


    Target_Criteria = pd.DataFrame(Target_Criteria_code)




    # target code

    df['target_code'] = Target_Criteria
    print(df['target_code'])
    df['target_code'].apply(lambda x: getCodeorNone(x,itemdict))
    print('CORRECTED')
    print(df['target_code'])

    #df=pd.DataFrame(df).fillna()
    df.fillna(df.mean(), inplace=True)

    train, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df['target_code'])
    df.head()

    featurecolumns = ['LeadID', 'ContaMediaAccount', 'Final', 'SemDistCorrel', 'DistVIPHamming', 'WeightSem', 'SemDistHamming']

    #'StoryLead-Code', 'StoryAgent-Code' , , ,,
    features_train = train[featurecolumns].values
    labels_train = train['target_code'].values
    features_test = test[featurecolumns].values
    labels_test = test['target_code'].values


    print(len(features_train))
    dtrain = xgb.DMatrix(features_train, label=labels_train)
    dtest = xgb.DMatrix(features_test, label=labels_test)

    # user defined evaluation function, return a pair metric_name, result
    # This function takes the top 10 predictions and checks to see if the target label is in that set.
    # The error is 1 - the fraction of rows where the label is in the top 10.
    def evalerror(preds, dtrain, topNvalue=10):
        labels = dtrain.get_label()
        vals = np.argpartition(preds, -topNvalue)[:, -topNvalue:]
        error = 1 - float(vals.size - np.count_nonzero((vals.transpose() - labels).transpose())) / len(labels)
        # return a pair metric_name, result
        return 'error', error

    num_round = 500
    print('labels')
    print (len(np.unique(labels_train)) + 1)
    print (labels_train)
    param = {'booster': 'dart',
             'max_depth': 1000, 'learning_rate': 0.0025,
             'objective': 'multi:softprob',
             'sample_type': 'uniform',
             'normalize_type': 'tree',
             'num_class': len(np.unique(labels_train)) + 1 ,
             'rate_drop': 0.05,
             'skip_drop': 0.25}

    '''

    param = {'max_depth': 1000,
        'eta': 0.2,
        'silent': 0,
        'gamma':2,
        'objective':'multi:softprob',
        'num_class':len(np.unique(labels_train))+1,
        'seed':32}
'''
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist, feval=evalerror, early_stopping_rounds=2000)
    pickle.dump(bst, open("xgb-python.model", "wb"))


def machine_learning_logic_keras(inputfileagent, inputfilelead, outputfile) :
    #MACHINE LEARNING LOGIC FOR SUPERVISED LEARNING ADJUSTMENT BASED ON TARGET COLUMNS AS FUNCTION FROM FEATURES, RULES
    #APPLIED FOR LEADS


    #df_leads = pd.read_csv( TEMP_STR_LEAD, dtype=str)
    #df_agents = pd.read_csv(TEMP_STR_AGENT, dtype=str)
    df_decisions = pd.read_csv(ml_inputfile, dtype=str)

    print (df_decisions.head())



    #df=df_leads
    df =df_decisions



    df = df.fillna('None')
    #df = df.groupby('target_id').filter(lambda x: len(x) >= 5)


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

    def getModel(n, num_epochs=Epochs):
        model = Sequential()
        model.add(Dense(n, activation='relu', input_dim=len(feature_columns)))
        model.add(Dropout(0.5))
        model.add(Dense(int(n / 2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nitems, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        callbacks = [EarlyStopping(monitor='acc', patience=10, verbose=0),
                     TQDMNotebookCallback()]
        model.fit(X_train, y_train,
                  epochs=num_epochs,
                  batch_size=128, verbose=1,
                  callbacks=callbacks)

        return model

    #itemdict = getItemDict(df['target_id'])

    StoryLeadCode=getItemDict(df['StoryLead'])
    StoryAgentCode=getItemDict(df['StoryAgent'])

    print(StoryLeadCode)
    print(StoryAgentCode)
    leadid = getItemDict(df['LeadID'])
    agentid= getItemDict(df['ContaMediaAccount'])

    print('new**********************88')
    df['StoryLead-Code'] = df['StoryLead']
    df['StoryAgent-Code'] = df['StoryAgent']

    df['StoryLead-Code'] = df['StoryLead'].apply(lambda x: getCodeorNone(x,StoryLeadCode))
    df['StoryAgent-Code'] = df['StoryAgent'].apply(lambda x: getCodeorNone(x, StoryAgentCode))
    print(StoryLeadCode)
    print(StoryAgentCode)

    #Target_Criteria=(df['Final'])
    Target_Criteria = (df['HumanSupervisionSTAR'])
    #Target_Criteria_code = getItemDict(df['Final'])
    #Target_Criteria_code = [float(i) for i in  Target_Criteria]()
    #Target_Criteria_code = [1+int(min(float(i), 300) /20) for i in Target_Criteria]
    Target_Criteria_code = [1 + int(min(float(i), 10)/3 ) for i in Target_Criteria]
    #Target_Criteria_code = [(1 + int(i) / 3) for i in Target_Criteria]

    #print('target')
    #print((np.array(Target_Criteria_code)))
    Target_Criteria=pd.DataFrame(Target_Criteria_code)
    Target_Criteria.to_csv('temp/ml_test.csv', header=True, index=True)







    # target code
    df['target_code'] = Target_Criteria
    itemdict = getItemDict(df['target_code'])
        #df['target_id'].apply(lambda x: getCodeorNone(x,itemdict))

    train, test = train_test_split(df, test_size=0.2, random_state=123,stratify=df['target_code'])
    df.head()


    featurecolumns = ['LeadID', 'ContaMediaAccount','StoryLead-Code', 'StoryAgent-Code',  'SemDistHamming', 'SemDistCorrel', 'DistVIPHamming', 'WeightSem']

    nitems = len(itemdict)

    features_train = train[featurecolumns].values
    labels_train = train['target_code'].values
    features_test = test[featurecolumns].values
    labels_test = test['target_code'].values


    udf = pd.get_dummies(df['LeadID'], prefix='f1')
    udf = pd.get_dummies(df['ContaMediaAccount'], prefix='f2')
    udf = pd.get_dummies(df['StoryLead-Code'], prefix='f3')
    udf = pd.get_dummies(df['StoryAgent-Code'], prefix='f4')
    udf = pd.get_dummies(df['SemDistHamming'], prefix='f5')
    udf = pd.get_dummies(df['SemDistCorrel'], prefix='f6')
    udf = pd.get_dummies(df['DistVIPHamming'], prefix='f7')
    udf = pd.get_dummies(df['WeightSem'], prefix='f8')



    # feature 7 is the 2nd previous user click, so we weight it by 0.5
    #udf = udf.join(
    #    pd.get_dummies(df['feature7-code'], prefix='pre') * 0.3 + pd.get_dummies(df['feature8-code'], prefix='pre'))
    feature_columns = udf.columns
    #feature_columns=featurecolumns
    # udf.head()




    one_hot_labels = to_categorical(df['target_code'], num_classes=nitems)

    # target code
    df['target_code'] = Target_Criteria
    # df['target_id'].apply(lambda x: getCodeorNone(x,itemdict))

    #X_train, X_test = train_test_split(df, test_size=0.2, random_state=123, stratify=df['target_code'])

    X_train, X_test, y_train, y_test = train_test_split(udf, one_hot_labels, test_size=0.2, random_state=23)



    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    layer1 = [256, 512, 1024, 2048]
    scores = np.zeros(len(layer1))
    print('start')

    for n in range(len(layer1)):
        model = getModel(layer1[n])
        print('X_test',n)
        print(X_test)
        preds = model.predict_proba(X_test)
        scores[n] = getScore(preds, y_test, Num)
        print(scores[n] )
    import matplotlib.pyplot as plt


    print(scores)
    plt.scatter(layer1, scores)
    plt.xlabel('Layer 1 Size')
    plt.ylabel('Top 10 Score')

    model1 = getModel(512, 300)
    preds = model1.predict_proba(X_test)
    score = getScore(preds, y_test, Num)
    display(HTML('<b> Score: {}</b>'.format(score)))

    from keras.models import load_model

    model1.save('model512.h5')  # creates a HDF5 file 'my_model.h5'
    del model1  # deletes the existing model

    Nmax=Num
    # returns a compiled model
    # identical to the previous one
    model1 = load_model('model512.h5')
    preds = model1.predict_proba(X_test)
    score_eval = pd.DataFrame({'topN': range(1, Nmax)})
    score_eval['error'] = score_eval['topN'].apply(lambda x: getScore(preds, y_test, x))
    score_eval.plot(x='topN', y='error', kind='scatter')
    plt.ylim(0, 1)




# user defined evaluation function, return a pair metric_name, result
    # This function takes the top 10 predictions and checks to see if the target label is in that set.
    # The error is 1 - the fraction of rows where the label is in the top 10.
    def evalerror(preds, dtrain,topNvalue=Num):
        labels = dtrain.get_label()
        vals = np.argpartition(preds,-topNvalue)[:,-topNvalue:]
        error = 1 - float(vals.size - np.count_nonzero((vals.transpose() - labels).transpose()))/len(labels)
        # return a pair metric_name, result
        return 'error', error













def connect_db():

    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    client = pymongo.MongoClient("mongodb+srv://towert5:towert5@cluster0-intti.mongodb.net/test?retryWrites=true&w=majority")

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

def main(argv):
   inputfileagent = sys.argv[1]
   inputfilelead = sys.argv[2]
   outputfile = sys.argv[3]



   print ('Input file 1 is "', inputfileagent )
   print('Input file 2 is "', inputfilelead)
   print ('Output file is "', outputfile)

   #connect_db()


   print('SEMANTIC Logic')


   #PRIMARY VERSION
   semantic_logic(inputfileagent, inputfilelead, outputfile)
   #print('ML Logic')

   # WORKING TRAINING VERSION XGBOOST, CHECK PROPER LABELS, WORKED ON 20 LABELS
   #xgboost version training
   #WORKING WITH
   #machine_learning_logic (inputfileagent, inputfilelead, outputfile)


   # tensorflow version training WORKING 
   #machine_learning_logic_keras(inputfileagent, inputfilelead, outputfile)



if __name__ == "__main__":
   main(sys.argv[1:])

