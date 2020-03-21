import pandas as pd
import numpy as np
import math
from scipy import spatial



#CONTROL PARAMS MID FILES
TEMP_STR_LEAD='temp/lead_encoder.csv'
TEMP_STR_AGENT='temp/agent_encoder.csv'
ml_inputfile = 'temp/predecisions_manual_humansupervised.csv'

#METIC SCORE
DIST_TOP_HIGH=100
#RIGID PARAM
DISTANCE_THRESHOLD=50

DEB_FLAG=1

#GENERIC
TOP_NUMBER_SEARCH=10
TOP_NUMBER_ALL_LEADS=160

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

def semantic_logic(df_inputagents, df_inputleads):
    #bunch = pd.read_csv(inputfilelead,dtype=str)
    bunch = df_inputleads
    df = df_inputleads
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

    """ Transform Dataset """

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
    #bunch = pd.read_csv(inputfileagent,dtype=str)
    #print ('****agents******')
    #X=bunch[['gender','famstatus','lang','profession','hobby', 'sport','travel_distance', 'age_group','lang_group']]
    #enc = OrdinalEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
    #enc.fit(X)
    #df= enc.transform(X)


    bunch = df_inputagents
    df = df_inputagents

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
                dataSetII_d =np.array(dataSetI_distance.iloc[j,0:2])
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
        
        #resultfr3.to_csv(outputfile, header=True, index=True)
        return resultfr3
        #resultfr3.to_csv(ml_inputfile, header=True, index=True)

        #resultfr4.to_csv('out/out2.csv', header=True, index=True)

"""
def main():
    # Connect to MongoDB Instance
    print("Connecting to MongoDB Instance")
    try:
        mongo_url = "mongodb://test:test123@ds121203.mlab.com:21203/geolocation?retryWrites=false"
        client = pymongo.MongoClient(mongo_url)
        db = client.geolocation
        agents = db.agents
        leads = db.leads
        print("Successfully connected to MongoDB")
    except:
        raise Exception("Unable to connect to MongoDB")

    df_leads = pd.DataFrame.from_dict(list(leads.find()))
    df_agents = pd.DataFrame.from_dict(list(agents.find()))

    print("Running Semantic Logic")
    #PRIMARY VERSION
    semantic_logic(df_agents, df_leads, "test.csv")


if __name__ == "__main__":
    main()
"""