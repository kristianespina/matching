import pandas as pd
from pymongo import MongoClient

from mlmodels import semantic_logic

#mongodb+srv://kristian:kristiancjcube@cluster0-intti.mongodb.net/test?retryWrites=true&w=majority
MONGO_URL = "mongodb://test:test123@ds121203.mlab.com:21203/geolocation?retryWrites=false"

class Mongo:

    def __init__(self, connection_string, database_name):
        # Connect to MongoDB Instance
        print("Connecting to MongoDB Instance")
        try:
            client = MongoClient(connection_string)
            self.database = client[database_name] # Database
            print("Successfully connected to MongoDB")
        except:
            raise Exception("Unable to connect to MongoDB")

    def read(self, collection_name):
        """Parses a collection from mongo database and returns a Pandas DataFrame
        
        Arguments:
            collection_name {string} -- name of the collection in the database. Analogous to table in SQL
        
        Returns:
            [pd.DataFrame] -- Pandas DataFrame
        """
        return pd.DataFrame.from_dict(list(self.database[collection_name].find()))

def clean_dtypes(df):
    """Cleans the dataframe column types
    
    Arguments:
        df {pd.DataFrame} -- Messy DataFrame
    
    Returns:
        pd.DataFrame -- Clean DataFrame
    """
    df['AgentLat'] = df['AgentLat'].astype(float)
    df['AgentLong'] = df['AgentLong'].astype(float)
    df['ContaMediaAccount'] = df['ContaMediaAccount'].astype(int)
    df['DistVIPHamming'] = df['DistVIPHamming'].astype(float)
    df['Distance'] = df['Distance'].astype(float)
    df['Final'] = df['Final'].astype(float)
    df['LeadID'] = df['LeadID'].astype(int)
    df['LeadLat'] = df['LeadLat'].astype(float)
    df['LeadLong'] = df['LeadLong'].astype(float)
    df['MLDecision'] = df['MLDecision'].astype(float)
    df['SemDistCorrel'] = df['SemDistCorrel'].astype(float)
    df['SemDistCosine'] = df['SemDistCosine'].astype(float)
    df['SemDistHamming'] = df['SemDistHamming'].astype(float)
    df['StarRating'] = df['StarRating'].astype(float)
    df['StoryAgent'] = df['StoryAgent'].tolist()
    df['StoryLead'] = df['StoryLead'].tolist()
    df['VIPAgentStory'] = df['VIPAgentStory'].tolist()
    df['VIPLeadStory'] = df['VIPLeadStory'].tolist()
    df['VIPAgentStory'] = df['VIPAgentStory'].astype(str)
    df['VIPLeadStory'] = df['VIPLeadStory'].astype(str)
    df['WeightSem'] = df['WeightSem'].astype(float)

    return df

def main():
    db = Mongo(MONGO_URL, 'geolocation')
    leads = db.read('leads')
    agents = db.read('agents')

    print("Running Semantic Logic...")
    df = semantic_logic(agents, leads)
    df = clean_dtypes(df)
    output = db.database.output
    output.drop()
    output.insert_many(df.to_dict(orient='records'))

    return True

if __name__ == "__main__":
    main()
