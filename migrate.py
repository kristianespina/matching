"""
    Performs Database Migration (CSV to MongoDB)
    Migrates CSV files to MongoDB Server

    Author: Kristian Espina (onelespina@gmail.com)
"""
import pymongo
import pandas as pd
import json

CONN_STRING = "mongodb://test:test123@ds121203.mlab.com:21203/geolocation?retryWrites=false"
LEADS_FILENAME = "input/leads_08.csv"
AGENTS_FILENAME = "input/agents_v2.csv"

LEADS_COLLECTION_NAME = "new_leads"
AGENTS_COLLECTION_NAME = "new_agents"


client = pymongo.MongoClient(CONN_STRING)
db = client.geolocation
agents = db[AGENTS_COLLECTION_NAME]
leads = db[LEADS_COLLECTION_NAME]

df_agents = pd.read_csv(AGENTS_FILENAME)
df_agents = df_agents.rename(columns={"alpha.1": "alpha1"})
agents.drop()
agents.insert_many(df_agents.to_dict(orient='records'))

df_leads = pd.read_csv(LEADS_FILENAME)
df_leads = df_leads.rename(columns={
    "Unnamed: 18": "unnamed18",
    "Unnamed: 19": "unnamed19",
})
leads.drop()
leads.insert_many(df_leads.to_dict(orient='records'))

df_leads = pd.DataFrame.from_dict(list(leads.find()))
df_agents = pd.DataFrame.from_dict(list(agents.find()))