import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

team_fullnames = ['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
    'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab', 'Sunrisers Hyderabad',
    'Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
team_shortnames = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW']
team_numbers = {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}
toss_decision_ids={'field':0, 'bat':1}
city_ids = {
    'Hyderabad':1,
    'Pune':2,
    'Rajkot':3,
    'Indore':4,
    'Bangalore':5,
    'Mumbai':6,
    'Kolkata':7,
    'Delhi':8,
    'Chandigarh':9,
    'Kanpur':10,
    'Jaipur':11,
    'Chennai':12,
    'Cape Town':13,
    'Port Elizabeth':14,
    'Durban':15,
    'Centurion':16,
    'East London':17,
    'Johannesburg':18,
    'Kimberley':19,
    'Bloemfontein':20,
    'Ahmedabad':21,
    'Cuttack':22,
    'Nagpur':23,
    'Dharamsala':24,
    'Kochi':25,
    'Visakhapatnam':26,
    'Raipur':27,
    'Ranchi':28,
    'Abu Dhabi':29,
    'Sharjah':30,
    'Dubai':31
}
matches_data = pd.read_csv("G:\CSUF Study\ADBMS\Project\DataSet\IPL\matches.csv")
deliveries_data = pd.read_csv("G:\CSUF Study\ADBMS\Project\DataSet\IPL\deliveries.csv")
#deliveries_data = deliveries_data.drop(columns =['inning'])

matches_data[pd.isnull(matches_data['winner'])]
matches_data['winner'].fillna('Draw', inplace=True)
matches_data[pd.isnull(matches_data['city'])]
matches_data['city'].fillna('Dubai',inplace=True)

matches_data.replace(team_fullnames,team_shortnames,inplace=True)
deliveries_data.replace(team_fullnames,team_shortnames,inplace=True)

winner_dict = team_numbers.copy()
winner_dict['Draw']=14

encode_matchesdata = {'team1': team_numbers,
         'team2': team_numbers,
         'toss_winner': team_numbers,
         'toss_decision':toss_decision_ids,
         'city':city_ids,
         'winner': winner_dict}

encode_deliveriesdata = {'batting_team': team_numbers,'bowling_team': team_numbers}

matches_data.replace(encode_matchesdata, inplace=True)
deliveries_data.replace(encode_deliveriesdata, inplace=True)

data = deliveries_data.groupby(['match_id', 'inning']).sum()

batting1_runs=[]
batting2_runs=[]

s=set()
for i in range(1,matches_data.shape[0]+1):
    t = tuple(data['total_runs'][i])
    batting1_runs.append(t[0])
    if len(t)==1:
        batting2_runs.append(0)
    else:
        batting2_runs.append(t[1])
    s.add(len(t))

matches_data['batting1_runs']=batting1_runs
matches_data['batting2_runs']=batting2_runs
matches_data=matches_data.set_index('id')