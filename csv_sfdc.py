import json
import csv
from langchain.tools import  tool

@tool
def update_survey_to_salesforce_account(filename):
    """Update Survey response to Accounts in Salesforce"""
# Re-open the file and process each row
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Check the headers again to confirm the email column, for safety
        id_column = None
        for header in reader.fieldnames:
            if "id" in header.lower():
                id_column = header
                break
                
        # Process each row if an email column is found
        if id_column:
            for row in reader:
                try:
                 process_row(id_column,row)
                except:
                    pass
        else:
            print("No id_column  found.")



from langchain.tools import  tool
import json
import pandas as pd
from simple_salesforce import Salesforce



def process_row(id_column,row):
    """
    Process a single row from the CSV file, printing the email id and a JSON representation of the row.
    """
    # Print the email id
    sf_id = row.get(id_column, "No email found")
    print(f"Salesforce ID: {sf_id}")
    
    # Convert the row to a JSON representation and print
    json_representation = json.dumps(row, ensure_ascii=False)
    #print(f"JSON Representation: {json_representation}\n")
  
    update_sfdc_account(sf_id, json_representation)

import os
def update_sfdc_account (salesforce_id,surveyjson):

    
    #loginInfo = json.load(open('login.json'))
    username = os.environ['SF_username'] # loginInfo['username']
    password = os.environ['SF_password'] # loginInfo['password']
    security_token = os.environ['SF_token'] #loginInfo['security_token']
    domain = os.environ['SF_domain'] #loginInfo['domain']
    #print(f'SALESFORCE LOGIN WITH user: {username} pwd:{password} domain {domain} token {security_token}')
    sf = Salesforce(username=username, password=password, security_token=security_token, domain=domain)

    #For updating a ACCOUNT record
    accInfo = {
    'survey_response__c':surveyjson,
    }


    #print (accInfo, salesforce_id)
   #sf.Contact.update('003e0000003GuNXAA0',{'LastName': 'Jones', 'FirstName': 'John'})
    object_update_result=sf.Account.update(salesforce_id, accInfo)

    print(f'Updated Salesforce Account {salesforce_id} status {object_update_result}')

    return object_update_result
