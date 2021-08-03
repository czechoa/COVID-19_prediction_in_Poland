# from __future__ import print_function
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import numpy as np
def get_spreadsheets_covid_situation_in_region_as_list():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SERVICE_ACCOUNT_FILE = 'prepareData/key.json'

    creds = None
    creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # If modifying these scopes, delete the file token.json.


    # The ID and range of a sample spreadsheet.
    # SAMPLE_SPREADSHEET_ID = '1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E'
    SAMPLE_SPREADSHEET_ID = '1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E'
    SAMPLE_RANGE_NAME = 'Sytuacja epidemiczna w województwach!A2:EG259'

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.


    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])
    return values

def get_spreadsheets_covid_situation_in_region_as_df():
    values = get_spreadsheets_covid_situation_in_region_as_list()
    # values = np.array(values)
    df = pd.DataFrame(data=values)
    df = df.rename(columns=df.iloc[0]).iloc[1:, :]
    return df


