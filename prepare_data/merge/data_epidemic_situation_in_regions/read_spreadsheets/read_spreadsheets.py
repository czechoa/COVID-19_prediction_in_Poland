from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd


def get_spreadsheets_covid_situation_in_region_as_list():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SERVICE_ACCOUNT_FILE = 'prepare_data/merge/data_epidemic_situation_in_regions/read_spreadsheets/key.json'

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    SAMPLE_SPREADSHEET_ID = '1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E'
    SAMPLE_RANGE_NAME = 'Sytuacja epidemiczna w województwach!A2:EG259'

    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])
    return values


def get_spreadsheets_covid_situation_in_region_as_df():
    values = get_spreadsheets_covid_situation_in_region_as_list()

    df = pd.DataFrame(data=values)
    df = df.rename(columns=df.iloc[0]).iloc[1:, :]
    return df
