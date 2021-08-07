import urllib.request
import zipfile
import  pandas as pd
from io import BytesIO

url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
filehandle, _ = urllib.request.urlretrieve(url)
zip_file_object = zipfile.ZipFile(filehandle, 'r')
w = [i for i in zip_file_object.namelist() if '2021_PL' in str(i)]
Poland_file = zip_file_object.namelist()[zip_file_object.namelist().index(w[0])]

file = zip_file_object.open(Poland_file)
content = file.read()
df = pd.read_csv(BytesIO(content))
# df_mobility = pd.read_csv(file)
