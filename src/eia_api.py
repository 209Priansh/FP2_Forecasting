import requests

def fetch_data():
    api_url = ("https://api.eia.gov/v2/international/data/?api_key=Z9AffCIxlrN77E0AfKdySRz2ayEShmj9DY1ytIVp"
               "&frequency=monthly"
               "&data[0]=value"
               "&facets[activityId][]=1"
               "&facets[productId][]=53"
               "&facets[countryRegionId][]=IND"
               "&facets[unit][]=TBPD"
               "&sort[0][column]=period&sort[0][direction]=desc"
               "&offset=0&length=5000")

    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data['response']['data'])
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None
# Fetch and save data
df = fetch_data()
if df is not None:
    file_path = r'C:\Prashant\ISB\TERM5\FP2\ASSIGN\Code\india_petroleum_data.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Data saved to {file_path}")
