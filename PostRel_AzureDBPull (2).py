import requests
import datetime
from azure.identity import DefaultAzureCredential

def get_azure_metrics():
    try:
        credential = DefaultAzureCredential()
        token = credential.get_token('https://management.azure.com/.default')
        print(token.token + "\n")
        bearer = token.token
        results_dict = {}

        # from 5am CST to now
        start_time = datetime.datetime.now().strftime('%Y-%m-%d') + "T10:00:00Z"
        # end time is now in UTC
        end_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        timespan = f"{start_time}/{end_time}"
        print(timespan)
        aggregation = "maximum"
        api_v = "2023-10-01"
        headers = {"Authorization":f"Bearer {bearer}"}

        metric = "cpu_percent,memory_percent"
        params = f"metricnames={metric}&timespan={timespan}&aggregation={aggregation}&api-version={api_v}"
        try:
            res = requests.get(f"https://management.azure.com/subscriptions/c3ca2369-06db-4f15-a401-532d029a563e/resourceGroups/gpdacq-prod-1-rg/providers/Microsoft.DBforMySQL/flexibleServers/gpdacq-prod-1-mysql/providers/microsoft.insights/metrics?{params}", headers=headers).json()

            for val in res['value']:
                results_dict[val['name']['value']] = -1
                for ts in val['timeseries']:
                    for datum in ts['data']:
                        if datum['maximum'] > results_dict[val['name']['value']]:
                            results_dict[val['name']['value']] = datum['maximum']
            try:
                metric = "NormalizedRUConsumption"
                params = f"metricnames={metric}&timespan={timespan}&aggregation={aggregation}&api-version={api_v}"
                res = requests.get(f"https://management.azure.com/subscriptions/c3ca2369-06db-4f15-a401-532d029a563e/resourceGroups/gpdacq-prod-1-rg/providers/Microsoft.DocumentDB/databaseAccounts/gpdacq-prod/providers/microsoft.insights/metrics?{params}", headers=headers).json()
                for val in res['value']:
                    results_dict[val['name']['value']] = -1
                    for ts in val['timeseries']:
                        for datum in ts['data']:
                            if datum['maximum'] > results_dict[val['name']['value']]:
                                results_dict[val['name']['value']] = datum['maximum']

                print(results_dict)
                return results_dict
            except Exception as e:
                print("Failed to get Cosmos metrics", e)
                return results_dict
        except Exception as e:
            print("Failed to get MySQL metrics", e)
            return None
    except Exception as e:
        print(e)
        return None
    
# get_azure_metrics()