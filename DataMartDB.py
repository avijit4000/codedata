import psycopg2
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import requests
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host="itsm_dm_prod.optum.com",
    database="itsm_dm",
    user= os.getenv('DATA_MART_DB_USER'),
    password= os.getenv('DATA_MART_DB_PASSWORD'),
    port="5432"
)

cur = conn.cursor()

def format_date(date):
    return date.strftime('%m/%d/%Y')

def generate_monthly_date_ranges():
    today_date_time = datetime.today()
    today = today_date_time.date()

    this_month_start_date_time = today_date_time.replace(day=1)
    this_month_start = this_month_start_date_time.date()

    date_ranges = [
            (this_month_start, today)
    ]
    for i in range(36):
        i=i+1
        start_date_of_prev_month = this_month_start - relativedelta(months=i)
        end_date_of_prev_month = start_date_of_prev_month + relativedelta(months=1) - timedelta(days=1)
        date_ranges.append((start_date_of_prev_month, end_date_of_prev_month))

    return date_ranges

#function to generate start day and end day of the previous month
def generate_prev_month_date_ranges():
    today = datetime.today()
    last_month = today - relativedelta(months=1)
    start_date = datetime(last_month.year, last_month.month, 1).date()
    end_date = datetime(today.year, today.month, 1).date() - timedelta(days=1)
    return [(start_date, end_date)]

def generate_YTD_date_ranges():
    today = datetime.today()

    last_3_years = []

    for i in range(3):
        year = today.year - i
        today_day = today.day
        today_month = today.month
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, today_month, today_day).date()
        last_3_years.append((start_date, end_date))

    return last_3_years

def generate_yearly_date_ranges():
    today = datetime.today()
    last_3_years = []

    for i in range(3):
        year = today.year - (i+1)
        start_date = datetime(year, 1, 1).date()
        # print(start_date)
        end_date = datetime(year, 12, 31).date()
        # print(end_date)
        last_3_years.append((start_date, end_date))

    return last_3_years


def generate_sql(start_date, end_date):
    sql = """WITH MAIN_QUERY AS (
        SELECT DISTINCT
            I.IN_ID,
            I.PROBLEM_STATUS,
            I.PRIORITY_CODE,
            I.OPEN_TIME as "Incident Open Time",
            I.CLOSE_TIME as "Incident Close Time",
            I.SN_PROBLEM_ID,
            I.BRIEF_DESCRIPTION AS "Incident Description",
            I.SN_FAILED_SERVICE AS "Failed Service",
            I.ASSIGNMENT "Incident Assignment",
            I.SN_WG_SYS_ID,
            ssc.SM_SCT_SRV_NAME,
            ssc.SM_SCT_ROLLUP_NAME AS "Service Grouping",
            IA.IA_NUMBER,
            IA.IA_CLOSED_AT,
            IA.IA_RESOLVED_AT,
            IA.IA_OPENED_AT,
            IA.IA_WAR_ROOM_SOURCE,
            IA.IA_MANAGING_TEAM,
            IA.IA_START_TIME,
            IA.IA_MEAN_TIME_TO_IDENTIFY,
            IA.IA_PIR_FIRST_IND_TIME,
            P.BRIEF_DESCRIPTION AS "Problem Description",
            P.OPEN_TIME AS "Problem Open Time",
            P.CLOSE_TIME AS "Problem Close Time",
            P.ASSIGNMENT,
            sls.SUPPORTING_LOB,
            sls.IT_SUPPORTING_SUB_LOB,
            sls.IT_SUPPORTING_LOB
        FROM SM_DM.SN_IA_INCIDENT_ALERT IA
        JOIN SM_DM.SM_INCIDENTS_v2 I ON I.IN_ID=IA.IA_SOURCE_INCIDENT
        LEFT JOIN SM_DM.sm_incidents_text_cols T ON I.IN_ID = T.IN_ID
        LEFT JOIN SM_DM.SM_PROBLEMS P ON I.SN_PROBLEM_ID = P.PR_ID
        LEFT JOIN SM_DM.SM_SERVICE_CATEGORY ssc ON ssc.SM_SCT_LOGICAL_NAME = I.SN_FAILED_SERVICE
        LEFT JOIN SM_DM.SM_LOB_SUPPORTING sls ON sls.LOGICAL_NAME = I.SN_FAILED_SERVICE
    ),
    CBA_FLAG AS (
        SELECT LOGICAL_NAME, UH_CBA_SERVICE_FLAG AS CBA_FLAG
        FROM SM_DM.SM_DEVICE sd
        WHERE TYPE = 'Service'
    ),
    FIRST_PAGE_SENT AS (
        SELECT IAP_SOURCE, MIN(SYS_CREATED_ON) AS FIRST_Page_Sent
        FROM SM_DM.SN_IA_WEBEX_PARTICIPANT siwp
        GROUP BY IAP_SOURCE
    )
    SELECT
        AVG(MEAN_TO_DETECT) as MTD,
        AVG(MEAN_TO_ENGAGE) as MTE,
        AVG(MEAN_TO_IDENTIFY) as MTI,
        AVG(MEAN_TO_RESOLVE) as MTR,
        AVG(MEAN_TO_WARROOM) as MTW
    FROM (
        SELECT
            IN_ID,
            AVG(IA_PIR_FIRST_IND_TIME - IA_START_TIME) MEAN_TO_DETECT,
            AVG(FIRST_Page_Sent - IA_START_TIME) MEAN_to_engage,
            AVG(FIRST_Page_Sent - IA_MEAN_TIME_TO_IDENTIFY) MEAN_TO_IDENTIFY,
            AVG(IA_START_TIME - IA_RESOLVED_AT) MEAN_TO_RESOLVE,
            AVG(IA_PIR_FIRST_IND_TIME - IA_OPENED_AT) MEAN_TO_WARROOM
        FROM (
            SELECT *
            FROM MAIN_QUERY Q
            LEFT JOIN CBA_FLAG F ON Q."Failed Service" = F.LOGICAL_NAME
            LEFT JOIN FIRST_PAGE_SENT P ON Q.IA_NUMBER = P.IAP_SOURCE
            WHERE Q."Incident Assignment" = 'GOVERNMENT PROGRAMS DIGITAL (UNT) - APP'
            AND Q.PRIORITY_CODE in ('1', '2')
            AND Q."Incident Open Time" between """ + start_date + """ and """ + end_date + """
        )ZZZ
        GROUP BY IN_ID
    )XX"""
    return sql

def format_timedetlta_obj(time_delta_obj):
    if time_delta_obj == None or time_delta_obj == 0:
        total_minutes = 0
        return str(total_minutes)


    total_seconds = time_delta_obj.total_seconds()
    total_minutes = total_seconds / 60
    total_minutes += time_delta_obj.microseconds / (60 * 10**6)
    return str(total_minutes)


merged_data = []
def run_sql_with_date_ranges(date_ranges, source):
    i = 0
    for date_range in date_ranges:
        # print(date_range)
        start_date = format_date(date_range[0])
        end_date = format_date(date_range[1])
        start_date_string = f"'{start_date}'"
        end_date_string = f"'{end_date}'"
        sql = generate_sql(start_date_string, end_date_string)

        cur.execute(sql)
        results = cur.fetchall()

        for row in results:
            MTTD = format_timedetlta_obj(row[0])
            MTTE = format_timedetlta_obj(row[1])
            MTTI = format_timedetlta_obj(row[2])
            MTTR = format_timedetlta_obj(row[3])
            MTTW = format_timedetlta_obj(row[4])
            if source == "monthly":
                data = {"source": source,"start_date": start_date, "end_date" : end_date ,"MTTD": MTTD, "MTTE": MTTE, "MTTI": MTTI, "MTTR": MTTR, "MTTW": MTTW}
            elif source == "ytd":
                data = {"source": source,"start_date": start_date, "end_date" : end_date ,"MTTD": MTTD, "MTTE": MTTE, "MTTI": MTTI, "MTTR": MTTR, "MTTW": MTTW}
            elif source == "yearly":
                data = {"source": source,"start_date": start_date, "end_date" : end_date ,"MTTD": MTTD, "MTTE": MTTE, "MTTI": MTTI, "MTTR": MTTR, "MTTW": MTTW}
            output_data = json.loads(json.dumps(data))
            merged_data.append(output_data)

today_date = datetime.today()
# print(today_date.day, today_date.month)
if today_date.day == 1:
    date_ranges_monthly = generate_prev_month_date_ranges()
    run_sql_with_date_ranges(date_ranges_monthly, "monthly")

if today_date.month == 1 and today_date.day == 1:
    date_ranges_yearly = generate_yearly_date_ranges()
    run_sql_with_date_ranges(date_ranges_yearly, "yearly")

date_ranges_ytd = generate_YTD_date_ranges()
run_sql_with_date_ranges(date_ranges_ytd, "ytd")

cur.close()
conn.close()

json_data = json.dumps(merged_data)
with open('./monitoring_dashboard/cronjob/DataMartDB.json', 'w') as f:
    f.write(json_data)


# DEV_LOGSTASH_ENDPOINT = "http://rn000124779:8080/generic_export"
PROD_LOGSTASH_ENDPOINT =  "http://rn000124779:8080/generic_export"


def sendToLogstash(report, endpoint):
    try:
        print("Sending to Logstash")
        url = endpoint
        headers = {'Content-type': 'application/json'}
        r = requests.post(url, data=report, headers=headers)

    except Exception as e:
        print("Error occured while sending data to Logstash")


f = open('./monitoring_dashboard/cronjob/DataMartDB.json')
data = json.loads(f.read())


timestamp = datetime.utcnow()

for doc in data:
    doc["_index_tag"] = "gpd_data_mart"
    doc["@doc_id"] = hashlib.sha1((str(doc['source']) + str(doc['start_date'])).encode("utf-8")).hexdigest()

data_to_post = '\n'.join(json.dumps(d) for d in data)

sendToLogstash(data_to_post, PROD_LOGSTASH_ENDPOINT)