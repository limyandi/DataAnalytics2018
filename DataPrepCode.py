import pandas as pd
import numpy as np
import re

##TODO: Write a function that checks the value counts, if the value counts of the first and second value exceeded 70% of the data, change the rest becomes 'Others'
def data_preprocessing(filename):

    #Data reading.
    df = pd.read_csv(filename, encoding='latin-1', index_col=0)

    ## Remove values with too many null.
    missing_values_treshold = 0.4
    null_columns = df.isnull().mean()
    # Get the columns with missing values more than 40%
    missing_instances = null_columns[null_columns > missing_values_treshold].index

    df.drop(missing_instances, axis='columns', inplace=True)

    ### TODO: DATE COLUMNS TAKES TOO MUCH TIME.
    # ## Date columns ##
    date_columns = []
    date_columns += columns_date_handler(df, r'date$')
    date_columns += columns_date_handler(df, r'start$')
    date_columns += columns_date_handler(df, r'end$')
    date_columns += columns_date_handler(df, r'to$')
    date_columns += columns_date_handler(df, r'from$')

    df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%d/%m/%Y', errors='coerce')

    #TODO: NOT WORKING.
    df[date_columns] = df[date_columns].apply(lambda x: x.dt.year)

    # This one makes the date makes much more important!!! (VERY IMPORTANT, USE THIS AS EXPLANATION FOR THE REPORT).
    # df[date_columns] = df[date_columns].apply(lambda x: x.dt.values.astype(np.int64))

    #Data cleaning
    category_columns = []

    us_state_abbrev = {
        'ALABAMA': 'AL',
        'ALASKA': 'AK',
        'ARIZONA': 'AZ',
        'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA',
        'COLORADO': 'CO',
        'CONNECTICUT': 'CT',
        'DELAWARE': 'DE',
        'FLORIDA': 'FL',
        'GEORGIA': 'GA',
        'HAWAII': 'HI',
        'IDAHO': 'ID',
        'ILLINOIS': 'IL',
        'INDIANA': 'IN',
        'IOWA': 'IA',
        'KANSAS': 'KS',
        'KENTUCKY': 'KY',
        'LOUISIANA': 'LA',
        'MAINE': 'ME',
        'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA',
        'MICHIGAN': 'MI',
        'MINNESOTA': 'MN',
        'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO',
        'MONTANA': 'MT',
        'NEBRASKA': 'NE',
        'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH',
        'NEW JERSEY': 'NJ',
        'NEW MEXICO': 'NM',
        'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC',
        'NORTH DAKOTA': 'ND',
        'OHIO': 'OH',
        'OKLAHOMA': 'OK',
        'OREGON': 'OR',
        'PENNSYLVANIA': 'PA',
        'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD',
        'TENNESSEE': 'TN',
        'TEXAS': 'TX',
        'UTAH': 'UT',
        'VERMONT': 'VT',
        'VIRGINIA': 'VA',
        'WASHINGTON': 'WA',
        'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI',
        'WYOMING': 'WY',
        'DISTRICT OF COLUMBIA': 'DC',
        'GUAM': 'GU',
        'PUERTO RICO': 'PR',
        'VIRGIN ISLANDS': 'VI',
        'NORTHERN MARIANA ISLANDS': 'MP',
        'MARSHALL ISLANDS': 'MH',

    }

    ## Categorical ##
    category_columns = []
    category_columns += df.loc[:, df.dtypes == object].columns.values.tolist()
    numeric_columns = []
    numeric_columns += df.select_dtypes(include=[np.number]).columns.values.tolist()

    state_cols = [col for col in df.columns if 'state' in col]
    df[state_cols] = df[state_cols].replace(us_state_abbrev)

    ##STARTING FROM HERE IS REPLACING DATA INTO COLUMNS THAT WE FOUND THAT ARE PRETTY IMPORTANT. ###
    #STATE
    allowed_vals_agent_state = ['CA', 'NY', 'TX']
    df.loc[~df['agent_state'].isin(allowed_vals_agent_state), 'agent_state'] = "Others"

    allowed_vals_employer_state = ['CA', 'TX', 'NY', 'NJ']
    df.loc[~df['employer_state'].isin(allowed_vals_employer_state), 'employer_state'] = "Others"

    allowed_vals_job_info_state = ['CA', 'TX', 'NY', 'NJ']
    df.loc[~df['job_info_work_state'].isin(allowed_vals_job_info_state), 'job_info_work_state'] = "Others"

    allowed_vals_class_of_admission = ['H-1B']
    df.loc[~df['class_of_admission'].isin(
        allowed_vals_class_of_admission), 'class_of_admission'] = "Others"

    #COUNTRY_OF_CITIZENSHIP
    allowed_vals_country_of_citizenship = ['INDIA', 'CHINA', 'SOUTH KOREA', 'CANADA']
    df.loc[~df['country_of_citizenship'].isin(allowed_vals_country_of_citizenship), 'country_of_citizenship'] = "Others"

    #EMPLOYER_COUNTRY
    allowed_vals_employer_country = ['UNITED STATES OF AMERICA']
    df.loc[~df['employer_country'].isin(allowed_vals_employer_country), 'employer_country'] = "Others"

    #FW_INFO_BIRTH_COUNTRY
    # allowed_vals_fw_info_birth_country = ['INDIA', 'CHINA', 'SOUTH KOREA', 'PHILLIPINES', 'MEXICO', 'CANADA', 'TAIWAN']
    # df.loc[~df['fw_info_birth_country'].isin(allowed_vals_fw_info_birth_country), 'fw_info_birth_country'] = 'Others'

    #JOB INFO EDUCATION
    df.loc[:, 'job_info_education'] = df['job_info_education'].mask(
        df.job_info_education == 'Associate\'s', 'Others')
    df.loc[:, 'job_info_education'] = df['job_info_education'].mask(
        df.job_info_education == 'High School', 'Others')
    df.loc[:, 'job_info_education'] = df['job_info_education'].mask(
        df.job_info_education == 'Doctorate', 'Others')
    df.loc[:, 'job_info_education'] = df['job_info_education'].mask(
        df.job_info_education == 'None', 'Others')
    df.loc[:, 'job_info_education'] = df['job_info_education'].mask(
        df.job_info_education == 'Other', 'Others')

    #FOREIGN_WORKER_INFO_EDUCATION
    df.loc[:, 'foreign_worker_info_education'] = df['foreign_worker_info_education'].mask(
        df.foreign_worker_info_education == 'Associate\'s', 'Others')
    df.loc[:, 'foreign_worker_info_education'] = df['foreign_worker_info_education'].mask(
        df.foreign_worker_info_education == 'High School', 'Others')
    df.loc[:, 'foreign_worker_info_education'] = df['foreign_worker_info_education'].mask(
        df.foreign_worker_info_education == 'Doctorate', 'Others')

    ## CONVERT SOME IMPORTANT INTEGER (ACTUAL) DATA TO INTEGER. --> Anomaly data, I dont know why, it is treated as float in real data, but when processed in python, it becomes object data.
    df['wage_offer_from_9089'] = pd.to_numeric(df['wage_offer_from_9089'], errors='coerce')
    df['pw_amount_9089'] = pd.to_numeric(df['pw_amount_9089'], errors='coerce')
    category_columns.remove('pw_amount_9089')
    category_columns.remove('wage_offer_from_9089')

    #RI_COLL_TEACH_PRO_JNL
    # allowed_vals_ri_coll_teach_pro_jnl = ['None']
    # df.loc[~df['ri_coll_teach_pro_jnl'].isin(allowed_vals_ri_coll_teach_pro_jnl), 'ri_coll_teach_pro_jnl'] = 'Others'

    ## PW SOC CODE ##
    allowed_vals_pw_soc_code = ['15-1132', '15-1121', '15-1133']
    df.loc[~df['pw_soc_code'].isin(allowed_vals_pw_soc_code), 'pw_soc_code'] = 'Others'

    df[category_columns] = df[category_columns].apply(lambda x: x.str.upper().replace(r'[^\w\s]', '', regex=True))

    # EMPLOYER_CITY (DO REPLACING ONLY AFTER WE HAVE DONE THE STR UPPER SO WE DONT FIND DUPLICATE VALUE WITH DIFFERENT IN CASE.
    allowed_vals_employer_city = ['COLLEGE STATION', 'NEW YORK', 'SANTA CLARA', 'MOUNTAIN VIEW', 'SAN JOSE', 'REDMOND',
                                  'SAN FRANSISCO', 'SUNNYVALE', 'PLANO', 'SEATTLE', 'CHICAGO', 'HOUSTON']

    df.loc[~df['employer_city'].isin(allowed_vals_employer_city), 'employer_city'] = "OTHERS"

    allowed_vals_agent_city = ['SAN FRANCISCO', 'NEW YORK', 'BOSTON', 'SANTA CLARA', 'CHICAGO']
    df.loc[~df['agent_city'].isin(allowed_vals_employer_city), 'agent_city'] = "OTHERS"

    # JOB_INFO_WORK_CITY
    allowed_vals_job_info_work_city = ['COLLEGE STATION', 'NEW YORK', 'SAN JOSE', 'SAN FRANCISCO', 'MOUNTAIN VIEW', 'REDMOND', 'HOUSTON', 'SANTA CLARA', 'SEATTLE', 'SUNNYVALE']
    df.loc[~df['job_info_work_city'].isin(allowed_vals_job_info_work_city), 'job_info_work_city'] = "OTHERS"

    #NAICS_US_TITLE
    # allowed_vals_naics_us_title = ['CUSTOM COMPUTER PROGRAMMING SERVICES', 'COMPUTER SYSTEMS DESIGN SERVICES', 'SOFTWARE PUBLISHERS']
    # df.loc[~df['naics_us_title'].isin(allowed_vals_naics_us_title), 'naics_us_title'] = "OTHERS"

    # ri_1st_ad_newspaper_name
    allowed_vals_ri_1st_ad_newspaper_name = ['SAN JOSE MERCURY NEWS', 'SAN FRANSISCO CHRONICLE', 'THE EAGLE', 'THE SEATTLE TIMES', 'THE NEW YORK TIMES']
    df.loc[~df['ri_1st_ad_newspaper_name'].isin(allowed_vals_ri_1st_ad_newspaper_name), 'ri_1st_ad_newspaper_name'] = "OTHERS"

    ##FILL MODE
    df[category_columns] = df[category_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

    ###END  ###

    ## Encode data that only has two values to 0 and 1.
    for col in category_columns:
        if len(df[col].unique()) == 2:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes

    limit_of_unique_values = 50
    for col in category_columns:
        if len(df[col].unique()) > limit_of_unique_values:
            df.drop(col, inplace=True, axis=1, errors='ignore')

    ## STARTING FROM HERE, DELETE COLUMNS THAT IS NOT IMPORTANT AT ALL FOUND OUT FROM THE RANDOM FOREST FEATURE IMPORTANCES.
    # df = df.drop(['preparer_info_title', 'pw_soc_title', 'fw_info_education_other', 'schd_a_sheepherder', 'pw_unit_of_pay_9089',
    #               'pw_source_name_9089', 'wage_offer_unit_of_pay_9089', 'ji_offered_to_sec_j_fw', 'recr_info_employer_rec_payment',
    #               'ri_1st_ad_newspaper_name', 'ji_live_in_domestic_service', 'ri_posted_notice_at_worksite', 'ji_fw_live_on_premises',
    #               'employer_country', 'agent_city', 'fw_info_training_comp', 'ri_2nd_ad_newspaper_or_journal', 'fw_ownership_interest',
    #               'job_info_combo_occupation', 'fw_info_birth_country', 'recr_info_barg_rep_notified', 'recr_info_sunday_newspaper',
    #               'employer_city', 'job_info_work_city', 'recr_info_coll_univ_teacher', 'naics_us_code'
    #               ], axis=1, errors='ignore')
    # ## Drop row ID.
    # df.drop(['row ID'], axis=1)

    ## Numerical ##
    numeric_columns = []
    numeric_columns += df.select_dtypes(include=[np.number]).columns.values.tolist()

    # Because it does not make sense to be established before 1500.
    # TODO: Fix this, takes the real data.
    df[numeric_columns].loc[df['employer_yr_estab'] < 1500, 'employer_yr_estab'] = np.nan

    df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))

    ## Convert float to int
    floatcol = []
    floatcol += df.select_dtypes(include=[np.float]).columns.values.tolist()
    df[floatcol] = df[floatcol].astype(int)


    # df = df[['case_status', 'pw_expire_date', 'case_received_date', 'decision_date', 'employer_num_employees',
    #                'employer_yr_estab', 'fw_info_yr_rel_edu_completed', 'pw_determ_date', 'recr_info_second_ad_start',
    #                'recr_info_swa_job_order_end', 'job_info_alt_occ_num_months', 'recr_info_first_ad_start',
    #                'recr_info_swa_job_order_start', 'ri_job_search_website_from', 'ri_job_search_website_to',
    #                'naics_us_code', 'ri_employer_web_post_from', 'ri_employer_web_post_to', 'fw_info_birth_country',
    #                'job_info_education', 'job_info_experience', 'job_info_work_state', 'pw_level_9089', 'pw_soc_code',
    #                'ri_coll_teach_pro_jnl']]

    df = df[['decision_date', 'case_received_date', 'recr_info_swa_job_order_end', 'recr_info_second_ad_start', 'ri_job_search_website_from',
             'ri_job_search_website_to', 'recr_info_swa_job_order_start', 'recr_info_first_ad_start', 'pw_determ_date',
             'pw_expire_date', 'wage_offer_from_9089', 'pw_amount_9089', 'employer_yr_estab', 'job_info_experience', 'job_info_alt_field',
             'job_info_education', 'foreign_worker_info_education']]

    ## Done ##

    return df

# *args define that we can use several different arguments.
def columns_date_handler(df, *args):
    columns = []
    for arg in args:
        columns += [col for col in df.columns.values.tolist()
                    if re.search(arg, col)]

        return columns

data_preprocessing('~/Desktop/TrainingSet.csv').to_csv('TrainingSet_clean.csv')
data_preprocessing('~/Desktop/TestingSet.csv').to_csv('TestingSet_clean.csv')





###     # Filter categorical data, if over 70 unique values, drop the data in that columns.
   ### limit_of_unique_values = 70
   ### df[category_columns] = df[category_columns].loc[:, df[category_columns].nunique() < limit_of_unique_values]



