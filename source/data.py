import numpy as np
import pandas as pd
from responsibly.dataset import COMPASDataset
from responsibly.dataset import GermanDataset
from responsibly.dataset import AdultDataset
from sklearn.preprocessing import StandardScaler

def compas():
    # Get the whole dataset, already nicely filtered for us from this library
    compas_ds = COMPASDataset()

    # Make the dataframe
    cdf = compas_ds.df

    """
    There are some columns that need to be adjusted, and a bunch that need to be dropped
    - length jail sentence becomes one column instead of c_jail_in and c_jail_out  
    - time in custody becomes one column instead of cusotdy_in and custody_out 
    - I encode binary attributes 0,1 where 0 is majority class 1 is minority class 
    Male => 0 Female => 1, 
    Misdemeanor => 0, Felony => 1 
    """

    # Turn the length of jail sentence a single variable
    c_jail_out = pd.to_datetime(cdf['c_jail_out'])
    c_jail_in = pd.to_datetime(cdf['c_jail_in'])
    c_jail_time = (c_jail_out - c_jail_in).apply(lambda x: x.days + x.seconds / 3600)
    cdf["c_jail_time"] = c_jail_time

    # Turn the length of custody into a single variable
    custody_in = pd.to_datetime(cdf['in_custody'])
    custody_out = pd.to_datetime(cdf['out_custody'])
    custody_delta = (custody_out - custody_in).apply(lambda x: x.days + x.seconds / 3600)
    cdf["custody_length"] = custody_delta

    # Encode Male Female
    cdf = cdf.replace({'sex': {'Male': 0, 'Female': 1}})

    # Encode Charge Degree
    cdf = cdf.replace({'c_charge_degree': {'M': 0, 'F': 1}})

    # One Hot Encode Race
    cdf = one_hot(cdf, "race")

    # Remove Nans (not even sure how those show up for crimes?)
    cdf = cdf.replace({np.nan: "other"})

    charges = cdf["c_charge_desc"].unique()

    # I dropped all of these columns because they didn't seem useful (idk what I was saying earlier)
    # If you disagree just commit it out I guess idrc (this is still true)
    cdf = cdf.drop(["name", "id", "dob", "first", "last", "compas_screening_date",
                    "age_cat", "c_case_number", "r_case_number", "vr_case_number", "decile_score.1",
                    "type_of_assessment", "score_text", "screening_date", "v_type_of_assessment", "priors_count.1",
                    "v_score_text", "v_screening_date", "in_custody", "out_custody", "length_of_stay",
                    "c_jail_out", "c_jail_in", "age_cat", "c_charge_desc", "c_offense_date",
                    "c_arrest_date", "c_offense_date", "r_charge_degree", "r_days_from_arrest",
                    "r_offense_date", "r_charge_desc", "r_jail_in", "r_jail_out", "violent_recid",
                    "vr_charge_degree", "vr_offense_date", "score_factor", "vr_charge_desc",
                    "v_decile_score", "c_days_from_compas", "start", "end", "event",
                    "days_b_screening_arrest"], axis=1)
    return cdf

def adultDataset():
    #Get the whole dataset, already nicely filtered for us from this library
    adult_ds = AdultDataset()

    #Make the dataframe
    adf = adult_ds.df

    #Make the occupations 1-hot
    adf = one_hot(adf, "occupation", drop=True)


    #Clean up Education
    adf = adf.replace({'education': {
        'Assoc-acdm': "HS-grad",
        'Some-college': "HS-grad",
        'Assoc-voc': "HS-grad",
        "Prof-school": "HS-grad",
        "11th": "No-HSD",
        "9th": "No-HSD",
        "10th": "No-HSD",
        "12th": "No-HSD",
        "5th-6th": "No-HSD",
        "7th-8th": "No-HSD",
        "1st-4th": "No-HSD",
        "Preschool": "No-HSD"
    }})

    #Drop smaller racial categories for now
    adf = adf[adf["race"] != "Amer-Indian-Eskimo"]
    adf = adf[adf["race"] != "Asian-Pac-Islander"]
    adf = adf[adf["race"] != "Other"]

    #Only united states (~40k/48k)
    # United States non US (~40k/48k)
    adf.loc[adf['native_country'] != ' United-States', 'native_country'] = 'Non-US'
    adf.loc[adf['native_country'] == ' United-States', 'native_country'] = 'US'
    adf['native_country'] = adf['native_country'].map({'US': 1, 'Non-US': 0}).astype(int)
    adf.head()


    #mMake race and gender binary (yuck)
    adf = adf.replace({'sex': {
        'Female': 1,
        'Male': 0,
    }})

    adf = adf.replace({'race': {
        'Black': 1,
        'White': 0,
    }})

    #Make all categorical 1-hot
    adf = one_hot(adf, "education", drop=True)

    #Simply work classes
    adf['workclass'] = adf['workclass'].replace(['State-gov', 'Local-gov', 'Federal-gov'], 'Gov')
    adf['workclass'] = adf['workclass'].replace(['Self-emp-inc', 'Self-emp-not-inc'], 'Self_employed')
    adf = one_hot(adf, "workclass", drop=True)




    #Make Marital status binary
    adf['marital_status'] = adf['marital_status'].replace(['Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'Single')
    adf['marital_status'] = adf['marital_status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Couple')
    adf = one_hot(adf, "marital_status", drop=True)

    #Make relationships one-hot
    adf = one_hot(adf, "relationship", drop=True)


    #Make the income variables binary
    adf = adf.replace({'income_per_year': {
        '<=50K': 0,
        '>50K': 1,
    }})



    return adf

def germanDataset():
    # import data
    german_ds = GermanDataset()
    # Make the dataframe
    cdf = german_ds.df

    # Rename the columns of status and sex as they seem to be swapped
    cdf = cdf.rename(columns = {'sex': 'marital_status', 'status': 'sex'}, inplace = False)
    # Encode credit classification
    cdf = cdf.replace({'credit': {'good': 1, 'bad': 0}})
    # Encode Male Female 
    cdf = cdf.replace({'sex': {'male': 0, 'female': 1}})
    # redo age factor so it's not an interval
    cdf = cdf.drop(["age_factor"], axis=1)  # remove age factor since age is also a variable
    cdf['age_factor'] = cdf['age'] >= 25
    
    # remove the random columns that are duplicated
    cdf = cdf.loc[:,~cdf.columns.duplicated()]
    
    # deal with categorical values by one hot encoding
    catvars = ['credit_history', 'purpose', 'savings', 
               'present_employment','marital_status', 'other_debtors', 'property', 
               'installment_plans', 'housing', 'job']
    for x in catvars:
        cdf = one_hot(cdf, x)
        
    cdf = cdf.rename(columns = {'none': 'no_guarantor_co-applicant', 'stores': 'store_installment', 
                                'bank':'bank_installment'}, inplace = False)
    return cdf

def one_hot(df, column, drop=True):
    """
    :param df: the dataframe we're manipulating (pandas)
    :param column: the name of the column we wanna 1-hot (string)
    :param drop: drop the column we encoded in the return df (bool)
    :return:
    """

    #Get all the possible values, these will be the new colums in the new encoding
    values = df[column].unique()

    #Go through the values and create the encodings, i think this is straight forward idk
    for v in values:
        one_hot = df[column].apply(lambda x: x == v)
        df[v] = one_hot
    if drop:
        df = df.drop([column], axis=1)
    return df


#TODO: Is this what i want? not sure
# def write_compas():
#     # Write to CSV file
#     os.chdir("..")
#     cdf.to_csv("./data/compas.csv")
