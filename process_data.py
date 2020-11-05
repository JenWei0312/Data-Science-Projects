import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import logging
logging.basicConfig(filename="data_processing.log", level=logging.INFO)

# Params
file_name = "data/ALTAML_HipFractures July 17.csv"
#file_name = "ALTAML_HipFractures July 17.csv"
clean_file_location = "data/cleaned_data.p"
encoded_file_location = "data/encoded_data.p"


logging.info("Data processing started")
try:
    data = pd.read_csv(file_name)
except:
    logging.info(e)
    raise Exception(e)
logging.info(f"Data read from {file_name}")

# Drop nans
data.dropna(thresh=len(data) // 2, axis=1, inplace=True)
logging.info("Dropped nan")


# Encode the targets to two classes
data["episodeDispositionCode"] = data["episodeDispositionCode"].dropna().astype("int")
target_dict = {}
for i, (code, desc) in enumerate(
    zip(data["episodeDispositionCode"], data["episodeDispositionShortDesc"])
):
    if not np.isnan(code):
        target_dict[code] = desc
# -------------class 0, Acute Care ---------------------------------
# '1': 'To Acute Care',
# '10': 'Inpatient Care',
# '4': 'Home with support',
# '5': 'Home without support',
# ------------ class 1, Continuing Care-----------------------------
# '2': 'To Cont Care',
# '30': 'Residential Care',
# '40': 'Group/Supportive Living',
# ---------------These classes are removed -------------------------
# '7': 'Died',
# '12': 'Did not return from pass',
# '72': 'Died in Facility',
# '3': 'To Other',
# '6': 'AMA',
# '62': 'Left Against Medical Advice (LAMA)'
class_names = ["Acute Care", "Continuing Care"]
classes_to_be_removed = [7, 12, 72, 3.0, 6.0, 20.0, 61.0, 62.0, 65.0, 73.0, 90.0]
data = data[~data["episodeDispositionCode"].isin(classes_to_be_removed)].reset_index(
    drop=True
)
data["episodeDispositionCode"].replace([1, 10, 4, 5], 1, inplace=True)
data["episodeDispositionCode"].replace([30, 40, 2], 2, inplace=True)
data.groupby("episodeDispositionCode")["episodeDispositionCode"].count()


def binarize(x):
    if x == 1:
        return 0
    elif x == 2:
        return 1


data["episodeDispositionCode"] = data["episodeDispositionCode"].apply(binarize)
logging.info("Labels encoded")

# Mannualy select features that should be dropped
patient_feats = ["ipStayId", "patientId"]
institution_specific_features = [
    "institutionId",
    "institutionIdFrom",
    "institutionIdTo",
    "mostRespDoctorId",
    "episodeStartInstId",
    "episodeInstToId",
    "postalCode",
]
date_features = [
    "admitDateTime",
    "dischargeDateTime",
    "episodeStartDateTime",
    "episodeEndDateTime",
]
leaking_feats = [
    "cmg",
    "riw",
    "mccCode",
    "expectedLOS",
    "transferLos",
    "transfer",
    "transferIpStayId",
    "dischargeDateTime",
    "dischargeFiscalYearEnd",
    "dischargeFiscalQuarter",
    "dischargeFiscalMonth",
    "dispositionCode",
    "dispositionShortDesc",
    "alcDays",
    "subacuteDays",
    "acuteDays",
    "episodeDispositionShortDesc",
    "losEpisode",
    "episodeEndDateTime",
    "episodeInstToId",
    "surgeryCount",
    "procedureCount",
    "aeDeathIP",
    "aeDeathEpisode",
    "institutionTypeIdTo",
    "episode_hours",
    "losAcute",
    "adminTOdischarge_hours",
    "riw",
    "expectedLOS",
    "transfer",
    "transferIpStayId",
    "transferLos",
    "losAcute",
    "episodeEdVisit30",
    "episodeEdVisit60",
    "episodeEdVisit90",
    "aeUnexpectedReturnToOR",
    "aeIntraOpFracture",
    "aePostOpFracture",
    "aeMechanicalComplication",
    "aeDislocation",
    "aeCVA",
    "aeMI",
    "aePE",
    "aeDVT",
    "aePneumonia",
    "aeGIBleed",
    "aeIleus",
    "aeBleeding",
    "aeOther",
    "aeUTI",
    "aeDementia",
    "aeDelirium",
    "aePressureUlcer",
    "aeARF",
    "aeMedicalEventCount",
    "aeMechanicalEventCount",
    "aeCareRelatedEventCount",
    "adminTOdischarge_hours",
    "episode_hours",
]


def get_feats_with_code(feat_list, code):
    """Helper function that gets feature names with a specific 'code' at in it. 
    This is used to detect cases like features with `Desc` at the end (siginif-
    -ying that it is a description)."""
    desc_feats = []
    for feat_name in feat_list:
        if feat_name.find(code) != -1:
            desc_feats.append(feat_name)
    return desc_feats


feats_with_ae = get_feats_with_code(list(data.keys()), "ae")
feats_with_Desc = get_feats_with_code(list(data.keys()), "Desc")
feats_with_Ed = get_feats_with_code(list(data.keys()), "Ed")
date_features = [
    "admitDateTime",
    "dischargeDateTime",
    "episodeStartDateTime",
    "episodeEndDateTime",
]
feats_to_remove = (
    feats_with_ae
    + feats_with_Desc
    + feats_with_Ed
    + patient_feats
    + institution_specific_features
    + leaking_feats
    + date_features
)
logging.info(f"Features removed: {feats_to_remove}")

# Engineer new date features
for date in date_features:
    data[date] = pd.to_datetime(data[date])
data["adminTOdischarge_hours"] = (data.dischargeDateTime - data.admitDateTime).astype(
    "timedelta64[h]"
)
data["episode_hours"] = (data.episodeEndDateTime - data.episodeStartDateTime).astype(
    "timedelta64[h]"
)

# Drop features
for col in feats_to_remove:
    try:
        data.drop(col, axis=1, inplace=True)
    except Exception as e:
        logging.info(e)
        pass
data.dropna(inplace=True)
# save cleaned up features
pickle.dump(data, open(clean_file_location, "wb"))
logging.info(f"Data shape after cleaning: {data.shape}")
logging.info(f"Cleaned data saved at {clean_file_location}")

# encode categorical features
categorical_feats = [
    "procedureTypeCode",
    "mostRespDxCode",
    "institutionZone",
    "entryCode",
    "admitByAmbulanceCode",
    "admitCategoryCode",
    "sex",
    "institutionZoneId",
    "institutionTypeId",
    "institutionTypeIdFrom",
]
logging.info("Feature encoding started")
encoder = ce.OneHotEncoder(cols=categorical_feats, handle_unknown="ignore")
data = encoder.fit_transform(data)
logging.info("Feature encoding Ended")
logging.info(f"Data shape after encoding: {data.shape}")
# save encoded features
pickle.dump(data, open(encoded_file_location, "wb"))
logging.info(f"File written at {encoded_file_location}")
