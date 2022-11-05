# main script to train the model on the e-POCT data
# There are 4 different ways to run this script. Either a single model
# is trained and evaluated or 5 X 2 fold CV can be performed. Also, one
# can choose whether to also train the model to perform feature decoding, or not.
# The metrics for the 5 x 2 CV are saved in the saved_objects folder.
# When training and evaluationg a single model, several plots are computed
# when applying the model to the unseen test set. These can be found in
# the folders saved_plots and heatmaps.

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

from baselines import Logistic_regression_EPOCT, MLP
from dataset_generation import EPOCTQuestionnaireDataset
from training_procedures_epoct import train_and_test_modules
from utils import preprocess_data_epoct
from utils_epoct import get_targets, process_features

# parameters (to adapt to perform 5 X 2 CV or feature decoding)
FiveFoldCv = False
FeatureDecoding = False
# to reload a pretrained model
reload = True
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# load complete unlabeled data
os.chdir("data")
print(os.getcwd())
path = os.path.join(os.getcwd(), "data.csv")
print(path)
data = pd.read_csv(path)
print(os.getcwd())
os.chdir("../models")
# select features and targets
df, categorical_features_dict = process_features(data)
features, targets, categorical_features_dict = get_targets(
    df, categorical_features_dict
)

num_classes = targets.shape[1]
possible_target_values = targets.columns
# instantiate questionnaire data set object
qst_obj = EPOCTQuestionnaireDataset(
    features, targets, num_classes, possible_target_values, categorical_features_dict
)
# save for use in later scripts
pickle.dump(qst_obj, open("saved_objects/qst_obj.pt", "wb"))

if FiveFoldCv:
    folds = qst_obj.ml_2x5fold()
    macro_f1_log = np.empty((5, 2))
    macro_f1_mlp = np.empty((5, 2))
    results_knn_all = {}
    results_log_all = {}
    results_mlp_all = {}
    for j in range(5):
        fold = folds[j]
        results_log_all[j] = {}
        results_mlp_all[j] = {}
        # to switch train and test sets within fold
        for k in range(2):
            if k == 0:
                pass
            else:
                # switch train and test
                fold = tuple([fold[1], fold[0]])
            (
                X_train,
                X_valid,
                X_test,
                X_cont_train,
                X_cont_valid,
                X_cont_test,
                X_cat_train,
                X_cat_valid,
                X_cat_test,
                y_train,
                y_valid,
                y_test,
                mean,
                std,
                min_,
                max_,
            ) = preprocess_data_epoct(
                qst_obj,
                valid_size=0.2,
                test_size=0.2,
                tensors=False,
                imput_strategy="mean",
                fold=fold,
            )
            print(
                f"length train : {len(X_train)}, length valid : {len(X_valid)}, length test : {len(X_test)}"
            )

            # compute baselines
            (
                macro_f1_log[j, k],
                results_log_all[j][k],
                base_logist_proba,
            ) = Logistic_regression_EPOCT(
                X_train, y_train, X_test, y_test, penalty=20, targets=targets
            )
            macro_f1_mlp[j, k], results_mlp_all[j][k], base_mlp_proba = MLP(
                X_train, y_train, X_valid, y_valid, X_test, y_test, targets=targets
            )
else:
    folds = None
    (
        X_train,
        X_valid,
        X_test,
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
        mean,
        std,
        min_,
        max_,
    ) = preprocess_data_epoct(
        qst_obj,
        valid_size=0.2,
        test_size=0.2,
        tensors=False,
        imput_strategy="mean",
        fold=folds,
    )

    # # compute baselines
    macro_f1_log, results_log_all, base_logist_proba = Logistic_regression_EPOCT(
        X_train, y_train, X_test, y_test, penalty=20, targets=targets
    )
    macro_f1_mlp, results_mlp_all, base_mlp_proba = MLP(
        X_train, y_train, X_valid, y_valid, X_test, y_test, targets=targets
    )

print("Using encoder modules")
# train the modules
lr_feature_decoders = {
    "dem_agem": 1e-1,
    "signv_waz_d0": 2e-3,
    "dem_sex": 4e-3,
    "sign_convulscomplex_d0": 2e-4,
    "sign_pallor_d0": 2e-3,
    "sign_respdistress_d0": 2e-3,
    "sign_skin_sev_d0": 2e-3,
    "signv_hypox_d0": 2e-3,
    "signv_muaclow_d0": 2e-3,
    "signv_hr1_d0": 2e-3,
    "signv_rr1_d0": 2e-3,
    "signv_temp_d0": 2e-3,
    "hist_hiv_mom": 4e-3,
    "hist_ttt_before": 4e-5,
    "hist_vac_pcv1": 1e-5,
    "hist_vac_pcv3": 5e-5,
    "lab_bcx_id_d0": 2e-8,
    "lab_pglucose_d0": 7e-8,
    "lab_udip_ket_d0": 2e-8,
    "lab_udip_nit_d0": 6e-8,
    "lab_udip_spec_d0": 3e-5,
    "lab_urine_cx_id": 5e-8,
    "lab_urine_type_d0": 7e-7,
    "symp_complaint_o_d0": 2e-4,
    "symp_eye_d0": 2e-3,
    "sympcc_abdopain": 2e-3,
    "sympcc_dyspnea": 2e-3,
    "sympcc_dysuria": 2e-3,
    "sympcc_fev": 2e-4,
    "sympcc_feveronly": 6e-3,
    "sympcc_loa": 2e-3,
    "sympcc_pharyngitis": 2e-3,
    "sympcc_uri": 7e-3,
}

lr_encoders = {
    "dem_agem": 1e-2,
    "signv_waz_d0": 1e-2,
    "dem_sex": 1e-2,
    "sign_convulscomplex_d0": 1e-2,
    "sign_pallor_d0": 1e-2,
    "sign_respdistress_d0": 1e-2,
    "sign_skin_sev_d0": 1e-2,
    "signv_hypox_d0": 1e-2,
    "signv_muaclow_d0": 1e-2,
    "signv_hr1_d0": 1e-2,
    "signv_rr1_d0": 1e-2,
    "signv_temp_d0": 1e-2,
    "hist_hiv_mom": 1e-2,
    "hist_ttt_before": 1e-2,
    "hist_vac_pcv1": 1e-2,
    "hist_vac_pcv3": 1e-2,
    "lab_bcx_id_d0": 1e-2,
    "lab_pglucose_d0": 1e-2,
    "lab_udip_ket_d0": 1e-2,
    "lab_udip_nit_d0": 1e-2,
    "lab_udip_spec_d0": 1e-2,
    "lab_urine_cx_id": 1e-2,
    "lab_urine_type_d0": 1e-2,
    "symp_complaint_o_d0": 1e-2,
    "symp_eye_d0": 1e-2,
    "sympcc_abdopain": 1e-2,
    "sympcc_dyspnea": 1e-2,
    "sympcc_dysuria": 1e-2,
    "sympcc_fev": 1e-2,
    "sympcc_feveronly": 1e-2,
    "sympcc_loa": 1e-2,
    "sympcc_pharyngitis": 1e-2,
    "sympcc_uri": 1e-2,
}

if FiveFoldCv:
    reload = False
    if FeatureDecoding:
        n_epochs = 280
        # 250
    else:
        n_epochs = 270
else:
    if FeatureDecoding:
        n_epochs = 350
    else:
        n_epochs = 230
training_parameters = {
    "n_epochs": n_epochs,
    "lr_encoders": lr_encoders,
    "lr_feature_decoders": lr_feature_decoders,
    "lr_diseases_decoder": 1e-2,
    "lr": 2e-3,
    "decay_factor": 0.9,
    "step_size": 150,
    "batch_size": 500,
    "aux_loss_encoded_weight": 1,
    "aux_loss_unknown_weight": 1,
    "diseases_loss_weight": 1,
    "idemp_loss_weight": 1,
    "gradient_clipping": 10000,
}

qst_obj.training_parameters = training_parameters
macro_f1, aggregated_results = train_and_test_modules(
    qst_obj, folds, reload=reload, feature_decoding=FeatureDecoding
)

print(f"MoDN : {macro_f1}")
print(f"Macro f1 log : {macro_f1_log}")
metrics = {
    "mlp": {"f1": macro_f1_mlp, "results": results_mlp_all},
    "log_reg": {"f1": macro_f1_log, "results": results_log_all},
    "modn": {"f1": macro_f1, "results": aggregated_results},
}
# save metrics to produce plots in using statistical_tests.py
if FiveFoldCv:
    if FeatureDecoding:
        pickle.dump(metrics, open("saved_objects/metrics_decoding.p", "wb"))
    else:
        pickle.dump(metrics, open("saved_objects/metrics.p", "wb"))
print("End of script")
