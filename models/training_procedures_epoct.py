# Training and testing functions

import os
import warnings
from time import process_time

from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR

import wandb

from graph_functions import plot_calibration, plot_heatmap_after_encoder_epoct_all_steps
from utils import *

warnings.filterwarnings("ignore")

STATE_SIZE = 30

# preprocessing params


def get_optimizer_with_feature_decoding(
    qst,
    initial_state,
    encoders,
    continuous_decoders,
    categorical_decoders,
    binary_decoders,
    lr_encoders,
    lr_decoders,
    lr_binary_decoders,
    lr,
):
    """
    instantiate the optimizer with the parameters for the modules it has to optimize
    :param qst: questionnaire data set object
    :param encoders: dict of encoders
    :param continuous_decoders: dict of decoders for continuous features
    :param categorical_decoders: dict of decoders for categorical features
    :param binary_decoders: binary disease decoders
    :param lr_encoders: learning rates for encoders
    :param lr_decoders: learning rates for decoders
    :param lr_binary_decoders: learning rates for disease decoders
    :param lr: learning rate for remaining parameters
    :return:
    """

    parameters_optimizer = [{"params": [initial_state.state_value], "lr": lr}]

    parameters_optimizer.extend(
        {
            "params": list(encoders[feature_name].parameters()),
            "lr": lr_encoders[feature_name],
        }
        for feature_name in encoders.keys()
    )
    parameters_optimizer.extend(
        [
            {
                "params": list(continuous_decoders[feature_name].parameters()),
                "lr": lr_decoders[feature_name],
            }
            for feature_name in qst.continuous_features_names
        ]
    )
    parameters_optimizer.extend(
        [
            {
                "params": list(categorical_decoders[feature_name].parameters()),
                "lr": lr_decoders[feature_name],
            }
            for feature_name in qst.categorical_features_names
        ]
    )
    parameters_optimizer.extend(
        [
            {
                "params": list(binary_decoders[disease].parameters()),
                "lr": lr_binary_decoders,
            }
            for disease in binary_decoders.keys()
        ]
    )

    optimizer = torch.optim.Adam(parameters_optimizer, lr=lr)

    return optimizer


def get_optimizer(
    encoders_to_train,
    initial_state,
    encoders,
    binary_decoders,
    lr_encoders,
    lr_binary_decoders,
    lr,
    diseases=True,
):
    """
    instantiate the optimizer with the parameters for the modules it has to optimize
    :param encoders_to_train: which encoders to train
    :param initial_state: initial state
    :param encoders: encoder modules
    :param binary_decoders: binary decoder modules
    :param lr_encoders: learning rate for encoders
    :param lr_binary_decoders: learning rate for disease decoders
    :param lr: learning rate for other parameters (unused)
    :param diseases: boolean, whether to train disease decoders (usually true)
    :return:
    """
    parameters_optimizer = [{"params": [initial_state.state_value], "lr": lr}]

    parameters_optimizer.extend(
        {
            "params": list(encoders[feature_name].parameters()),
            "lr": lr_encoders[feature_name],
        }
        for feature_name in encoders_to_train
    )
    if diseases:
        parameters_optimizer.extend(
            [
                {
                    "params": list(binary_decoders[disease].parameters()),
                    "lr": lr_binary_decoders,
                }
                for disease in binary_decoders.keys()
            ]
        )

    optimizer = torch.optim.Adam(parameters_optimizer, lr=lr)

    return optimizer


def train_modules_with_decoding(
    qst,
    X_cont_train,
    y_train,
    X_cont_valid,
    y_valid,
    X_cat_train,
    X_cat_valid,
    initial_state,
    encoders,
    continuous_feature_decoders,
    categorical_feature_decoders,
    disease_decoders,
    idempotence=True,
    save=False,
):
    """
    Final Training process with feature decoding also
    :param qst: questionnaire object
    :param X_cont_train: continuous training features
    :param y_train: targets train
    :param X_cont_valid: continuous validation features
    :param y_valid: targets valid
    :param X_cat_train: categorical training features
    :param X_cat_valid: categorical validation features
    :param initial_state: initial state
    :param encoders: encoder modules
    :param: continuous_feature_decoders: decoder modules for continuous features
    :param: categorical_feature_decoders: decoder modules for categorical features
    :param disease_decoders: disease decoder modules
    :param idempotence: whether to randomly reapply some encoders
    :param save: boolean to save the model
    :return:
    """

    # parameters for classical encoders and decoders
    training_parameters = qst.training_parameters
    lr_feature_decoders = training_parameters["lr_feature_decoders"]
    lr_encoders = training_parameters["lr_encoders"]
    lr_diseases_decoder = training_parameters["lr_diseases_decoder"]
    lr = training_parameters["lr"]
    decay_factor = training_parameters["decay_factor"]
    step_size = training_parameters["step_size"]
    n_epochs = training_parameters["n_epochs"]

    # batch_size = 500
    batch_size = training_parameters["batch_size"]
    aux_loss_encoded_weight = training_parameters["aux_loss_encoded_weight"]
    aux_loss_unknown_weight = training_parameters["aux_loss_unknown_weight"]
    diseases_loss_weight = training_parameters["diseases_loss_weight"]
    idemp_loss_weight = training_parameters["idemp_loss_weight"]

    # for logs

    wandb.init(
        project="stage_igh",
        config={
            "dataset_size": len(qst),
            "epochs": n_epochs,
            "lr_encoders": lr_encoders,
            "lr_diseases_decoder": lr_diseases_decoder,
            "lr_feature_decoders": lr_feature_decoders,
            "STATE_SIZE": STATE_SIZE,
            "batch_size": batch_size,
            "idempotence": idempotence,
            "aux_loss_encoded_weight": aux_loss_encoded_weight,
            "aux_loss_unknown_weight": aux_loss_unknown_weight,
            "disease_loss_weight": diseases_loss_weight,
        },
    )

    # loss for classification
    criterion = nn.NLLLoss()

    # non nan features and indices of patients
    non_nan_cont_features, non_nan_cont_patients = get_non_nan_features(X_cont_train)
    non_nan_cat_features, non_nan_cat_patients = get_non_nan_features(
        X_cat_train, continuous=False
    )

    print(f"\n Loss :")
    time_start = process_time()

    # get the optimizer for the parameters we train in that level
    optimizer = get_optimizer_with_feature_decoding(
        qst,
        initial_state,
        encoders,
        continuous_feature_decoders,
        categorical_feature_decoders,
        disease_decoders,
        lr_encoders,
        lr_feature_decoders,
        lr_diseases_decoder,
        lr,
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=decay_factor)

    for j in range(n_epochs):
        scheduler.step()
        # feature encoding loss for encoded continuous features
        aux_loss_cont_encoded = 0
        # feature encoding loss for not yet encoded continuous features
        aux_loss_cont_unknown = 0
        # feature encoding loss for encoded categorical features
        aux_loss_cat_encoded = 0
        # feature encoding loss for not yet encoded categorical features
        aux_loss_cat_unknown = 0
        # to store the sum of all the disease decoding losses after each encoding step
        disease_loss = 0
        # idempotence loss
        idemp_loss = 0
        # state_changes_loss
        state_changes_loss = 0
        # select random batch of indices (without replacement)
        batch_indices = np.random.choice(
            range(len(X_cont_train)), batch_size, replace=False
        )
        # init state
        state = initial_state(len(X_cont_train))
        # decode all diseases
        for name in qst.disease_names:
            disease_loss += criterion(
                disease_decoders[name](state[batch_indices, :]),
                y_train[batch_indices, qst.disease_names.index(name)],
            )
        # iterate over levels in tree
        for level in qst.train_group_order_features.keys():
            # apply encoders
            for feature_group in shuffle(qst.train_group_order_features[level].keys()):
                # select patients
                patients = np.intersect1d(
                    qst.train_group_order_features[level][feature_group], batch_indices
                )
                if len(patients) > 0:
                    # apply shuffled encoders (we shuffle encoders in same level in tree)
                    for feature_name in shuffle(feature_group):
                        state_before = state.clone()
                        # apply continuous feature encoders
                        if feature_name in qst.continuous_features_names:
                            state[patients, :] = encoders[feature_name](
                                state[patients, :],
                                X_cont_train[
                                    patients,
                                    qst.continuous_features_names.index(feature_name),
                                ].view(-1, 1),
                            )
                        else:
                            # apply categorical feature encoders
                            state[patients, :] = encoders[feature_name](
                                state[patients, :],
                                X_cat_train[
                                    patients,
                                    qst.categorical_features_names.index(feature_name),
                                ].view(-1, 1),
                            )
                        # compute state_change_loss
                        state_changes_loss += mse(
                            state[patients, :], state_before[patients, :]
                        )

                        # after having encoded each feature in the tree, we try to predict the diseases
                        for name in qst.disease_names:
                            disease_loss += criterion(
                                disease_decoders[name](state[batch_indices, :]),
                                y_train[batch_indices, qst.disease_names.index(name)],
                            )

            # decode all the features
            for decode_level in qst.train_group_order_features.keys():
                for feature_group in shuffle(
                    qst.train_group_order_features[decode_level].keys()
                ):
                    # select patients for which feature gets encoded at that level
                    patients = np.intersect1d(
                        qst.train_group_order_features[decode_level][feature_group],
                        batch_indices,
                    )
                    if len(patients) > 0:
                        for feature_name in shuffle(feature_group):
                            # features already have been encoded
                            if decode_level <= level:
                                # continuous feature
                                if feature_name in qst.continuous_features_names:
                                    aux_loss_cont_encoded += neg_log_likelihood_1D(
                                        X_cont_train[
                                            patients,
                                            qst.continuous_features_names.index(
                                                feature_name
                                            ),
                                        ].view(-1, 1),
                                        *continuous_feature_decoders[feature_name](
                                            state[patients, :]
                                        ),
                                    )
                                # categorical feature
                                else:
                                    aux_loss_cat_encoded += criterion(
                                        categorical_feature_decoders[feature_name](
                                            state[patients, :]
                                        ),
                                        X_cat_train[
                                            patients,
                                            qst.categorical_features_names.index(
                                                feature_name
                                            ),
                                        ].long(),
                                    )
                            # features that havent been encoded
                            else:
                                if feature_name in qst.continuous_features_names:
                                    aux_loss_cont_unknown += neg_log_likelihood_1D(
                                        X_cont_train[
                                            patients,
                                            qst.continuous_features_names.index(
                                                feature_name
                                            ),
                                        ].view(-1, 1),
                                        *continuous_feature_decoders[feature_name](
                                            state[patients, :]
                                        ),
                                    )
                                else:
                                    aux_loss_cat_unknown += criterion(
                                        categorical_feature_decoders[feature_name](
                                            state[patients, :]
                                        ),
                                        X_cat_train[
                                            patients,
                                            qst.categorical_features_names.index(
                                                feature_name
                                            ),
                                        ].long(),
                                    )

        # train for idempotence
        state_before_idemp = state.clone()
        if idempotence:
            for feature in shuffle(qst.categorical_features_names):
                index_of_feature = qst.categorical_features_names.index(feature)
                patients = np.intersect1d(
                    non_nan_cat_patients[index_of_feature], batch_indices
                )
                # reapply the categorical encoders
                if len(patients) > 0:
                    state[patients, :] = encoders[feature](
                        state[patients, :],
                        X_cat_train[patients, index_of_feature].view(-1, 1),
                    )
                    # categorical feature decoding loss
                    aux_loss_cat_encoded += criterion(
                        categorical_feature_decoders[feature](state[patients, :]),
                        X_cat_train[patients, index_of_feature].long(),
                    )
                    # idempotence loss
                    idemp_loss += mse(
                        state_before_idemp[patients, :], state[patients, :]
                    )
                    # get the state before having re-applied an encoder
                    state = state_before_idemp.clone()
            for feature in shuffle(qst.continuous_features_names):
                index_of_feature = qst.continuous_features_names.index(feature)
                patients = np.intersect1d(
                    non_nan_cont_patients[index_of_feature], batch_indices
                )
                # reapply the continuous encoders
                if len(patients) > 0:
                    state[patients, :] = encoders[feature](
                        state[patients, :],
                        X_cont_train[patients, index_of_feature].view(-1, 1),
                    )
                    # continuous feature decoding loss
                    # aux_loss_cont_encoded += mse(X_cont_train[patients, index_of_feature].view(-1, 1),
                    #                              continuous_feature_decoders[feature](state[patients, :]))
                    aux_loss_cont_encoded += neg_log_likelihood_1D(
                        X_cont_train[patients, index_of_feature].view(-1, 1),
                        *continuous_feature_decoders[feature](state[patients, :]),
                    )
                    # idempotence loss
                    idemp_loss += mse(
                        state_before_idemp[patients, :], state[patients, :]
                    )
                    state = state_before_idemp.clone()
        # for logs
        with torch.no_grad():
            for index, name in enumerate(qst.continuous_features_names):
                wandb.log(
                    {
                        "loss_cont_feat"
                        + str(name): neg_log_likelihood_1D(
                            X_cont_train[
                                np.intersect1d(
                                    batch_indices, non_nan_cont_patients[index]
                                ),
                                index,
                            ].view(-1, 1),
                            *continuous_feature_decoders[name](
                                state[
                                    np.intersect1d(
                                        batch_indices, non_nan_cont_patients[index]
                                    ),
                                    :,
                                ]
                            ),
                        ),
                        "epoch": j,
                    }
                )
            for index, name in enumerate(qst.categorical_features_names):
                wandb.log(
                    {
                        "loss_cat_feat"
                        + str(name): criterion(
                            categorical_feature_decoders[name](
                                state[
                                    np.intersect1d(
                                        batch_indices, non_nan_cat_patients[index]
                                    ),
                                    :,
                                ]
                            ),
                            X_cat_train[
                                np.intersect1d(
                                    batch_indices, non_nan_cat_patients[index]
                                ),
                                index,
                            ].long(),
                        ),
                        "epoch": j,
                    }
                )

        wandb.log(
            {
                "epoch": j,
                "aux_loss_cont_encoded": aux_loss_cont_encoded,
                "aux_loss_cont_unknown": aux_loss_cont_unknown,
                "aux_loss_cat_encoded": aux_loss_cat_encoded,
                "aux_loss_cat_unknown": aux_loss_cat_unknown,
                "disease_loss": disease_loss,
                "idemp_loss": idemp_loss,
                "state_changes_loss": state_changes_loss,
            }
        )

        # Differentiate loss and make an optimizer step
        optimizer.zero_grad()
        (
            (aux_loss_cont_encoded + aux_loss_cat_encoded) * aux_loss_encoded_weight
            + disease_loss * diseases_loss_weight
            + (aux_loss_cont_unknown + aux_loss_cat_unknown) * aux_loss_unknown_weight
            + idemp_loss * idemp_loss_weight
            + state_changes_loss
        ).backward()
        for param_dict in optimizer.param_groups:
            for p in param_dict["params"]:
                torch.nn.utils.clip_grad_norm_(
                    p, max_norm=training_parameters["gradient_clipping"]
                )

        optimizer.step()

        if j % 50 == 0:
            print(
                f"epoch : {j} : disease_loss {disease_loss} aux_loss_cont : {aux_loss_cont_encoded}"
            )

        # MONITORING ON VALIDATION DATA
        if j % 10 == 0:
            with torch.no_grad():
                state_changes_loss_valid = 0
                disease_loss_valid = 0
                aux_loss_cat_encoded_valid = 0
                aux_loss_cont_encoded_valid = 0
                aux_loss_cat_unknown_valid = 0
                aux_loss_cont_unknown_valid = 0
                idemp_loss_valid = 0
                valid_state = initial_state(len(X_cont_valid))
                # non nan features
                (
                    non_nan_cont_features_valid,
                    non_nan_cont_patients_valid,
                ) = get_non_nan_features(X_cont_valid)
                (
                    non_nan_cat_features_valid,
                    non_nan_cat_patients_valid,
                ) = get_non_nan_features(X_cat_valid, continuous=False)

                for level in qst.valid_group_order_features.keys():
                    for feature_group in qst.valid_group_order_features[level].keys():
                        patients = qst.valid_group_order_features[level][feature_group]
                        if len(patients) > 0:
                            state_valid_before = valid_state.clone()
                            for feature_name in shuffle(feature_group):
                                if feature_name in qst.continuous_features_names:
                                    valid_state[patients, :] = encoders[feature_name](
                                        valid_state[patients, :],
                                        X_cont_valid[
                                            patients,
                                            qst.continuous_features_names.index(
                                                feature_name
                                            ),
                                        ].view(-1, 1),
                                    )

                                else:
                                    valid_state[patients, :] = encoders[feature_name](
                                        valid_state[patients, :],
                                        X_cat_valid[
                                            patients,
                                            qst.categorical_features_names.index(
                                                feature_name
                                            ),
                                        ].view(-1, 1),
                                    )
                                state_changes_loss_valid += mse(
                                    state_valid_before[patients, :],
                                    valid_state[patients, :],
                                )

                                for index, name in enumerate(qst.disease_names):
                                    disease_loss_valid += criterion(
                                        disease_decoders[name](valid_state),
                                        y_valid[:, index],
                                    )

                    for decode_level in qst.valid_group_order_features.keys():
                        for feature_group in qst.valid_group_order_features[
                            decode_level
                        ].keys():
                            # select patients for which feature gets encoded at that level
                            patients = qst.valid_group_order_features[decode_level][
                                feature_group
                            ]
                            if len(patients) > 0:
                                for feature_name in shuffle(feature_group):
                                    # features already have been encoded
                                    if decode_level <= level:
                                        # continuous feature
                                        if (
                                            feature_name
                                            in qst.continuous_features_names
                                        ):
                                            aux_loss_cont_encoded_valid += neg_log_likelihood_1D(
                                                X_cont_valid[
                                                    patients,
                                                    qst.continuous_features_names.index(
                                                        feature_name
                                                    ),
                                                ].view(-1, 1),
                                                *continuous_feature_decoders[
                                                    feature_name
                                                ](valid_state[patients, :]),
                                            )
                                        # categorical feature
                                        else:
                                            aux_loss_cat_encoded_valid += criterion(
                                                categorical_feature_decoders[
                                                    feature_name
                                                ](valid_state[patients, :]),
                                                X_cat_valid[
                                                    patients,
                                                    qst.categorical_features_names.index(
                                                        feature_name
                                                    ),
                                                ].long(),
                                            )
                                    # features havent been encoded
                                    else:
                                        if (
                                            feature_name
                                            in qst.continuous_features_names
                                        ):
                                            aux_loss_cont_unknown_valid += neg_log_likelihood_1D(
                                                X_cont_valid[
                                                    patients,
                                                    qst.continuous_features_names.index(
                                                        feature_name
                                                    ),
                                                ].view(-1, 1),
                                                *continuous_feature_decoders[
                                                    feature_name
                                                ](valid_state[patients, :]),
                                            )
                                        else:
                                            aux_loss_cat_unknown_valid += criterion(
                                                categorical_feature_decoders[
                                                    feature_name
                                                ](valid_state[patients, :]),
                                                X_cat_valid[
                                                    patients,
                                                    qst.categorical_features_names.index(
                                                        feature_name
                                                    ),
                                                ].long(),
                                            )
                # idempotence, re-encode features
                if idempotence:
                    state_before_idemp = valid_state.clone()
                    for feature in shuffle(qst.categorical_features_names):
                        index_of_feature = qst.categorical_features_names.index(feature)
                        patients = non_nan_cat_patients_valid[index_of_feature]
                        if len(patients) > 0:
                            valid_state[patients, :] = encoders[feature](
                                valid_state[patients, :],
                                X_cat_valid[patients, index_of_feature].view(-1, 1),
                            )
                            aux_loss_cat_encoded_valid += criterion(
                                categorical_feature_decoders[feature](
                                    valid_state[patients, :]
                                ),
                                X_cat_valid[patients, index_of_feature].long(),
                            )
                            idemp_loss_valid += mse(
                                state_before_idemp[patients, :],
                                valid_state[patients, :],
                            )
                            valid_state = state_before_idemp.clone()

                    for feature in shuffle(qst.continuous_features_names):
                        # get index of feature in list of features
                        index_of_feature = qst.continuous_features_names.index(feature)
                        patients = non_nan_cont_patients_valid[index_of_feature]
                        if len(patients) > 0:
                            valid_state[patients, :] = encoders[feature](
                                valid_state[patients, :],
                                X_cont_valid[patients, index_of_feature].view(-1, 1),
                            )
                            # aux_loss_cont_encoded_valid += mse(X_cont_valid[patients, index_of_feature].view(-1, 1),
                            #                                    continuous_feature_decoders[feature](valid_state[patients, :]))
                            aux_loss_cont_encoded_valid += neg_log_likelihood_1D(
                                X_cont_valid[patients, index_of_feature].view(-1, 1),
                                *continuous_feature_decoders[feature](
                                    valid_state[patients, :]
                                ),
                            )
                            idemp_loss_valid += mse(
                                state_before_idemp[patients, :],
                                valid_state[patients, :],
                            )
                            valid_state = state_before_idemp.clone()

                # loss per disease
                for name in qst.disease_names:
                    # double sum : we add up each disease decoder at each encoding step
                    wandb.log(
                        {
                            "valid_loss_dis_"
                            + str(name): criterion(
                                disease_decoders[name](valid_state),
                                y_valid[:, qst.disease_names.index(name)],
                            ),
                            "epoch": j,
                        }
                    )
                for index, name in enumerate(qst.continuous_features_names):
                    wandb.log(
                        {
                            "loss_cont_feat_valid"
                            + str(name): neg_log_likelihood_1D(
                                X_cont_valid[
                                    non_nan_cont_patients_valid[index], index
                                ].view(-1, 1),
                                *continuous_feature_decoders[name](
                                    valid_state[non_nan_cont_patients_valid[index], :]
                                ),
                            ),
                            "epoch": j,
                        }
                    )
                for index, name in enumerate(qst.categorical_features_names):
                    wandb.log(
                        {
                            "loss_cat_feat_valid"
                            + str(name): criterion(
                                categorical_feature_decoders[name](
                                    valid_state[non_nan_cat_patients_valid[index], :]
                                ),
                                X_cat_valid[
                                    non_nan_cat_patients_valid[index], index
                                ].long(),
                            ),
                            "epoch": j,
                        }
                    )
                # f1 scores
                for index, name in enumerate(qst.disease_names):
                    _, prediction = torch.max(disease_decoders[name](valid_state), 1)
                    f1score_class0 = f1_score(
                        y_valid[:, index], prediction, average=None, labels=[0, 1]
                    )[0]
                    f1score_class1 = f1_score(
                        y_valid[:, index], prediction, average=None, labels=[0, 1]
                    )[1]
                    wandb.log(
                        {
                            "epoch": j,
                            name + "_f1_class0_valid": f1score_class0,
                            name + "_f1_class1_valid": f1score_class1,
                        }
                    )

                wandb.log(
                    {
                        "epoch": j,
                        "disease_loss_valid": disease_loss_valid,
                        "aux_loss_cont_encoded_valid": aux_loss_cont_encoded_valid,
                        "aux_loss_cat_encoded_valid": aux_loss_cat_encoded_valid,
                        "aux_loss_cont_unknown_valid": aux_loss_cont_unknown_valid,
                        "aux_loss_cat_unknown_valid": aux_loss_cat_unknown_valid,
                        "idemp_loss_valid": idemp_loss_valid,
                        "state_changes_loss_valid": state_changes_loss_valid,
                    }
                )
    # save model
    if save:
        model = {
            "encoders": encoders,
            "cont_decoders": continuous_feature_decoders,
            "cat_decoders": categorical_feature_decoders,
            "dis_decoders": disease_decoders,
            "initial_state": initial_state,
        }
        torch.save(model, "trained_model_350_decoding.pt")
    time_stop = process_time()
    total_time = time_stop - time_start
    # wandb.log({'training time': total_time})
    return


def train_modules_without_decoding(
    qst,
    X_cont_train,
    y_train,
    X_cont_valid,
    y_valid,
    X_cat_train,
    X_cat_valid,
    initial_state,
    encoders,
    disease_decoders,
    encoders_to_apply,
    encoders_to_train=[],
    optimize_diseases=True,
    subset=None,
    save=False,
    seed=None,
    percentage=None,
):
    """
    Training process with no feature decoding
    :param qst: questionnaire object
    :param X_cont_train: continuous training features
    :param y_train: targets train
    :param X_cont_valid: continuous validation features
    :param y_valid: targets valid
    :param X_cat_train: categorical training features
    :param X_cat_valid: categorical validation features
    :param initial_state: initial state
    :param encoders: encoder modules
    :param disease_decoders: disease decoder modules
    :param encoders_to_apply: encoders to apply
    :param encoders_to_train: encoders we want to train
    :param optimize_diseases: train disease decoders
    :param subset: specific subset of qst to use (e.g. centralized or local) if any
    :param save: boolean to save the model
    :param seed: random seed
    :param percentage: fraction of centralized data
    :return:
    """

    # parameters for classical encoders and decoders
    training_parameters = qst.training_parameters
    lr_feature_decoders = training_parameters["lr_feature_decoders"]
    lr_encoders = training_parameters["lr_encoders"]
    lr_diseases_decoder = training_parameters["lr_diseases_decoder"]
    lr = training_parameters["lr"]
    decay_factor = training_parameters["decay_factor"]
    step_size = training_parameters["step_size"]
    n_epochs = training_parameters["n_epochs"]

    # batch_size = 500
    batch_size = training_parameters["batch_size"]
    aux_loss_encoded_weight = training_parameters["aux_loss_encoded_weight"]
    aux_loss_unknown_weight = training_parameters["aux_loss_unknown_weight"]
    diseases_loss_weight = training_parameters["diseases_loss_weight"]
    idemp_loss_weight = training_parameters["idemp_loss_weight"]

    # for logs

    wandb.init(
        project="stage_igh",
        config={
            "dataset_size": len(qst),
            "epochs": n_epochs,
            "lr_encoders": lr_encoders,
            "lr_diseases_decoder": lr_diseases_decoder,
            "lr_feature_decoders": lr_feature_decoders,
            "STATE_SIZE": STATE_SIZE,
            "batch_size": batch_size,
            "aux_loss_encoded_weight": aux_loss_encoded_weight,
            "aux_loss_unknown_weight": aux_loss_unknown_weight,
            "disease_loss_weight": diseases_loss_weight,
        },
    )

    # loss for classification
    criterion = nn.NLLLoss()

    print(f"\n Loss :")
    time_start = process_time()

    # get the optimizer for the parameters we train in that level
    optimizer = get_optimizer(
        encoders_to_train,
        initial_state,
        encoders,
        disease_decoders,
        lr_encoders,
        lr_diseases_decoder,
        lr,
        diseases=optimize_diseases,
    )

    # learning rate decay
    scheduler = StepLR(optimizer, step_size=step_size, gamma=decay_factor)

    for j in range(n_epochs):
        scheduler.step()
        # to store the sum of all the disease decoding losses after each encoding step
        disease_loss = 0
        # select random batch of indices (without replacement)
        # state_changes_loss
        state_changes_loss = 0
        if subset is None:
            batch_indices = np.random.choice(
                range(len(X_cont_train)), batch_size, replace=False
            )
        elif subset == "specific":
            batch_indices = np.random.choice(
                qst.specific_features_train, batch_size, replace=False
            )
        elif subset == "common":
            batch_indices = np.random.choice(
                qst.common_features_train, batch_size, replace=False
            )
        # init state
        state = initial_state(len(X_cont_train))
        # decode all diseases
        for name in qst.disease_names:
            disease_loss += criterion(
                disease_decoders[name](state[batch_indices, :]),
                y_train[batch_indices, qst.disease_names.index(name)],
            )
        # iterate over levels in tree
        for level in qst.train_group_order_features.keys():
            # apply encoders
            for feature_group in shuffle(qst.train_group_order_features[level].keys()):
                # select patients
                patients = np.intersect1d(
                    qst.train_group_order_features[level][feature_group], batch_indices
                )
                if len(patients) > 0:
                    # apply shuffled encoders (we shuffle encoders in same level in tree)
                    for feature_name in shuffle(feature_group):
                        if feature_name in encoders_to_apply:
                            state_before = state.clone()
                            # apply continuous feature encoders
                            if feature_name in qst.continuous_features_names:
                                state[patients, :] = encoders[feature_name](
                                    state[patients, :],
                                    X_cont_train[
                                        patients,
                                        qst.continuous_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            else:
                                # apply categorical feature encoders
                                state[patients, :] = encoders[feature_name](
                                    state[patients, :],
                                    X_cat_train[
                                        patients,
                                        qst.categorical_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            # compute state_change_loss
                            state_changes_loss += mse(
                                state[patients, :], state_before[patients, :]
                            )

                            # after having encoded each feature in the tree, we try to predict the diseases
                            for name in qst.disease_names:
                                disease_loss += criterion(
                                    disease_decoders[name](state[batch_indices, :]),
                                    y_train[
                                        batch_indices, qst.disease_names.index(name)
                                    ],
                                )
                        else:
                            pass

        wandb.log(
            {
                "epoch": j,
                "disease_loss": disease_loss,
                "state_changes_loss": state_changes_loss,
            }
        )
        # Differentiate loss and make an optimizer step
        optimizer.zero_grad()
        (disease_loss + state_changes_loss).backward()
        for param_dict in optimizer.param_groups:
            for p in param_dict["params"]:
                torch.nn.utils.clip_grad_norm_(
                    p, max_norm=training_parameters["gradient_clipping"]
                )
                # param_norm = p.grad.data.norm(2)
                # if param_norm > 5000 :
                #     print(f'epoch : {j} param norm {param_norm.data}')

        optimizer.step()

        if j % 50 == 0:
            print(f"epoch : {j} : disease_loss {disease_loss}")

        # MONITORING ON VALIDATION DATA
        if j % 10 == 0:
            with torch.no_grad():
                state_changes_loss_valid = 0
                disease_loss_valid = 0
                valid_state = initial_state(len(X_cont_valid))

                for level in qst.valid_group_order_features.keys():
                    for feature_group in qst.valid_group_order_features[level].keys():
                        patients = qst.valid_group_order_features[level][feature_group]
                        if len(patients) > 0:
                            state_valid_before = valid_state.clone()
                            for feature_name in feature_group:
                                if feature_name in encoders_to_apply:
                                    if feature_name in qst.continuous_features_names:
                                        valid_state[patients, :] = encoders[
                                            feature_name
                                        ](
                                            valid_state[patients, :],
                                            X_cont_valid[
                                                patients,
                                                qst.continuous_features_names.index(
                                                    feature_name
                                                ),
                                            ].view(-1, 1),
                                        )

                                    else:
                                        valid_state[patients, :] = encoders[
                                            feature_name
                                        ](
                                            valid_state[patients, :],
                                            X_cat_valid[
                                                patients,
                                                qst.categorical_features_names.index(
                                                    feature_name
                                                ),
                                            ].view(-1, 1),
                                        )
                                    state_changes_loss_valid += mse(
                                        state_valid_before[patients, :],
                                        valid_state[patients, :],
                                    )

                                    for index, name in enumerate(qst.disease_names):
                                        disease_loss_valid += criterion(
                                            disease_decoders[name](valid_state),
                                            y_valid[:, index],
                                        )
                                else:
                                    pass

                # loss per disease
                for name in qst.disease_names:
                    # double sum : we add up each disease decoder at each encoding step
                    wandb.log(
                        {
                            "valid_loss_dis_"
                            + str(name): criterion(
                                disease_decoders[name](valid_state),
                                y_valid[:, qst.disease_names.index(name)],
                            ),
                            "epoch": j,
                            "state_changes_loss_valid": state_changes_loss_valid,
                        }
                    )

                # f1 scores
                for index, name in enumerate(qst.disease_names):
                    _, prediction = torch.max(disease_decoders[name](valid_state), 1)
                    f1score_class0 = f1_score(
                        y_valid[:, index], prediction, average=None, labels=[0, 1]
                    )[0]
                    f1score_class1 = f1_score(
                        y_valid[:, index], prediction, average=None, labels=[0, 1]
                    )[1]
                    wandb.log(
                        {
                            "epoch": j,
                            name + "_f1_class0_valid": f1score_class0,
                            name + "_f1_class1_valid": f1score_class1,
                        }
                    )

                wandb.log({"epoch": j, "disease_loss_valid": disease_loss_valid})
    # save model for distributed learning common
    if save:
        model = {
            "encoders": encoders,
            "dis_decoders": disease_decoders,
            "initial_state": initial_state,
        }
        torch.save(
            model,
            "updated_centralized/central_" + str(percentage) + str(seed) + ".pt",
        )
        # torch.save(
        #     model, 'model.pt')
    time_stop = process_time()
    total_time = time_stop - time_start
    wandb.log({"training time": total_time})
    return


def train_modules_fine_tuning(
    qst,
    X_cont_train,
    y_train,
    X_cont_valid,
    y_valid,
    X_cat_train,
    X_cat_valid,
    initial_state,
    encoders,
    disease_decoders,
    encoders_to_train,
    all_encoders,
):
    """
    Simplified training process with no feature decoding
    :param qst: questionnaire object
    :param X_cont_train: continuous training features
    :param y_train: targets train
    :param X_cont_valid: continuous validation features
    :param y_valid: targets valid
    :param X_cat_train: categorical training features
    :param X_cat_valid: categorical validation features
    :param initial_state: initial state
    :param encoders: encoder modules
    :param disease_decoders: disease decoder modules
    :param encoders_to_train: encoders we want to train (usually all of them)
    :param all_encoders: all the encoders
    :return:
    """
    # training loop
    def training_loop(start, stop):
        # learning rate decay
        scheduler = StepLR(optimizer, step_size=step_size, gamma=decay_factor)
        for j in range(start, stop):
            scheduler.step()
            # to store the sum of all the disease decoding losses after each encoding step
            disease_loss = 0
            # select random batch of indices (without replacement)
            # state_changes_loss
            state_changes_loss = 0
            batch_indices = np.random.choice(
                qst.specific_features_train, batch_size, replace=False
            )
            # init state
            state = initial_state(len(X_cont_train))
            # decode all diseases
            for name in qst.disease_names:
                disease_loss += criterion(
                    disease_decoders[name](state[batch_indices, :]),
                    y_train[batch_indices, qst.disease_names.index(name)],
                )
            # iterate over levels in tree
            for level in qst.train_group_order_features.keys():
                # apply encoders
                for feature_group in shuffle(
                    qst.train_group_order_features[level].keys()
                ):
                    # select patients
                    patients = np.intersect1d(
                        qst.train_group_order_features[level][feature_group],
                        batch_indices,
                    )
                    if len(patients) > 0:
                        # apply shuffled encoders (we shuffle encoders in same level in tree)
                        for feature_name in shuffle(feature_group):
                            state_before = state.clone()
                            # apply continuous feature encoders
                            if feature_name in qst.continuous_features_names:
                                state[patients, :] = encoders[feature_name](
                                    state[patients, :],
                                    X_cont_train[
                                        patients,
                                        qst.continuous_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            else:
                                # apply categorical feature encoders
                                state[patients, :] = encoders[feature_name](
                                    state[patients, :],
                                    X_cat_train[
                                        patients,
                                        qst.categorical_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            # compute state_change_loss
                            state_changes_loss += mse(
                                state[patients, :], state_before[patients, :]
                            )

                            # after having encoded each feature in the tree, we try to predict the diseases
                            for name in qst.disease_names:
                                disease_loss += criterion(
                                    disease_decoders[name](state[batch_indices, :]),
                                    y_train[
                                        batch_indices, qst.disease_names.index(name)
                                    ],
                                )

            wandb.log(
                {
                    "epoch": j,
                    "disease_loss": disease_loss,
                    "state_changes_loss": state_changes_loss,
                }
            )

            # Differentiate loss and make an optimizer step
            optimizer.zero_grad()
            (disease_loss + state_changes_loss).backward()
            for param_dict in optimizer.param_groups:
                for p in param_dict["params"]:
                    torch.nn.utils.clip_grad_norm_(
                        p, max_norm=training_parameters["gradient_clipping"]
                    )
                    # param_norm = p.grad.data.norm(2)
                    # if param_norm > 5000 :
                    #     print(f'epoch : {j} param norm {param_norm.data}')

            optimizer.step()

            # MONITORING ON VALIDATION DATA
            if j % 10 == 0:
                with torch.no_grad():
                    state_changes_loss_valid = 0
                    disease_loss_valid = 0
                    valid_state = initial_state(len(X_cont_valid))

                    for level in qst.valid_group_order_features.keys():
                        for feature_group in qst.valid_group_order_features[
                            level
                        ].keys():
                            patients = qst.valid_group_order_features[level][
                                feature_group
                            ]
                            if len(patients) > 0:
                                state_valid_before = valid_state.clone()
                                for feature_name in feature_group:
                                    if feature_name in qst.continuous_features_names:
                                        valid_state[patients, :] = encoders[
                                            feature_name
                                        ](
                                            valid_state[patients, :],
                                            X_cont_valid[
                                                patients,
                                                qst.continuous_features_names.index(
                                                    feature_name
                                                ),
                                            ].view(-1, 1),
                                        )

                                    else:
                                        valid_state[patients, :] = encoders[
                                            feature_name
                                        ](
                                            valid_state[patients, :],
                                            X_cat_valid[
                                                patients,
                                                qst.categorical_features_names.index(
                                                    feature_name
                                                ),
                                            ].view(-1, 1),
                                        )
                                    state_changes_loss_valid += mse(
                                        state_valid_before[patients, :],
                                        valid_state[patients, :],
                                    )

                                    for index, name in enumerate(qst.disease_names):
                                        disease_loss_valid += criterion(
                                            disease_decoders[name](valid_state),
                                            y_valid[:, index],
                                        )

                    # loss per disease
                    for name in qst.disease_names:
                        # double sum : we add up each disease decoder at each encoding step
                        wandb.log(
                            {
                                "valid_loss_dis_"
                                + str(name): criterion(
                                    disease_decoders[name](valid_state),
                                    y_valid[:, qst.disease_names.index(name)],
                                ),
                                "epoch": j,
                                "state_changes_loss_valid": state_changes_loss_valid,
                            }
                        )

                    # f1 scores
                    for index, name in enumerate(qst.disease_names):
                        _, prediction = torch.max(
                            disease_decoders[name](valid_state), 1
                        )
                        f1score_class0 = f1_score(
                            y_valid[:, index], prediction, average=None, labels=[0, 1]
                        )[0]
                        f1score_class1 = f1_score(
                            y_valid[:, index], prediction, average=None, labels=[0, 1]
                        )[1]
                        wandb.log(
                            {
                                "epoch": j,
                                name + "_f1_class0_valid": f1score_class0,
                                name + "_f1_class1_valid": f1score_class1,
                            }
                        )

                    wandb.log({"epoch": j, "disease_loss_valid": disease_loss_valid})

        return

    # parameters for classical encoders and decoders
    training_parameters = qst.training_parameters
    lr_feature_decoders = training_parameters["lr_feature_decoders"]
    lr_encoders = training_parameters["lr_encoders"]
    lr_diseases_decoder = training_parameters["lr_diseases_decoder"]
    lr = training_parameters["lr"]
    decay_factor = training_parameters["decay_factor"]
    step_size = training_parameters["step_size"]
    n_epochs = training_parameters["n_epochs"]

    # batch_size = 500
    batch_size = training_parameters["batch_size"]
    aux_loss_encoded_weight = training_parameters["aux_loss_encoded_weight"]
    aux_loss_unknown_weight = training_parameters["aux_loss_unknown_weight"]
    diseases_loss_weight = training_parameters["diseases_loss_weight"]
    idemp_loss_weight = training_parameters["idemp_loss_weight"]

    # for logs

    wandb.init(
        project="stage_igh",
        config={
            "dataset_size": len(qst),
            "epochs": n_epochs,
            "lr_encoders": lr_encoders,
            "lr_diseases_decoder": lr_diseases_decoder,
            "lr_feature_decoders": lr_feature_decoders,
            "STATE_SIZE": STATE_SIZE,
            "batch_size": batch_size,
            "aux_loss_encoded_weight": aux_loss_encoded_weight,
            "aux_loss_unknown_weight": aux_loss_unknown_weight,
            "disease_loss_weight": diseases_loss_weight,
        },
    )

    # loss for classification
    criterion = nn.NLLLoss()

    print(f"\n Loss :")
    time_start = process_time()

    # get the optimizer for the new modules
    optimizer = get_optimizer(
        encoders_to_train,
        initial_state,
        encoders,
        disease_decoders,
        lr_encoders,
        lr_diseases_decoder,
        lr,
        diseases=False,
    )

    # train only new modules in that loop

    training_loop(start=0, stop=qst.training_parameters["n_epochs"][0])
    # add new modules to optimizer
    for elem in all_encoders:
        if elem not in encoders_to_train:
            optimizer.param_groups.append(
                {
                    "params": encoders[elem].parameters(),
                    "lr": lr_encoders[elem],
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                }
            )
    for disease in qst.disease_names:
        optimizer.param_groups.append(
            {
                "params": disease_decoders[disease].parameters(),
                "lr": lr_diseases_decoder,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False,
            }
        )

    training_loop(
        start=qst.training_parameters["n_epochs"][0] + 1,
        stop=qst.training_parameters["n_epochs"][1],
    )

    return


def valid_diseases_decoders(
    qst, X_cont_test, X_cat_test, y_test, initial_state, encoders, diseases_decoders
):
    """
    validate disease decoders on unseen test set
    :param qst: questionnaire data set object
    :param X_cont_test: tensor with continuous features of test data
    :param X_cat_test: tensor with categorical features of test data
    :param y_test: tensor with targets of test data
    :param initial_state: initial state
    :param encoders: trained encoders
    :param diseases_decoders: trained disease decoder
    :return:
    """

    seed = 0
    random.seed(seed)
    with torch.no_grad():
        test_state = torch.tile(initial_state.state_value, (len(X_cont_test), 1))
        # save states after each level
        test_state_dict = {}
        # arrays to store metrics
        predicted_diseases = torch.empty((len(y_test), len(diseases_decoders)))
        decoder_output = torch.empty(
            (len(qst.test_group_order_features), len(y_test), len(qst.disease_names))
        )
        calibrations = {}
        ECEs = {}
        # apply trained encoders for each level in the tree
        for level in qst.test_group_order_features.keys():
            calibrations[level] = {}
            ECEs[level] = {}
            for feature_group in qst.test_group_order_features[level].keys():
                patients = qst.test_group_order_features[level][feature_group]
                for feature_name in shuffle(feature_group):
                    if feature_name in (list(encoders.keys())):
                        if len(patients) > 0:
                            if feature_name in qst.continuous_features_names:
                                test_state[patients, :] = encoders[feature_name](
                                    test_state[patients, :],
                                    X_cont_test[
                                        patients,
                                        qst.continuous_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            else:
                                test_state[patients, :] = encoders[feature_name](
                                    test_state[patients, :],
                                    X_cat_test[
                                        patients,
                                        qst.categorical_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
            test_state_dict[level] = test_state.clone()

            # predict diseases after encoding of each level
            for index, name in enumerate(qst.disease_names):
                disease_pred = diseases_decoders[name](test_state)
                _, predicted_diseases[:, index] = torch.max(disease_pred, 1)
                decoder_output[level, :, index] = torch.exp(disease_pred[:, 1])
                # calibration
                n_bins = 13
                accuracies, confidences, obs_per_bin, ECE = calibration(
                    qst, name, level, decoder_output, y_test, n_bins
                )
                calibrations[level][name] = (accuracies, confidences, obs_per_bin)
                ECEs[level][name] = ECE
            accuracies, confidences, obs_per_bin, ECE = calibration(
                qst, "all", level, decoder_output, y_test, n_bins
            )
            ECEs[level]["all"] = ECE

        plot_calibration(accuracies, confidences)

        get_heatmaps_epoct(
            qst,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            diseases_decoders,
        )

        # f1 scores

        print("f1 scores on whole test set")
        for index, name in enumerate(qst.disease_names):
            _, prediction = torch.max(diseases_decoders[name](test_state), 1)
            f1score_class0 = f1_score(
                y_test[:, index], prediction, average=None, labels=[0, 1]
            )[0]
            f1score_class1 = f1_score(
                y_test[:, index], prediction, average=None, labels=[0, 1]
            )[1]
            wandb.log(
                {
                    name + "_f1_class0_test": f1score_class0,
                    name + "_f1_class1_test": f1score_class1,
                }
            )
        # decode diseases once all features have been encoded for patients in batch
        results = pd.DataFrame(
            columns=["Disease", "Class", "Precision", "Recall", "F_1", "Num_samples"]
        )
        aggregated_results = pd.DataFrame(
            columns=["Disease", "Macro Precision", "Macro Recall", "Macro F_1"]
        )
        f1_macro_all = 0
        for index, name in enumerate(qst.disease_names):
            _, prediction = torch.max(diseases_decoders[name](test_state), 1)
            (
                precision_0,
                recall_0,
                F1_0,
                precision_1,
                recall_1,
                F1_1,
                F1_average,
            ) = compute_metrics(prediction, y_test[:, index])
            print(f"Disease {name}, f1_all : {(F1_0 + F1_1)/2}")
            f1_macro_all += (F1_0 + F1_1) / (2 * len(qst.disease_names))
            results = results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Class": 0,
                        "Precision": np.around(precision_0, decimals=2),
                        "Recall": np.around(recall_0, decimals=2),
                        "F_1": np.around(F1_0, decimals=2),
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 0
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            results = results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Class": 1,
                        "Precision": np.around(precision_1, decimals=2),
                        "Recall": np.around(recall_1, decimals=2),
                        "F_1": np.around(F1_1, decimals=2),
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 1
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            aggregated_results = aggregated_results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Macro Precision": np.around(
                            (precision_0 + precision_1) / 2, decimals=2
                        ),
                        "Macro Recall": np.around(
                            (recall_0 + recall_1) / 2, decimals=2
                        ),
                        "Macro F_1": np.around((F1_1 + F1_0) / 2, decimals=2),
                    }
                ),
                ignore_index=True,
            )
        print(f"F1 macro all : {f1_macro_all}")
        aggr_results = wandb.Table(dataframe=aggregated_results)
        all_results = wandb.Table(dataframe=results)
        wandb.log({"results_dataset (whole)": aggr_results})
        wandb.log({"results_data (whole)": all_results})
        f1 = pd.Series(
            {
                "Macro_F1": results["F_1"].sum() / len(results),
                "Weighted_F1": (results["F_1"] * results["Num_samples"]).sum()
                / results["Num_samples"].sum(),
            }
        )
        wandb.log(
            {
                "f1_macro": results["F_1"].sum() / len(results),
                "f1_weighted": (results["F_1"] * results["Num_samples"]).sum()
                / results["Num_samples"].sum(),
            }
        )
        print(results)
        latex_table_from_df(results)
        print(aggregated_results)
        latex_table_from_df(aggregated_results)
        print(f1)
        print(f1.to_latex())
        print(ECEs)

    return results["F_1"].sum() / len(results), aggregated_results


def valid_diseases_decoders_distributed(
    qst,
    X_cont_test,
    X_cat_test,
    y_test,
    initial_state,
    encoders,
    diseases_decoders,
    encoders_to_apply,
):
    """
    Use disease and question decoders on unseen test set
    :param qst: questionnaire data set object
    :param X_cont_test: tensor with continuous features of test data
    :param X_cat_test: tensor with categorical features of test data
    :param y_test: tensor with targets of test data
    :param initial_state: initial state
    :param encoders: dict of trained encoders
    :param diseases_decoders: trained disease decoder
    :param encoders_to_apply: encoders to apply
    :return:
    """

    seed = 0
    random.seed(seed)
    with torch.no_grad():
        test_state = torch.tile(initial_state.state_value, (len(X_cont_test), 1))
        print(initial_state.state_value.sum())
        # save states after each level
        test_state_dict = {}
        # apply trained encoders for each level in the tree
        for level in qst.test_group_order_features.keys():
            for feature_group in qst.test_group_order_features[level].keys():
                patients = qst.test_group_order_features[level][feature_group]
                for feature_name in shuffle(feature_group):
                    if feature_name in encoders_to_apply:
                        if len(patients) > 0:
                            if feature_name in qst.continuous_features_names:
                                test_state[patients, :] = encoders[feature_name](
                                    test_state[patients, :],
                                    X_cont_test[
                                        patients,
                                        qst.continuous_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
                            else:
                                test_state[patients, :] = encoders[feature_name](
                                    test_state[patients, :],
                                    X_cat_test[
                                        patients,
                                        qst.categorical_features_names.index(
                                            feature_name
                                        ),
                                    ].view(-1, 1),
                                )
            test_state_dict[level] = test_state.clone()

        # decode diseases once all features have been encoded for patients in batch
        results = pd.DataFrame(
            columns=["Disease", "Class", "Precision", "Recall", "F_1", "Num_samples"]
        )
        aggregated_results = pd.DataFrame(
            columns=["Disease", "Macro Precision", "Macro Recall", "Macro F_1"]
        )
        for index, name in enumerate(qst.disease_names):
            _, prediction = torch.max(diseases_decoders[name](test_state), 1)
            (
                precision_0,
                recall_0,
                F1_0,
                precision_1,
                recall_1,
                F1_1,
                F1_average,
            ) = compute_metrics(prediction, y_test[:, index])
            results = results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Class": 0,
                        "Precision": np.around(precision_0, decimals=2),
                        "Recall": np.around(recall_0, decimals=2),
                        "F_1": np.around(F1_0, decimals=2),
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 0
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            results = results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Class": 1,
                        "Precision": np.around(precision_1, decimals=2),
                        "Recall": np.around(recall_1, decimals=2),
                        "F_1": np.around(F1_1, decimals=2),
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 1
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            aggregated_results = aggregated_results.append(
                pd.Series(
                    {
                        "Disease": name,
                        "Macro Precision": np.around(
                            (precision_0 + precision_1) / 2, decimals=2
                        ),
                        "Macro Recall": np.around(
                            (recall_0 + recall_1) / 2, decimals=2
                        ),
                        "Macro F_1": np.around((F1_1 + F1_0) / 2, decimals=2),
                    }
                ),
                ignore_index=True,
            )
        aggr_results = wandb.Table(dataframe=aggregated_results)
        all_results = wandb.Table(dataframe=results)
        wandb.log({"results_dataset (whole)": aggr_results})
        wandb.log({"results_data (whole)": all_results})
        f1 = pd.Series(
            {
                "Macro_F1": results["F_1"].sum() / len(results),
                "Weighted_F1": (results["F_1"] * results["Num_samples"]).sum()
                / results["Num_samples"].sum(),
            }
        )
        wandb.log(
            {
                "f1_macro": results["F_1"].sum() / len(results),
                "f1_weighted": (results["F_1"] * results["Num_samples"]).sum()
                / results["Num_samples"].sum(),
            }
        )
        print(results)
        latex_table_from_df(results)
        print(aggregated_results)
        latex_table_from_df(aggregated_results)
        print(f1)
        print(f1.to_latex())

    return results["F_1"].sum() / len(results)


def get_heatmaps_epoct(
    qst, X_cont_test, X_cat_test, y_test, initial_state, encoders, diseases_decoders
):
    """
    get heatmaps for some patients
    :param qst: questionnaire data set object
    :param X_cont_test: tensor with continuous features of test data
    :param X_cat_test: tensor with categorical features of test data
    :param y_test: tensor with targets of test data
    :param initial_state: initial state
    :param encoders: trained encoders
    :param diseases_decoders: trained disease decoders
    :return:
    """

    seed = 0
    random.seed(seed)
    for file in os.scandir("heatmaps"):
        os.remove(file.path)
    with torch.no_grad():
        patients = list(np.random.choice(range(len(y_test)), 40, replace=False))
        for patient_id in [568, 472, 340]:
            if patient_id not in patients:
                patients.append(patient_id)

        for index, patient in enumerate(patients):
            test_state = torch.tile(initial_state.state_value, (1, 1))
            # save states after each level
            test_state_dict = {}
            # arrays to store metrics
            predicted_diseases = torch.empty((1, len(diseases_decoders)))
            decoder_output = torch.empty(
                (
                    qst.num_available_features[
                        qst.testing_data_indices_reverse_mapping[patient]
                    ]
                    + 1,
                    len(qst.disease_names),
                )
            )
            feature_rank = 1
            list_features = ["No information"]
            # before anything has been encoded
            for index, name in enumerate(qst.disease_names):
                disease_pred = diseases_decoders[name](test_state)
                decoder_output[0, index] = torch.exp(disease_pred[:, 1])
            # apply trained encoders for each level in the tree
            for level in qst.order_test_features[patient].keys():
                feature_group = shuffle(qst.order_test_features[patient][level])
                for feature_name in feature_group:
                    if feature_name in ["lab_urine_type_d0", "sympcc_dysuria"]:
                        continue
                    if feature_name in qst.continuous_features_names:
                        test_state = encoders[feature_name](
                            test_state,
                            X_cont_test[
                                patient,
                                qst.continuous_features_names.index(feature_name),
                            ].view(-1, 1),
                        )
                    else:
                        test_state = encoders[feature_name](
                            test_state,
                            X_cat_test[
                                patient,
                                qst.categorical_features_names.index(feature_name),
                            ].view(-1, 1),
                        )

                    for index, name in enumerate(qst.disease_names):
                        disease_pred = diseases_decoders[name](test_state)
                        decoder_output[feature_rank, index] = torch.exp(
                            disease_pred[:, 1]
                        )
                    feature_rank += 1
                    list_features.append(feature_name)

            fig = plot_heatmap_after_encoder_epoct_all_steps(
                decoder_output, y_test, patient, list_features, qst
            )

    return


def train_and_test_modules(qst_obj, folds, reload=True, feature_decoding=False):
    """
    Trains and tests encoder/decoder modules on epoct data
    :param qst_obj : questionnaire object dataset
    :param folds: train/test splits for 5foldcv
    :param feature_decoding: boolean, whether to train with or without feature decoding

    """
    macro_f1 = np.empty((5, 2))
    aggregated_results = {}
    if folds is None:
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
            mean_cont,
            std_cont,
            min_cont,
            max_cont,
        ) = preprocess_data_epoct(
            qst_obj,
            valid_size=0.2,
            test_size=0.2,
            tensors=True,
            imput_strategy=None,
            fold=None,
        )
        # uncomment to save qst_obj
        # torch.save(qst_obj, 'saved_objects/qst_epoct.pt')

        if reload:
            # decoding
            if feature_decoding:
                model_dict = torch.load(
                    "saved_objects/model_decoding.pt",
                    map_location=torch.device("cpu"),
                )
                continuous_feature_decoders = model_dict["cont_decoders"]
                categorical_feature_decoders = model_dict["cat_decoders"]
            else:
                model_dict = torch.load(
                    "saved_objects/model.pt", map_location=torch.device("cpu")
                )
            encoders = model_dict["encoders"]
            single_disease_decoders = model_dict["dis_decoders"]
            initial_state = model_dict["initial_state"]
            wandb.init(
                project="stage_igh",
                config={
                    "dataset_size": len(qst_obj),
                    "num_continuous_features": qst_obj.num_continuous_features,
                },
            )

        else:
            save = False
            initial_state = InitState(STATE_SIZE)
            encoders = get_feature_encoders(qst_obj, STATE_SIZE)
            (
                continuous_feature_decoders,
                categorical_feature_decoders,
            ) = get_feature_decoders(qst_obj, STATE_SIZE)
            single_disease_decoders = get_binary_disease_decoders(qst_obj, STATE_SIZE)
            initial_state = InitState(STATE_SIZE)
            if feature_decoding:
                train_modules_with_decoding(
                    qst_obj,
                    X_cont_train,
                    y_train,
                    X_cont_valid,
                    y_valid,
                    X_cat_train,
                    X_cat_valid,
                    initial_state,
                    encoders,
                    continuous_feature_decoders,
                    categorical_feature_decoders,
                    single_disease_decoders,
                    save=save,
                )
            else:
                train_modules_without_decoding(
                    qst_obj,
                    X_cont_train,
                    y_train,
                    X_cont_valid,
                    y_valid,
                    X_cat_train,
                    X_cat_valid,
                    initial_state,
                    encoders,
                    single_disease_decoders,
                    list(encoders.keys()),
                    list(encoders.keys()),
                    optimize_diseases=True,
                    save=save,
                )

        macro_f1, aggregated_results = valid_diseases_decoders(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
        )
    # 5x2foldcv
    else:
        for j in range(5):
            print(f"fold {j}")
            fold = folds[j]
            aggregated_results[j] = {}
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
                    mean_cont,
                    std_cont,
                    min_cont,
                    max_cont,
                ) = preprocess_data_epoct(
                    qst_obj,
                    valid_size=0.2,
                    test_size=0.2,
                    tensors=True,
                    imput_strategy=None,
                    fold=fold,
                )
                print(f"length of training data {len(X_train)}")
                qst_obj.training_parameters["batch_size"] = len(X_train)

                # save final model
                save = False
                initial_state = InitState(STATE_SIZE)
                encoders = get_feature_encoders(qst_obj, STATE_SIZE)
                (
                    continuous_feature_decoders,
                    categorical_feature_decoders,
                ) = get_feature_decoders(qst_obj, STATE_SIZE)
                single_disease_decoders = get_binary_disease_decoders(
                    qst_obj, STATE_SIZE
                )
                initial_state = InitState(STATE_SIZE)
                if feature_decoding:
                    train_modules_with_decoding(
                        qst_obj,
                        X_cont_train,
                        y_train,
                        X_cont_valid,
                        y_valid,
                        X_cat_train,
                        X_cat_valid,
                        initial_state,
                        encoders,
                        continuous_feature_decoders,
                        categorical_feature_decoders,
                        single_disease_decoders,
                        save=save,
                    )
                else:
                    train_modules_without_decoding(
                        qst_obj,
                        X_cont_train,
                        y_train,
                        X_cont_valid,
                        y_valid,
                        X_cat_train,
                        X_cat_valid,
                        initial_state,
                        encoders,
                        single_disease_decoders,
                        list(encoders.keys()),
                        list(encoders.keys()),
                        optimize_diseases=True,
                        save=save,
                    )

                # # apply trained modules to unseen data to test the disease decoders
                (macro_f1[j, k], aggregated_results[j][k],) = valid_diseases_decoders(
                    qst_obj,
                    X_cont_test,
                    X_cat_test,
                    y_test,
                    initial_state,
                    encoders,
                    single_disease_decoders,
                )

    return macro_f1, aggregated_results


def train_and_test_modules_distributed(
    qst_obj,
    X_cont_train,
    X_cont_valid,
    X_cont_test,
    X_cat_train,
    X_cat_valid,
    X_cat_test,
    y_train,
    y_valid,
    y_test,
    percentage=80,
    method="upper-baseline",
    seed=None,
):
    """train and test modules for iio experiments

    Args:
        qst_obj: questionnaire object
        X_cont_train : training tensor continuous features
        X_cont_valid : valid tensor continuous features
        X_cont_test : test tensor continuous features
        X_cat_train : train tensor categorical features
        X_cat_valid : valid tensor categorical features
        X_cat_test : test tensor categorical features
        y_train : train targets
        y_valid : valid targets
        y_test : test targets
        percentage (int, optional): percentage of centralized data. Defaults to 80.
        method (str, optional): which method for iio experiment. Defaults to "upper-baseline".
        seed (_type_, optional): seed. Defaults to None.

    Returns:
        _type_: _description_
    """
    if method == "upper-baseline":
        print("upper baseline")
        qst_obj.training_parameters["batch_size"] = len(X_cont_train)
        save = False

        initial_state = InitState(STATE_SIZE)
        encoders = get_feature_encoders(qst_obj, STATE_SIZE)
        single_disease_decoders = get_binary_disease_decoders(qst_obj, STATE_SIZE)
        train_modules_without_decoding(
            qst_obj,
            X_cont_train,
            y_train,
            X_cont_valid,
            y_valid,
            X_cat_train,
            X_cat_valid,
            initial_state,
            encoders,
            single_disease_decoders,
            encoders_to_apply=qst_obj.feature_names,
            encoders_to_train=qst_obj.feature_names,
            optimize_diseases=True,
            subset=None,
            save=save,
            seed=seed,
        )

        f1_macro = valid_diseases_decoders_distributed(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.feature_names,
        )

    elif method == "fine-tune":
        print("fine tuning")
        qst_obj.training_parameters["batch_size"] = len(qst_obj.specific_features_train)
        encoders = get_feature_encoders(qst_obj, STATE_SIZE)
        model_dict = torch.load(
            "updated_centralized/central" + str(percentage) + str(seed) + ".pt",
            map_location=torch.device("cpu"),
        )
        trained_encoders = model_dict["encoders"]
        single_disease_decoders = model_dict["dis_decoders"]
        initial_state = model_dict["initial_state"]
        # load pre-trained encoders
        for feature in qst_obj.common_features:
            encoders[feature] = trained_encoders[feature]
        print(f"new features {qst_obj.specific_features}")

        wandb.init(
            project="stage_igh",
            config={
                "dataset_size": len(qst_obj),
                "num_continuous_features": qst_obj.num_continuous_features,
            },
        )
        # add training subset
        train_modules_fine_tuning(
            qst_obj,
            X_cont_train,
            y_train,
            X_cont_valid,
            y_valid,
            X_cat_train,
            X_cat_valid,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.specific_features,
            all_encoders=qst_obj.feature_names,
        )

        f1_macro = valid_diseases_decoders_distributed(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.feature_names,
        )
    elif method == "compartm":
        print("Compartmentalization")
        qst_obj.training_parameters["batch_size"] = len(qst_obj.specific_features_train)
        encoders = get_feature_encoders(qst_obj, STATE_SIZE)
        model_dict = torch.load(
            "updated_centralized/central_" + str(percentage) + str(seed) + ".pt",
            map_location=torch.device("cpu"),
        )
        trained_encoders = model_dict["encoders"]
        single_disease_decoders = model_dict["dis_decoders"]
        initial_state = model_dict["initial_state"]
        # load pre-trained encoders
        for feature in qst_obj.common_features:
            encoders[feature] = trained_encoders[feature]
        print(f"new features {qst_obj.specific_features}")
        wandb.init(
            project="stage_igh",
            config={
                "dataset_size": len(qst_obj),
                "num_continuous_features": qst_obj.num_continuous_features,
            },
        )
        # train only new modules
        train_modules_without_decoding(
            qst_obj,
            X_cont_train,
            y_train,
            X_cont_valid,
            y_valid,
            X_cat_train,
            X_cat_valid,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.feature_names,
            qst_obj.specific_features,
            optimize_diseases=False,
            subset="specific",
            save=False,
        )

        f1_macro = valid_diseases_decoders_distributed(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.feature_names,
        )

    elif method == "common":
        print("Common tuning")
        qst_obj.training_parameters["batch_size"] = len(qst_obj.common_features_train)
        save = True
        initial_state = InitState(STATE_SIZE)
        encoders = get_feature_encoders(qst_obj, STATE_SIZE)
        single_disease_decoders = get_binary_disease_decoders(qst_obj, STATE_SIZE)
        # add training subset
        train_modules_without_decoding(
            qst_obj,
            X_cont_train,
            y_train,
            X_cont_valid,
            y_valid,
            X_cat_train,
            X_cat_valid,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.common_features,
            qst_obj.common_features,
            optimize_diseases=True,
            subset="common",
            save=save,
            seed=seed,
            percentage=percentage,
        )

        # # apply trained modules to unseen data to test the disease decoders
        f1_macro = valid_diseases_decoders_distributed(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.common_features,
        )
    elif method == "lower-baseline":
        print("lower baseline")
        qst_obj.training_parameters["batch_size"] = len(qst_obj.specific_features_train)
        save = False
        initial_state = InitState(STATE_SIZE)
        encoders = get_feature_encoders(qst_obj, STATE_SIZE)
        single_disease_decoders = get_binary_disease_decoders(qst_obj, STATE_SIZE)
        train_modules_without_decoding(
            qst_obj,
            X_cont_train,
            y_train,
            X_cont_valid,
            y_valid,
            X_cat_train,
            X_cat_valid,
            initial_state,
            encoders,
            single_disease_decoders,
            encoders_to_apply=qst_obj.feature_names,
            encoders_to_train=qst_obj.feature_names,
            optimize_diseases=True,
            subset="specific",
            save=save,
            seed=seed,
        )

        f1_macro = valid_diseases_decoders_distributed(
            qst_obj,
            X_cont_test,
            X_cat_test,
            y_test,
            initial_state,
            encoders,
            single_disease_decoders,
            qst_obj.feature_names,
        )
    else:
        print("No valid method")

    return f1_macro


def training_baseline_mlp(
    X_train, y_train, X_valid, y_valid, model, training_parameters
):

    lr = training_parameters["lr"]
    n_epochs = training_parameters["n_epochs"]
    batch_size = training_parameters["batch_size"]
    # for logs
    wandb.init(
        project="stage_igh", config={"epochs": n_epochs, "batch_size": batch_size}
    )
    # loss for multilabel classification
    criterion = nn.CrossEntropyLoss()

    # get the optimizer for the parameters we train in that level
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for j in range(n_epochs):

        # select random batch of indices (without replacement)

        batch_indices = np.random.choice(range(len(X_train)), batch_size, replace=False)

        # apply model
        y_pred = model(X_train[batch_indices, :])
        loss = criterion(y_pred, y_train[batch_indices])
        # Differentiate loss and make an optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"epoch": j, "loss": loss})
        # MONITORING ON VALIDATION DATA
        if j % 10 == 0:
            with torch.no_grad():
                y_pred_valid = model(X_valid)
                loss_valid = criterion(y_pred_valid, y_valid)
                wandb.log({"epoch": j, "loss_valid": loss_valid})
                # print(f'epoch {j} loss {loss} validation loss {loss_valid}')

    return model


if __name__ == "__main__":
    print("End of script")
