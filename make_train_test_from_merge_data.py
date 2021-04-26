from merge_data_mobility_epidemic_situation import get_merge_data
import pandas as pd


# %%
def reshape_data_merge_to_get_rows_with_n_days_history_and_target(data_merge: pd.DataFrame, number_of_days_in_one_rows,
                                                                  number_day_ahead_to_predition):
    data_merge_with_history = pd.DataFrame()
    data_merge_with_history_test = pd.DataFrame()
    n = number_of_days_in_one_rows + number_day_ahead_to_predition
    number_of_days = len(data_merge.loc[:, 'date'].unique())
    while n < data_merge.shape[0]:
        if n % number_of_days == 0:  # new region
            data_merge_stack: pd.Series = data_merge.iloc[
                                          n - 1 - number_of_days_in_one_rows:n - 1,
                                          3:].stack()
            data_merge_stack = pd.concat([data_merge.iloc[n - 1, :3], data_merge_stack]).reset_index(
                drop=True)
            data_merge_with_history_test = data_merge_with_history_test.append(data_merge_stack, ignore_index=True)

            n += number_of_days_in_one_rows + number_day_ahead_to_predition
        else:
            data_merge_stack: pd.Series = data_merge.iloc[
                                          (
                                                      n - number_of_days_in_one_rows - number_day_ahead_to_predition):n - number_day_ahead_to_predition,
                                          3:].stack()
            data_merge_stack = pd.concat([data_merge.iloc[n, :3], data_merge_stack]).reset_index(
                drop=True)
            data_merge_stack['target'] = data_merge.iloc[n, -1]
            data_merge_with_history = data_merge_with_history.append(data_merge_stack, ignore_index=True)
            n += 1
    return data_merge_with_history, data_merge_with_history_test


# %%
def make_train_and_target(data_merge_with_history: pd.DataFrame):
    train = data_merge_with_history.iloc[:, :-1]
    train: pd.DataFrame = train.set_index([train.columns[0], train.columns[1]])
    train = train.astype(float)
    target = data_merge_with_history.iloc[:, -1]
    target = target.astype(float)
    return train, target


# %%
def make_test(data_merge_with_history_without_target: pd.DataFrame):
    data_merge_with_history_without_target = data_merge_with_history_without_target.set_index(
        [data_merge_with_history_without_target.columns[0], data_merge_with_history_without_target.columns[1]])
    test = data_merge_with_history_without_target.astype(float)
    return test


# %%
def get_train_target_test(data_merge:pd.DataFrame, number_of_days_in_one_rows=14, number_day_ahead_to_prediction=7):

    data_merge_with_history, data_merge_with_history_without_target = reshape_data_merge_to_get_rows_with_n_days_history_and_target(
        data_merge, number_of_days_in_one_rows, number_day_ahead_to_prediction)

    train, target = make_train_and_target(data_merge_with_history)
    test = make_test(data_merge_with_history_without_target)
    return train, target, test


# %%
data_merge = get_merge_data()
train, target, test = get_train_target_test(data_merge,1,1)

# make_all(train, target, 'results/prediction_7')
#
# # %%
# test = make_test(data_merge_with_history_without_target)
# submission = make_submission(test, 7)
#
# submission = add_prediction_to_submission(test, submission, 7)
# submission_to_cvs(submission, 'results/test_7')
