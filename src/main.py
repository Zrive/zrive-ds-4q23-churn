import configparser
from data_gathering import data_gathering
from data_cleaning import data_cleaning
from feature_computation import feature_computation
from modeling import modeling
from evaluation import evaluation


def get_initial_params():
    config = configparser.ConfigParser()
    config.read("src/params.ini")
    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")

    return train_from, train_to


def main_orchestrator():
    train_from, train_to = get_initial_params()
    print(train_from, train_to)
    raw_data = data_gathering()
    clean_data = data_cleaning()
    features = feature_computation()
    model = modeling()
    eval_result = evaluation()

    return eval_result


if __name__ == "__main__":
    print(main_orchestrator())
