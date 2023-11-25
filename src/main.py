import configparser
from data_gathering import data_gathering
from data_cleaning import data_cleaning
from feature_computation import feature_computation
from modeling import modeling
from evaluation import evaluation


def get_initial_params():
    """
    Loads all relevant parameters into a global variable. These parameters are then accessible to other functions in the script.
    Each function can consume the parameters it requires for its operation.
    This approach ensures centralized management and consistency of parameters across different functions.

    Returns:
        None: This function does not return a value but populates a global variable
        with necessary parameters.
    """

    config = configparser.ConfigParser()
    config.read("src/params.ini")
    global train_from, train_to
    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")


def main_orchestrator():
    """
    Orchestrates the entire ML pipe.
    It starts by initializing parameters and then sequentially executes the following steps:
    data gathering, data cleaning, feature computation, model training, and model evaluation.
    Each step is handled by a dedicated function, and the flow of data between these functions
    is managed within this orchestrator.

    Steps:
    1. Initialize parameters using `get_initial_params()`.
    2. Gather raw data with `data_gathering()`.
    3. Clean the gathered data using `data_cleaning()`.
    4. Compute relevant features from the clean data in `feature_computation()`.
    5. Train the machine learning model in `modeling()`.
    6. Evaluate the model's performance with `evaluation()`.

    Returns:
        dict: Evaluation results of the trained model, containing key performance metrics.
    """

    get_initial_params()
    print(train_from, train_to)
    raw_data = data_gathering()
    clean_data = data_cleaning()
    features = feature_computation()
    model = modeling()
    eval_result = evaluation()

    return eval_result


if __name__ == "__main__":
    print(main_orchestrator())
