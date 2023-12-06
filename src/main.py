import configparser
from data_gathering import data_gathering
from data_cleaning import data_cleaning
from feature_computation import feature_computation
from modeling import modeling
from evaluation import evaluation
from utils.logger import get_logger

# Instantiate logger
logger = get_logger(__name__)


def get_initial_params():
    """
    Loads all relevant parameters into a global variable. These parameters are then accessible to other functions in the script.
    Each function can consume the parameters it requires for its operation.
    This approach ensures centralized management and consistency of parameters across different functions.

    Returns:
        None: This function does not return a value but populates a global variable
        with necessary parameters.
    """
    global train_from, train_to, logistic_regression_params

    config = configparser.ConfigParser()
    config.read("src/params.ini")

    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")

    # Load Logistic Regression parameters
    logistic_regression_params = {
        "penalty": config.get("LOGISTIC_REGRESSION", "penalty", fallback="l2"),
        "C": config.getfloat("LOGISTIC_REGRESSION", "C", fallback=1.0),
        "solver": config.get("LOGISTIC_REGRESSION", "solver", fallback="saga"),
        "max_iter": config.getint("LOGISTIC_REGRESSION", "max_iter", fallback=10000),
    }


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

    query = """
    WITH selectable_customer AS
    (
            SELECT   customer_id
            FROM     `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
            GROUP BY customer_id ), customer_selected AS
    (
        SELECT customer_id AS selected_customer
        FROM   selectable_customer
        WHERE  RAND() < 0.1 )
    SELECT     customer_id,
            MONTH,
            YEAR,
            pago_final_0,
            dif_pago_final_prev_month,
            dif_pago_final_prev_2_month,
            dif_pago_final_prev_3_month,
            periodica_0,
            dif_periodica_prev_month,
            dif_periodica_prev_2_month,
            dif_periodica_prev_3_month,
            consumo_0,
            dif_consumo_prev_month,
            dif_consumo_prev_2_month,
            dif_consumo_prev_3_month,
            aperiodica_0,
            dif_aperiodica_prev_month,
            dif_aperiodica_prev_2_month,
            dif_aperiodica_prev_3_month,
            discount_0,
            dif_discount_prev_month,
            dif_discount_prev_2_month,
            dif_discount_prev_3_month,
            ajuste_0,
            dif_ajuste_prev_month,
            dif_ajuste_prev_2_month,
            dif_ajuste_prev_3_month,
            Tota_Compra_disp,
            Curr_Compra_disp,
            Curr_Compra_Finanz_disp,
            Curr_Finanz_disp,
            Month_purchase_disp,
            Modelo_disp,
            Import_Rest_quota_disp,
            pvp_total_disp,
            pvp_total_disp_movil,
            Curr_cancel_disp,
            Tota_cancel_disp NUM_GB_OWNN_CURR,
            NUM_GB_2G_CURR,
            NUM_GB_3G_CURR,
            NUM_GB_4G_CURR,
            NUM_GB_5G_CURR,
            NUM_SESS_CURR,
            NUM_SECS_CURR,
            NUM_CALL_CURR,
            NUM_CALL_WEEK_CURR,
            NUM_CALL_WEEKEND_CURR,
            NUM_SECS_WEEK_CURR,
            NUM_SECS_WEEKEND_CURR,
            NUM_CALL_WEEK,
            NUM_CALL_WEEKEND,
            NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    FROM       `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
    INNER JOIN customer_selected
    ON         customer_id = selected_customer
    WHERE      IS_CUST_SEGM_RESI > 0
    AND        IS_CUST_BILL_POST_CURR = TRUE
    AND        CUST_BUNDLE_CURR = 'FMC'
    AND        NUM_IMPAGOS = 0
    AND        pago_final_0 IS NOT NULL
    """
    # TO-DO: PARAMETRIZE THIS
    save_curves_path = "src/models"

    get_initial_params()
    raw_data = data_gathering(query, logger)
    clean_data = data_cleaning(raw_data, logger)
    features, target, features_test, target_test = feature_computation(
        clean_data, train_from, train_to, logger
    )
    model = modeling(features, target, logistic_regression_params, logger)
    model_metrics, precision_decile, uplift_by_decile, feature_importance = evaluation(
        model, features, target, logger, save_curves_path
    )


if __name__ == "__main__":
    print(main_orchestrator())
