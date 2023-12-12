# COLS to be used across functions

# user-info cols to aggregate data later on
users_cols = ["customer_id", "MONTH", "YEAR"]

# pre-cooked features
diff_cols = [
    "dif_pago_final_prev_month",
    "dif_pago_final_prev_2_month",
    "dif_pago_final_prev_3_month",
    "dif_consumo_prev_month",
    "dif_consumo_prev_2_month",
    "dif_consumo_prev_3_month",
    "dif_discount_prev_month",
    "dif_discount_prev_2_month",
    "dif_discount_prev_3_month",
    "dif_periodica_prev_month",
    "dif_periodica_prev_2_month",
    "dif_periodica_prev_3_month",
    "dif_aperiodica_prev_month",
    "dif_aperiodica_prev_2_month",
    "dif_aperiodica_prev_3_month",
    "dif_ajuste_prev_month",
    "dif_ajuste_prev_2_month",
    "dif_ajuste_prev_3_month",
    "service_mobile_pending_install",
    "service_fix_pending_install",
    "service_mobile_cancelled",
    "service_fix_cancelled",
    "service_mobile_pending_install_3month",
    "service_fix_pending_install_3month",
    "service_mobile_cancelled_3month",
    "service_fix_cancelled_3month",
    "service_mobile_pending_install_6month",
    "service_fix_pending_install_6month",
    "service_mobile_cancelled_6month",
    "service_fix_cancelled_6month",
]

# to-be-cooked features
transform_cols = [
    "pago_final_0",
    "consumo_0",
    "aperiodica_0",
    "periodica_0",
    "discount_0",
    "ajuste_0",
    "NUM_GB_OWNN_CURR",
    "NUM_GB_2G_CURR",
    "NUM_GB_3G_CURR",
    "NUM_GB_4G_CURR",
    "NUM_GB_5G_CURR",
    "NUM_SESS_CURR",
    "NUM_SECS_CURR",
    "PERC_SECS_TYPE_IN_CURR",
    "PERC_SECS_TYPE_OUT_CURR",
    "PERC_SECS_OWNN_CURR",
    "PERC_SECS_NATR_CURR",
    "PERC_SECS_SERV_MOBI_CURR",
    "PERC_SECS_TYPE_IN_OWNN_CURR",
    "PERC_SECS_TYPE_OUT_OWNN_CURR",
    "PERC_SECS_TYPE_IN_NATR_CURR",
    "PERC_SECS_TYPE_OUT_NATR_CURR",
    "NUM_PLAT_GMM_CURR",
    "NUM_PLAT_OMV_CURR",
    "NUM_NETW_OWNN_CURR",
    "NUM_CALL_CURR",
    "PERC_CALL_TYPE_IN_CURR",
    "PERC_CALL_TYPE_OUT_CURR",
    "PERC_CALL_OWNN_CURR",
    "PERC_CALL_NATR_CURR",
    "NUM_CALL_WEEK_CURR",
    "NUM_CALL_WEEKEND_CURR",
    "NUM_SECS_WEEK_CURR",
    "NUM_SECS_WEEKEND_CURR",
    "NUM_CALL_WEEK",
    "NUM_CALL_WEEKEND",
]

# direct-to-model features
keep_cols = [
    "NUM_DAYS_ACT",
    "order_mobile_from_new_alta",
    "MIN_DAYS_PERM_CURR",
    "PREV_FINISHED_PERM",
    "NUM_DAYS_LINE_TYPE_MAIN_POST_ACT",
]

# target
target_col = ["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]
