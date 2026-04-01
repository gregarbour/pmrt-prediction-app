import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model coefficients
# ---------------------------------------------------------------------------
COEFS = {
    "PRE_age_at_dx":                          -0.002378081258571237,
    "PRE_dximg_mammography":                   0.2558198328695431,
    "PRE_dximg_ultrasound":                    0.3303786236334959,
    "PRE_tumor_max_size_composite":            0.011773323529951186,
    "PRE_susp_LN_prsnt_composite":             0.5324024058161426,
    "PRE_susp_LN_size_composite":              0.029215523393665697,
    "PRE_pre_op_biopsy":                      -0.41168883522228455,
    "PRE_his_subtype_is_invasive_composite":   0.06126810489065488,
    "PRE_his_subtype_idc":                     0.34185730894131894,
    "PRE_his_subtype_dcis":                   -0.4953718204992072,
    "PRE_lymphovascular_invasion":             0.13777100843249082,
    "PRE_er_status":                          -0.031370606454014995,
    "PRE_her_status":                          0.49394214162080885,
    "PRE_axillary_lymph_node_core_b":          0.14828588214628263,
    "PRE_metastatic_carcinoma_on_ax":          0.7952427520012665,
    "PRE_surg_indicat_prim_primary_tx":        0.21577450955206823,
    "PRE_surg_indicat_prim_recurrent_cancer": -0.25817460777443113,
    "intercept":                              -1.8060538608212995,
}

IMPUTE_VALUES = {
    "PRE_age_at_dx":                          50.656410256410254,
    "PRE_dximg_mammography":                   0.88,
    "PRE_dximg_ultrasound":                    0.7733333333333333,
    "PRE_tumor_max_size_composite":            27.34041095890411,
    "PRE_susp_LN_prsnt_composite":             0.24666666666666667,
    "PRE_susp_LN_size_composite":              2.6683333333333334,
    "PRE_pre_op_biopsy":                       1.1416526138279932,
    "PRE_his_subtype_is_invasive_composite":   0.7333333333333333,
    "PRE_his_subtype_idc":                     0.6383333333333333,
    "PRE_his_subtype_dcis":                    0.2683333333333333,
    "PRE_lymphovascular_invasion":             0.035,
    "PRE_er_status":                           1.2008333333333334,
    "PRE_her_status":                          1.73,
    "PRE_axillary_lymph_node_core_b":          0.2,
    "PRE_metastatic_carcinoma_on_ax":          0.13916666666666666,
    "PRE_surg_indicat_prim_primary_tx":        0.7266666666666667,
    "PRE_surg_indicat_prim_recurrent_cancer":  0.04666666666666667,
}

# Columns the user must supply (original display names → internal names)
REQUIRED_COLUMNS = {
    "Presence of carcinoma on axillary lymph node biopsy": "PRE_metastatic_carcinoma_on_ax",
    "Histological subtype: DCIS":                          "PRE_his_subtype_dcis",
    "Presence of pre-operative palpable axillary lymph node": "PRE_susp_LN_prsnt_composite",
    "HER status":                                          "PRE_her_status",
    "Maximum dimension/size of pre-operative tumor on any imaging modality (mm)": "PRE_tumor_max_size_composite",
    "Histological subtype: IDC":                           "PRE_his_subtype_idc",
    "Maximum size of pre-operative suspicious lymph node on any imaging modality (mm) - enter 0 if none specified": "PRE_susp_LN_size_composite",
    "Diagnostic imaging: US":                              "PRE_dximg_ultrasound",
    "Pre-operative biopsy method":                         "PRE_pre_op_biopsy",
    "Lymphovascular invasion":                             "PRE_lymphovascular_invasion",
    "Age at diagnosis":                                    "PRE_age_at_dx",
    "ER status":                                           "PRE_er_status",
}

ENCODED_COLS = list(REQUIRED_COLUMNS.values())

# Human-readable valid values for error messages
VALID_VALUES = {
    "PRE_metastatic_carcinoma_on_ax": '"Y", "N", or "NA"',
    "PRE_his_subtype_dcis":           '"Y" or "N"',
    "PRE_susp_LN_prsnt_composite":    '"Y" or "N"',
    "PRE_his_subtype_idc":            '"Y" or "N"',
    "PRE_dximg_ultrasound":           '"Y" or "N"',
    "PRE_lymphovascular_invasion":    '"Y" or "N"',
    "PRE_her_status":                 '"positive" or "negative"',
    "PRE_er_status":                  '"positive" or "negative"',
    "PRE_pre_op_biopsy":              '"core needle biopsy", "surgical biopsy", or "fine needle aspirate"',
    "PRE_tumor_max_size_composite":   "a numeric value (mm)",
    "PRE_susp_LN_size_composite":     "a numeric value (mm)",
    "PRE_age_at_dx":                  "a numeric value (years)",
}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def yn(val):
    if val == "N":
        return 0
    elif val == "Y":
        return 1
    return np.nan


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def encode_row(row):
    PRE_metastatic_carcinoma_on_ax = (
        0.5 if str(row["PRE_metastatic_carcinoma_on_ax"]).strip() == "NA"
        else yn(str(row["PRE_metastatic_carcinoma_on_ax"]).strip())
    )
    PRE_his_subtype_dcis        = yn(str(row["PRE_his_subtype_dcis"]).strip())
    PRE_susp_LN_prsnt_composite = yn(str(row["PRE_susp_LN_prsnt_composite"]).strip())
    PRE_his_subtype_idc         = yn(str(row["PRE_his_subtype_idc"]).strip())
    PRE_dximg_ultrasound        = yn(str(row["PRE_dximg_ultrasound"]).strip())
    PRE_lymphovascular_invasion = yn(str(row["PRE_lymphovascular_invasion"]).strip())
    PRE_tumor_max_size_composite = to_float(row["PRE_tumor_max_size_composite"])
    PRE_susp_LN_size_composite   = to_float(row["PRE_susp_LN_size_composite"])
    PRE_age_at_dx                = to_float(row["PRE_age_at_dx"])

    her = str(row["PRE_her_status"]).strip().lower()
    PRE_her_status = 1 if her == "positive" else 2 if her == "negative" else np.nan

    er = str(row["PRE_er_status"]).strip().lower()
    PRE_er_status = 1 if er == "positive" else 2 if er == "negative" else np.nan

    biopsy_map = {"core needle biopsy": 1, "surgical biopsy": 2, "fine needle aspirate": 3}
    PRE_pre_op_biopsy = biopsy_map.get(str(row["PRE_pre_op_biopsy"]).strip().lower(), np.nan)

    return pd.Series({
        "PRE_metastatic_carcinoma_on_ax": PRE_metastatic_carcinoma_on_ax,
        "PRE_his_subtype_dcis":           PRE_his_subtype_dcis,
        "PRE_susp_LN_prsnt_composite":    PRE_susp_LN_prsnt_composite,
        "PRE_her_status":                 PRE_her_status,
        "PRE_tumor_max_size_composite":   PRE_tumor_max_size_composite,
        "PRE_his_subtype_idc":            PRE_his_subtype_idc,
        "PRE_susp_LN_size_composite":     PRE_susp_LN_size_composite,
        "PRE_dximg_ultrasound":           PRE_dximg_ultrasound,
        "PRE_pre_op_biopsy":              PRE_pre_op_biopsy,
        "PRE_lymphovascular_invasion":    PRE_lymphovascular_invasion,
        "PRE_age_at_dx":                  PRE_age_at_dx,
        "PRE_er_status":                  PRE_er_status,
    })


def calc_prob(row):
    log_odds = (
        COEFS["intercept"] +
        row["PRE_metastatic_carcinoma_on_ax"] * COEFS["PRE_metastatic_carcinoma_on_ax"] +
        row["PRE_his_subtype_dcis"]            * COEFS["PRE_his_subtype_dcis"] +
        row["PRE_susp_LN_prsnt_composite"]     * COEFS["PRE_susp_LN_prsnt_composite"] +
        row["PRE_her_status"]                  * COEFS["PRE_her_status"] +
        row["PRE_tumor_max_size_composite"]    * COEFS["PRE_tumor_max_size_composite"] +
        row["PRE_his_subtype_idc"]             * COEFS["PRE_his_subtype_idc"] +
        row["PRE_susp_LN_size_composite"]      * COEFS["PRE_susp_LN_size_composite"] +
        row["PRE_dximg_ultrasound"]            * COEFS["PRE_dximg_ultrasound"] +
        row["PRE_pre_op_biopsy"]               * COEFS["PRE_pre_op_biopsy"] +
        row["PRE_lymphovascular_invasion"]     * COEFS["PRE_lymphovascular_invasion"] +
        row["PRE_age_at_dx"]                   * COEFS["PRE_age_at_dx"] +
        row["PRE_er_status"]                   * COEFS["PRE_er_status"]
    )
    return 1 / (1 + np.exp(-log_odds))


# ---------------------------------------------------------------------------
# Validation: detect clearly wrong encodings (non-NaN rows with bad values)
# ---------------------------------------------------------------------------

def check_encoding_errors(df_raw):
    """
    Returns a list of dicts, one per bad cell:
      { "row": int, "column": str (display name), "value": str, "expected": str }
    """
    errors = []

    yn_cols = {
        "PRE_metastatic_carcinoma_on_ax": {"Y", "N", "NA"},
        "PRE_his_subtype_dcis":           {"Y", "N"},
        "PRE_susp_LN_prsnt_composite":    {"Y", "N"},
        "PRE_his_subtype_idc":            {"Y", "N"},
        "PRE_dximg_ultrasound":           {"Y", "N"},
        "PRE_lymphovascular_invasion":    {"Y", "N"},
    }
    categorical_cols = {
        "PRE_her_status":    {"positive", "negative"},
        "PRE_er_status":     {"positive", "negative"},
        "PRE_pre_op_biopsy": {"core needle biopsy", "surgical biopsy", "fine needle aspirate"},
    }
    numeric_cols = ["PRE_tumor_max_size_composite", "PRE_susp_LN_size_composite", "PRE_age_at_dx"]

    display_name = {v: k for k, v in REQUIRED_COLUMNS.items()}

    for col, valid_set in yn_cols.items():
        for i, val in enumerate(df_raw[col], start=2):
            s = str(val).strip()
            if s in ("", "nan", "NaN"):
                continue
            if s not in valid_set:
                errors.append({
                    "row": i, "column": display_name[col],
                    "value": str(val), "expected": VALID_VALUES[col],
                })

    for col, valid_set in categorical_cols.items():
        for i, val in enumerate(df_raw[col], start=2):
            s = str(val).strip().lower()
            if s in ("", "nan", "NaN"):
                continue
            if s not in valid_set:
                errors.append({
                    "row": i, "column": display_name[col],
                    "value": str(val), "expected": VALID_VALUES[col],
                })

    for col in numeric_cols:
        for i, val in enumerate(df_raw[col], start=2):
            s = str(val).strip()
            if s in ("", "nan", "NaN"):
                continue
            try:
                float(s)
            except ValueError:
                errors.append({
                    "row": i, "column": display_name[col],
                    "value": str(val), "expected": VALID_VALUES[col],
                })

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Read CSV or XLSX
    filename = f.filename.lower()
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(f, engine="openpyxl")
        elif filename.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a .csv or .xlsx file."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    # --- Column validation ---
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return jsonify({
            "error": (
                "The following required column(s) are missing or misspelled in your CSV:\n"
                + "\n".join(f"  • {c}" for c in missing_cols)
            )
        }), 400

    # Rename to internal names
    df.rename(columns=REQUIRED_COLUMNS, inplace=True)

    # --- Encoding validation (before encoding) ---
    encoding_errors = check_encoding_errors(df)
    if encoding_errors:
        total_errors = len(encoding_errors)
        if total_errors > 10:
            # Build a downloadable error CSV
            err_df = pd.DataFrame(encoding_errors)[["row", "column", "value", "expected"]]
            err_df.columns = ["Row", "Column", "Invalid Value", "Expected Values"]
            err_csv = io.BytesIO()
            err_df.to_csv(err_csv, index=False)
            err_csv.seek(0)
            app.config["_last_error_csv"] = err_csv.getvalue()
            return jsonify({
                "error_type": "encoding",
                "total_errors": total_errors,
                "download_errors": True,
                "errors": [],
            }), 400
        else:
            app.config["_last_error_csv"] = None
            return jsonify({
                "error_type": "encoding",
                "total_errors": total_errors,
                "download_errors": False,
                "errors": encoding_errors,
            }), 400

    # --- Count missing before encoding ---
    missing_before = df[ENCODED_COLS].isna().sum()
    pct_before     = df[ENCODED_COLS].isna().mean() * 100
    before_df = pd.DataFrame({"Missing (n)": missing_before, "Missing (%)": pct_before.round(1)})
    before_df.index = [
        next(k for k, v in REQUIRED_COLUMNS.items() if v == col)
        for col in ENCODED_COLS
    ]

    # --- Encode ---
    df[ENCODED_COLS] = df[ENCODED_COLS].apply(encode_row, axis=1)

    # --- Impute ---
    df.fillna(IMPUTE_VALUES, inplace=True)

    # --- Predict ---
    df["Predicted Probability"] = df.apply(calc_prob, axis=1).round(4)

    # --- Build output CSV ---
    output = io.BytesIO()
    df[["Predicted Probability"]].to_csv(output, index=True, index_label="Row")
    output.seek(0)

    # --- Build summary payload ---
    summary = [
        {"column": idx, "missing_n": int(row["Missing (n)"]), "missing_pct": float(row["Missing (%)"])}
        for idx, row in before_df.iterrows()
    ]
    total_rows = len(df)

    app.config["_last_output"] = output.getvalue()

    return jsonify({
        "success": True,
        "total_rows": total_rows,
        "summary": summary,
    })


@app.route("/download")
def download():
    data = app.config.get("_last_output")
    if not data:
        return "No results available.", 404
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv",
    )


@app.route("/download-errors")
def download_errors():
    data = app.config.get("_last_error_csv")
    if not data:
        return "No error report available.", 404
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="encoding_errors.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
