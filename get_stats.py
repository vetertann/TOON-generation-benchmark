import pandas as pd

df = pd.read_csv("eval_runs.csv")

# =====================================================
# 1) MODEL TABLE
# Using existing aggregate columns:
#   json_plain_*
#   json_*
#   toon_*
# Produce fields:
#   J1S, JF, JT,  JSO1S, JSOF, JSOT,  T1S, TF, TT
# =====================================================

model_cols = {
    # JSON (plain)
    "json_plain_one_shot_accuracy": "J1S",
    "json_plain_final_accuracy": "JF",
    "json_plain_total_tokens": "JT",

    # JSON-SO
    "json_one_shot_accuracy": "JSO1S",
    "json_final_accuracy": "JSOF",
    "json_total_tokens": "JSOT",

    # TOON
    "toon_one_shot_accuracy": "T1S",
    "toon_final_accuracy": "TF",
    "toon_total_tokens": "TT",
}

eval_results_by_model = (
    df.groupby("model")[list(model_cols.keys())]
      .mean()
      .rename(columns=model_cols)
      .reset_index()
)

eval_results_by_model.to_csv("eval_results_by_model.csv", index=False)


# =====================================================
# 2) CASE TABLE
# For each case (users/order/company/invoice),
#      take mean over ALL runs+models
# We rename columns the same way.
# =====================================================

case_prefixes = ["users", "order", "company", "invoice"]

# Mapping for case-level column renaming
case_cols = {
    # JSON (plain)
    "{}_json_plain_one_shot": "J1S",
    "{}_json_plain_final": "JF",
    "{}_json_plain_prompt_tokens": "JT_prompt",
    "{}_json_plain_completion_tokens": "JT_completion",

    # JSON-SO
    "{}_json_one_shot": "JSO1S",
    "{}_json_final": "JSOF",
    "{}_json_prompt_tokens": "JSOT_prompt",
    "{}_json_completion_tokens": "JSOT_completion",

    # TOON
    "{}_toon_one_shot": "T1S",
    "{}_toon_final": "TF",
    "{}_toon_prompt_tokens": "TT_prompt",
    "{}_toon_completion_tokens": "TT_completion",
}

rows = []
for prefix in case_prefixes:
    subset = {}
    for pattern, newname in case_cols.items():
        col = pattern.format(prefix)
        if col in df.columns:
            subset[newname] = df[col].mean()
    row = {"case": prefix}

    # collapse prompt + completion → token budget
    row["JT"] = subset["JT_prompt"] + subset["JT_completion"]
    row["JSOT"] = subset["JSOT_prompt"] + subset["JSOT_completion"]
    row["TT"] = subset["TT_prompt"] + subset["TT_completion"]

    # rest values stay as-is
    for k in ["J1S", "JF", "JSO1S", "JSOF", "T1S", "TF"]:
        row[k] = subset[k]

    rows.append(row)

eval_results_by_case = pd.DataFrame(rows)
eval_results_by_case.to_csv("eval_results_by_case.csv", index=False)

print("Saved eval_results_by_model.csv")
print("Saved eval_results_by_case.csv")
