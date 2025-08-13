import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import os
from pathlib import Path
from typing import Generator, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
# ---------- 1.  response (label) column for every data set ----------

LABEL_COL = {
    "PhDArticles":          "Articles",
    "Pokemon":              "Use",
    "puffin":               "class",
    "Titanic":              "Survived",
    "covid_patients":       "death",
    "trajectories":         "OUT",
    "cad1":                 "CAD",
    "cad2":                 "CAD",
    "HouseVotes84":         "Class",
}

SRC = {
    "PhDArticles":          {"pkg": "stagedtrees", "rname": "PhDArticles"},
    "Pokemon":              {"pkg": "stagedtrees", "rname": "Pokemon"},
    "puffin":               {"pkg": "MBCbook",     "rname": "puffin"},
    "Titanic":              {"pkg": "datasets",    "rname": "Titanic", "is_table": True},
    "covid_patients":       {"pkg": "stagedtrees", "rname": "covid_patients"},
    "trajectories":         {"pkg": "stagedtrees", "rname": "trajectories"},
    "cad1":                 {"pkg": "gRbase",      "rname": "cad1"},
    "cad2":                 {"pkg": "gRbase",      "rname": "cad2"},
    "HouseVotes84":         {"pkg": "mlbench",     "rname": "HouseVotes84",        "mlbench": True},
}

def iter_datasets():
    """
    Generator → yields (name, dataframe, label_col) tuples on-demand.
    Downstream consumers can iterate without front-loading every R
    asset into memory.

    Usage:
        for name, df, y in iter_datasets():
            # do work
    """
    for ds, meta in SRC.items():
        try:
            # load R package at runtime
            pkg = importr(meta["pkg"])

            if meta.get("is_table"):                  # Titanic contingency table
                ro.r("library(datasets)")
                r_df = ro.r("as.data.frame(as.table(Titanic))")
                with localconverter(ro.default_converter + pandas2ri.converter):
                    df = ro.conversion.rpy2py(r_df)

            elif meta.get("mlbench"):                   # mlbench objects
                ro.r(f"data({meta['rname']}, package = '{meta['pkg']}')")
                r_df = ro.r(f"as.data.frame({meta['rname']})")
                with localconverter(ro.default_converter + pandas2ri.converter):
                    df = ro.conversion.rpy2py(r_df)

            else:                                       # vanilla data frames
                r_df = data(pkg).fetch(meta["rname"])[meta["rname"]]
                with localconverter(ro.default_converter + pandas2ri.converter):
                    df = ro.conversion.rpy2py(r_df)

            label = LABEL_COL[ds]
            yield ds, df, label                         # 🚚 ship it downstream

        except Exception as e:
            # fail fast but keep the iterator alive
            print(f"{ds}: ⚠️  {e}")



DATASET_DESCRIPTIONS = {

    "PhDArticles":
        "Biochemistry PhD-student cohort from the 1950-60 s.  "
        "Features: Articles – categorical research output during the last 3 years "
        "of the PhD {‘0’, ‘1-2’, ‘>2’}; Gender {‘male’, ‘female’}; Kids {‘yes’, ‘no’}; "
        "Married {‘yes’, ‘no’}; Mentor – mentor’s publication volume {‘low’, ‘medium’, "
        "‘high’}; Prestige – university prestige {‘low’, ‘high’}.  "
        ,

    "Pokemon":
        "Demographics of potential Pokémon Go users.  "
        "Features: Use – whether the person plays the game {‘Y’, ‘N’}; "
        "Age {‘<=30’, ‘>30’}; Degree – completed higher-education degree {‘Yes’, ‘No’}; "
        "Gender {‘Male’, ‘Female’}; Activity – physically active in the week before "
        "survey {‘Yes’, ‘No’}. "
        ,

    "puffin":
        "Morphological descriptions of Atlantic puffins (two sub-species).  "
        "Features: \nclass – taxonomic sub-class 1 and 2 which are {1:‘lherminieri’, 2:‘subalaris’}; "
        "gender: ['Male', 'Female'] "
        "eyebrow - eyebrow-stripe pattern: ['Poor pronounced', 'None', 'Pronounced', 'Very pronounced'] "
        "collar – neck-collar pattern: ['Dashed', 'Longdashed', 'None', 'Continuous'] "
        "sub.caudal – tail covert pattern: ['White', 'Black', 'Black & white', 'Black & WHITE'] "
        "border – bill/face border pattern: ['Few', 'None', 'Many'] "
        ,

    "Titanic":
        "Classical Titanic survival dataset based on the following variables "
        "Variables: Class {‘1st’, ‘2nd’, ‘3rd’, ‘Crew’}; Sex {‘Male’, ‘Female’}; "
        "Age {‘Child’, ‘Adult’}; Survived {‘No’, ‘Yes’}; "
        ,

    "covid_patients":
        "Trajectories of French SARS-CoV-2 hospitalisations.  "
        "Features: Sex {‘Female’, ‘Male’}; Age {‘0-39’, ‘40-49’, ‘50-59’, ‘60-69’, "
        "‘70-79’, ‘80+’}; ICU – admitted to intensive care {‘yes’, ‘no’}; "
        "death – in-hospital death {‘yes’, ‘no’}. "
        ,

    "trajectories":
        "Generated imaginary hospital trajectories.  "
        "possible values per column: "
        "SEX: ['female', 'male'] "
        "AGE: ['elder', 'adult', 'child'] "
        "ICU: ['0', '1'] "
        "RSP - respiratory support: ['mask', 'intub', 'no'] "
        "OUT - outcome: ['survived', 'death'] "
        ,

    "cad1":
        "Complete coronary-artery-disease clinic data.  "
        "14 categorical predictors plus response CAD (heart-attack) {‘No’, ‘Yes’}.  "
        "possible values per feature: "
        "Sex: ['Male', 'Female'] "
        "AngPec: ['None', 'Atypical', 'Typical'] "
        "AMI: ['NotCertain', 'Definite'] "
        "QWave: ['No', 'Yes'] "
        "QWavecode: ['Usable', 'Nonusable'] "
        "STcode: ['Usable', 'Nonusable'] "
        "STchange: ['Yes', 'No'] "
        "SuffHeartF: ['Yes', 'No'] "
        "Hypertrophi: ['No', 'Yes'] "
        "Hyperchol: ['No', 'Yes'] "
        "Smoker: ['No', 'Yes'] "
        "Inherit: ['No', 'Yes'] "
        "Heartfail: ['No', 'Yes'] "
        "CAD: ['No', 'Yes'] ",

    "cad2":
        "Complete coronary-artery-disease clinic data.  "
        "14 categorical predictors plus response CAD (heart-attack) {‘No’, ‘Yes’}.  "
        "possible values per feature: "
        "Sex: ['Male', 'Female'] "
        "AngPec: ['None', 'Atypical', 'Typical'] "
        "AMI: ['NotCertain', 'Definite'] "
        "QWave: ['No', 'Yes'] "
        "QWavecode: ['Usable', 'Nonusable'] "
        "STcode: ['Usable', 'Nonusable'] "
        "STchange: ['Yes', 'No'] "
        "SuffHeartF: ['Yes', 'No'] "
        "Hypertrophi: ['No', 'Yes'] "
        "Hyperchol: ['No', 'Yes', nan] "
        "Smoker: [nan, 'No', 'Yes'] "
        "Inherit: ['No', nan, 'Yes'] "
        "Heartfail: ['No', 'Yes'] "
        "CAD: ['No', 'Yes'] ",

    "HouseVotes84":
        "1984 U.S. House of Representatives roll-call votes.  "
        "Variables: Class – party {‘democrat’, ‘republican’}; 16 key bills, each "
        "categorical with vote {‘y’, ‘n’, ‘?’ (unknown/present/absent)}.  "
        "V1	handicapped-infants: 2 (y,n,‘?’) "
        "V2	water-project-cost-sharing: 2 (y,n,‘?’) "
        "V3	adoption-of-the-budget-resolution: 2 (y,n,‘?’) "
        "V4	physician-fee-freeze: 2 (y,n,‘?’) "
        "V5	el-salvador-aid: 2 (y,n,‘?’) "
        "V6	religious-groups-in-schools: 2 (y,n,‘?’) "
        "V7	anti-satellite-test-ban: 2 (y,n,‘?’) "
        "V8	aid-to-nicaraguan-contras: 2 (y,n,‘?’) "
        "V9	mx-missile: 2 (y,n,‘?’) "
        "V10	immigration: 2 (y,n,‘?’) "
        "V11	synfuels-corporation-cutback: 2 (y,n,‘?’) "
        "V12	education-spending: 2 (y,n,‘?’) "
        "V13	superfund-right-to-sue: 2 (y,n,‘?’) "
        "V14	crime: 2 (y,n,‘?’) "
        "V15	duty-free-exports: 2 (y,n,‘?’) "
        "V16	export-administration-act-south-africa: 2 (y,n,‘?’) "

}

def get_dataset_descriptions(name):
    base = name.rsplit("_", 1)[0] if "_" in name else name
    return DATASET_DESCRIPTIONS[base]

def iter_train_test(
    sample: int = -1,
    train_pct: float = 0.8,
    save_dir: str | os.PathLike = "data_splits",
    random_state: int = 42,
    force: bool = False,
    run_id: int = 0
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame, str], None, None]:
    """
    Yields (dataset_name, train_df, test_df, label_col) for every dataset
    returned by `iter_datasets()`, caching splits to CSV.  All values are cast
    to string and NaNs are replaced by the literal 'unknown'.

    Parameters
    ----------
    train_pct : float
        Fraction of rows that go to *train* (e.g. 0.8 → 80 % train).
    save_dir : str | PathLike
        Directory for cached CSVs.
    random_state : int
        Seed for reproducible splits.
    force : bool
        Re-create the split even if cached files exist.
    """

    test_pct = 1.0 - train_pct

    split_tag = f"{int(train_pct*100):02d}-{int(test_pct*100):02d}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, df, target in iter_datasets():
        train_path = save_dir / f"{name}_train_{split_tag}.csv"
        test_path  = save_dir / f"{name}_test_{split_tag}.csv"

        # ---------------------------------------------------
        # Either load cached CSVs or make a new split
        # ---------------------------------------------------
        if not force and train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
        else:
            clean_df = df.dropna(subset=[target])

            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_pct,
                random_state=random_state,
                stratify=clean_df[target],
            )

            train_df = train_df.astype("string").fillna("unknown").apply(lambda col: col.str.lower())
            test_df = test_df.astype("string").fillna("unknown").apply(lambda col: col.str.lower())

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

        train_df = train_df.astype("string").fillna("unknown").apply(lambda col: col.str.lower())
        test_df = test_df.astype("string").fillna("unknown").apply(lambda col: col.str.lower())

        if 'Freq' in train_df.columns: train_df = train_df.drop(columns='Freq')
        if 'Freq' in test_df.columns: test_df = test_df.drop(columns='Freq')

        desc = get_dataset_descriptions(f"{name}_{train_pct}")
        df_merged = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        target_values = [str(val) for val in sorted(df_merged[target].unique())]
        features = [col for col in df_merged.columns if col != target]
        feature_values = {}
        for feature in features:
            feature_values[feature] = [str(val) for val in sorted(df_merged[feature].dropna().unique())]
        dev_df = None
        if sample > 0:
            idx = train_df.sample(n=sample, random_state=run_id).index
            dev_df = train_df.drop(idx)
            train_df = train_df.loc[idx]
        yield f"{name}_{train_pct}", train_df, dev_df, test_df, target, desc, features, feature_values, target_values, df_merged

if __name__ == "__main__":

    for name, df, target in iter_datasets():
        before_rows = df.shape[0]
        clean_df   = df.dropna(subset=[target])
        after_rows = clean_df.shape[0]

        print(f"{name:18s} → label: {target}")
        print(f"   rows before NA-drop: {before_rows}")
        print(f"   rows after  NA-drop: {after_rows}")

        print("   possible values per column:")
        for col in clean_df.columns:
            values = clean_df[col].unique()
            # Convert NumPy scalars to native Python types for nicer printing
            values = [v.item() if hasattr(v, "item") else v for v in values]
            print(f"      • {col}: {values}")
        print()  # blank line between datasets

    for name, tr, td, te, label, desc, features, feature_values, target_values, df_merged in iter_train_test(-1,0.8):
        total_rows = tr.shape[0] + te.shape[0]
        print(f"{name:18s} → label: {label}  (rows after drop: {total_rows})")
        print("   possible values per column:")
        for col in tr.columns:
            values = pd.concat([tr[col], te[col]]).unique()
            print(f"      • {col}: {values}")
        print()