import pandas as pd
from pathlib import Path
import os


def processing(pc_base_dir):
    data_dir = pc_base_dir / "rawdata"
    split1_df = pd.read_csv(data_dir / "adult.data", header=None, skipinitialspace=True)
    split1_df.replace(to_replace="?", value=pd.NA, inplace=True)
    split1_df = split1_df.apply(
        lambda col: col.fillna(col.mode()[0] if not col.mode().empty else None)
    )

    split2_df = pd.read_csv(data_dir / "adult.test", header=None, skipinitialspace=True)
    split2_df.replace(to_replace="?", value=pd.NA, inplace=True)
    split2_df = split2_df.apply(
        lambda col: col.fillna(col.mode()[0] if not col.mode().empty else None)
    )
    split2_df.replace(to_replace="<=50K.", value="<=50K", inplace=True)
    split2_df.replace(to_replace=">50K.", value=">50K", inplace=True)

    combined_df = pd.concat([split1_df, split2_df], ignore_index=True)

    def split_to_train_test_stratified(df, label_column, train_frac=0.8):
        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        labels = df[label_column].unique()
        for lbl in labels:
            lbl_df = df[df[label_column] == lbl]
            lbl_train_df = lbl_df.sample(frac=train_frac)
            lbl_test_df = lbl_df.drop(lbl_train_df.index)
            train_df = pd.concat([train_df, lbl_train_df])
            test_df = pd.concat([test_df, lbl_test_df])
        return train_df, test_df

    train_df, test_df = split_to_train_test_stratified(combined_df, 14)

    (pc_base_dir / "test").mkdir(parents=True, exist_ok=True)
    test_df.to_csv(
        pc_base_dir / "test" / "test_with_labels.csv", index=False, header=False
    )
    test_labels_df = test_df[14]
    test_labels_df.to_csv(
        pc_base_dir / "test" / "test_labels.csv", index=False, header=False
    )
    test_df = test_df.drop(columns=[14])
    test_df.to_csv(
        pc_base_dir / "test" / "test_without_labels.csv", index=False, header=False
    )

    train_df = train_df[[14] + list(range(14))]
    (pc_base_dir / "training").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(pc_base_dir / "training" / "train.csv", index=False, header=False)


if __name__ == "__main__":
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    processing(pc_base_dir)
