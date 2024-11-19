import pandas as pd
from sklearn.utils import resample

def augment_data(train_df, label_column='label', upsample_classes=None, random_state=42):
    """
    Perform data augmentation by upsampling specified classes in the dataset.

    Parameters:
    - train_df (DataFrame): The training dataset containing at least one column for class labels.
    - label_column (str): The name of the column containing the class labels. Default is 'label'.
    - upsample_classes (dict): A dictionary where keys are class labels to be upsampled, 
                               and values are the desired number of samples for each class.
    - random_state (int): Seed value for reproducibility.

    Returns:
    - DataFrame: The augmented dataset with upsampled classes.
    """
    dfs_upsampled = []
    for label, n_samples in upsample_classes.items():
        df_class = train_df[train_df[label_column] == label]
        df_upsampled = resample(df_class, replace=True, n_samples=n_samples, random_state=random_state)
        dfs_upsampled.append(df_upsampled)
    
    df_majority = train_df[~train_df[label_column].isin(upsample_classes.keys())]
    df_balanced = pd.concat([df_majority] + dfs_upsampled)

    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_balanced
