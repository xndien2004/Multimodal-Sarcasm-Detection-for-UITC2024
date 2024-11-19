import pandas as pd
import numpy as np

class CaptionProcessor:
    def __init__(self, df):
        self.df = df.copy()

    def _get_duplicate_captions(self):
        """Find rows with duplicate 'caption'."""
        return self.df[self.df.duplicated('caption', keep=False)]

    def _select_label(self, group):
        """Select the appropriate label for duplicate rows based on 'caption'."""
        label_counts = group['label'].value_counts()

        if len(label_counts) == 1:
            # Case where all labels are the same
            return label_counts.idxmax()

        if label_counts.iloc[0] > label_counts.iloc[1]:
            # Case 1: One label has a higher count
            return label_counts.idxmax()
        
        # Case 2: Labels have equal counts, compare average probabilities
        mean_probs = group.groupby('label')['prob'].mean()
        return mean_probs.idxmax()

    def process(self):
        """Process the dataframe to ensure duplicate 'caption' rows have the correct label."""
        duplicate_captions = self._get_duplicate_captions()

        for caption, group in duplicate_captions.groupby('caption'):
            selected_label = self._select_label(group)
            self.df.loc[self.df['caption'] == caption, 'label'] = selected_label

        return self.df
