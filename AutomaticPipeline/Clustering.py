import pandas as pd
import csv
import numpy as np
import argparse
import json
def assign_pitch_group(row, male_percentiles, female_percentiles):
    if row['gender'] == 'male':
        if row['pitch'] <= male_percentiles[0]:
            return 'low'
        elif row['pitch'] <= male_percentiles[1]:
            return 'normal'
        else:
            return 'high'
    else: 
        if row['pitch'] <= female_percentiles[0]:
            return 'low'
        elif row['pitch'] <= female_percentiles[1]:
            return 'normal'
        else:
            return 'high'

def replace_age_with_text(row):
    age = row['age']
    if age < 14:
        return "Child"
    elif age < 24:
        return "Youth"
    elif age < 65:
        return "Adult"
    else:
        return "Elderly"


def main(args):

    input_path = args.input_path
    df = pd.read_csv(input_path, encoding = 'utf-8', sep='\t', header=None,  quoting=csv.QUOTE_NONE)
    df.columns = ['filename1', 'filename2', 'age', 'gender', 'pitch', 'energy', 'speed', 'emotion', 'transcript']
    df = df.dropna()
    # df = df.dropna(subset=['pitch'])

    male_percentiles = np.percentile(df[df['gender'] == 'male']['pitch'], [40, 90])
    female_percentiles = np.percentile(df[df['gender'] == 'female']['pitch'], [10, 60])

    df['age'] = df.apply(replace_age_with_text, axis=1)
    df['pitch_group'] = df.apply(assign_pitch_group, axis=1, male_percentiles = male_percentiles, female_percentiles = female_percentiles)
    df['energy_group'] = pd.qcut(df['energy'], 3, labels=["low", "normal", "high"])
    df['speed_group'] = pd.qcut(df['speed'], 3, labels=["fast", "normal", "slow"])

    df_to_save = df[['filename1', 'filename2',  'age', 'gender', 'pitch_group', 'energy_group', 'speed_group', 'emotion', 'transcript']]
    df_to_save.to_csv(input_path.replace('.scp', '_clusterd.scp'), sep='\t', header=0, index=False,  quoting=csv.QUOTE_NONE)
    
    
    result_dict = {}
    for index, row in df_to_save.iterrows():
        key = f"{row['filename1']}_{row['filename2']}"
        value = (
            f"age:{row['age']}\t"
            f"gender:{row['gender']}\t"
            f"pitch:{row['pitch_group']}\t"
            f"volume:{row['energy_group']}\t"
            f"speed:{row['speed_group']}\t"
            f"emotion:{row['emotion']}\t"
            f"transcription:{row['transcript']}"
        )
        text_style_target = f"The speaker speaks in {row['pitch_group']} pitch, {row['energy_group']} volume and {row['speed_group']} speed."

        result_dict[key] = {}
        result_dict[key]['labels'] = value
        result_dict[key]['text_style_target'] = text_style_target

    with open(input_path.replace('.scp', '.json'), 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default = './outputs/labels_CTTS_0.scp')
    args = parser.parse_args()
    main(args)