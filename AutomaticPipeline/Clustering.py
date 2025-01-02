import pandas as pd
import csv
import numpy as np
import argparse
import json
def assign_pitch_group(row, language, male_percentiles, female_percentiles):
    if row['gender'] == 'male':
        if row['pitch'] <= male_percentiles[0]:
            return 'low' if language == 'en' else '低'
        elif row['pitch'] <= male_percentiles[1]:
            return 'normal' if language == 'en' else '中'
        else:
            return 'high' if language == 'en' else '高'
    else: 
        if row['pitch'] <= female_percentiles[0]:
            return 'low' if language == 'en' else '低'
        elif row['pitch'] <= female_percentiles[1]:
            return 'normal' if language == 'en' else '中'
        else:
            return 'high' if language == 'en' else '高'

def replace_age_with_text(row, language):
    age = row['age']
    if age < 14:
        return "Child" if language == 'en' else '小孩'
    elif age < 26:
        return "Teenager" if language == 'en' else '少年'
    elif age < 40:
        return "Young Adult" if language == 'en' else '青年'
    elif age < 55:
        return "Middle-aged" if language == 'en' else '中年'
    else:
        return "Elderly" if language == 'en' else '老年'


def main(args):

    input_path = args.input_path
    language = args.language
    df = pd.read_csv(input_path, encoding = 'utf-8', sep='\t', header=None,  quoting=csv.QUOTE_NONE)
    df.columns = ['filename1', 'filename2', 'age', 'gender', 'pitch', 'energy', 'speed', 'emotion', 'transcript']
    df = df.dropna()
    # df = df.dropna(subset=['pitch'])

    male_percentiles = np.percentile(df[df['gender'] == 'male']['pitch'], [40, 90])
    female_percentiles = np.percentile(df[df['gender'] == 'female']['pitch'], [10, 60])

    df['age'] = df.apply(replace_age_with_text, axis=1, language = language)
    df['pitch_group'] = df.apply(assign_pitch_group, axis=1, language = language, male_percentiles = male_percentiles, female_percentiles = female_percentiles)
    df['energy_group'] = pd.qcut(df['energy'], 3, labels=["low", "normal", "high"] if language=='en' else ["低", "中", "高"])
    df['speed_group'] = pd.qcut(df['speed'], 3, labels=["fast", "normal", "slow"] if language=='en' else ["快", "中", "慢"])

    df_to_save = df[['filename1', 'filename2',  'age', 'gender', 'pitch_group', 'energy_group', 'speed_group', 'emotion', 'transcript']]
    df_to_save.to_csv(input_path.replace('.scp', '_clusterd.scp'), sep='\t', header=0, index=False,  quoting=csv.QUOTE_NONE)
    
    
    result_dict = {}
    for index, row in df_to_save.iterrows():
        key = f"{row['filename1']}_{row['filename2']}"
        if language == 'en':
            value = (
                f"age:{row['age']}\t"
                f"gender:{row['gender']}\t"
                f"pitch:{row['pitch_group']}\t"
                f"volume:{row['energy_group']}\t"
                f"speed:{row['speed_group']}\t"
                f"emotion:{row['emotion']}\t"
                f"transcription:{row['transcript']}"
            )
        else:
            value = (
                f"语气：{row['emotion']}\t"
                f"年龄：{row['age']}\t"
                f"性别：{row['gender']}\t"
                f"音高：{row['pitch_group']}\t"
                f"音量：{row['energy_group']}\t"
                f"语速：{row['speed_group']}"
                f"文本：{row['transcript']}\t"
            )
        result_dict[key] = {}
        result_dict[key]['labels'] = value

    with open(input_path.replace('.scp', '.json'), 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default = 'en')
    parser.add_argument('--input_path', type=str, default = './outputs/labels_LJspeech_0.scp')
    args = parser.parse_args()
    main(args)