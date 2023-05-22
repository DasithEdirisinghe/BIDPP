from propy import PyPro
import pandas as pd
import numpy as np

dir =  'Data/'
raw_data_dir = 'raw_data'
processed_data_dir = 'processed_data'

def extractProteinSequenceFromFasta(file):
    # read text file in FASTA format
    with open(file, 'r') as f:
        lines = f.readlines()
    # remove new line characters
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if line != '']
    # odd ids are protein sequences
    protein_sequences = lines[1::2]
    # even ids are protein ids
    protein_ids = lines[::2]
    # return protein sequences
    return protein_ids, protein_sequences

# function to extract AAC features from a given FASTA format txt file using propy3
def extractFeatureDF(protein_ids, protein_sequences, feature_type, negative):
    df = pd.DataFrame()
    # iterate over protein sequences
    for i in range(len(protein_sequences)):
        try:
            # get protein sequence
            protein = PyPro.GetProDes(protein_sequences[i])
            if feature_type == 'AAC':
                extractedFeatures = protein.GetAAComp()
            # convert dictionary to pandas dataframe
            df1 = pd.DataFrame.from_dict(extractedFeatures, orient='index').transpose()
            df1['id'] = protein_ids[i][1:]
            # add dataframe to main dataframe with df.concat
            df = pd.concat([df, df1], ignore_index=True)
            # print(feature_type, f"Extracted features for sequence {i}", negative)
        except ZeroDivisionError:
            print(f"Skipping sequence {i} due to ZeroDivisionError")
            continue
    if negative:
        df['label'] = 0
    else:
        df['label'] = 1
    # return AAC features dataframe
    return df

def combineNegativeAndPositiveDFs(negativeFile, positiveFile, feature_type):
    # extract protein ids and sequences from negative FASTA file
    negative_ids, negative_sequences = extractProteinSequenceFromFasta(negativeFile)
    # extract protein ids and sequences from positive FASTA file
    positive_ids, positive_sequences = extractProteinSequenceFromFasta(positiveFile)
    # extract feature_type from negative FASTA file
    negativeDF = extractFeatureDF(negative_ids, negative_sequences, feature_type, negative=True)
    # extract feature_type from positive FASTA file
    positiveDF = extractFeatureDF(positive_ids, positive_sequences, feature_type, negative=False)
    # combine positive and negative dataframes
    combinedDF = pd.concat([negativeDF, positiveDF], ignore_index=True)
    # shuffle dataframe
    combinedDF = combinedDF.sample(frac=1).reset_index(drop=True)
    # return combined dataframe
    return combinedDF

def run(TR_pos, TR_neg, TS_pos, TS_neg):
    combineNegativeAndPositiveDFs(f'{dir + raw_data_dir}/{TR_neg}', f'{dir + raw_data_dir}/{TR_pos}', 'AAC').to_csv(f'{dir + processed_data_dir}/TR_AAC.csv', index=False)
    combineNegativeAndPositiveDFs(f'{dir + raw_data_dir}/{TS_neg}', f'{dir + raw_data_dir}/{TS_pos}', 'AAC').to_csv(f'{dir + processed_data_dir}/TR_AAC.csv', index=False)