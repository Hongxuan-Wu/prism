import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import subprocess
import warnings

root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
blast_dir = ''  # Use executables on PATH unless --blast-dir is supplied.

predict_dir = osp.join(root_dir, 'results', 'predicts')
select_dir = osp.join(root_dir, 'results', 'select')

def sort_promoters():
    """
    Sorts promoter sequences based on authenticity and transcript level classifications.
    
    This function reads promoter sequence data and classification results from CSV files, combines them into a single DataFrame,
    and sorts them in descending order based on authenticity and transcript level classifications. The sorted results are then
    saved to a new CSV file.
    """
    sequences = pd.read_csv(osp.join(predict_dir, 'gen_seqs.csv'), delimiter=',', header=None)
    authenticity_cls = pd.read_csv(osp.join(predict_dir, 'authenticity_cls.csv'), delimiter=',', header=None)
    transcript_level_cls2 = pd.read_csv(osp.join(predict_dir, 'transcript_level_cls2.csv'), delimiter=',', header=None)
    transcript_level_cls4 = pd.read_csv(osp.join(predict_dir, 'transcript_level_cls4.csv'), delimiter=',', header=None)

    inputs = [sequences, authenticity_cls, transcript_level_cls2, transcript_level_cls4]
    if sequences.empty or any(frame.shape != (len(sequences), 1) or frame.isna().any().any() for frame in inputs):
        raise ValueError('Sequence and prediction files must be nonempty, single-column, and have matching row counts without missing values')
    
    df = pd.concat([sequences, authenticity_cls, transcript_level_cls2, transcript_level_cls4], axis=1)
    df.columns = ['sequence', 'authenticity_cls', 'transcript_level_cls2', 'transcript_level_cls4']
    scores = df.iloc[:, 1:].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError('Prediction scores must be finite numbers')
    df[df.columns[1:]] = scores
    
    df_sorted = df.sort_values(by=['authenticity_cls', 'transcript_level_cls2', 'transcript_level_cls4'], ascending=[False, False, False])
    df_sorted.to_csv(osp.join(select_dir, 'predicts_sorted.csv'), sep=',', header=True)

def cutATG():
    """
    Reads data from 'predicts_sorted.csv', cuts each sequence at the start of a specified target sequence (ATG), 
    and saves the processed data to 'predicts_sorted_cut.csv'.
    
    The function identifies the position of the target sequence within each DNA sequence and retains only the portion 
    of the sequence before this target. The target sequence is defined by the variable `target_seq`.
    """
    file = pd.read_csv(osp.join(select_dir, 'predicts_sorted.csv'))
    sequences = file['sequence'].tolist()
    
    # atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa
    target_seq = 'ATGGTGAGCAAGGGCGAGGA'
    # target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGT'
    # target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGG'

    cutSequences = []
    boundary_found = []
    for row, seq in enumerate(sequences):
        if not isinstance(seq, str) or not seq or not set(seq.upper()).issubset(set('ACGT')):
            raise ValueError(f'Sorted row {row}: sequence must be nonempty A/C/G/T DNA')
        seq = seq.upper()
        position = seq.find(target_seq)
        promoter = seq if position < 0 else seq[:position]
        if not promoter:
            raise ValueError(f'Sorted row {row}: reporter boundary leaves an empty promoter')
        cutSequences.append(promoter)
        boundary_found.append(position >= 0)

    missing = len(boundary_found) - sum(boundary_found)
    if missing:
        warnings.warn(f'{missing} sequences lack the reporter boundary; full sequences are retained and flagged', UserWarning)
    
    file['sequence'] = cutSequences
    file['reporter_boundary_found'] = boundary_found
    file.to_csv(os.path.join(select_dir, 'predicts_sorted_cut.csv'), index=False)

def csv2fasta():
    """
    Converts sequence data from a CSV file into FASTA format.

    This function reads sequence data from 'predicts_sorted_cut.csv' and writes it to a new file named 
    'predicts_sorted_cut.fasta' in FASTA format, using identifiers from the first column of the CSV as sequence headers.
    """
    file = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut.csv'))
    sequences = file['sequence'].tolist()
        
    f = open(osp.join(select_dir, 'predicts_sorted_cut.fasta'), 'w')
    # number = 1
    for i in range(len(sequences)):
        seq_cut = sequences[i].replace('\n', '')
        f.write(f'>{file.iloc[i,0]}\n')
        f.write(f'{seq_cut}\n')
        # number += 1
    f.close()

def run_command(command):
    subprocess.run(command, check=True, env=dict(os.environ, BLAST_USAGE_REPORT='false'))

def blast():
    """
    Performs BLAST analysis of query sequences against a reference genome database.

    This function runs BLASTN to align sequences from 'predicts_sorted_cut.fasta' against a nucleotide database created 
    from 'GCF_000005845.2_ASM584v2_genomic.fna'. Results are saved in a tabular format to 'predicts_sorted_cut_blast.txt'.
    """
    fasta_path = osp.join(select_dir, 'predicts_sorted_cut.fasta')
    
    # Escherichia coli K-12 MG1655 T00007
    fna_path = osp.join(root_dir, '2select/fasta/GCF_000005845.2_ASM584v2_genomic.fna')

    bin_dir = osp.join(blast_dir, 'bin') if blast_dir else ''
    database = osp.join(select_dir, 'reference_db')
    run_command([osp.join(bin_dir, 'makeblastdb'), '-in', fna_path, '-dbtype', 'nucl', '-out', database])
    run_command([osp.join(bin_dir, 'blastn'), '-query', fasta_path, '-db', database,
                 '-out', osp.join(select_dir, 'predicts_sorted_cut_blast.txt'), '-outfmt', '6'])

def blast_filter():
    """
    Filters BLAST results to retain only the first occurrence of each query sequence.

    This function reads BLAST output from 'predicts_sorted_cut_blast.txt', removes duplicate entries for the same query,
    and saves the filtered results to 'predicts_sorted_cut_blast.csv' with appropriate column headers.
    """
    path = osp.join(select_dir, 'predicts_sorted_cut_blast.txt')
    f = open(path)
    file = f.readlines()
    f.close()
    
    selected_list = []
    selected_lines = []
    for line in file:
        if not line.strip():
            continue
        line_cut = line.replace('\n', '').split('\t')
        if len(line_cut) != 12:
            raise ValueError('Expected 12 tab-separated fields in legacy BLAST output')
        if not line_cut[0] in selected_list:
            selected_list.append(line_cut[0])
            selected_lines.append(line_cut)
        else:
            continue

    columns = [
        'Query ID',
        'Subject ID',
        '% Identity',
        'Alignment Length',
        'Mismatches',
        'Gap Opens',
        'Query Start',
        'Query End',
        'Subject Start',
        'Subject End',
        'E-value',
        'Bit Score',
    ]
    df = pd.DataFrame(selected_lines, columns=columns)
    df.to_csv(os.path.join(osp.join(select_dir, 'predicts_sorted_cut_blast.csv')), index=False, header=True)

def select_blast():
    """
    Merges BLAST results with the original sequence data based on query identifiers.

    This function combines data from 'predicts_sorted_cut.csv' and 'predicts_sorted_cut_blast.csv' using a left join on 
    query identifiers, and saves the merged dataset to 'predicts_sorted_cut_blast_merge.csv'.
    """
    df_blast = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut_blast.csv'), dtype={'Query ID': str})
    df_seqs = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut.csv'), dtype={'Unnamed: 0': str})
    for identifiers in (df_seqs['Unnamed: 0'], df_blast['Query ID']):
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError('Query identifiers must be present and unique before merging')
    if not set(df_blast['Query ID']).issubset(set(df_seqs['Unnamed: 0'])):
        raise ValueError('BLAST results contain query identifiers absent from the input sequences')
    
    df_merged = pd.merge(
        df_seqs,
        df_blast,
        left_on='Unnamed: 0',
        right_on='Query ID',
        how='left',
        validate='one_to_one'
    )    
    df_merged['blast_hit_reported'] = df_merged['Subject ID'].notna()
    df_merged.to_csv(osp.join(select_dir, 'predicts_sorted_cut_blast_merge.csv'), index=False)


def main(argv=None):
    global root_dir, blast_dir, predict_dir, select_dir
    parser = argparse.ArgumentParser(description='Legacy single-genome selection; core_nt audits are in 3evaluate/blast.py')
    parser.add_argument('--root-dir', default=root_dir, help='Repository root containing 2select/fasta')
    parser.add_argument('--blast-dir', default=blast_dir, help='Local BLAST+ installation containing bin/; otherwise use PATH')
    parser.add_argument('--predict-dir', help='Directory containing the four aligned, headerless prediction/input CSV files')
    parser.add_argument('--output-dir', help='Fresh output directory; defaults to ROOT/results/select')
    args = parser.parse_args(argv)
    root_dir = osp.abspath(args.root_dir)
    blast_dir = osp.abspath(args.blast_dir) if args.blast_dir else ''
    predict_dir = osp.abspath(args.predict_dir) if args.predict_dir else osp.join(root_dir, 'results', 'predicts')
    select_dir = osp.abspath(args.output_dir) if args.output_dir else osp.join(root_dir, 'results', 'select')
    os.makedirs(select_dir, exist_ok=False)
    sort_promoters()
    cutATG()
    csv2fasta()
    blast()
    blast_filter()
    select_blast()


if __name__ == '__main__':
    main()
