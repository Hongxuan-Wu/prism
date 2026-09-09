import os
import pandas as pd
import torch
import pdb

from torch.utils.data import Dataset, DataLoader

# fromlist30
# {'[0.00, 10.00)': 54208,
# '[10.00, 100.00)': 52026,
# '[100.00, 500.00)': 26059,
# '[500.00, )': 8244}

# manual66
# {'[0.00, 10.00)': 138125,
# '[10.00, 100.00)': 112255,
# '[100.00, 500.00)': 52128,
# '[500.00, )': 20115}

# streptomyces
# {'[0.00, 10.00)': 83068,
#  '[10.00, 100.00)': 59346,
#  '[100.00, 500.00)': 20701,
#  '[500.00, )': 6725}

# vibrio
# {'[0.00, 10.00)': 32118,
#  '[10.00, 100.00)': 31736,
#  '[100.00, 500.00)': 20322,
#  '[500.00, )': 5356}

# genes_prokaryotes
# {'[0.00, 10.00)': 138125,
# '[10.00, 100.00)': 112255,
# '[100.00, 500.00)': 52128,
# '[500.00, )': 20115}

def load_dataframe(args, config):
    # read dataframe
    if config['mode'] == 'test':        
        if '_GeneExpression' in config['val_dataset']:
            if config['val_dataset'] == 'EColi_GeneExpression':
                df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/EColi_tpm_gene.txt'))
            elif config['val_dataset'] == 'Bsub_GeneExpression':
                df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Bsub_tpm_gene.txt'))
            elif config['val_dataset'] == 'Cglu_GeneExpression':
                df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Cglu_tpm_gene.txt'))
            elif config['val_dataset'] == 'Vibr_GeneExpression':
                df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Vibr_tpm_gene.txt'))
            elif config['val_dataset'] == 'WCFS1_GeneExpression':
                df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/WCFS1_tpm_gene.txt'))
            
            df_tmp.columns=['id', 'promoters', 'genes', 'intensity']
            
            df_tmp['intensity_cls'] = -1
            if config['num_class'] == 4:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[(df_tmp['intensity'] >= 100) & (df_tmp['intensity'] < 500), 'intensity_cls'] = 2
                df_tmp.loc[df_tmp['intensity'] >= 500, 'intensity_cls'] = 3
            elif config['num_class'] == 3:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 2
            elif config['num_class'] == 2:
                if not config['strong_weak']:
                    # activity
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 10, 'intensity_cls'] = 1
                else:
                    # strong & weak
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = -1
                    df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 1
                    df_tmp = df_tmp[df_tmp.intensity_cls != -1]
            val_dataset = df_tmp

        return val_dataset
    elif config['mode'] == 'predict':
        df_predict = pd.read_csv(config['predict_filepath'], delimiter=',', header=None)
        return df_predict

def make_loader(args, config, df_valid, tokenizer):
    if config['num_class'] not in (2, 3, 4):
        raise ValueError('num_class must be 2, 3, or 4')
    if df_valid.empty or not df_valid['intensity_cls'].isin(range(config['num_class'])).all():
        raise ValueError('Test labels must be nonempty integer class indices within num_class')
    valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['valid_batch_size'], 
        shuffle=False, 
        drop_last=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    data_loader = {
        'val': valid_loader
    }
    return data_loader

def make_loader_predict(args, config, df_predict, tokenizer):
    predict_dataset = PredictDataset(df=df_predict, tokenizer=tokenizer)
    predict_loader = DataLoader(
            predict_dataset, 
            batch_size=config['valid_batch_size'], 
            shuffle=False, 
            drop_last=False,
            num_workers=args.num_workers, 
            pin_memory=True
    )
    data_loader = {
        'predict': predict_loader
    }
    return data_loader 
    
class PredictDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PredictDataset, self).__init__()
        self.df = df
        
        file = df.to_numpy()
        sequence = file[:, 0].tolist()
        
        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx]

class PromoterDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PromoterDataset, self).__init__()
        self.df = df

        sequence = (df['promoters'] + df['genes']).tolist()
        
        # The configured loss, not the classes observed in a subset, determines the target dtype.
        self.labels = torch.tensor(df['intensity_cls'].tolist(), dtype=torch.float32)
        
        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.labels[idx]
