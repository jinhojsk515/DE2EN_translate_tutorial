from torch.utils.data import Dataset


class DE_EN_dataset(Dataset):
    def __init__(self, de_path, en_path=None, data_length=None):
        with open(de_path, 'r') as f:
            de = f.readlines()
        self.de = [line.strip() for line in de]
        with open(en_path, 'r') as f:
            en = f.readlines()
        self.en = [line.strip() for line in en]
        if data_length:
            self.de = self.de[data_length[0]:data_length[1]]
            self.en = self.en[data_length[0]:data_length[1]]
        assert len(self.de) == len(self.en)

    def __len__(self):
        return len(self.de)

    def __getitem__(self, index):
        return self.de[index], self.en[index]
