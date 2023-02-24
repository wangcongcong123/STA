from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encoded_examples):
        self.encoded_examples = encoded_examples

    def __getitem__(self, index):
        selected_to_return = {}
        for k, v in self.encoded_examples.items():
            selected_to_return[k] = v[index]
        return selected_to_return

    def __len__(self):
        return len(self.encoded_examples["input_ids"])
