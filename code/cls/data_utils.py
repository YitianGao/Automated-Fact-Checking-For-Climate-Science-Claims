from torch.utils.data import Dataset
import json
import torch


def process(text):
    return text.lower()


def to_cuda(batch):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].cuda()


class ClsDataset(Dataset):
    def __init__(self, mode, label2ids, tok, max_length=512):
        self.max_length = max_length
        if mode == "train":
            f = open("data/{}-claims.json".format(mode), "r")
        else:
            f = open("data/retrieval-{}-claims.json".format(mode), "r")
        self.dataset = json.load(f)
        f.close()
        f = open("data/evidence.json", "r")
        self.evidences = json.load(f)
        f.close()

        self.label2ids = label2ids
        self.tokenizer = tok
        # self.label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [process(data["claim_text"])]
        for evidence_id in data["evidences"]:
            input_text.append(process(self.evidences[evidence_id]))
        input_text = self.tokenizer.sep_token.join(input_text)
        if self.mode != "test":
            label = self.label2ids[data["claim_label"]]
        else:
            label = None
        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        labels = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)
            if self.mode != "test":
                labels.append(label)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        if self.mode != "test":
            batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding
