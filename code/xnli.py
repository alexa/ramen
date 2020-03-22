# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from src.logger import get_logger
from opt import get_parser

torch.manual_seed(528491)
parser = get_parser()
params = parser.parse_args()


class XNLIDataset(Dataset):
    def __init__(self, bin_path, max_length=312):
        data = torch.load(bin_path)
        self.xs = data['xs']
        self.ys = data['ys']

        if self.xs.size(1) > max_length:
            self.xs = self.xs[:, :max_length]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return {'x': self.xs[idx], 'y': self.ys[idx]}


class XNLINet(nn.Module):
    def __init__(self, params):
        super(XNLINet, self).__init__()
        self.encoder = AutoModel.from_pretrained(params.pretrained_path)
        # calculate the last hidden size
        out_dim = self.encoder.embeddings.word_embeddings.weight.size(-1)

        # make it float32
        for layer in self.encoder.modules():
            layer.float()
        # classifier
        self.pred = nn.Linear(out_dim, 3)

    def forward(self, tokens):
        # x: (bs, slen) tensor
        output = self.encoder(tokens)  # (bs, slen, dim)
        h = output[0]
        y = self.pred(h[:, 0])
        return y

def truncate(x, pad_index=0):
    length = x.ne(pad_index).sum(dim=1).max().item()
    return x[:, :length]

def evaluate(model, loader):
    # eval mode
    model.eval()
    cor, tot = 0, 0
    for batch in loader:
        with torch.no_grad():
            inputs = truncate(batch['x'])
            logit = model(inputs.cuda())
            _, yhat = logit.max(1)
        match = (yhat == batch['y'].cuda())
        cor += match.sum().item()
        tot += match.numel()
    # back to training mode
    model.train()
    return cor * 100 / tot


def test(params):
    """
    zero-shot testing of XNLI
    """
    logger = get_logger(params, f'test_{params.tgt_lang}_xnli.log')
    model = XNLINet(params)
    logger.info(f"| load: {params.xnli_model}")
    model.load_state_dict(torch.load(params.xnli_model))
    model.cuda()

    test_file = os.path.join(params.data_path, f"{params.tgt_lang}.test.pth")
    logger.info(f"| load test data: {test_file}")
    data = XNLIDataset(test_file)

    test_loader = DataLoader(data, num_workers=4, batch_size=params.batch_size)
    acc = evaluate(model, test_loader)

    logger.info(f'Zero-shot XNLI-{params.tgt_lang} accuracy: {acc:.1f}')


def train(params):
    """
    Tuning English model for XNLI task
    """
    # logging the results
    logger = get_logger(params, 'tune_{}_xnli.log'.format(params.src_lang))
    model = XNLINet(params)
    model = model.cuda()

    optimizer = torch.optim.Adam(
        list(model.encoder.encoder.layer.parameters()) +
        list(model.pred.parameters()), lr=params.lr)

    train_file = os.path.join(params.data_path, f"{params.src_lang}.train.pth")
    valid_file = os.path.join(params.data_path, f"{params.src_lang}.valid.pth")

    train_data = XNLIDataset(train_file)
    valid_data = XNLIDataset(valid_file)

    train_loader = DataLoader(train_data, num_workers=4,
                              batch_size=params.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, num_workers=4,
                              batch_size=params.batch_size)

    best_valid_acc = 0
    n_iter = 0
    for epoch in range(1, params.max_epoch):
        for batch in train_loader:
            inputs = truncate(batch['x'])
            output = model(inputs.cuda())
            loss = F.cross_entropy(output, batch['y'].cuda())
            loss.backward()
            if n_iter % params.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            n_iter += 1
            if n_iter % 50 == 0:
                logger.info("epoch {} - iter {} | XNLI loss {:.4f}"
                            .format(epoch, n_iter, loss.item()))
            if n_iter % params.epoch_size == 0:
                logger.info('run evaluation')
                val_acc = evaluate(model, valid_loader)
                logger.info("epoch {} - iter {} | XNLI validation acc {:.4f}"
                            .format(epoch, n_iter, val_acc))
                if val_acc > best_valid_acc:
                    logger.info(f'save best model: {params.xnli_model}')
                    best_valid_acc = val_acc
                    torch.save(model.state_dict(), params.xnli_model)
    logger.info('=== End of epoch ===')


def main():
    print(params)
    if params.src_lang:
        train(params)
    else:
        test(params)


if __name__ == '__main__':
    main()
