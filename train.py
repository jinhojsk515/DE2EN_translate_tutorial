from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
from dataset import DE_EN_dataset
from torch.utils.data import DataLoader
from models import DE2EN
import torch.optim as optim
import random
import numpy as np
from scheduler import create_scheduler
import time
import datetime
import os
from nltk.translate.bleu_score import corpus_bleu
from pathlib import Path


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train(model, data_loader, optimizer, de_tokenizer, en_tokenizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (de, en) in enumerate(tqdm_data_loader):
        de_input = de_tokenizer(de, padding='longest', return_tensors="pt").to(device)
        en_input = en_tokenizer(en, padding='longest', return_tensors="pt").to(device)

        loss = model(de_input, en_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


@torch.no_grad()
def generate(model, data_loader, de_tokenizer, en_tokenizer, device):
    model.eval()
    print('Generating samples for evaluation...')
    start_time = time.time()
    reference, candidate = [], []
    for de, en in data_loader:
        de_input = de_tokenizer(de, padding='longest', return_tensors="pt").to(device)
        de_embeds = model.de(de_input.input_ids, attention_mask=de_input.attention_mask,
                             return_dict=True).last_hidden_state

        text_input = torch.tensor([101]).expand(de_embeds.size(0), 1).to(device)  # batch*1

        for _ in range(100):
            output = model.generate(de_embeds, de_input.attention_mask, text_input)
            if output.sum() == 0:
                break
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(len(en)):
            reference.append(en[i])
            sentence = text_input[i, 1:]
            if 102 in sentence:
                sentence = sentence[:(sentence == 102).nonzero(as_tuple=True)[0][0].item()]
            cdd = en_tokenizer.convert_tokens_to_string(en_tokenizer.convert_ids_to_tokens(sentence))
            candidate.append(cdd.replace(' .', '.'))
    print(f'generation complete, time: {time.time()-start_time:.2f}s')
    return reference, candidate


@torch.no_grad()
def evaluate(generated):
    ref, cand = generated
    references = [[x.strip().strip(".").split()] for x in ref]
    candidates = [x.strip().strip(".").split() for x in cand]
    score = corpus_bleu(references, candidates)
    print('BLEU:', score)
    return score


def main():
    evaluation = False
    cp = None if evaluation is False else './output/checkpoint_best.pth'
    output_dir = './output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ### Dataset ### #
    print("Creating dataset")
    dataset_train = DE_EN_dataset('./data/train.de', './data/train.en')
    dataset_val = DE_EN_dataset('./data/test_2016_flickr.de', './data/test_2016_flickr.en')
    train_loader = DataLoader(dataset_train, batch_size=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=64, pin_memory=True, drop_last=False)
    print('#data', len(dataset_train))

    en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    de_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

    # ### Model ### #
    print("Creating model")
    model = DE2EN()
    print('total #parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if cp:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(cp, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % cp)
        print(msg)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    arg_sche = AttrDict({'sched': 'cosine', 'lr': 1e-4, 'epochs': 5, 'min_lr': 1e-5,
                         'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0})
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = 5
    warmup_steps = 1
    best = 0
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not evaluation:
            print('TRAIN', epoch)
            train(model, train_loader, optimizer, de_tokenizer, en_tokenizer, epoch, warmup_steps, device, lr_scheduler)

        print('VALIDATION')
        val_generated = generate(model, val_loader, de_tokenizer, en_tokenizer, device)
        val_stats = evaluate(val_generated)

        if not evaluation:
            if val_stats > best:
                save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))
                best = val_stats
                reference, candidate = val_generated
                with open('gt.txt', 'w') as w:
                    for r in reference:
                        w.write(r + '\n')
                with open('gen.txt', 'w') as w:
                    for c in candidate:
                        w.write(c + '\n')
        if evaluation:
            break
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
