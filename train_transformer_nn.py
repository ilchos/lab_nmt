import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

import torchtext as tt
import sentencepiece as sp

from datetime import datetime
from pathlib import Path
import shutil
import inspect

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from utils.init_weights import init_weights
import utils.backup as backup
from utils.translate import translate_batch, translation2writer

from utils.moving_average import MovingAverage
from utils.utils_sp import *

from models.transformer_nn.transformer_nn_nopos import *

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 100 # 100
crop_len = 60
n_epochs = 100
blue_corpus_size = 500

# Load data
sp_procs = {ln: sp.SentencePieceProcessor(f".stor/sp_{ln}.model")
            for ln in [SRC_LN, TRG_LN]}

text2idx = {ln: text2idx_sentpiece(sp_procs[ln])
            for ln in [SRC_LN, TRG_LN]}
idx2text = {ln: sp_procs[ln].decode
            for ln in [SRC_LN, TRG_LN]}

fields = {ln: tt.data.Field(use_vocab=False,
                            tokenize=text2idx[ln],
                            is_target=(ln==TRG_LN),
                            # init_token=SOS_IDX, eos_token=EOS_IDX,
                            pad_token=PAD_IDX,
                            # batch_first=True,
                            fix_length=crop_len
                            )
          for ln in [SRC_LN, TRG_LN]}

train_data, val_data = tt.datasets.TranslationDataset.splits(
    path="./data/", train="train", validation="valid", test=None,
    exts=(".en", ".ru"),
    fields=[("src", fields[SRC_LN]), ("trg", fields[TRG_LN])]
)

# train_data, _ = train_data.split(0.01)
# val_data, _ = val_data.split(0.1)

val_text_data = tt.datasets.TranslationDataset(path="./data/valid.", exts=["en", "ru"],
                                               fields=[tt.data.Field(sequential=False, lower=True)]*2)
val_text_data = [(e.src, e.trg) for e in val_text_data.examples][:blue_corpus_size]

train_iter, val_iter = tt.data.BucketIterator.splits(
    (train_data, val_data), batch_sizes=(batch_size, batch_size),
    sort_key=lambda x: (len(x.src), len(x.trg)),
    device=device)
exit()
# Initialize model
model = Seq2Seq(*[sp_procs[ln].vocab_size()
                for ln in [SRC_LN, TRG_LN]]).to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {n_params:,} trainable parameters")
# ---------------

# Set up training
lr = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

scheduler_plateu = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                  factor=0.25, patience=10,
                                                  threshold=0.01)

def learning_rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

scheduler_warmup = lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate(step, 512, 1, 4000)
        )

# ---------------

### Checkpoint and Backups
load_model = False
dt = datetime.now()
dt_str = dt.strftime("%b%d_%H_%M_%S")

backup_location = Path("./backup")
backup_folder = backup_location / model_title / dt_str
backup_folder.mkdir(parents=True, exist_ok=True)

shutil.copy(Path(__file__), backup_folder/(f"train_{dt_str}.py")) # backup train file
shutil.copy(inspect.getsourcefile(type(model)), backup_folder) # backup model source file

start_epoch = 0
if load_model == True:
    dt_str_backup = "Mar08_13_52_24"
    checkpoint_fname = "checkpoint_19.pth"
    checkpoint_path = Path(backup_location) / model_title / dt_str_backup / checkpoint_fname
    start_epoch = backup.load_checkpoint(checkpoint_path, model,
                                         optimizer,
                                         scheduler_warmup
                                         )
    scheduler_warmup.last_epoch = start_epoch*len(train_iter)-1
    print("Scheduler last lr:", scheduler_warmup.get_last_lr())
    # scheduler_warmup.step()
    # scheduler_step.last_epoch = start_epoch-1
    # Manually change lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lr

print("Starting from epoch", start_epoch)
### ----------

writer = SummaryWriter(log_dir=backup_folder)
# writer.add_hparams(
#     dict(train_size=len(train_data),
#          val_size=len(val_data),
#          batch_size=batch_size,
#          train_batches=len(train_iter),
#          val_batches=len(val_iter)),
#     dict()
# )

clip = 1
ma_train = MovingAverage(50)
ma_val = MovingAverage(200)
for epoch in trange(start_epoch, n_epochs, desc="Epochs", total=n_epochs, initial=start_epoch):
    backup.save_checkpoint(backup_folder/f"checkpoint_{epoch}.pth", epoch,
                           model, optimizer, scheduler_warmup)

    last_lr = optimizer.state_dict()["param_groups"][0]["lr"]
    writer.add_scalar("Epochs/learning_rate", last_lr, epoch)

    model.train()
    train_loss = 0
    for i, (src, trg) in enumerate(tqdm(train_iter, desc="Train", leave=False)):
        trg_input = trg[:-1, :]

        logits = model(src, trg_input)

        trg_out = trg[1:, :].reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        # step optimizer
        optimizer.zero_grad()
        loss = criterion(logits, trg_out)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()

        scheduler_warmup.step()

        global_step = epoch*len(train_iter) + i
        writer.add_scalar("Training/train_loss", loss.item(), global_step)
        writer.add_scalar("Training/train_loss_ma", ma_train(loss.item()), global_step)

        last_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        writer.add_scalar("Training/learning_rate", last_lr, global_step)

    train_loss /= len(train_iter)
    writer.add_scalar("Epochs/train_loss", train_loss, epoch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(tqdm(val_iter, desc="Val", leave=False)):
            trg_input = trg[:-1, :]

            logits = model(src, trg_input)

            trg_out = trg[1:, :].reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])

            loss = criterion(logits, trg_out)
            val_loss += loss.item()

            ma_val(loss.item())
            # scheduler.step(ma_valid.get_moving_average())

            global_step = epoch*len(val_iter) + i
            writer.add_scalar("Validation/val_loss", loss.item(), global_step)
            writer.add_scalar("Validation/val_loss_ma", ma_val.value, global_step)
            # if global_step % lr_patience == 0:
            #     current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            #     writer.add_scalar("Validation/learning_rate", current_lr, global_step)

    val_loss /= len(val_iter)
    writer.add_scalar("Epochs/val_loss", val_loss, epoch)

    # sample translation of validation dataset
    # translations = [translate(model, src, sp_procs) for src, trg in val_text_data]
    # current_bleu_score = evaluate_bleu_score([trg for src, trg in val_text_data],
    #                                             translations)
    # translation_tuples = [(src, trg, out) for (src, trg), out in zip(val_text_data,
    #                                                                   translations)]
    translation_tuples, current_bleu_score = translate_batch(model, val_text_data,
                                                             text2idx[SRC_LN], idx2text[TRG_LN])

    # scheduler_plateu.step(current_bleu_score)

    writer.add_scalar("Epochs/bleu_score", current_bleu_score, epoch)
    writer.add_text("Sample Translation",
                    translation2writer(translation_tuples[:100]),
                    epoch)
    translation2file(backup_folder/"out.txt", translation_tuples, epoch)


backup.save_checkpoint(backup_folder/"final", epoch, model, optimizer, scheduler_plateu)
