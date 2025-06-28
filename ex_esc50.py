import os
import sys

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred.config_helpers import DynamicIngredient, CMD
from torch.nn import functional as F
import numpy as np

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader

from config_updates import add_configs
from helpers.mixup import my_mixup
from helpers.models_size import count_non_zero_params
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from sklearn import metrics
from pytorch_lightning import Trainer as plTrainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import precision_score, recall_score, f1_score



ex = Experiment("passt_esc50")

# Example call with all the default config:
# python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"
# with 2 gpus:
# DDP=2 python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"

# capture the config of the trainer with the prefix "trainer", this allows to use sacred to update PL trainer config
# get_trainer = ex.command(plTrainer, prefix="trainer")
@ex.capture(prefix="trainer")
def get_trainer(_config):
    logger = TensorBoardLogger("logs/", default_hp_metric=False)

    # Patch logger to skip saving hparams (this avoids the OmegaConf crash)
    logger.log_hyperparams = lambda *args, **kwargs: None

    return plTrainer(logger=logger, **_config)



# capture the WandbLogger and prefix it with "wandb", this allows to use sacred to update WandbLogger config from the command line
# get_logger = ex.command(WandbLogger, prefix="wandb")
@ex.capture(prefix="wandb")
def get_logger(_config):
    return WandbLogger(**_config)


# define datasets and loaders
get_train_loader = ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=12,
                          num_workers=16, shuffle=None, dataset=CMD("/basedataset.get_training_set"),
                          )

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=20, num_workers=16,
                                            dataset=CMD("/basedataset.get_test_set"))


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    saque_cmd = os.environ.get("SAQUE_CMD", "").strip()
    saque_id = os.environ.get("SAQUE_ID", "").strip()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if os.environ.get("SLURM_ARRAY_JOB_ID", False):
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip() + "_" + os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    process_id = os.getpid()

    models = {
        "net": DynamicIngredient("models.passt.model_ing", n_classes=50, s_patchout_t=10, s_patchout_f=3),
        "mel": DynamicIngredient("models.preprocess.model_ing",
                                 instance_cmd="AugmentMelSTFT",
                                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                                 timem=80,
                                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                                 fmax_aug_range=2000)
    }

    basedataset = DynamicIngredient("esc50.dataset.dataset")

    trainer = dict(
        max_epochs=10,
        gpus=1,
        benchmark=True,
        num_sanity_val_steps=0,
        num_nodes=1
        # add more trainer args here if needed
    )

    wandb = dict(project="passt_esc50", log_model=True)

    lr = 0.00001
    use_mixup = True
    mixup_alpha = 0.3


# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.01,
                         schedule_mode="exp_lin"):
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")


@ex.command
def get_lr_scheduler(optimizer, schedule_mode):
    if schedule_mode in {"exp_lin", "cos_cyc"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_lambda())
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


@ex.command
def get_optimizer(params, lr, adamw=True, weight_decay=0.0001):
    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


class M(Ba3lModule):
    def __init__(self, experiment):
        self.mel = None
        self.da_net = None
        super(M, self).__init__(experiment)

        self.use_mixup = self.config.use_mixup or False
        self.mixup_alpha = self.config.mixup_alpha

        desc, sum_params, sum_non_zero = count_non_zero_params(self.net)
        self.experiment.info["start_sum_params"] = sum_params
        self.experiment.info["start_sum_params_non_zero"] = sum_non_zero

        # in case we need embedings for the DA
        self.net.return_embed = True
        self.dyn_norm = self.config.dyn_norm
        self.do_swa = False

        self.distributed_mode = self.config.trainer.num_nodes > 1

    def forward(self, x):
        return self.net(x)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        if self.dyn_norm:
            if not hasattr(self, "tr_m") or not hasattr(self, "tr_std"):
                tr_m, tr_std = get_dynamic_norm(self)
                self.register_buffer('tr_m', tr_m)
                self.register_buffer('tr_std', tr_std)
            x = (x - self.tr_m) / self.tr_std
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, f, y = batch
        assert y.dtype == torch.long, f"Expected torch.long labels, got {y.dtype}"
        if self.mel:
            x = self.mel_forward(x)

        orig_x = x
        batch_size = len(y)

        rn_indices, lam = None, None
        if self.use_mixup:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

        y_hat, embed = self.forward(x)

        if self.use_mixup:
            # y_mix = y * lam.reshape(batch_size, 1) + y[rn_indices] * (1. - lam.reshape(batch_size, 1))
            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
            loss = samples_loss.mean()
            loss = samples_loss.mean()
            samples_loss = samples_loss.detach()
        else:
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = samples_loss.mean()
            samples_loss = samples_loss.detach()

        results = {"loss": loss, }

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'train.loss': avg_loss, 'step': self.current_epoch}

        self.log_dict(logs, sync_dist=True)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        y_hat, _ = self.forward(x)
        return f, y_hat

    def validation_step(self, batch, batch_idx):
        x, f, y = batch
        assert y.dtype == torch.long, f"Expected torch.long labels, got {y.dtype}"
        if self.mel:
            x = self.mel_forward(x)

        results = {}
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]

        for net_name, net in model_name:
            y_hat, _ = net(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = samples_loss.mean()
            _, preds = torch.max(y_hat, dim=1)

            n_correct_pred_per_sample = (preds == y)
            n_correct_pred = n_correct_pred_per_sample.sum()

            # Include predictions and targets for metric computation
            results = {
                **results,
                net_name + "val_loss": loss,
                net_name + "n_correct_pred": torch.as_tensor(n_correct_pred),
                net_name + "n_pred": torch.as_tensor(len(y)),
                "preds": preds.detach().cpu(),
                "targets": y.detach().cpu()
            }

        results = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
        return results

    def validation_epoch_end(self, outputs):
        print(outputs)
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack([x[net_name + 'val_loss'] for x in outputs]).mean()
            val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

            preds = torch.cat([x['preds'] for x in outputs])
            targets = torch.cat([x['targets'] for x in outputs])

            precision = precision_score(targets, preds, average='macro', zero_division=0)
            recall = recall_score(targets, preds, average='macro', zero_division=0)
            f1 = f1_score(targets, preds, average='macro', zero_division=0)

            print(f"[Epoch {self.current_epoch}] {net_name}Val Acc: {val_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            # Show all in progress bar
            self.log(net_name + 'val.loss', avg_loss.cuda(), prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(net_name + 'val.acc', val_acc.cuda(), prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(net_name + 'precision', precision, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(net_name + 'recall', recall, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(net_name + 'f1', f1, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('step', torch.tensor(self.current_epoch).cuda(), sync_dist=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.parameters())
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

    def configure_callbacks(self):
        return get_extra_checkpoint_callback() + get_extra_swa_callback()


@ex.command
def get_dynamic_norm(model, dyn_norm=False):
    if not dyn_norm:
        return None, None
    raise RuntimeError('no dynamic norm supported yet.')


@ex.command
def get_extra_checkpoint_callback(save_last_n=None):
    if save_last_n is None:
        return []
    return [ModelCheckpoint(monitor="step", verbose=True, save_top_k=save_last_n, mode='max')]


@ex.command
def get_extra_swa_callback(swa=True, swa_epoch_start=2,
                           swa_freq=1):
    if not swa:
        return []
    print("\n Using swa!\n")
    from helpers.swa_callback import StochasticWeightAveraging
    return [StochasticWeightAveraging(swa_epoch_start=swa_epoch_start, swa_freq=swa_freq)]


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    trainer = get_trainer()
    train_loader = get_train_loader()
    val_loader = get_validate_loader()

    modul = M(ex)

    trainer.fit(
        modul,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # ✅ Save a proper checkpoint
    trainer.save_checkpoint("epoch_9.ckpt")
    print("✅ Saved model checkpoint to epoch_9.ckpt")

    return {"done": True}


@ex.command
def model_speed_test(_run, _config, _log, _rnd, _seed, speed_test_batch_size=100):
    '''
    Test training speed of a model
    @param _run:
    @param _config:
    @param _log:
    @param _rnd:
    @param _seed:
    @param speed_test_batch_size: the batch size during the test
    @return:
    '''

    modul = M(ex)
    modul = modul.cuda()
    batch_size = speed_test_batch_size
    print(f"\nBATCH SIZE : {batch_size}\n")
    test_length = 100
    print(f"\ntest_length : {test_length}\n")

    x = torch.ones([batch_size, 1, 128, 998]).cuda()
    target = torch.ones([batch_size, 527]).cuda()
    # one passe
    net = modul.net
    # net(x)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    # net = torch.jit.trace(net,(x,))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    print("warmup")
    import time
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(10):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('warmup done:', (t2 - t1))
    torch.cuda.synchronize()
    t1 = time.time()
    print("testing speed")

    for i in range(test_length):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('test done:', (t2 - t1))
    print("average speed: ", (test_length * batch_size) / (t2 - t1), " specs/second")


# @ex.command
# def evaluate_only(_run, _config, _log, _rnd, _seed):
#     # force overriding the config, not logged = not recommended
#     trainer = get_trainer()
#     train_loader = get_train_loader()
#     val_loader = get_validate_loader()

#     modul = M(ex)
#     modul.val_dataloader = None
#     #trainer.val_dataloaders = None
#     print(f"\n\nValidation len={len(val_loader)}\n")
#     res = trainer.validate(modul, dataloaders=val_loader)
#     print("\n\n Validtaion:")
#     print(res)

@ex.command
def evaluate_only(_run, _config, _log, _rnd, _seed):
    import torch
    from sklearn.metrics import precision_score, recall_score, f1_score

    # 1. Define the 7 class IDs
    target_ids = {23, 24, 28, 29, 31, 32, 38}
    target_id_to_index = {tid: idx for idx, tid in enumerate(sorted(target_ids))}

    # 2. Load val data
    val_loader = get_validate_loader()

    # 3. Filter only samples with target in our 7-class subset
    filtered = []
    for x, f, y in val_loader.dataset:
        if int(y) in target_ids:
            new_label = target_id_to_index[int(y)]  # remap to [0–6]
            filtered.append((x, new_label))

    print(f"✅ Using {len(filtered)} samples for 7-class evaluation.")

    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __getitem__(self, idx):
            return self.data[idx]
        def __len__(self):
            return len(self.data)

    subset_loader = torch.utils.data.DataLoader(SubsetDataset(filtered), batch_size=8, shuffle=False)

    # 4. Load model and weights from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    model = M(ex)
    model.eval().to(device)

    # ✅ Manually load state dict from ckpt
    ckpt_path = "epoch_9.ckpt"  # Adjust path as needed
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    print("📦 Checkpoint loaded.")

    # 5. Run inference
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in subset_loader:
            x = x.to(device)
            y = y.to(device)
            if model.mel:
                x = model.mel_forward(x)
            logits, _ = model.forward(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 6. Metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n🎯 7-Class Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

@ex.command
def evaluate_selected_classes(ckpt_path, csv_path, audio_dir):
    import pandas as pd
    import torchaudio
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import precision_score, recall_score, f1_score

    TARGET_CLASSES = ["Breathing", "Coughing", "Snoring", "Drinking", "Mouse click", "Keyboard typing", "Clock tick"]

    class ESC50Subset(Dataset):
        def __init__(self, csv_path, audio_dir, target_classes):
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["category"].isin(target_classes)]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
            self.audio_dir = Path(audio_dir)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            audio_path = self.audio_dir / row["filename"]
            waveform, sr = torchaudio.load(audio_path)
            waveform = torchaudio.functional.resample(waveform, sr, 32000)
            waveform = waveform.mean(0).unsqueeze(0)
            label = self.class_to_idx[row["category"]]
            return waveform, label

    # Start Sacred run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init model via Sacred
    model = M(ex)
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    dataset = ESC50Subset(csv_path, audio_dir, TARGET_CLASSES)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if model.mel:
                x = model.mel_forward(x)
            logits, _ = model.forward(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

@ex.command
def test_loaders():
    '''
    get one sample from each loader for debbuging
    @return:
    '''
    for i, b in enumerate(ex.datasets.training.get_iter()):
        print(b)
        break

    for i, b in enumerate(ex.datasets.test.get_iter()):
        print(b)
        break


def set_default_json_pickle(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError



def multiprocessing_run(rank, word_size):
    print("rank ", rank, os.getpid())
    print("word_size ", word_size)
    os.environ['NODE_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[rank]
    argv = sys.argv
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        argv = argv + ["-u"]  # only rank 0 is observed
    if "with" not in argv:
        argv = argv + ["with"]

    argv = argv + [f"trainer.num_nodes={word_size}", f"trainer.accelerator=ddp"]
    print(argv)

    @ex.main
    def default_command():
        return main()

    ex.run_commandline(argv)


if __name__ == '__main__':
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    word_size = os.environ.get("DDP", None)
    if word_size:
        import random

        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = f"{9999 + random.randint(0, 9999)}"  # plz no collisions
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'

        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked ")
                multiprocessing_run(rank, word_size)
                exit(0)

        pid, exit_code = os.wait()
        print(pid, exit_code)
        exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)


@ex.automain
def default_command():
    return main()
