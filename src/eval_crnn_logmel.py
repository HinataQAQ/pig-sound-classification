import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset_logmel import LogMelDataset
from src.models.crnn import CRNNClassifier


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_csv', type=str, default='eval_crnn_predictions.csv')
    ap.add_argument('--thr', type=float, default=None, help='binary-only: prob(pos_label)>=thr => pos_label')
    ap.add_argument('--pos_label', type=str, default='cough')
    return ap.parse_args()


def build_model(cfg: dict, labels):
    mcfg = cfg.get('model', {})
    return CRNNClassifier(
        n_mels=int(cfg.get('logmel', {}).get('n_mels', 64)),
        num_classes=len(labels),
        cnn_channels=tuple(mcfg.get('cnn_channels', [16, 32, 64])),
        rnn_hidden=int(mcfg.get('rnn_hidden', 128)),
        rnn_layers=int(mcfg.get('rnn_layers', 2)),
        dropout=float(mcfg.get('dropout', 0.0)),
    )


def main():
    args = parse()
    cfg_file = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] device =', device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', cfg_file)
    if not isinstance(cfg, dict):
        cfg = cfg_file

    labels = [str(x).strip().lower() for x in ckpt.get('labels', cfg['data']['labels'])]
    labels_lower = [str(x).lower() for x in labels]

    ds = LogMelDataset(
        manifest_path=args.manifest,
        labels=labels,
        sr=int(cfg['data'].get('sample_rate', 32000)),
        segment_seconds=float(cfg['data'].get('segment_seconds', 1.0)),
        n_mels=int(cfg.get('logmel', {}).get('n_mels', 64)),
        fmin=float(cfg.get('logmel', {}).get('fmin', 50)),
        fmax=float(cfg.get('logmel', {}).get('fmax', 8000)),
        hop_ms=float(cfg.get('logmel', {}).get('hop_ms', 10.0)),
        win_ms=float(cfg.get('logmel', {}).get('win_ms', 25.0)),
        n_fft=int(cfg.get('logmel', {}).get('n_fft', 1024)),
        use_specaug=False,
        seed=int(cfg.get('train', {}).get('seed', 3407)),
        train=False,
        do_denoise=bool(cfg.get('preprocess', {}).get('do_denoise', False)),
        ref_mode=str(cfg.get('logmel', {}).get('ref_mode', 'fixed')),
        db_ref=float(cfg.get('logmel', {}).get('db_ref', 1.0)),
        top_db=float(cfg.get('logmel', {}).get('top_db', 80.0)),
        cmvn=str(cfg.get('logmel', {}).get('cmvn', 'none')),
        return_path=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg, labels).to(device)
    incompatible = model.load_state_dict(ckpt['state_dict'], strict=True)
    print('[INFO] load_state_dict =', incompatible)
    model.eval()

    pos_label = str(args.pos_label).lower()
    pos_idx = labels_lower.index(pos_label) if pos_label in labels_lower else 0
    neg_idx = 1 - pos_idx if len(labels) == 2 else None

    id2label = {i: l for i, l in enumerate(labels)}
    rows = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            if args.thr is not None and len(labels) == 2:
                p_pos = probs[:, pos_idx]
                pred_id = torch.where(
                    p_pos >= float(args.thr),
                    torch.full_like(y, pos_idx),
                    torch.full_like(y, neg_idx),
                )
            else:
                pred_id = probs.argmax(dim=1)

            y_cpu = y.cpu().numpy()
            pred_cpu = pred_id.cpu().numpy()
            prob_cpu = probs.cpu().numpy()

            for i in range(len(paths)):
                rec = {
                    'filepath': paths[i],
                    'y_true': id2label[int(y_cpu[i])],
                    'y_pred': id2label[int(pred_cpu[i])],
                }
                for k, lab in enumerate(labels):
                    rec[f'prob_{lab}'] = float(prob_cpu[i, k])
                rows.append(rec)

            y_true_all.extend(list(y_cpu))
            y_pred_all.extend(list(pred_cpu))

    y_true_all = np.asarray(y_true_all, dtype=np.int64)
    y_pred_all = np.asarray(y_pred_all, dtype=np.int64)
    acc = float((y_true_all == y_pred_all).mean())
    print(f'[OK] ACC={acc:.4f} ({int((y_true_all == y_pred_all).sum())}/{len(y_true_all)})')
    print('\n[REPORT]\n' + classification_report(
        y_true_all,
        y_pred_all,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    ))

    print('\n[CONFUSION]\n', confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=list(range(len(labels)))
    ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print('[OK] wrote ->', os.path.abspath(args.out_csv))

    if len(labels) == 2 and f'prob_{labels[pos_idx]}' in df.columns:
        print('\n== prob_pos mean by true ==')
        print(df.groupby('y_true')[f'prob_{labels[pos_idx]}'].mean().to_dict())


if __name__ == '__main__':
    main()
