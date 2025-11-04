# cafa6_max.py — Fast k‑mer KNN baseline with GO‑DAG closure
# Public domain / CC0

import os, math, gc
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable
import numpy as np
import pandas as pd
import networkx as nx
from Bio import SeqIO

class Config:
    ROOT = "/kaggle/input/cafa-6-protein-function-prediction" if os.path.exists("/kaggle/input") else "./data"
    TRAIN_SEQ = f"{ROOT}/Train/train_sequences.fasta"
    TRAIN_TERMS = f"{ROOT}/Train/train_terms.tsv"
    OBO_FILE = f"{ROOT}/Train/go-basic.obo"
    TEST_SEQ = f"{ROOT}/Test/testsuperset.fasta"
    OUTPUT = "/kaggle/working/submission.tsv" if os.path.exists("/kaggle/working") else "./submission.tsv"
    USE_EMBEDDINGS = False
    EMB_DIR = f"{ROOT}/Embeddings"
    KMER = 3
    HASHDIM = 1<<15
    TFIDF_ALPHA = 0.75
    TOPK = 50
    MIN_NEIGHBORS = 5
    MAX_TERMS_PER_PROT = 200
    THRESH = {"BP":0.35, "MF":0.30, "CC":0.30}
    GLOBAL_THRESH = 0.32

cfg = Config()

def _ingest_term_block(block, G, term_aspect):
    tid=None; is_obsolete=False; aspect=None; parents=[]
    for ln in block:
        if ln.startswith("id: GO:"):
            tid = ln.split("id: ")[1].strip()
        elif ln.startswith("is_obsolete:"):
            is_obsolete = (ln.split(":")[1].strip()=="true")
        elif ln.startswith("namespace:"):
            ns = ln.split("namespace:")[1].strip()
            aspect = {"biological_process":"BP","molecular_function":"MF","cellular_component":"CC"}.get(ns)
        elif ln.startswith("is_a: GO:"):
            parents.append(ln.split("is_a: ")[1].split(" ! ")[0].strip())
        elif ln.startswith("relationship: part_of GO:"):
            parents.append(ln.split("relationship: ")[1].split(" ! ")[0].split(" ")[1].strip())
    if tid and not is_obsolete:
        G.add_node(tid)
        if aspect: term_aspect[tid]=aspect
        for p in parents:
            G.add_edge(tid, p)

def load_go_obo(path):
    G = nx.DiGraph(); term_aspect={}
    with open(path, "r", encoding="utf-8") as f:
        block=[]
        for line in f:
            if line.strip()=="[Term]":
                if block: _ingest_term_block(block, G, term_aspect); block=[]
            elif line.strip()=="" and block:
                _ingest_term_block(block, G, term_aspect); block=[]
            else:
                block.append(line.rstrip("\n"))
        if block: _ingest_term_block(block, G, term_aspect)
    return G, term_aspect

def go_ancestors_closure(G, terms):
    out=set()
    for t in terms:
        if t in G:
            out.add(t); out.update(nx.algorithms.dag.ancestors(G, t))
    return list(out)

def read_fasta_ids_and_seq(path):
    recs={}
    with open(path, "r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            recs[str(rec.id)] = str(rec.seq)
    return recs

def load_train_terms(path):
    df = pd.read_csv(path, sep="\\t", dtype=str)
    prot_col = None
    for c in df.columns:
        if c.lower() in {"proteinid","protein_id","uniprot","id","sequence_id"}:
            prot_col=c; break
    if prot_col is None:
        prot_col = df.columns[0]
    go_cols = [c for c in df.columns if df[c].astype(str).str.contains(r"GO:\\d+").any()]
    if not go_cols:
        raise ValueError("No GO term column found")
    go_col = go_cols[0]
    out = df[[prot_col, go_col]].rename(columns={prot_col:"ProteinID", go_col:"GO_Term"}).dropna()
    out = out[out["GO_Term"].str.startswith("GO:")]
    out["ProteinID"]=out["ProteinID"].astype(str)
    out["GO_Term"]=out["GO_Term"].astype(str)
    return out

AA = set(list("ACDEFGHIKLMNPQRSTVWY"))
def hash_kmer(kmer, mod):
    h = 2166136261
    for ch in kmer:
        h ^= ord(ch); h = (h * 16777619) & 0xffffffff
    return h % mod

def seq_to_kmer_vec(seq, k, dim):
    x = np.zeros(dim, dtype=np.float32)
    s = seq.upper()
    for i in range(len(s)-k+1):
        kmer = s[i:i+k]
        if all((c in AA) for c in kmer):
            idx = hash_kmer(kmer, dim)
            x[idx]+=1.0
    return x

def build_feature_matrix(id2seq, k, dim):
    ids = list(id2seq.keys())
    X = np.zeros((len(ids), dim), dtype=np.float32)
    for i, pid in enumerate(ids):
        X[i] = seq_to_kmer_vec(id2seq[pid], k, dim)
        if (i+1)%20000==0:
            print(f"[feat] {i+1}/{len(ids)}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)+1e-8
    X = X / norms
    return X, ids

def compute_tfidf_weights(train_labels, all_terms, alpha=0.75):
    df = Counter()
    for terms in train_labels.values():
        df.update(set(terms))
    N = len(train_labels); w={}
    for t in all_terms:
        d = max(1, df.get(t, 0))
        w[t] = math.log(N / (d ** alpha))
    return w

def cosine_sim(a,b): return (a @ b.T)

def knn_predict(X_train, X_test, train_ids, test_ids, train_labels, term_weights, topk):
    B=4096; out={}
    for i in range(0, X_test.shape[0], B):
        Xt = X_test[i:i+B]
        sims = cosine_sim(Xt, X_train)
        idx = np.argpartition(-sims, kth=min(topk, sims.shape[1]-1), axis=1)[:, :topk]
        row_idx = np.arange(idx.shape[0])[:,None]
        top_sim = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-top_sim, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)
        for r in range(idx.shape[0]):
            pid = test_ids[i+r]
            votes = defaultdict(float)
            for j in range(idx.shape[1]):
                tr_pid = train_ids[idx[r, j]]
                s = float(top_sim[r, j])
                for term in train_labels.get(tr_pid, []):
                    votes[term] += s * term_weights.get(term, 1.0)
            out[pid] = dict(votes)
        print(f"[knn] {min(i+B, X_test.shape[0])}/{X_test.shape[0]}")
        del sims, idx, row_idx, top_sim, order; gc.collect()
    return out

def infer_aspect(term, term_aspect): return term_aspect.get(term, "BP")

def postprocess_predictions(raw_scores, G, term_aspect, per_aspect_thresh, global_thresh, max_terms):
    rows=[]
    for pid, d in raw_scores.items():
        keep=[]
        for t,s in d.items():
            asp = infer_aspect(t, term_aspect)
            th = per_aspect_thresh.get(asp, global_thresh)
            if s>=th: keep.append((t,s))
        keep_terms = [t for t,_ in sorted(keep, key=lambda x: -x[1])[:max_terms]]
        closed = go_ancestors_closure(G, keep_terms)
        max_score = defaultdict(float)
        for t,s in keep: max_score[t]=max(max_score[t], s)
        for anc in closed:
            if anc not in max_score:
                child_scores=[s for (t,s) in keep if nx.has_path(G,t,anc) and t!=anc]
                if child_scores: max_score[anc]=max(child_scores)*0.5
        for t,s in sorted(max_score.items(), key=lambda x: -x[1]):
            rows.append((pid, t, float(s)))
    return rows

def run_pipeline():
    print("[load] GO OBO"); G, term_aspect = load_go_obo(cfg.OBO_FILE)
    print("[load] Train terms"); df_terms = load_train_terms(cfg.TRAIN_TERMS)
    prot2terms = defaultdict(list)
    for pid, go in df_terms[["ProteinID","GO_Term"]].itertuples(index=False):
        prot2terms[pid].append(go)
    all_terms = sorted(df_terms["GO_Term"].unique().tolist())
    print("[load] Train sequences"); train_seq = read_fasta_ids_and_seq(cfg.TRAIN_SEQ)
    prot2terms = {pid:go for pid,go in prot2terms.items() if pid in train_seq}
    print("[load] Test sequences"); test_seq = read_fasta_ids_and_seq(cfg.TEST_SEQ)
    print("[feat] Hashed k‑mers")
    train_ids = list(prot2terms.keys()); test_ids = list(test_seq.keys())
    X_train,_ = build_feature_matrix({pid:train_seq[pid] for pid in train_ids}, cfg.KMER, cfg.HASHDIM)
    X_test,_  = build_feature_matrix({pid:test_seq[pid]  for pid in test_ids},  cfg.KMER, cfg.HASHDIM)
    term_w = compute_tfidf_weights(prot2terms, all_terms, alpha=cfg.TFIDF_ALPHA)
    raw = knn_predict(X_train, X_test, train_ids, test_ids, prot2terms, term_w, cfg.TOPK)
    rows = postprocess_predictions(raw, G, term_aspect, cfg.THRESH, cfg.GLOBAL_THRESH, cfg.MAX_TERMS_PER_PROT)
    sub = pd.DataFrame(rows, columns=["ProteinID","GO_Term","Score"])
    sub = sub[sub["GO_Term"].str.startswith("GO:")]
    sub = sub.sort_values(["ProteinID","Score"], ascending=[True, False])
    sub.to_csv(cfg.OUTPUT, sep="\\t", header=False, index=False)
    print(f"[done] Wrote {cfg.OUTPUT} with {len(sub):,} rows)")

if __name__=="__main__":
    ok = all(os.path.exists(p) for p in [cfg.OBO_FILE, cfg.TRAIN_SEQ, cfg.TRAIN_TERMS, cfg.TEST_SEQ])
    if ok: run_pipeline()
    else:
        print("Data not found. Set Config.ROOT to your local data dir or run on Kaggle.")
