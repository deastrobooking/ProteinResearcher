import subprocess
from pathlib import Path
from typing import Optional
import pandas as pd
from Bio import SeqIO
import tempfile


class DiamondRunner:
    def __init__(self, db_path: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path('cafa6_predictor/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path) if db_path else self.cache_dir / 'diamond.dmnd'
        
    def build_database(self, sequences: dict[str, str], force_rebuild: bool = False):
        if self.db_path.exists() and not force_rebuild:
            print(f"✓ DIAMOND database already exists: {self.db_path}")
            return
        
        print(f"Building DIAMOND database from {len(sequences)} sequences...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            fasta_path = f.name
            for protein_id, seq in sequences.items():
                f.write(f">{protein_id}\n{seq}\n")
        
        try:
            cmd = [
                'diamond', 'makedb',
                '--in', fasta_path,
                '-d', str(self.db_path.with_suffix('')),
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ DIAMOND database created: {self.db_path}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DIAMOND makedb failed: {e.stderr}")
        finally:
            Path(fasta_path).unlink()
    
    def run_blastp(
        self,
        query_sequences: dict[str, str],
        output_path: Optional[Path] = None,
        max_target_seqs: int = 100,
        evalue: float = 1e-3,
        sensitivity: str = 'sensitive',
        threads: int = 4
    ) -> pd.DataFrame:
        if output_path is None:
            output_path = self.cache_dir / 'diamond_blastp.tsv'
        
        output_path = Path(output_path)
        
        if output_path.exists():
            print(f"Loading cached DIAMOND results: {output_path}")
            return pd.read_csv(output_path, sep='\t', names=[
                'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
                'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'
            ])
        
        print(f"Running DIAMOND BLASTP on {len(query_sequences)} sequences...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            query_path = f.name
            for protein_id, seq in query_sequences.items():
                f.write(f">{protein_id}\n{seq}\n")
        
        try:
            cmd = [
                'diamond', 'blastp',
                '-q', query_path,
                '-d', str(self.db_path.with_suffix('')),
                '-o', str(output_path),
                '--outfmt', '6',
                '--max-target-seqs', str(max_target_seqs),
                '--evalue', str(evalue),
                f'--{sensitivity}',
                '--threads', str(threads),
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ DIAMOND BLASTP complete: {output_path}")
            
            df = pd.read_csv(output_path, sep='\t', names=[
                'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
                'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'
            ])
            
            return df
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DIAMOND blastp failed: {e.stderr}")
        finally:
            Path(query_path).unlink()
    
    def load_sequences_from_fasta(self, fasta_path: Path) -> dict[str, str]:
        sequences = {}
        for record in SeqIO.parse(fasta_path, 'fasta'):
            sequences[record.id] = str(record.seq)
        return sequences
