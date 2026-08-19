import numpy as np
import pandas as pd
from functools import partial
import ast
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import csv
import glob
import os
import json
import argparse
import logging
from typing import Dict, Generator, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
import signal
import sys
import sqlite3
import gc

csv.field_size_limit(10 ** 7)

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qdrant_upload.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class UploadProgress:
    """Track upload progress for resumability"""
    folder: str
    file_index: int
    total_counter: int
    timestamp: str
    completed_files: List[str]

    def save(self, filepath: str):
        """Save progress to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> Optional['UploadProgress']:
        """Load progress from JSON file"""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return cls(**data)
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None


@dataclass
class Config:
    """Configuration for the upload process"""
    host: str
    port: int
    index: str
    scheme: str
    root_path: str
    folders: List[str]
    chunk_size: int = 1000
    max_retries: int = 10
    base_delay: float = 1.0
    vector_size: int = 768
    shard_number: int = 9
    replication_factor: int = 1
    id_offset: int = 0
    progress_file: str = "upload_progress.json"
    delete_existing: bool = False
    resume: bool = True
    prefetch_size: int = 5  # Number of vectors to prefetch
    exclude_prefixes: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create config from command line arguments"""
        return cls(
            host=args.host,
            port=args.port,
            index=args.index,
            scheme=args.scheme,
            root_path=args.root_path,
            folders=args.folders,
            chunk_size=args.chunk_size,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            vector_size=args.vector_size,
            shard_number=args.shard_number,
            replication_factor=args.replication_factor,
            id_offset=args.id_offset,
            progress_file=args.progress_file,
            delete_existing=args.delete_existing,
            resume=args.resume,
            prefetch_size=args.prefetch_size,
            exclude_prefixes=args.exclude_prefixes
        )


class GracefulShutdown:
    """Handle graceful shutdown on SIGINT/SIGTERM"""

    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.warning(f"Shutdown signal received ({signum}). Finishing current batch...")
        self.shutdown_requested = True


def open_numpy_pickle(file_path: str, sub_path: str = "") -> np.ndarray:
    """Open numpy file with memory mapping for efficiency"""
    arr = np.load(file_path, mmap_mode='r')
    if file_path.endswith('npz'):
        if not sub_path:
            for k in arr.files:
                sub_path = k
                break
        logger.info(f"Using NPZ subpath: {sub_path}")
        return arr[sub_path]
    return arr


def get_id_type_dict(file_path: str) -> Dict[str, str]:
    """Load ID to type mapping with progress tracking"""
    result = {}
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            total = sum(1 for _ in csvfile) - 1
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, total=total, desc=f"Loading {Path(file_path).name}"):
                result[row['id']] = ast.literal_eval(row['type'])[0]
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise
    return result


class NameIdSQLite:
    """Folder-scoped SQLite manager for name-to-ID lookups."""

    FILENAME = "name_ids.db"

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.csv_path = os.path.join(folder_path, "name_ids.csv")
        self.db_path = os.path.join(folder_path, self.FILENAME)
        self.conn: Optional[sqlite3.Connection] = None
        self.cur: Optional[sqlite3.Cursor] = None

    def prepare(self, batch_insert_size: int = 100_000):
        """Build SQLite DB from name_ids.csv if not already present."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No name_ids.csv found in {self.folder_path}")

        if os.path.exists(self.db_path):
            logger.info(f"Reusing existing SQLite DB: {self.db_path}")
            self.conn = sqlite3.connect(self.db_path)
            self.cur = self.conn.cursor()
            return

        logger.info(f"Building SQLite DB for folder: {self.folder_path}")
        # Use a temporary DB in same folder
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS name_ids (Name TEXT PRIMARY KEY, ID TEXT);")

        with open(self.csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            batch = []
            count = 0
            for row in reader:
                name = row.get('Name')
                id_val = row.get('ID')
                batch.append((name, id_val))
                count += 1
                if count % batch_insert_size == 0:
                    self.cur.executemany("INSERT OR REPLACE INTO name_ids VALUES (?, ?);", batch)
                    self.conn.commit()
                    batch.clear()
                    logger.info(f"  inserted {count:,} rows...")
            if batch:
                self.cur.executemany("INSERT OR REPLACE INTO name_ids VALUES (?, ?);", batch)
                self.conn.commit()

        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON name_ids(Name);")
        self.conn.commit()
        logger.info(f"SQLite DB ready: {self.db_path}")

    def get(self, name: str) -> Optional[str]:
        """Fetch ID for a given Name, or None if not found."""
        if not self.cur:
            raise RuntimeError("SQLite DB not prepared yet. Call .prepare() first.")
        row = self.cur.execute("SELECT ID FROM name_ids WHERE Name = ?", (name,)).fetchone()
        if not row:
            return None
        try:
            return ast.literal_eval(row[0])[0]
        except Exception:
            return row[0]

    def cleanup(self):
        """Close connection and delete DB file."""
        if self.conn:
            self.conn.close()
        gc.collect()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                logger.info(f"Deleted SQLite DB: {self.db_path}")
            except Exception as e:
                logger.warning(f"Could not delete sqlite DB {self.db_path}: {e}")


def iter_files(np_file: str,
               name_id_csv_path: str,
               id_type_file: str,
               type_id_dict: Dict,
               name_lookup: Optional[NameIdSQLite] = None,
               start_index: int = 0,
               exclude_prefixes: Optional[List[str]] = None) -> Generator:
    """Generate documents from files with optional start index for resume. This is streaming and memory-safe."""
    excluded = {p.upper() for p in (exclude_prefixes or [])}
    logger.info(f"Opening numpy array file: {np_file}")
    np_arr = open_numpy_pickle(np_file)
    logger.info(f"Found {np_arr.shape[0]} vector rows")

    with open(name_id_csv_path) as name_id_csv:
        total_rows = sum(1 for _ in name_id_csv) - 1
        name_id_csv.seek(0)
        reader = csv.DictReader(name_id_csv)

        # Skip to start_index if resuming
        for _ in range(start_index):
            next(reader)

        index = start_index
        for row in tqdm(reader, total=total_rows - start_index,
                        desc="Processing rows", initial=start_index):
            try:
                # Prefer explicit ID in row; otherwise try name lookup
                id_field = row.get('ID')
                if id_field and id_field.strip():
                    curies = ast.literal_eval(id_field)
                elif name_lookup:
                    lookup_id = name_lookup.get(row.get('Name'))
                    curies = [lookup_id] if lookup_id else []
                else:
                    curies = []

                if excluded:
                    curies = [c for c in curies if c.split(':', 1)[0].upper() not in excluded]
                    if not curies:
                        index += 1
                        continue

                vector = np_arr[index].astype(np.float32, copy=False)

                # Validate vector
                if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                    logger.warning(f"Invalid vector at index {index}, skipping")
                    index += 1
                    continue

                doc = {
                    "curies": curies,
                    "embedding": vector,
                    "name": row.get('Name', '').strip('"'),
                    "categories": [type_id_dict.get(c, "unknown") for c in curies]
                }
                index += 1
                yield doc
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                index += 1
                continue


class SAPQdrant:
    """Enhanced Qdrant client with retry logic and error handling"""

    def __init__(self, config: Config):
        self.client = AsyncQdrantClient(
            url=f"{config.scheme}://{config.host}:{config.port}",
            timeout=300  # 5 minute timeout for large operations
        )
        self.index = config.index
        self.config = config

    async def close(self):
        """Close the client connection"""
        await self.client.close()

    async def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            return await self.client.collection_exists(collection_name=self.index)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    async def count(self) -> int:
        """Number of points currently in the collection (0 if it does not exist)"""
        if not await self.collection_exists():
            return 0
        return (await self.client.count(collection_name=self.index, exact=True)).count

    async def delete_index(self):
        """Delete the index/collection"""
        if await self.collection_exists():
            logger.info(f"Deleting collection: {self.index}")
            return await self.client.delete_collection(collection_name=self.index)
        logger.info(f"Collection {self.index} does not exist")

    async def create_index(self):
        """Create the index/collection with optimized settings"""
        logger.info(f'Creating collection: {self.index}')
        try:
            return await self.client.create_collection(
                collection_name=self.index,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE,
                    on_disk=True
                ),
                shard_number=self.config.shard_number,
                replication_factor=self.config.replication_factor,
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    # Storage is NetApp NFS, so on_disk=True makes every HNSW
                    # graph hop a network round trip: measured 137.9ms vs 3.0ms
                    # p50. Recreating the collection with on_disk=True silently
                    # undoes that.
                    on_disk=False
                ),
                optimizers_config=OptimizersConfigDiff(
                    memmap_threshold=200000,
                    max_segment_size=10_000_000,
                    indexing_threshold=0,  # Start with indexing disabled
                    max_optimization_threads=4
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type="int8",
                        quantile=0.99,
                        # ~1KB per vector, against 64Gi per pod.
                        always_ram=True
                    )
                )
            )
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def disable_indexing(self):
        """Disable indexing for faster bulk upload"""
        logger.info("Disabling indexing for bulk upload")
        await self.client.update_collection(
            collection_name=self.index,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )

    async def enable_indexing(self):
        """Enable indexing after bulk upload"""
        logger.info("Enabling indexing")
        await self.client.update_collection(
            collection_name=self.index,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000
            )
        )

    async def safe_upsert(self, points: List[models.PointStruct]) -> bool:
        """Upsert with exponential backoff retry logic"""
        retries = 0
        delay = self.config.base_delay

        while retries < self.config.max_retries:
            try:
                await self.client.upsert(
                    collection_name=self.index,
                    wait=True,
                    points=points
                )
                return True
            except Exception as e:
                retries += 1
                if retries == self.config.max_retries:
                    logger.error(f"Failed to upsert after {retries} attempts: {e}")
                    raise

                logger.warning(f"Upsert failed (attempt {retries}/{self.config.max_retries}): {e}")
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        return False

    async def populate_index(self, generator: Generator,
                             start_counter: int = 0,
                             shutdown_handler: GracefulShutdown = None) -> int:
        """Populate index with batch processing and progress tracking"""
        await self.disable_indexing()

        to_insert = []
        counter = start_counter
        batch_start_time = datetime.now()

        try:
            for data in generator:
                if shutdown_handler and shutdown_handler.shutdown_requested:
                    logger.warning("Shutdown requested, finishing current batch...")
                    break

                vector = np.array(data['embedding']).astype(np.float32)
                payload = {
                    "curie": data['curies'][0] if data['curies'] else None,
                    "name": data['name'],
                    "categories": data['categories'][0] if data['categories'] else None
                }

                to_insert.append(
                    models.PointStruct(id=counter, vector=vector, payload=payload)
                )

                if len(to_insert) >= self.config.chunk_size:
                    await self.safe_upsert(to_insert)

                    # Log batch performance
                    batch_duration = (datetime.now() - batch_start_time).total_seconds()
                    rate = self.config.chunk_size / batch_duration if batch_duration > 0 else 0
                    logger.info(f"Upserted batch at counter {counter}, rate: {rate:.1f} vectors/sec")

                    to_insert = []
                    batch_start_time = datetime.now()

                counter += 1

            # Insert remaining
            if len(to_insert) > 0:
                await self.safe_upsert(to_insert)
                logger.info(f"Upserted final batch of {len(to_insert)} vectors")

        except Exception as e:
            logger.error(f"Error during populate_index: {e}")
            raise

        return counter


async def recreate_index(client: SAPQdrant, delete: bool = False):
    """Recreate the index"""
    if delete:
        try:
            await client.delete_index()
            logger.info("Index deleted")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")

    if not await client.collection_exists():
        await client.create_index()
        logger.info("Index created")
    else:
        logger.info("Index already exists")


async def main(config: Config):
    """Main execution function with resume capability"""
    shutdown_handler = GracefulShutdown()

    # Load progress if resuming
    progress = None
    if config.resume:
        progress = UploadProgress.load(config.progress_file)
        if progress:
            logger.info(f"Resuming from folder: {progress.folder}, "
                        f"file: {progress.file_index}, "
                        f"counter: {progress.total_counter}")

    # Initialize client
    client = SAPQdrant(config)

    try:
        # Create index if needed
        await recreate_index(client, delete=config.delete_existing)

        # Point ids are sequential integers from `counter`, and upsert replaces
        # on collision. Without a progress file to carry the previous run's
        # counter forward, starting at an unconsidered 0 is how you overwrite
        # points. Make the caller say what they mean.
        if not progress and config.id_offset == 0 and not config.delete_existing:
            existing = await client.count()
            if existing:
                raise SystemExit(
                    f"Refusing to start: collection '{config.index}' already holds "
                    f"{existing:,} points, there is no progress file to resume from, "
                    f"and --id-offset is 0. Pass --id-offset above the highest existing "
                    f"id, or --delete-existing to deliberately wipe the collection."
                )

        counter = progress.total_counter if progress else config.id_offset
        logger.info(f"First point id for this run: {counter}")
        start_folder_idx = 0

        # Find resume point
        if progress:
            try:
                start_folder_idx = config.folders.index(progress.folder)
            except ValueError:
                logger.warning(f"Progress folder {progress.folder} not found, starting from beginning")
                progress = None

        # Process folders
        for folder_idx, folder in enumerate(config.folders[start_folder_idx:], start=start_folder_idx):
            if shutdown_handler.shutdown_requested:
                break

            logger.info(f"Processing folder {folder_idx + 1}/{len(config.folders)}: {folder}")

            folder_path = os.path.join(config.root_path, folder)
            embeddings = sorted(glob.glob(os.path.join(folder_path, "embedding", "*.np*")))
            name_id_chunk_files = sorted(glob.glob(os.path.join(folder_path, "metadata", "name_ids*")))
            id_type_file = os.path.join(folder_path, "metadata", "id_types.csv")
            master_name_id_file = os.path.join(folder_path, "name_ids.csv")

            if not embeddings or not name_id_chunk_files:
                logger.warning(f"No files found in {folder_path}, skipping")
                continue

            # Build or reuse sqlite for this folder (from master name_ids.csv)
            name_lookup = None
            try:
                name_lookup = NameIdSQLite(folder_path)
                name_lookup.prepare()
            except FileNotFoundError:
                # If there's no master name_ids.csv, we'll still process using per-chunk metadata
                logger.info(f"No master name_ids.csv for {folder_path}; proceeding without sqlite")
                name_lookup = None

            # Load id->type mapping
            logger.info(f"Loading id_types from {id_type_file}")
            type_id_dict = get_id_type_dict(id_type_file)

            # Determine start file index
            start_file_idx = 0
            completed_files = []
            if progress and progress.folder == folder:
                start_file_idx = progress.file_index
                completed_files = progress.completed_files

            # Process files
            for file_idx, (name_id, emb) in enumerate(
                    zip(name_id_chunk_files[start_file_idx:], embeddings[start_file_idx:]),
                    start=start_file_idx
            ):
                if shutdown_handler.shutdown_requested:
                    break

                if name_id in completed_files:
                    logger.info(f"Skipping already completed file: {name_id}")
                    continue

                logger.info(f"Processing file {file_idx + 1}/{len(name_id_chunk_files)}: {name_id}")

                try:
                    counter = await client.populate_index(
                        iter_files(emb, name_id, id_type_file, type_id_dict, name_lookup,
                                   exclude_prefixes=config.exclude_prefixes),
                        counter,
                        shutdown_handler
                    )

                    completed_files.append(name_id)

                    # Save progress
                    progress = UploadProgress(
                        folder=folder,
                        file_index=file_idx + 1,
                        total_counter=counter,
                        timestamp=datetime.now().isoformat(),
                        completed_files=completed_files
                    )
                    progress.save(config.progress_file)
                    logger.info(f"Progress saved. Total vectors uploaded: {counter}")

                except Exception as e:
                    logger.error(f"Failed to process file {name_id}: {e}")
                    # Save progress before exiting
                    if progress:
                        progress.save(config.progress_file)
                    raise

            # Cleanup sqlite for this folder to reclaim disk space
            if name_lookup:
                try:
                    name_lookup.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up sqlite for {folder_path}: {e}")

            # Clear completed files for next folder
            completed_files = []

        # Enable indexing after all uploads
        if not shutdown_handler.shutdown_requested:
            logger.info("Upload complete. Enabling indexing...")
            await client.enable_indexing()
            logger.info(f"Total vectors uploaded: {counter}")
        else:
            logger.warning("Upload interrupted. Progress saved. Re-run with --resume to continue.")

    finally:
        await client.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Upload embeddings to Qdrant with resume capability"
    )

    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--scheme", default="http", choices=["http", "https"], help="Connection scheme")
    parser.add_argument("--index", default="sapbert", help="Index/collection name")
    parser.add_argument("--root-path", required=True, help="Root path to data")
    parser.add_argument("--folders", nargs="+", required=True, help="Folders to process")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Batch size for upsert")
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum retry attempts")
    parser.add_argument("--base-delay", type=float, default=1.0, help="Base delay for retry backoff")
    parser.add_argument("--vector-size", type=int, default=768, help="Vector dimension size")
    parser.add_argument("--shard-number", type=int, default=9, help="Number of shards")
    parser.add_argument("--replication-factor", type=int, default=1, help="Replication factor")
    parser.add_argument("--id-offset", type=int, default=0,
                        help="First point id to use when not resuming. Point ids are "
                             "sequential integers and upsert replaces on collision, so "
                             "loading into a populated collection must start above the "
                             "highest existing id.")
    parser.add_argument("--progress-file", default="upload_progress.json", help="Progress file path")
    parser.add_argument("--delete-existing", action="store_true", help="Delete existing index")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume")
    parser.add_argument("--prefetch-size", type=int, default=5, help="Prefetch buffer size")
    parser.add_argument("--exclude-prefixes", nargs="*", default=[],
                        help="Skip rows whose curie prefix matches (e.g. PUBCHEM.COMPOUND UMLS). Case-insensitive.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config.from_args(args)

    logger.info("=" * 80)
    logger.info("Starting Qdrant Upload Process")
    logger.info(f"Configuration: {config}")
    logger.info("=" * 80)

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
