import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = "all-mpnet-base-v2"
DB_TABLE_NAME = "summar_docs"

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
retriever = SentenceTransformer(EMB_MODEL_NAME)

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
print(db_uri)
db = lancedb.connect(db_uri)
table = db.open_table(DB_TABLE_NAME)
