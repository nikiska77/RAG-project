import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

EMB_MODEL_NAME = "all-mpnet-base-v2"
CR_ENC_EMB_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
DB_TABLE_NAME = "summar_docs"


# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
retriever = SentenceTransformer(EMB_MODEL_NAME)
cross_encoder = CrossEncoder(CR_ENC_EMB_MODEL_NAME)

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
table = db.open_table(DB_TABLE_NAME)
