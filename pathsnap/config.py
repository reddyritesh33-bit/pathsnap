import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get(
    "PATHSNAP_DATA",
    os.path.join(os.path.dirname(ROOT), "Student folder", "Student folder"),
)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
STEST_DIR = os.path.join(DATA_DIR, "student_test")
CACHE_DIR = os.path.join(ROOT, "cache")
ARTIFACTS = os.path.join(ROOT, "artifacts")
RESULTS = os.path.join(ROOT, "results")
SEED = 42
