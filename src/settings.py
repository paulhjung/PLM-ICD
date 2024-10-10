from omegaconf import OmegaConf

PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"

ID_COLUMN = "_id"
TEXT_COLUMN = "TEXT"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

DOWNLOAD_DIRECTORY_MIMICIII = (
    "~/Desktop/mimic-iii-1.4"  # Path to the MIMIC-III data. Example: ~/mimiciii/1.4
)
DOWNLOAD_DIRECTORY_MIMICIV = "~/Desktop/mimic/mimic-iv-2.2"  # Path to the MIMIC-IV data. Example: ~/physionet.org/files/mimiciv/2.2
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "~/Desktop/mimic/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/"  # Path to the MIMIC-IV-Note data. Example: ~/physionet.org/files/mimic-iv-note/2.2
DATA_DIRECTORY_MIMICIV_ICD10 = "~/Documents/Github/PLM-ICD/data/mimic4"

PROJECT = "<your project name>" # this variable is used for genersating plots and tables from wandb
EXPERIMENT_DIR = "files/"  # Path to the experiment directory. Example: ~/experiments
PALETTE = {
    "PLM-ICD": "#E69F00",
    "LAAT": "#009E73",
    "MultiResCNN": "#D55E00",
    "CAML": "#56B4E9",
    "CNN": "#CC79A7",
    "Bi-GRU": "#F5C710",
}
HUE_ORDER = ["PLM-ICD", "LAAT", "MultiResCNN", "CAML", "Bi-GRU", "CNN"]
MODEL_NAMES = {"PLMICD": "PLM-ICD", "VanillaConv": "CNN", "VanillaRNN": "Bi-GRU"}
