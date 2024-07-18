
PROJECT_DIR = '../../misc/nature_comm_2023_reproduce/'  # base dir to save outputs of analysis
R_LABELS = ['Asian', 'Black', 'White']
CXP_JPG_DIR = '../../../datasets/'
MIMIC_BASE_DIR = '../../../datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/'
MIMIC_JPG_DIR = MIMIC_BASE_DIR + 'physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
MIMIC_JPG_DIR_SMALL = MIMIC_JPG_DIR.replace('2.0.0/files', '2.0.0/files_small')  # containing pre-resized files
MODEL_SAVE_DIR = '../torchxrayvision/outputs/'
MIMIC_DCM_DIR = MIMIC_JPG_DIR.replace('/mimic-cxr-jpg/', '/mimic-cxr/')

CXP_LABELS = ["Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            'No Finding']

CXP_LABELS = sorted(CXP_LABELS)

NO_FINDINGS_TAG = 'No Finding'
ALL_VIEWS = ['AP', 'PA', 'LATERAL', 'LL']