# Copy and paste your OpenAI API Key
openai_api_key = "[OPENAI_API_KEY]"
GOOGLE_API_KEY = "[GOOGLE_API_KEY]"
# Put your name
key_owner = "[NAME]"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True

ORIGIN_PERSONA = "base_the_ville_persona_10"
SAVE_DIR = "john_eddy_fast_log_nomem_10_cuda6"

AGENT_MODEL = "llama-3.3-70b" # {"llama-3.3-70b", "gpt-4o-mini"}
TRAIN_METHOD = "wo_train" # {"DPO", "KTO", "SFT", "wo_train"}
PERSONA_GIVEN = "" # {"givenpersona", ""}
REASON_ = "noreason" # {"noreason", "reasonright", "reasonleft"}
MEMORY_RET = "wo_mem_suggest" # {"wo_mem_suggest", "ret_suggest"}
RANKING = "" # {"ranking", ""}
AGENT_MEMORY_LEVEL = "" # {"", "A", "AS", "ASC", "AS9C1", "AS8C2", "ASCR", "ASC9R1", "ASC8R2"} A: Action, S: Suggestion, C: Score, R: Reason
EVALUATOR = "gemini-2.0" # {"gpt-4o", "gpt-4o-mini", "gemini-2.0", "gemini-2.5", "o3-mini", "o4-mini"}
EVALUATE_SEPERATE = "seperate" # {"all", "seperate"}
FORCE_NOREC_RATIO = "norec0" # {"norec100", "norec70", "norec30", "norec0"}
VISUALIZATION = "novis" # {"vis", "novis"}
REPLAY_BUFFER = "equally200" # {"equally200", "equally400", "halftoday"}
REC_REVISION = "recnorev" # {"recrev", "recnorev"}
MMLU_EVAL = "" # {"mmlueval", ""}

TRAIN_MODE = f"{AGENT_MODEL}_{TRAIN_METHOD}_{PERSONA_GIVEN}_{REASON_}_{MEMORY_RET}_{RANKING}_{AGENT_MEMORY_LEVEL}_{EVALUATOR}_{EVALUATE_SEPERATE}_{FORCE_NOREC_RATIO}_{VISUALIZATION}_{REPLAY_BUFFER}_{REC_REVISION}_{MMLU_EVAL}"