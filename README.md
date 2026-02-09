# ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation

## Setting Up the Environment 
To set up your environment, you will need to generate a `utils.py` file that contains your OpenAI, Google API keys and download the necessary packages.

### Step 1. Generate Utils File
In the `reverie/backend_server` folder (where `reverie.py` is located), create a new file titled `utils.py` and copy and paste the content below into the file:
```
# Copy and paste your OpenAI API Key
openai_api_key = <<YOUR OPENAI API KEY>>
GOOGLE_API_KEY = <<YOUR GEMINI API KEY>>
# Put your name
key_owner = <name>

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True

ORIGIN_PERSONA = "base_the_ville_persona_1"
SAVE_DIR = "john_eddy_fast_log_test_1"

AGENT_MODEL = "llama-3.3-70b" # {"llama-3.3-70b", "gpt-4o-mini"}
TRAIN_METHOD = "DPO" # {"DPO", "KTO", "wo_train"}
PERSONA_GIVEN = "" # {"givenpersona", ""}
REASON_ = "noreason" # {"noreason", "reasonright", "reasonleft"}
MEMORY_RET = "ret_suggest" # {"wo_mem_suggest", "ret_suggest"}
RANKING = "ranking" # {"ranking", ""}
AGENT_MEMORY_LEVEL = "AS" # {"", "A", "AS", "ASC", "AS9C1", "AS8C2", "ASCR", "ASC9R1", "ASC8R2"} A: Action, S: Suggestion, C: Score, R: Reason
EVALUATOR = "gemini-2.0" # {"gpt-4o", "gpt-4o-mini", "gemini-2.0", "gemini-2.5", "o3-mini", "o4-mini"}
EVALUATE_SEPERATE = "seperate" # {"all", "seperate"}
FORCE_NOREC_RATIO = "norec0" # {"norec100", "norec70", "norec30", "norec0"}
VISUALIZATION = "novis" # {"vis", "novis"}
REPLAY_BUFFER = "equally200" # {"equally200", "equally400", "halftoday"}
REC_REVISION = "recnorev" # {"recrev", "recnorev"}
MMLU_EVAL = "" # {"mmlueval", ""}

TRAIN_MODE = f"{AGENT_MODEL}_{TRAIN_METHOD}_{PERSONA_GIVEN}_{REASON_}_{MEMORY_RET}_{RANKING}_{AGENT_MEMORY_LEVEL}_{EVALUATOR}_{EVALUATE_SEPERATE}_{FORCE_NOREC_RATIO}_{VISUALIZATION}_{REPLAY_BUFFER}_{REC_REVISION}_{MMLU_EVAL}"

HF_CACHE_DIR = "./"

# Verbose 
debug = True
```
Replace `<<YOUR OPENAI API KEY>>`, `<<YOUR GEMINI API KEY>>` with your keys, and `<name>` with your name.
 
### Step 2. Install requirements.txt
1. Create a conda environment using requirements.txt.
`conda create -n propersim python=3.10`

2. `conda activate propersim`

3. `pip install -r requirements.txt`

## Running a Simulation (Non-visualization Mode)
When visualization is disabled, the simulation can run efficiently. If you set VISUALIZATION in utils.py to "novis", the simulation will run without visualization. Navigate to `reverie/backend_server` and run `reverie.py`.

    python reverie.py

## Running a Simulation (Visualization Mode) 
To run a new simulation, you will need to concurrently start two servers: the environment server and the agent simulation server. 

### Step 1. Starting the Environment Server
Again, the environment is implemented as a Django project, and as such, you will need to start the Django server. To do this, first navigate to `environment/frontend_server` (this is where `manage.py` is located) in your command line. Then run the following command:

    python manage.py runserver

Then, on your favorite browser, go to [http://localhost:8000/](http://localhost:8000/). If you see a message that says, "Your environment server is up and running," your server is running properly. Ensure that the environment server continues to run while you are running the simulation, so keep this command-line tab open! (Note: I recommend using either Chrome or Safari. Firefox might produce some frontend glitches, although it should not interfere with the actual simulation.)

If there are some errors, try ``python manage.py runserver 7777``


### Step 2. Starting the Simulation Server
Open up another command line (the one you used in Step 1 should still be running the environment server, so leave that as it is). Navigate to `reverie/backend_server` and run `reverie.py`.

    python reverie.py
This will start the simulation server. 

### Step 3. Running and Saving the Simulation
On your browser, navigate to [http://localhost:8000/simulator_home](http://localhost:8000/simulator_home). You should see the map of Smallville, along with a list of active agents on the map. You can move around the map using your keyboard arrows. Please keep this tab open. 
If you run ``python manage.py runserver 7777`` in step 1, then navigate to [http://localhost:7777/simulator_home](http://localhost:7777/simulator_home)

### Step 4. Replaying a Simulation
You can replay a simulation that you have already run simply by having your environment server running and navigating to the following address in your browser: `http://localhost:8000/replay/<simulation-name>/<starting-time-step>`. Please make sure to replace `<simulation-name>` with the name of the simulation you want to replay, and `<starting-time-step>` with the integer time-step from which you wish to start the replay.

For instance, by visiting the following link, you will initiate a pre-simulated example, starting at time-step 1:  
[http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/](http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/)

### Step 5. Demoing a Simulation
You may have noticed that all character sprites in the replay look identical. We would like to clarify that the replay function is primarily intended for debugging purposes and does not prioritize optimizing the size of the simulation folder or the visuals. To properly demonstrate a simulation with appropriate character sprites, you will need to compress the simulation first. To do this, open the `compress_sim_storage.py` file located in the `reverie` directory using a text editor. Then, execute the `compress` function with the name of the target simulation as its input. By doing so, the simulation file will be compressed, making it ready for demonstration.

To start the demo, go to the following address on your browser: `http://localhost:8000/demo/<simulation-name>/<starting-time-step>/<simulation-speed>`. Note that `<simulation-name>` and `<starting-time-step>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)


## Acknowledgement
The foundation of this code is based on [Generative Agents](https://github.com/joonspk-research/generative_agents), and we would like to express our gratitude to the authors for providing their code.


