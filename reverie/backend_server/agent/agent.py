import math
import sys
import datetime
import random
import torch
import gc
import numpy as np
import os
import pickle
from collections import deque

sys.path.append('../')

from global_methods import *
from opensource_inference import *
from opensource_training_methods import *

from agent.memory_structures.spatial_memory import *
from agent.memory_structures.associative_memory import *
from agent.memory_structures.eval_memory import *
from agent.memory_structures.scratch import *

from agent.cognitive_modules.perceive import *
from agent.cognitive_modules.retrieve import *
from agent.cognitive_modules.plan import *
from agent.cognitive_modules.reflect import *
from agent.cognitive_modules.execute import *
from agent.cognitive_modules.converse import *
from agent.cognitive_modules.suggestion import *

class Agent: 
  def __init__(self, name, suggestion_model_name=None, folder_mem_saved=False, client=None, hf_cache_dir="./cache"):
    # PERSONA BASE STATE 
    # <name> is the full name of the agent. This is a unique identifier for
    # the agent within Reverie. 
    self.name = name

    # PERSONA MEMORY 
    # If there is already memory in folder_mem_saved, we load that. Otherwise,
    # we create new memory instances. 
    # <s_mem> is the agent's spatial memory. 
    f_s_mem_saved = f"{folder_mem_saved}/bootstrap_memory/spatial_memory.json"
    self.s_mem = MemoryTree(f_s_mem_saved)
    # <a_mem> is the agent's associative memory. 
    f_a_mem_saved = f"{folder_mem_saved}/bootstrap_memory/associative_memory"
    self.a_mem = AssociativeMemory(f_a_mem_saved)
    # <e_mem> is the persona's associative memory. 
    f_e_mem_saved = f"{folder_mem_saved}/bootstrap_memory/evaluation_memory"
    self.e_mem = EvaluationMemory(f_e_mem_saved)
    # <scratch> is the agent's scratch (short term memory) space. 
    scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
    self.scratch = Scratch(scratch_saved)
    
    self.client = client
    self.suggestion_model_name = suggestion_model_name ### modelname_trainingmethod_reason
    self.suggestion_model = None
    self.current_prompt = None
    self.current_no_recommendation_prompt = None
    self.preference_records = {
      "chosen": [],
      "rejected": [],
      "score": []
    }
    self.total_preference_records = {
      "chosen": [],
      "rejected": [],
      "score": []
    }
    self.total_evaluation_records = {}
    self.recent_events_hourly = deque(maxlen=24)
    self.summary_recent_events_hourly = ""
    self.memory_hourly = []
    self.events_hourly = []
    self.memory_4hourly = deque(maxlen=1)
    self.hf_cache_dir = hf_cache_dir



  def save(self, save_folder): 
    """
    Save agent's current state (i.e., memory). 

    INPUT: 
      save_folder: The folder where we wil be saving our agent's state. 
    OUTPUT: 
      None
    """
    # Spatial memory contains a tree in a json format. 
    # e.g., {"double studio": 
    #         {"double studio": 
    #           {"bedroom 2": 
    #             ["painting", "easel", "closet", "bed"]}}}
    f_s_mem = f"{save_folder}/spatial_memory.json"
    self.s_mem.save(f_s_mem)
    
    # Associative memory contains a csv with the following rows: 
    # [event.type, event.created, event.expiration, s, p, o]
    # e.g., event,2022-10-23 00:00:00,,Isabella Rodriguez,is,idle
    f_a_mem = f"{save_folder}/associative_memory"
    self.a_mem.save(f_a_mem)

    f_e_mem = f"{save_folder}/evaluation_memory"
    self.e_mem.save(f_e_mem)

    # Scratch contains non-permanent data associated with the agent. When 
    # it is saved, it takes a json form. When we load it, we move the values
    # to Python variables. 
    f_scratch = f"{save_folder}/scratch.json"
    self.scratch.save(f_scratch)


  def perceive(self, maze):
    """
    This function takes the current maze, and returns events that are 
    happening around the agent. Importantly, perceive is guided by 
    two key hyper-parameter for the  agent: 1) att_bandwidth, and 
    2) retention. 

    First, <att_bandwidth> determines the number of nearby events that the 
    agent can perceive. Say there are 10 events that are within the vision
    radius for the agent -- perceiving all 10 might be too much. So, the 
    agent perceives the closest att_bandwidth number of events in case there
    are too many events. 

    Second, the agent does not want to perceive and think about the same 
    event at each time step. That's where <retention> comes in -- there is 
    temporal order to what the agent remembers. So if the agent's memory
    contains the current surrounding events that happened within the most 
    recent retention, there is no need to perceive that again. xx

    INPUT: 
      maze: Current <Maze> instance of the world. 
    OUTPUT: 
      a list of <ConceptNode> that are perceived and new. 
        See associative_memory.py -- but to get you a sense of what it 
        receives as its input: "s, p, o, desc, agent.scratch.curr_time"
    """
    return perceive(self, maze, self.client)


  def retrieve(self, perceived):
    """
    This function takes the events that are perceived by the agent as input
    and returns a set of related events and thoughts that the agent would 
    need to consider as context when planning. 

    INPUT: 
      perceive: a list of <ConceptNode> that are perceived and new.  
    OUTPUT: 
      retrieved: dictionary of dictionary. The first layer specifies an event,
                 while the latter layer specifies the "curr_event", "events", 
                 and "thoughts" that are relevant.
    """
    return retrieve(self, perceived)


  def plan(self, maze, agents, new_day, retrieved, user_action, train_mode):
    """
    Main cognitive function of the chain. It takes the retrieved memory and 
    perception, as well as the maze and the first day state to conduct both 
    the long term and short term planning for the agent. 

    INPUT: 
      maze: Current <Maze> instance of the world. 
      agents: A dictionary that contains all agent names as keys, and the 
                Persona instance as values. 
      new_day: This can take one of the three values. 
        1) <Boolean> False -- It is not a "new day" cycle (if it is, we would
           need to call the long term planning sequence for the agent). 
        2) <String> "First day" -- It is literally the start of a simulation,
           so not only is it a new day, but also it is the first day. 
        2) <String> "New day" -- It is a new day. 
      retrieved: dictionary of dictionary. The first layer specifies an event,
                 while the latter layer specifies the "curr_event", "events", 
                 and "thoughts" that are relevant.
    OUTPUT 
      The target action address of the agent (agent.scratch.act_address).
    """
    return plan(self, maze, agents, new_day, retrieved, self.client, user_action, train_mode)

  def execute(self, maze, agents, plan):
    """
    This function takes the agent's current plan and outputs a concrete 
    execution (what object to use, and what tile to travel to). 

    INPUT: 
      maze: Current <Maze> instance of the world. 
      agents: A dictionary that contains all agent names as keys, and the 
                Persona instance as values. 
      plan: The target action address of the agent  
            (agent.scratch.act_address).
    OUTPUT: 
      execution: A triple set that contains the following components: 
        <next_tile> is a x,y coordinate. e.g., (58, 9)
        <pronunciatio> is an emoji.
        <description> is a string description of the movement. e.g., 
        writing her next novel (editing her novel) 
        @ double studio:double studio:common room:sofa
    """
    return execute(self, maze, agents, plan)


  def reflect(self):
    """
    Reviews the agent's memory and create new thoughts based on it. 

    INPUT: 
      None
    OUTPUT: 
      None
    """
    reflect(self, self.client)

  def move(self, maze, agents, curr_tile, curr_time, user_action=None, train_mode=None):
    """
    This is the main cognitive function where our main sequence is called. 

    INPUT: 
      maze: The Maze class of the current world. 
      agents: A dictionary that contains all agent names as keys, and the 
                Persona instance as values. 
      curr_tile: A tuple that designates the agent's current tile location 
                 in (row, col) form. e.g., (58, 39)
      curr_time: datetime instance that indicates the game's current time. 
    OUTPUT: 
      execution: A triple set that contains the following components: 
        <next_tile> is a x,y coordinate. e.g., (58, 9)
        <pronunciatio> is an emoji.
        <description> is a string description of the movement. e.g., 
        writing her next novel (editing her novel) 
        @ double studio:double studio:common room:sofa
    """
    # Updating agent's scratch memory with <curr_tile>. 
    self.scratch.curr_tile = curr_tile

    # We figure out whether the agent started a new day, and if it is a new
    # day, whether it is the very first day of the simulation. This is 
    # important because we set up the agent's long term plan at the start of
    # a new day. 
    new_day = False
    if not self.scratch.curr_time: 
      new_day = "First day"
    elif (self.scratch.curr_time.strftime('%A %B %d')
          != curr_time.strftime('%A %B %d')):
      new_day = "New day"
    self.scratch.curr_time = curr_time

    # Main cognitive sequence begins here. 
    perceived = self.perceive(maze)
    retrieved = self.retrieve(perceived)
    plan = self.plan(maze, agents, new_day, retrieved, user_action, train_mode)
    # self.reflect()

    # <execution> is a triple set that contains the following components: 
    # <next_tile> is a x,y coordinate. e.g., (58, 9)
    # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
    # <description> is a string description of the movement. e.g., 
    #   writing her next novel (editing her novel) 
    #   @ double studio:double studio:common room:sofa
    return self.execute(maze, agents, plan)

  def update_preference_data(self, train_mode, prompt, suggestion, reason, score, revision, suggestion_1=None, suggestion_2=None):
    if 'reasonright' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rr.txt"
    elif 'reasonleft' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rl.txt"
    else:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_wo_r.txt"
    f = open(prompt_template, "r")
    system_prompt = f.read()
    f.close()

    conversation_chosen = [
    {
      "content": system_prompt,
      "role": "system"
    },
    {
      "content": prompt,
      "role": "user"
    }]

    conversation_rejected = [
    {
      "content": system_prompt,
      "role": "system"
    },
    {
      "content": prompt,
      "role": "user"
    }]


    if 'ranking' in train_mode:
      response_num1 = f'"Suggestion": {suggestion_1}'
      response_num2 = f'"Suggestion": {suggestion_2}'
      
      suggestion1_dict = {
        "content": '{'+response_num1+'}',
        "role": "assistant"
      }
      suggestion2_dict = {
        "content": '{'+response_num2+'}',
        "role": "assistant"
      }
      
      if '1' in score:
        conversation_chosen.append(suggestion1_dict)
        conversation_rejected.append(suggestion2_dict)
      elif '2' in score:
        conversation_chosen.append(suggestion2_dict)
        conversation_rejected.append(suggestion1_dict)
      else:
        return
      score = float(reason[0])

    else:
      if 'reasonright' in train_mode:
        response_suggestion = f'"Suggestion": {suggestion}, "Reason": {reason}'
      elif 'reasonleft' in train_mode:
        response_suggestion = f'"Reason": {reason}, "Suggestion": {suggestion}'
      else:
        response_suggestion = f'"Suggestion": {suggestion}'
      
      suggestion_dict = {
        "content": '{'+response_suggestion+'}',
        "role": "assistant"
      }
      
      if 'reasonright' in train_mode:
        response_no_recommendation = f'"Suggestion": "No Recommendation", "Reason": "No Recommendation"'
      elif 'reasonleft' in train_mode:
        response_no_recommendation = f'"Reason": "No Recommendation", "Suggestion": "No Recommendation"'
      else:
        response_no_recommendation = f'"Suggestion": "No Recommendation"'
      
      no_recommendation_dict = {
        "content": '{'+response_no_recommendation+'}',
        "role": "assistant"
      }


      if 'revision' in train_mode:
        if 'reasonright' in train_mode:
          revision_suggestion = f'"Suggestion": {revision}, "Reason": {reason}'
        elif 'reasonleft' in train_mode:
          revision_suggestion = f'"Reason": {reason}, "Suggestion": {revision}'
        else:
          revision_suggestion = f'"Suggestion": {revision}'
        
        revision_dict = {
          "content": '{'+revision_suggestion+'}',
          "role": "assistant"
        }
      #### If you want to train the agent using revision result, you should implement the code.
      conversation_chosen = ""
      conversation_rejected = ""
      score = ""

    self.preference_records["chosen"].append(conversation_chosen)
    self.preference_records["rejected"].append(conversation_rejected)
    self.preference_records["score"].append(score)

  def initialize_models(self):
    del self.suggestion_model
    self.suggestion_model = None

  def update_suggestion_model(self, train_mode):
    
    def sample_indices_with_ratio(lst, ratio=(4, 1)):
      suggestion_values = np.array([item[2]['content'] for item in lst])
      no_recommendation_indices = np.where(np.char.find(suggestion_values.astype(str), "No Recommendation") >= 0)[0]
      non_no_recommendation_indices = np.where(np.char.find(suggestion_values.astype(str), "No Recommendation") == -1)[0]
      
      num_non_no_recommendation = min(len(non_no_recommendation_indices), len(no_recommendation_indices) // ratio[0] * ratio[1])
      num_no_recommendation = num_non_no_recommendation * ratio[0]

      sampled_no_recommendation = np.random.choice(no_recommendation_indices, num_no_recommendation, replace=False)
      sampled_non_no_recommendation = np.random.choice(non_no_recommendation_indices, num_non_no_recommendation, replace=False)

      sampled_indices = np.concatenate([sampled_no_recommendation, sampled_non_no_recommendation])

      return sampled_indices


    if 'llama' in self.suggestion_model_name or 'mixtral' in self.suggestion_model_name:
      if self.suggestion_model == None:
        self.suggestion_model = llm_model_upload(train_mode, self.hf_cache_dir, 2048+256)
      else:
        if 'wo_train' not in train_mode:
          if 'lora' not in self.suggestion_model_name:
            model, base_model, tokenizer = self.suggestion_model
            self.suggestion_model = peft_lm_model_upload(base_model, tokenizer, train_mode, self.hf_cache_dir, 2048+256)
            self.suggestion_model_name = self.suggestion_model_name + '_lora'

          model, base_model, tokenizer = self.suggestion_model

          combined_preference_data = {
                                      "chosen": [],
                                      "rejected": [],
                                      "score": []
                                      }
          
          sampled_indices = [si for si in range(len(self.preference_records['chosen']))]

          self.preference_records['chosen'] = [self.preference_records['chosen'][si] for si in sampled_indices]
          self.preference_records['rejected'] = [self.preference_records['rejected'][si] for si in sampled_indices]
          self.preference_records['score'] = [self.preference_records['score'][si] for si in sampled_indices]

          preference_len = len(self.preference_records['chosen'])
          total_preference_len = len(self.total_preference_records['chosen'])

          if '_equally' in train_mode:
            if '_equally200' in train_mode:
              extract_data_len = 200
            elif '_equally400' in train_mode:
              extract_data_len = 400

            self.total_preference_records['chosen'] += self.preference_records['chosen']
            self.total_preference_records['rejected'] += self.preference_records['rejected']
            self.total_preference_records['score'] += self.preference_records['score']

            total_chosen = self.total_preference_records['chosen']
            total_rejected = self.total_preference_records['rejected']
            total_score = self.total_preference_records['score']

            if preference_len + total_preference_len > extract_data_len:
              indices_prefer = random.sample(range(preference_len + total_preference_len), int(extract_data_len))
              combined_preference_data['chosen'] = [total_chosen[i] for i in indices_prefer]
              combined_preference_data['rejected'] = [total_rejected[i] for i in indices_prefer]
              combined_preference_data['score'] = [total_score[i] for i in indices_prefer]
            else:
              combined_preference_data['chosen'] = total_chosen
              combined_preference_data['rejected'] = total_rejected
              combined_preference_data['score'] = total_score

          elif 'halftoday':
            total_chosen = self.total_preference_records['chosen']
            total_rejected = self.total_preference_records['rejected']
            total_score = self.total_preference_records['score']
            
            if preference_len > total_preference_len:
              combined_preference_data['chosen'] = self.preference_records['chosen'] + total_chosen
              combined_preference_data['rejected'] = self.preference_records['rejected'] + total_rejected
              combined_preference_data['score'] = self.preference_records['score'] + total_score
            else:
              indices_prefer = random.sample(range(total_preference_len), int(preference_len))
              combined_preference_data['chosen'] = [total_chosen[i] for i in indices_prefer] + self.preference_records['chosen']
              combined_preference_data['rejected'] = [total_rejected[i] for i in indices_prefer] + self.preference_records['rejected']
              combined_preference_data['score'] = [total_score[i] for i in indices_prefer] + self.preference_records['score']
            
            self.total_preference_records['chosen'] += self.preference_records['chosen']
            self.total_preference_records['rejected'] += self.preference_records['rejected']
            self.total_preference_records['score'] += self.preference_records['score']

          if 'DPO' in train_mode:
            self.suggestion_model = DPO_train(self.total_evaluation_records, combined_preference_data, model, base_model, train_mode, tokenizer, self.hf_cache_dir, 2048+256)
          elif 'KTO' in train_mode:
            self.suggestion_model = KTO_train(self.total_evaluation_records, combined_preference_data, model, base_model, train_mode, tokenizer, self.hf_cache_dir, 2048+256)
          elif 'SFT' in train_mode:
            self.suggestion_model = SFT_train(self.total_evaluation_records, combined_preference_data, model, base_model, train_mode, tokenizer, self.hf_cache_dir, 2048+256)
        
        # self.total_preference_records['chosen'] += self.preference_records['chosen']
        # self.total_preference_records['rejected'] += self.preference_records['rejected']
        # self.total_preference_records['score'] += self.preference_records['score']

        # Initialize the data
        self.preference_records = {
          "chosen": [],
          "rejected": [],
          "score": []
        }
    
    for i_name in range(100):
      if os.path.exists(f'./total_data_{i_name}.pkl'):
          continue
      if 'mmlueval' in train_mode:
          mmlu_eval = self.mmlu_eval_(train_mode)
          self.total_evaluation_records['mmlu'] = mmlu_eval
      with open(f'./total_data_{i_name}.pkl', 'wb') as f:
          pickle.dump(self.total_evaluation_records, f)
      break

  def suggest(self, agents, train_mode, is_john_walking, movement_path, step, curr_time):
    return suggest(self, agents, self.client, train_mode, is_john_walking, movement_path, step, curr_time)

  def mmlu_eval_(self, train_mode):
    return model_mmlu_eval(self, train_mode)

  def open_convo_session(self, convo_mode): 
    open_convo_session(self, convo_mode, self.client)
  
  def update_total_records(self, date, action, suggestion, score, reason, deducted_category):
    try:
      self.total_evaluation_records[date].append({'action': action, 'suggestion':suggestion, 'score':score, 'reason':reason, 'deducted_category':deducted_category})
    except:
      self.total_evaluation_records[date] = [{'action': action, 'suggestion':suggestion, 'score':score, 'reason':reason, 'deducted_category':deducted_category}]
