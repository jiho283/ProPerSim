import sys
import random
sys.path.append('../../')

from global_methods import *
from utils import *
from persona.prompt_template.run_gpt_prompt import *
from persona.prompt_template.gpt_structure import *
from persona.cognitive_modules.retrieve import *
import datetime
import Levenshtein
from collections import deque
import time

def sort_dates(date_strings):
    def parse_date(date_string):
        return datetime.datetime.strptime(date_string, '%A %B %d -- %I:%M:%S %p')
    
    try:
        return sorted(date_strings, key=parse_date, reverse=True)
    except Exception as e:
        print("Error while parsing:", e)
        return date_strings


def evaluation(user_persona, agent_persona, client, train_mode, first_suggestion=None):
    if 'John is walking.' == agent_persona.scratch.user_action_for_suggestion:
      user_action = f"{user_persona.name} is making moves to do [{user_persona.scratch.act_description}]"
    else:
      user_action = f"{user_persona.name} is {user_persona.scratch.act_description}"
    focal_points = [f'{user_action}. The agent suggests "{agent_persona.scratch.suggestion}"']
    try:
        statements = ""
        retrieved = new_retrieve(user_persona, focal_points, n_count=0, client=client, is_eval_suggest=True)
        sorted_dates = []
        for key, val in retrieved.items():
            for i in val: 
                sorted_dates.append(str(i.created.strftime('%A %B %d -- %I:%M:%S %p')))
            
        sorted_dates = sort_dates(sorted_dates)
        
        for date_ in sorted_dates:
          for key, val in retrieved.items():
            for i in val: 
              if date_ == f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}":
                if f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" not in statements:
                  statements = f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" + statements
                  break
        # import pdb; pdb.set_trace()
        long_memory = ""
        for fourhour in user_persona.memory_4hourly:
          long_memory = long_memory + fourhour
        for dat_, onehour in user_persona.memory_hourly:
          long_memory = long_memory + onehour
        
        long_memory = long_memory + user_persona.summary_recent_events_hourly
        
        statements = long_memory + statements

        # recent_hour_reco = 0
        # for event_ in user_persona.recent_events_hourly:
        #   if 'no reco' not in event_.lower():
        #     recent_hour_reco += 1
        
        # statements = statements + f"\nNumber of Recommendations Received in the Past Hour: {recent_hour_reco}\n"
    
    except:
      if statements == "":
        statements = "No Relevant Memory"
    
    # import pdb;pdb.set_trace()
    if 'seperate' in train_mode:
      score, reason, revision = run_gpt_prompt_make_evaluation_seperate(user_persona, agent_persona, statements, train_mode, client=client, first_suggestion=first_suggestion)
    else:
      score, reason, revision = run_gpt_prompt_make_evaluation(user_persona, agent_persona, statements, train_mode, client=client, first_suggestion=first_suggestion)

    if 'evaltest' in train_mode:
      print('\n\n[LLM EVAL]')
      print(f'Score: {score}, Reason: {reason}\n')
      print('[User Evaluation]\nUser action:', user_action)
      print('Agent suggestion:', agent_persona.scratch.suggestion)
      score = input("Score: ")
      reason = input("Reason: ")
      
    return score, reason, revision    


def eval_suggestion(persona, personas, client=None, train_mode=None, first_suggestion=None): 
  if persona.scratch.act_description == 'sleeping' or (not persona.scratch.act_description):
    return None, None, None

  if first_suggestion:
    if Levenshtein.distance(first_suggestion, personas['Eddy Lin'].scratch.suggestion) > 3:
      score_1, reason_1, revision_1 = evaluation(persona, personas['Eddy Lin'], client, train_mode.replace('_ranking', ''), first_suggestion)
    else:
      score_1, reason_1, revision_1 = "", "", ""
    score_2, reason_2, revision_2 = evaluation(persona, personas['Eddy Lin'], client, train_mode.replace('_ranking', ''), None)
    
    if score_1 == "" and reason_1 == "":
      score_1, reason_1, revision_1 = score_2, reason_2, revision_2

    personas['Eddy Lin'].scratch.suggestion_eval_result = score_2
    ret_events = perceive_suggestion_eval(personas, persona.scratch.act_description, personas['Eddy Lin'].scratch.suggestion, score_2, reason_2, "", train_mode, client)
    return (score_1, reason_1), (score_2, reason_2), revision_2
  
  elif personas['Eddy Lin'].scratch.suggestion:
    score, reason, revision = evaluation(persona, personas['Eddy Lin'], client, train_mode)
    if 'secondsugg' not in train_mode:
      personas['Eddy Lin'].scratch.suggestion_eval_result = score
      ret_events = perceive_suggestion_eval(personas, persona.scratch.act_description, personas['Eddy Lin'].scratch.suggestion, score, reason, revision, train_mode, client)
    return score, reason, revision
  else:
    return None, None, None


def eval_suggestion_wo_record(persona, personas, client=None, train_mode=None): 
  if persona.scratch.act_description == 'sleeping' or (not persona.scratch.act_description):
    return None, None, None
  
  if personas['Eddy Lin'].scratch.suggestion:
    score, reason, revision = evaluation(persona, personas['Eddy Lin'], client, train_mode)
    return score, reason, revision
  else:
    return None, None, None


def perceive_suggestion_eval(personas, action, suggestion, score, score_reason, revision, train_mode, client=None): 
  # Storing events. 
  # <ret_events> is a list of <ConceptNode> instances from the persona's 
  # associative memory. 
  ret_events = []
  for persona_name, persona in personas.items():
    if persona_name == 'John Lin':
      user_action = personas['John Lin'].scratch.act_description
      if personas['Eddy Lin'].scratch.user_action_for_suggestion == 'John is walking.':
        user_action = f"{personas['John Lin'].name} is making moves to do [{personas['John Lin'].scratch.act_description}]"
      evaluation_result = f"""John Lin's Action: {user_action}\nAgent's Suggestion: {suggestion}\n"""

      date_time = persona.scratch.curr_time.strftime('%A %B %d -- %I:%M:%S %p')
      time_hour = persona.scratch.curr_time.replace(minute=0, second=0, microsecond=0)
      end_time_hour = time_hour + datetime.timedelta(hours=1)

      persona.events_hourly.append((date_time, evaluation_result))
      persona.recent_events_hourly.append(evaluation_result)

      sum_hourly = summarize_sugg_eval_hourly((time_hour, persona.scratch.curr_time), persona.events_hourly, train_mode, client)
      persona.summary_recent_events_hourly = sum_hourly

      if ":57:30" in date_time:
        persona.memory_hourly.append((date_time, sum_hourly))
        persona.events_hourly = []
        if len(persona.memory_hourly) == 4:
          sum_4hourly = summarize_sugg_eval_4hourly(end_time_hour, persona.memory_hourly, train_mode, client)
          persona.memory_4hourly.append(sum_4hourly)
          persona.memory_hourly = []
    
    elif persona_name == 'Eddy Lin':
      user_action = personas['Eddy Lin'].scratch.user_action_for_suggestion
      if "_ASCR_" in train_mode:
        evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4, Reason: {score_reason}"""
      
      elif "_ASC9R1_" in train_mode:
        rand_val = random.random()
        if rand_val < 0.1:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4, Reason: {score_reason}"""
        else:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4"""
      
      elif "_ASC8R2_" in train_mode:
        rand_val = random.random()
        if rand_val < 0.2:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4, Reason: {score_reason}"""
        else:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4"""
      
      elif "_ASC_" in train_mode:
        evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4"""
      
      elif "_AS9C1_" in train_mode:
        rand_val = random.random()
        if rand_val < 0.1:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4"""
        else:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\n"""

      elif "_AS8C2_" in train_mode:
        rand_val = random.random()
        if rand_val < 0.2:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\nScore: {score}/4"""
        else:
          evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\n"""

      elif "_AS_" in train_mode:
        evaluation_result = f"""John Lin's Action: {user_action}\n"Agent's Suggestion": {suggestion}\n"""
      
      elif "_A_" in train_mode:
        evaluation_result = f"""John Lin's Action: {user_action}\n"""
      
      date_time = personas['John Lin'].scratch.curr_time.strftime('%A %B %d -- %I:%M:%S %p')
      time_hour = personas['John Lin'].scratch.curr_time.replace(minute=0, second=0, microsecond=0)
      end_time_hour = time_hour + datetime.timedelta(hours=1)

      persona.events_hourly.append((date_time, evaluation_result))
      persona.recent_events_hourly.append(evaluation_result)

      if len(persona.events_hourly) > 4:
        sum_hourly = agent_summarize_sugg_eval_hourly((time_hour, personas['John Lin'].scratch.curr_time), persona.events_hourly, train_mode, client)
        persona.summary_recent_events_hourly = sum_hourly
      else:
        persona.summary_recent_events_hourly = ""

      if ":57:30" in date_time:
        # import pdb; pdb.set_trace()
        # sum_hourly = summarize_sugg_eval_hourly((time_hour, end_time_hour), persona.events_hourly, client)
        persona.memory_hourly.append((date_time, persona.summary_recent_events_hourly))
        persona.events_hourly = []
        if len(persona.memory_hourly) == 4:
          sum_4hourly = agent_summarize_sugg_eval_4hourly(end_time_hour, persona.memory_hourly, train_mode, client)
          persona.memory_4hourly.append(sum_4hourly)
          persona.memory_hourly = []

      # personas['Eddy Lin'].hourly_memory.append((personas['Eddy Lin'].scratch.curr_time, evaluation_result))
    
    if persona_name == 'John Lin' and 'no recommendation' in suggestion.lower():
      continue
    keywords = [action, suggestion]
    desc_embedding_in = evaluation_result
    event_poignancy = 5
    if desc_embedding_in in persona.e_mem.embeddings: 
        event_embedding = persona.e_mem.embeddings[desc_embedding_in]
    else: 
        for i in range(10):
          try:
            event_embedding = get_embedding(desc_embedding_in, client=client)
            break
          except:
            time.sleep(5)
            continue
    event_embedding_pair = (desc_embedding_in, event_embedding)
    ret_events += [persona.e_mem.add_event(personas['John Lin'].scratch.curr_time, None,
                        evaluation_result, keywords, event_poignancy, 
                        event_embedding_pair, [])]

  return ret_events


def summarize_sugg_eval_hourly(times, time_sugg_list, train_mode, client):
  recomm = 0
  for date, i in time_sugg_list:
    if 'no recomm' not in i.lower():
      recomm += 1
  if recomm == 0:
    summary = "There were no recommendations at all."
  else:
    summary = run_gpt_prompt_hourly_sum(time_sugg_list, train_mode, client=client)
  start = times[0].strftime('%B %d %I:%M:%S%p')
  end = times[1].strftime('%B %d %I:%M:%S%p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary

def summarize_sugg_eval_4hourly(time, hourly_sugg_list, train_mode, client):
  recomm = 0
  for date, i in hourly_sugg_list:
    num = int(i.split('Number of Recommendation: ')[1].split('\n')[0])
    recomm += num
  if recomm == 0:
    summary = "There were no recommendations at all."
  else:
    summary = run_gpt_prompt_4hourly_sum(hourly_sugg_list, train_mode, client=client)
  start_time = time - datetime.timedelta(hours=len(hourly_sugg_list))
  start = start_time.strftime('%B %d %I:%M:%S %p')
  end = time.strftime('%B %d %I:%M:%S %p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary


def agent_summarize_sugg_eval_hourly(times, time_sugg_list, train_mode, client):
  recomm = 0
  for date, i in time_sugg_list:
    if 'no recomm' not in i.lower():
      recomm += 1
  summary = run_gpt_prompt_hourly_sum_agent(time_sugg_list, train_mode, client=client)
  start = times[0].strftime('%A %B %d %I:%M:%S %p')
  end = times[1].strftime('%A %B %d %I:%M:%S %p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary


def agent_summarize_sugg_eval_4hourly(time, hourly_sugg_list, train_mode, client):
  recomm = 0
  for date, i in hourly_sugg_list:
    num = int(i.split('Number of Recommendation: ')[1].split('\n')[0])
    recomm += num
  summary = run_gpt_prompt_4hourly_sum_agent(hourly_sugg_list, train_mode, client=client)
  start_time = time - datetime.timedelta(hours=len(hourly_sugg_list))
  start = start_time.strftime('%B %d %I:%M:%S %p')
  end = time.strftime('%B %d %I:%M:%S %p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary


def revise_suggestion(user_persona, agent_persona, score, reason, train_mode, client):
    if 'John is walking.' == agent_persona.scratch.user_action_for_suggestion:
      user_action = f"{user_persona.name} is making moves to do [{user_persona.scratch.act_description}]"
    else:
      user_action = f"{user_persona.name} is {user_persona.scratch.act_description}"
    focal_points = [f'{user_action}. The agent suggests "{agent_persona.scratch.suggestion}"']
    try:
        statements = ""
        retrieved = new_retrieve(user_persona, focal_points, n_count=0, client=client, is_eval_suggest=True)
        sorted_dates = []
        for key, val in retrieved.items():
            for i in val: 
                sorted_dates.append(str(i.created.strftime('%A %B %d -- %I:%M:%S %p')))
            
        sorted_dates = sort_dates(sorted_dates)
        
        for date_ in sorted_dates:
          for key, val in retrieved.items():
            for i in val: 
              if date_ == f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}":
                if f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" not in statements:
                  statements = f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" + statements
                  break
        # import pdb; pdb.set_trace()
        long_memory = ""
        for fourhour in user_persona.memory_4hourly:
          long_memory = long_memory + fourhour
        for dat_, onehour in user_persona.memory_hourly:
          long_memory = long_memory + onehour
        
        long_memory = long_memory + user_persona.summary_recent_events_hourly
        
        statements = long_memory + statements

    except:
      if statements == "":
        statements = "No Relevant Memory"
    
    revision = run_gpt_prompt_make_revision(user_persona, agent_persona, score, reason, statements, train_mode, client)
    
    return revision

