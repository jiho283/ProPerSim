import sys
import random
import json
sys.path.append('../../')

from global_methods import *
from utils import *
from agent.prompt_template.run_gpt_prompt import *
from agent.cognitive_modules.retrieve import *
import datetime

def sort_dates(date_strings):
    def parse_date(date_string):
        return datetime.datetime.strptime(date_string, '%A %B %d -- %I:%M:%S %p')
    
    try:
        return sorted(date_strings, key=parse_date, reverse=True)
    except Exception as e:
        print("Error while parsing:", e)
        return date_strings

def make_suggestion(persona, user_persona, client, train_mode, is_john_walking, movement_path, step, curr_time):
  def make_sugg_eval(action, suggestion, score, score_reason):
    return f'"Suggestion": {suggestion}'+f'\nScore: {score}/5'
  
  def make_statements(curr_statement, curr_time, curr_action, action, suggestion, score, score_reason):
    year_ = curr_time.split(', ')[1].strip()
    date_ = curr_time.split(f', {year_}, ')[0]
    time_ = curr_time.split(f', {year_}, ')[1].strip()
    if date_ not in curr_statement:
      curr_statement = f"{curr_statement}\nDate: {date_}, {year_}".strip()

    lines = curr_statement.strip().splitlines()
    last_line = lines[-1]
    if curr_action in last_line:
      if re.search(r'\(\d{2}:\d{2}:\d{2}\)', last_line) and not re.search(r'\(\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\)', last_line):
        new_last_line = re.sub(r'\((\d{2}:\d{2}:\d{2})\)', rf'(\1-{time_})', last_line)
      else:
        new_last_line = re.sub(r'(\d{2}:\d{2}:\d{2})(?!.*\d{2}:\d{2}:\d{2})', time_, last_line)
      lines[-1] = new_last_line  
      curr_statement = "\n".join(lines)
    
    else:
      if 'ing' in curr_action.split(' ')[0]:
        curr_statement += f"\n({time_}) John is {curr_action}"
      else:
        curr_statement += f"\n({time_}) John {curr_action}"

    if suggestion:
      sugg_eval = make_sugg_eval(action, suggestion, score, score_reason)
      curr_statement += f"\n{sugg_eval}"
    
    return curr_statement

  statements = ""
  focal_points = [f"{user_persona.name} is {user_persona.scratch.act_description}."]
  if "ret_suggest" in train_mode:
    try:
      if 'givenpersona' in train_mode:
        retrieved = new_retrieve_agent(persona, focal_points, n_count=0, client=client, is_eval_suggest=True)
      else:
        if '_rag' in train_mode.lower():
          retrieved = new_retrieve_agent(persona, focal_points, n_count=5, client=client, is_eval_suggest=True)
        else:
          retrieved = new_retrieve_agent(persona, focal_points, n_count=3, client=client, is_eval_suggest=True)
      sorted_dates = []
      for key, val in retrieved.items():
          for i in val: 
              sorted_dates.append(f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}")
          
      sorted_dates = sort_dates(sorted_dates)
      # import pdb; pdb.set_trace()
      for date_ in sorted_dates:
        for key, val in retrieved.items():
          for i in val: 
            if date_ == f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}":
              if f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" not in statements:
                statements = f"{i.created.strftime('%A %B %d -- %I:%M:%S %p')}\n{i.embedding_key}\n" + statements
                break
    except:
      if statements == "":
        statements = "No Relevant Memory"

  if 'plantest' in train_mode:
    return "No Recommendation", "No Recommendation"
  
  # if len(persona.hourly_memory) > 13:
  #   hourly_mem = list(persona.hourly_memory)
  #   start_time = hourly_mem[0][0]
  #   end_time = hourly_mem[-13][0]
  #   sugg_range = []
  #   for hm in hourly_mem:
  #     sugg_range.append(hm)
  #     if hm[0] == end_time:
  #       break
  #   summarized_memory = summarize_sugg_eval_hourly((start_time, end_time), sugg_range, client)
  #   statements = summarized_memory + statements
  long_memory = ""
  for fourhour in persona.memory_4hourly:
    long_memory = long_memory + fourhour
  for dat_, onehour in persona.memory_hourly:
    long_memory = long_memory + onehour
  
  long_memory = long_memory + persona.summary_recent_events_hourly

  statements = long_memory + statements

  suggestion, reason = run_gpt_prompt_make_suggestion(persona, user_persona, train_mode, retrieved=statements, client=client, is_john_walking=is_john_walking, curr_time=curr_time)
  # import pdb; pdb.set_trace()
  return suggestion, reason 

def suggest(persona, personas, client, train_mode, is_john_walking, movement_path, step, curr_time): 
  if persona.scratch.act_description == 'sleeping' or (not persona.scratch.act_description):
    return None, None

  if not persona.scratch.suggestion:
    if is_john_walking == 'yes':
      persona.scratch.user_action_for_suggestion = 'John is walking.'
    else:
      persona.scratch.user_action_for_suggestion = personas['John Lin'].scratch.act_description
      if '(' in personas['John Lin'].scratch.act_description and ')' in personas['John Lin'].scratch.act_description:
        persona.scratch.user_action_for_suggestion = personas['John Lin'].scratch.act_description.split('(')[-1].split(')')[0]
    persona.scratch.suggestion, reason = make_suggestion(persona, personas['John Lin'], client, train_mode, is_john_walking, movement_path, step, curr_time)
    return persona.scratch.user_action_for_suggestion, persona.scratch.suggestion

def summarize_sugg_eval_hourly(times, time_sugg_list, client):
  recomm = 0
  for date, i in time_sugg_list:
    if 'no recomm' not in i.lower():
      recomm += 1
  summary = run_gpt_prompt_hourly_sum(time_sugg_list, client=client)
  start = times[0].strftime('%A %B %d %I:%M:%S %p')
  end = times[1].strftime('%A %B %d %I:%M:%S %p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary


def summarize_sugg_eval_4hourly(time, hourly_sugg_list, client):
  recomm = 0
  for date, i in hourly_sugg_list:
    num = int(i.split('Number of Recommendation: ')[1].split('\n')[0])
    recomm += num
  summary = run_gpt_prompt_4hourly_sum(hourly_sugg_list, client=client)
  start_time = time - datetime.timedelta(hours=len(hourly_sugg_list))
  start = start_time.strftime('%B %d %I:%M:%S %p')
  end = time.strftime('%B %d %I:%M:%S %p')
  total_summary = f"TIME: {start} - {end}\nNumber of Recommendation: {recomm}\nSUMMARY: {summary}\n\n"
  return total_summary


def model_mmlu_eval(agent_persona, train_mode):
  return run_mmlu_eval(agent_persona, train_mode)