import re
import datetime
import sys
import ast
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch

sys.path.append('../../')

from global_methods import *
from agent.prompt_template.gpt_structure import *
from agent.prompt_template.print_prompt import *

def get_random_alphanumeric(i=6, j=6): 
  """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
  k = random.randint(i, j)
  x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
  return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False, client=None): 
  """
  Given the persona, returns an integer that indicates the hour when the 
  persona wakes up.  

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    integer for the wake up hour.
  """
  def create_prompt_input(persona, test_input=None): 
    if test_input: return test_input
    prompt_input = [persona.scratch.get_str_iss(),
                    persona.scratch.get_str_lifestyle(),
                    persona.scratch.get_str_firstname()]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = int(gpt_response.strip().lower().split("am")[0])
    return cr
  
  def __func_validate(gpt_response, prompt=""): 
    try: __func_clean_up(gpt_response, prompt="")
    except: return False
    return True

  def get_fail_safe(): 
    fs = 8
    return fs

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 5, 
             "temperature": 0.8, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
  prompt_input = create_prompt_input(persona, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  
  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan(persona, 
                              wake_up_hour, 
                              test_input=None, 
                              verbose=False, client=None):
  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form: 
  'wake up and complete the morning routine at 6:00 am', 
  'eat breakfast at 7:00 am',.. 
  Note that the actions come without a period. 

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    a list of daily actions in broad strokes.
  """
  def create_prompt_input(persona, wake_up_hour, test_input=None):
    if test_input: return test_input
    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [persona.scratch.get_str_lifestyle()]
    prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [f"{str(wake_up_hour)}:00 am"]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = []
    _cr = gpt_response.split(")")
    for i in _cr: 
      if i[-1].isdigit(): 
        i = i[:-1].strip()
        if i[-1] == "." or i[-1] == ",": 
          cr += [i[:-1].strip()]
    return cr

  def __func_validate(gpt_response, prompt=""):
    try: __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True

  def get_fail_safe(): 
    fs = ['wake up and complete the morning routine at 6:00 am', 
          'eat breakfast at 7:00 am', 
          'read a book from 8:00 am to 12:00 pm', 
          'have lunch at 12:00 pm', 
          'take a nap from 1:00 pm to 4:00 pm', 
          'relax and watch TV from 7:00 pm to 8:00 pm', 
          'go to bed at 11:00 pm'] 
    return fs


  
  gpt_param = {"engine": "text-davinci-003", "max_tokens": 500, 
               "temperature": 1, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
  prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_hourly_schedule(persona, 
                                            curr_hour_str,
                                            p_f_ds_hourly_org, 
                                            hour_str,
                                            intermission2=None,
                                            test_input=None, 
                                            verbose=False, client=None): 
  def create_prompt_input(persona, 
                          curr_hour_str, 
                          p_f_ds_hourly_org,
                          hour_str,
                          intermission2=None,
                          test_input=None): 
    if test_input: return test_input
    schedule_format = ""
    for i in hour_str: 
      schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
      schedule_format += f" Activity: [Fill in]\n"
    schedule_format = schedule_format[:-1]

    # intermission_str = f"Here the originally intended hourly breakdown of"
    intermission_str = f"Create a schedule for {curr_hour_str}.\nHere the originally intended hourly breakdown of"
    intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
    for count, i in enumerate(persona.scratch.daily_req): 
      intermission_str += f"{str(count+1)}) {i}, "
    intermission_str = intermission_str[:-2]

    prior_schedule = ""
    if p_f_ds_hourly_org: 
      prior_schedule = "\n"
      for count, i in enumerate(p_f_ds_hourly_org): 
        prior_schedule += f"[(ID:{get_random_alphanumeric()})" 
        prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
        prior_schedule += f" {hour_str[count]}] Activity:"
        prior_schedule += f" {persona.scratch.get_str_firstname()}"
        prior_schedule += f" is {i}\n"

    prompt_ending = f"[(ID:{get_random_alphanumeric()})"
    prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
    prompt_ending += f" -- {curr_hour_str}] Activity:"
    prompt_ending += f" {persona.scratch.get_str_firstname()} is "

    if intermission2: 
      intermission2 = f"\n{intermission2}"

    prompt_input = []
    prompt_input += [schedule_format]
    prompt_input += [persona.scratch.get_str_iss()]

    prompt_input += [prior_schedule + "\n"]
    prompt_input += [intermission_str]
    if intermission2: 
      prompt_input += [intermission2]
    else: 
      prompt_input += [""]
    prompt_input += [prompt_ending]

    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if cr[-1] == ".":
      cr = cr[:-1]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: __func_clean_up(gpt_response, prompt="")
    except: return False
    return True

  def get_fail_safe(): 
    fs = "asleep"
    return fs

  # # ChatGPT Plugin ===========================================================
  # def __chat_func_clean_up(gpt_response, prompt=""): ############
  #   cr = gpt_response.strip()
  #   if cr[-1] == ".":
  #     cr = cr[:-1]
  #   return cr

  # def __chat_func_validate(gpt_response, prompt=""): ############
  #   try: __func_clean_up(gpt_response, prompt="")
  #   except: return False
  #   return True

  # gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
  #              "temperature": 0, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_hourly_schedule_v2.txt" ########
  # prompt_input = create_prompt_input(persona, 
  #                                    curr_hour_str, 
  #                                    p_f_ds_hourly_org,
  #                                    hour_str, 
  #                                    intermission2,
  #                                    test_input)  ########
  # prompt = generate_prompt(prompt_input, prompt_template)
  # example_output = "studying for her music classes" ########
  # special_instruction = "The output should ONLY include the part of the sentence that completes the last line in the schedule above." ########
  # fail_safe = get_fail_safe() ########
  # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
  #                                         __chat_func_validate, __chat_func_clean_up, True)
  # if output != False: 
  #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # # ChatGPT Plugin ===========================================================


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0.5, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
  prompt_input = create_prompt_input(persona, 
                                     curr_hour_str, 
                                     p_f_ds_hourly_org,
                                     hour_str, 
                                     intermission2,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()
  
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  
  if f'Activity: {persona.scratch.get_str_firstname()} is ' in output:
    output = output.split(f'Activity: {persona.scratch.get_str_firstname()} is ')[1]
  
  for wrong_num in range(10):
    wrong_num_ = wrong_num + 1
    if f'({wrong_num_}) ' in output:
      output = output.replace(f'({wrong_num_}) ', '(')
      break

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]








def run_gpt_prompt_task_decomp(persona, 
                               task, 
                               duration, 
                               test_input=None, 
                               verbose=False, client=None): 
  def create_prompt_input(persona, task, duration, test_input=None):

    """
    Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is 
    planning on sleeping, 06:00 ~ 07:00am, Maeve is 
    planning on waking up and doing her morning routine, 
    and from 07:00am ~08:00am, Maeve is planning on having breakfast.  
    """
      
    curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
    all_indices = []
    # if curr_f_org_index > 0: 
    #   all_indices += [curr_f_org_index-1]
    all_indices += [curr_f_org_index]
    if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org): 
      all_indices += [curr_f_org_index+1]
    if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org): 
      all_indices += [curr_f_org_index+2]

    curr_time_range = ""

    print ("DEBUG")
    print (persona.scratch.f_daily_schedule_hourly_org)
    print (all_indices)

    summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
    summ_str += f'From '
    for index in all_indices: 
      print ("index", index)
      if index < len(persona.scratch.f_daily_schedule_hourly_org): 
        start_min = 0
        for i in range(index): 
          start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
        end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
        start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                      + datetime.timedelta(minutes=start_min)) 
        end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                      + datetime.timedelta(minutes=end_min)) 
        start_time_str = start_time.strftime("%H:%M%p")
        end_time_str = end_time.strftime("%H:%M%p")
        summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
        if curr_f_org_index+1 == index:
          curr_time_range = f'{start_time_str} ~ {end_time_str}'
    summ_str = summ_str[:-2] + "."

    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [summ_str]
    # prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [task]
    prompt_input += [curr_time_range]
    prompt_input += [duration]
    prompt_input += [persona.scratch.get_str_firstname()]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    print ("TOODOOOOOO")
    print (gpt_response)
    print ("-==- -==- -==- ")

    # TODO SOMETHING HERE sometimes fails... See screenshot
    gpt_response = gpt_response.split('\n---\n')[0]
    temp = [i.strip() for i in gpt_response.split("\n")]
    _cr = []
    cr = []
    for count, i in enumerate(temp): 
      if len(i.strip()) == 0:
        continue
      if count != 0: 
        _cr += [" ".join([j.strip () for j in i.split(" ")][3:])]
      else: 
        _cr += [i]
    for count, i in enumerate(_cr): 
      if "(duration in minutes:" not in i:
        continue
      k = [j.strip() for j in i.split("(duration in minutes:")]
      
      try:
        task = k[0]
        if task[-1] == ".": 
          task = task[:-1]
        duration = int(k[1].split(",")[0].strip())
        cr += [[task, duration]]
      except:
        print('k is ', k)
        print('i is ', i)
        print('_cr is ', _cr)
        assert 0

    total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                   .split("):")[0].strip())
    
    # TODO -- now, you need to make sure that this is the same as the sum of 
    #         the current action sequence. 
    curr_min_slot = [["dummy", -1],] # (task_name, task_index)
    for count, i in enumerate(cr): 
      i_task = i[0] 
      i_duration = i[1]

      i_duration -= (i_duration % 5)
      if i_duration > 0: 
        for j in range(i_duration): 
          curr_min_slot += [(i_task, count)]       
    curr_min_slot = curr_min_slot[1:]   
    
    try:
      if len(curr_min_slot) > total_expected_min: 
        last_task = curr_min_slot[60]
        for i in range(1, 6): 
          curr_min_slot[-1 * i] = last_task
      elif len(curr_min_slot) < total_expected_min: 
        last_task = curr_min_slot[-1]
        for i in range(total_expected_min - len(curr_min_slot)):
          curr_min_slot += [last_task]
    except:
      print('curr_min_slot is ', curr_min_slot)
      assert 0

    cr_ret = [["dummy", -1],]
    for task, task_index in curr_min_slot: 
      if task != cr_ret[-1][0]: 
        cr_ret += [[task, 1]]
      else: 
        cr_ret[-1][1] += 1
    cr = cr_ret[1:]

    return cr

  def __func_validate(gpt_response, prompt=""): 
    # TODO -- this sometimes generates error 
    try: 
      __func_clean_up(gpt_response)
    except: 
      pass
      # return False
    return gpt_response

  def get_fail_safe(): 
    fs = ["asleep"]
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000, 
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
  prompt_input = create_prompt_input(persona, task, duration)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  print ("?????")
  print (prompt)
  output = safe_generate_response(prompt, gpt_param, 5, get_fail_safe(),
                                   __func_validate, __func_clean_up, client=client)

  # TODO THERE WAS A BUG HERE... 
  # This is for preventing overflows...

  print ("IMPORTANT VVV DEBUG")

  # print (prompt_input)
  # print (prompt)
  print (output)

  fin_output = []
  time_sum = 0
  for i_task, i_duration in output: 
    time_sum += i_duration
    if '1) John is ' in i_task:
      i_task = i_task.replace('1) John is ', '')
    # HM?????????
    # if time_sum < duration: 
    if time_sum <= duration: 
      fin_output += [[i_task, i_duration]]
    else: 
      break
  ftime_sum = 0
  for fi_task, fi_duration in fin_output: 
    ftime_sum += fi_duration
  
  # print ("for debugging... line 365", fin_output)
  fin_output[-1][1] += (duration - ftime_sum)
  output = fin_output 



  task_decomp = output
  ret = []
  for decomp_task, duration in task_decomp: 
    ret += [[f"{task} ({decomp_task})", duration]]
  output = ret


  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_action_sector(action_description, 
                                persona, 
                                maze, 
                                test_input=None, 
                                verbose=False, client=None):
  def create_prompt_input(action_description, persona, maze, test_input=None): 
    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    
    prompt_input = []
    
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [persona.scratch.living_area.split(":")[1]]
    x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]


    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
    x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

    if persona.scratch.get_str_daily_plan_req() != "": 
      prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
    else: 
      prompt_input += [""]


    # MAR 11 TEMP
    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    fin_accessible_sectors = []
    for i in curr: 
      if "'s house" in i: 
        if persona.scratch.last_name in i: 
          fin_accessible_sectors += [i]
      else: 
        fin_accessible_sectors += [i]
    accessible_sector_str = ", ".join(fin_accessible_sectors)
    # END MAR 11 TEMP

    prompt_input += [accessible_sector_str]



    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description: 
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [action_description_1]

    prompt_input += [action_description_2]
    prompt_input += [persona.scratch.get_str_name()]
    return prompt_input


    

    


  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.split("}")[0]
    return cleaned_response

  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    if "}" not in gpt_response:
      return False
    if "," in gpt_response: 
      return False
    return True
  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v1/action_location_sector_v1.txt"
  prompt_input = create_prompt_input(action_description, persona, maze)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                  #  __func_validate, __func_clean_up, client=client)
  output = "Lin family's house"
  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x: 
    output = persona.scratch.living_area.split(":")[1]

  print ("DEBUG", random.choice(x), "------", output)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_action_arena(action_description, 
                                persona, 
                                maze, act_world, act_sector,
                                test_input=None, 
                                verbose=False, client=None):
  def create_prompt_input(action_description, persona, maze, act_world, act_sector, test_input=None): 
    prompt_input = []
    prompt_input += [persona.scratch.get_str_name()]
    x = f"{act_world}:{act_sector}"
    prompt_input += [act_sector]

    # MAR 11 TEMP
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible_arena_str.split(", ")
    fin_accessible_arenas = []
    for i in curr: 
      if "'s room" in i: 
        if persona.scratch.last_name in i: 
          fin_accessible_arenas += [i]
      else: 
        fin_accessible_arenas += [i]
    accessible_arena_str = ", ".join(fin_accessible_arenas)
    # END MAR 11 TEMP


    prompt_input += [accessible_arena_str]


    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description: 
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]
    
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [action_description_1]
    prompt_input += [action_description_2]
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [act_sector]

    prompt_input += [accessible_arena_str]

    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.split("}")[0]
    return cleaned_response

  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    if "}" not in gpt_response:
      return False
    if "," in gpt_response: 
      return False
    return True
  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v1/action_location_object_vMar11.txt"
  prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  print (output)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_action_game_object(action_description, 
                                      persona, 
                                      maze,
                                      temp_address,
                                      test_input=None, 
                                      verbose=False, client=None): 
  def create_prompt_input(action_description, 
                          persona, 
                          temp_address, 
                          test_input=None): 
    prompt_input = []
    if "(" in action_description: 
      action_description = action_description.split("(")[-1][:-1]
      
    prompt_input += [action_description]
    prompt_input += [persona
                     .s_mem.get_str_accessible_arena_game_objects(temp_address)]
    return prompt_input
  
  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    return True

  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  def get_fail_safe(): 
    fs = ("bed")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v1/action_object_v2.txt"
  prompt_input = create_prompt_input(action_description, 
                                     persona, 
                                     temp_address, 
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
  if output not in x: 
    output = random.choice(x)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]




def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False, client=None): 
  def create_prompt_input(action_description): 
    if "(" in action_description: 
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [action_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if len(cr) > 3:
      cr = cr[:3]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) == 0: 
        return False
    except: return False
    return True 

  def get_fail_safe(): 
    fs = "😋"
    return fs


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    cr = gpt_response.strip()
    if len(cr) > 3:
      cr = cr[:3]
    return cr

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) == 0: 
        return False
    except: return False
    return True 
    return True

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt" ########
  prompt_input = create_prompt_input(action_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "🛁🧖‍♀️" ########
  special_instruction = "The value for the output must ONLY contain the emojis." ########
  fail_safe = get_fail_safe()
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_event_triple(action_description, persona, verbose=False, client=None): 
  def create_prompt_input(action_description, persona): 
    if "(" in action_description: 
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [persona.name, 
                    action_description,
                    persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) != 2: 
        return False
    except: return False
    return True 

  def get_fail_safe(persona): 
    fs = (persona.name, "is", "idle")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 30, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(action_description, persona)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(persona) ########
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  output = (persona.name, output[0], output[1])
  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False, client=None): 
  def create_prompt_input(act_game_object, act_desp, persona): 
    prompt_input = [act_game_object, 
                    persona.name,
                    act_desp,
                    act_game_object,
                    act_game_object]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = f"{act_game_object} is idle"
    return fs

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt" ########
  prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "being fixed" ########
  special_instruction = "The output should ONLY contain the phrase that should go in <fill in>." ########
  fail_safe = get_fail_safe(act_game_object) ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)

  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False, client=None): 
  def create_prompt_input(act_game_object, act_obj_desc): 
    prompt_input = [act_game_object, 
                    act_obj_desc,
                    act_game_object]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) != 2: 
        return False
    except: return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = (act_game_object, "is", "idle")
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 30, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(act_game_object, act_obj_desc)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(act_game_object)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)
  output = (act_game_object, output[0], output[1])

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_new_decomp_schedule(persona, 
                                       main_act_dur, 
                                       truncated_act_dur, 
                                       start_time_hour,
                                       end_time_hour, 
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None, 
                                       verbose=False, client=None): 
  def create_prompt_input(persona, 
                           main_act_dur, 
                           truncated_act_dur, 
                           start_time_hour,
                           end_time_hour, 
                           inserted_act,
                           inserted_act_dur,
                           test_input=None): 
    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur: 
      original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      original_plan += "\n"
      for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur): 
      new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      new_plan_init += "\n"
      if count < len(truncated_act_dur) - 1: 
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

    prompt_input = [persona_name, 
                    start_hour_str,
                    end_hour_str,
                    original_plan,
                    persona_name,
                    inserted_act,
                    inserted_act_dur,
                    persona_name,
                    start_hour_str,
                    end_hour_str,
                    end_hour_str,
                    new_plan_init]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    new_schedule = prompt + " " + gpt_response.strip()
    new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
    new_schedule = new_schedule.split("\n")

    ret_temp = []
    for i in new_schedule: 
      ret_temp += [i.split(" -- ")]

    ret = []
    for time_str, action in ret_temp:
      start_time = time_str.split(" ~ ")[0].strip()
      end_time = time_str.split(" ~ ")[1].strip()
      delta = datetime.datetime.strptime(end_time, "%H:%M") - datetime.datetime.strptime(start_time, "%H:%M")
      delta_min = int(delta.total_seconds()/60)
      if delta_min < 0: delta_min = 0
      ret += [[action, delta_min]]

    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt)
      dur_sum = 0
      for act, dur in gpt_response: 
        dur_sum += dur
        if str(type(act)) != "<class 'str'>":
          return False 
        if str(type(dur)) != "<class 'int'>":
          return False
      x = prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
      x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
      delta_min = int((x[1] - x[0]).total_seconds()/60)

      if int(dur_sum) != int(delta_min): 
        return False

    except: 
      return False
    return True 

  def get_fail_safe(main_act_dur, truncated_act_dur): 
    dur_sum = 0
    for act, dur in main_act_dur: dur_sum += dur

    ret = truncated_act_dur[:]
    ret += main_act_dur[len(ret)-1:]

    # If there are access, we need to trim... 
    ret_dur_sum = 0
    count = 0
    over = None
    for act, dur in ret: 
      ret_dur_sum += dur
      if ret_dur_sum == dur_sum: 
        break
      if ret_dur_sum > dur_sum: 
        over = ret_dur_sum - dur_sum
        break
      count += 1 

    if over: 
      ret = ret[:count+1]
      ret[-1][1] -= over

    return ret

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.txt"
  prompt_input = create_prompt_input(persona, 
                                     main_act_dur, 
                                     truncated_act_dur, 
                                     start_time_hour,
                                     end_time_hour, 
                                     inserted_act,
                                     inserted_act_dur,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False, client=None): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = ""
    last_chat_about = ""
    if last_chat: 
      last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
      last_chat_about = last_chat.description

    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    
    if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      init_p_desc = f"{init_persona.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
      init_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    
    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      target_p_desc = f"{target_persona.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
      target_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"


    prompt_input = []
    prompt_input += [context]

    prompt_input += [curr_time]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [last_chatted_time]
    prompt_input += [last_chat_about]


    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    return prompt_input
  
  def __func_validate(gpt_response, prompt=""): 
    try: 
      if gpt_response.split("Answer in yes or no:")[-1].strip().lower() in ["yes", "no"]: 
        return True
      return False     
    except:
      return False 

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split("Answer in yes or no:")[-1].strip().lower()

  def get_fail_safe(): 
    fs = "yes"
    return fs


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 20, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False, client=None): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 

    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    if len(init_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    if len(target_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

    prompt_input = []
    prompt_input += [context]
    prompt_input += [curr_time]
    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]

    prompt_input += [init_persona.name]
    prompt_input += [init_act_desc]
    prompt_input += [target_persona.name]
    prompt_input += [target_act_desc]

    prompt_input += [init_act_desc]
    return prompt_input
  
  def __func_validate(gpt_response, prompt=""): 
    try: 
      if gpt_response.split("Answer: Option")[-1].strip().lower() in ["3", "2", "1"]: 
        return True
      return False     
    except:
      return False 

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split("Answer: Option")[-1].strip().lower() 

  def get_fail_safe(): 
    fs = "3"
    return fs


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 20, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False, client=None): 
  def create_prompt_input(init_persona, target_persona, curr_loc, 
                          test_input=None): 

    prev_convo_insert = "\n"
    if init_persona.a_mem.seq_chat: 
      for i in init_persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((init_persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
          for row in i.filling: 
            prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if init_persona.a_mem.seq_chat: 
      if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""


    init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                target_persona.scratch.act_event[1],
                                target_persona.scratch.act_event[2])
    init_persona_thought = ""
    for i in init_persona_thought_nodes: 
      init_persona_thought += f"-- {i.description}\n"

    target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(init_persona.scratch.act_event[0],
                                init_persona.scratch.act_event[1],
                                init_persona.scratch.act_event[2])
    target_persona_thought = ""
    for i in target_persona_thought_nodes: 
      target_persona_thought += f"-- {i.description}\n"

    init_persona_curr_desc = ""
    if init_persona.scratch.planned_path: 
      init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
    else: 
      init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

    target_persona_curr_desc = ""
    if target_persona.scratch.planned_path: 
      target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
    else: 
      target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"
 

    curr_loc = curr_loc["arena"]

    prompt_input = []
    prompt_input += [init_persona.scratch.get_str_iss()]
    prompt_input += [target_persona.scratch.get_str_iss()]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [init_persona_thought]

    prompt_input += [target_persona.name]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona_thought]

    prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

    prompt_input += [init_persona_curr_desc]
    prompt_input += [target_persona_curr_desc]

    prompt_input += [prev_convo_insert]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]

    prompt_input += [curr_loc]
    prompt_input += [init_persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
    content = re.findall('"([^"]*)"', gpt_response)

    speaker_order = []
    for i in gpt_response.split("\n"): 
      name = i.split(":")[0].strip() 
      if name: 
        speaker_order += [name]

    ret = []
    for count, speaker in enumerate(speaker_order): 
      ret += [[speaker, content[count]]]

    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(init_persona, target_persona): 
    convo = [[init_persona.name, "Hi!"], 
             [target_persona.name, "Hi!"]]
    return convo


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, curr_loc, 
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(persona, target_persona)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False, client=None): 
  def create_prompt_input(conversation, test_input=None): 
    convo_str = ""
    for row in conversation: 
      convo_str += f'{row[0]}: "{row[1]}"\n'

    prompt_input = [convo_str]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    ret = "conversing about " + gpt_response.strip()
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "conversing with a housemate about morning greetings"


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    ret = "conversing about " + gpt_response.strip()
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt" ########
  prompt_input = create_prompt_input(conversation, test_input)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "conversing about what to eat for lunch" ########
  special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  

def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False, client=None): 
  def create_prompt_input(description, test_input=None): 
    if "\n" in description: 
      description = description.replace("\n", " <LINE_BREAK> ")
    prompt_input = [description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print ("???")
    print (gpt_response)
    gpt_response = gpt_response.strip().split("Emotive keywords:")
    factual = [i.strip() for i in gpt_response[0].split(",")]
    emotive = [i.strip() for i in gpt_response[1].split(",")]
    all_keywords = factual + emotive
    ret = []
    for i in all_keywords: 
      if i: 
        i = i.lower()
        if i[-1] == ".": 
          i = i[:-1]
        ret += [i]
    print (ret)
    return set(ret)

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return []

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
  prompt_input = create_prompt_input(description, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)


  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, keyword, concept_summary, test_input=None): 
    prompt_input = [keyword, concept_summary, persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 40, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(persona, keyword, concept_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_convo_to_thoughts(persona, 
                                    init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None, verbose=False, client=None): 
  def create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None): 
    prompt_input = [init_persona_name,
                    target_persona_name,
                    convo_str,
                    init_persona_name,
                    fin_target]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 40, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = int(gpt_response)
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4



  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = int(gpt_response)
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  try:
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  except:
    output = 5
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
 

def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = int(gpt_response)
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = int(gpt_response)
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __chat_func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = int(gpt_response)
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = int(gpt_response)
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = "1) " + gpt_response.strip()
    ret = []
    for i in gpt_response.split("\n"): 
      ret += [i.split(") ")[-1]]
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["Who am I"] * n


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    ret = ast.literal_eval(gpt_response)
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, n)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]' ########
  special_instruction = "Output must be a list of str." ########
  fail_safe = get_fail_safe(n) ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================






  gpt_param = {"engine": "text-davinci-003", "max_tokens": 150, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


  
def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = "1. " + gpt_response.strip()
    ret = dict()
    for i in gpt_response.split("\n"): 
      row = i.split(". ")[-1]
      thought = row.split("(because of ")[0].strip()
      evi_raw = row.split("(because of ")[1].split(")")[0].strip()
      evi_raw = re.findall(r'\d+', evi_raw)
      evi_raw = [int(i.strip()) for i in evi_raw]
      ret[thought] = evi_raw
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["I am hungry"] * n

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 150, 
               "temperature": 0.5, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None): 
    prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently, 
                    statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, target_persona, statements, test_input=None): 
    prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context, 
                               init_summ_idea, 
                               target_summ_idea, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None): 
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    print (prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"
    

    prompt_input = [persona.scratch.currently, 
                    target_persona.scratch.currently, 
                    prev_convo_insert,
                    curr_context, 
                    curr_location,

                    persona.scratch.name,
                    init_summ_idea, 
                    persona.scratch.name,
                    target_persona.scratch.name,

                    target_persona.scratch.name,
                    target_summ_idea, 
                    target_persona.scratch.name,
                    persona.scratch.name,

                    persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print (gpt_response)

    gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
    content = re.findall('"([^"]*)"', gpt_response)

    speaker_order = []
    for i in gpt_response.split("\n"): 
      name = i.split(":")[0].strip() 
      if name: 
        speaker_order += [name]

    ret = []
    for count, speaker in enumerate(speaker_order): 
      ret += [[speaker, content[count]]]

    return ret



  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."




  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    # ret = ast.literal_eval(gpt_response)

    for row in gpt_response: 
      print (row)

    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    return True


  # print ("HERE JULY 23 -- ----- ") ########
  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]' ########
  special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  # print ("HERE END JULY 23 -- ----- ") ########
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, statements, question, test_input=None): 
    prompt_input = [statements, persona.scratch.name, question]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, question)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
 

def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None): 
    prompt_input = [persona.scratch.name, 
                    persona.scratch.get_str_iss(),
                    persona.scratch.name, 
                    interlocutor_desc, 
                    prev_convo, 
                    persona.scratch.name,
                    retrieved_summary, 
                    persona.scratch.name,]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  gpt_param = {"engine": "text-davinci-003", "max_tokens": 250, 
               "temperature": 1, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
  prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]






def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, whisper, test_input=None): 
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False, client=None): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  gpt_param = {"engine": "text-davinci-002", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt" ########
  prompt_input = create_prompt_input(persona, all_utt)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe was interesting to talk to.' ########
  special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True, client=client)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/memo_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up, client=client)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]




def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False, client=None): 
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = json.loads(gpt_response)
    return gpt_response["output"]

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      fields = ["output"]
      response = json.loads(gpt_response)
      for field in fields: 
        if field not in response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  print ("11")
  prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt" 
  prompt_input = create_prompt_input(comment) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe = get_fail_safe() 
  output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose, client=client)
  print (output)
  
  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def extract_first_json_dict(data_str):
    # Find the first occurrence of a JSON object within the string
    start_idx = data_str.find('{')
    end_idx = data_str.find('}', start_idx) + 1

    # Check if both start and end indices were found
    if start_idx == -1 or end_idx == 0:
        return None

    # Extract the first JSON dictionary
    json_str = data_str[start_idx:end_idx]

    try:
        # Attempt to parse the JSON data
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        # If parsing fails, return None
        return None


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False, client=None): 
  def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    print (prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items(): 
      for v in vals: 
        retrieved_str += f"- {v.description}\n"


    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "": 
      convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
    prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
      curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
      convo_str, init_persona.scratch.name, target_persona.scratch.name,
      init_persona.scratch.name, init_persona.scratch.name,
      init_persona.scratch.name
      ]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)

    cleaned_dict = dict()
    cleaned = []
    for key, val in gpt_response.items(): 
      cleaned += [val]
    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
      cleaned_dict["end"] = False

    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("ugh...")
    try: 
      print (extract_first_json_dict(gpt_response))

      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict

  print ("11")
  prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" 
  prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe = get_fail_safe() 
  output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose, client=client)
  print (output)
  
  gpt_param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_make_suggestion(persona, user_persona, train_mode, retrieved=None, client=None, is_john_walking=None, curr_time=None):
  def create_prompt_input(user_persona, retrieved, train_mode): 
    if 'givenpersona_reasonright' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rr_persona.txt"
    elif 'givenpersona_reasonleft' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rl_persona.txt"
    elif 'givenpersona_noreason' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_wo_r_persona.txt"
    elif 'reasonright' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rr.txt"
    elif 'reasonleft' in train_mode:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_w_rl.txt"
    else:
      prompt_template = "agent/prompt_template/v1/system_prompt_suggestion_wo_r.txt"
    f = open(prompt_template, "r")
    system_prompt = f.read()
    f.close()

    if 'givenpersona' in train_mode:
      # system_prompt = system_prompt.replace("<<PERSONA>>", user_persona.scratch.get_str_iss())
      preference_rubric = """Personal Preference: <<Personal Preference>>\nTiming: <<Timing>>\nFrequency: <<Frequency>>\nCommunication & Safety: <<Communication & Safety>>"""
      for category in user_persona.scratch.eval_criteria:
        repl_key = category.replace('_', ' ')
        preference_rubric = preference_rubric.replace(f'<<{repl_key}>>', user_persona.scratch.eval_criteria[category])
      system_prompt = system_prompt.replace("<<PREFERENCES>>", preference_rubric)


    prompt = """[MEMORY]
<<AGENT_MEMORY>>
[ACTION]
<<CURR_TIME>>
John Lin's Action: <<USER_ACTION>>

"""
    if 'reasonright' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Suggestion": "...", "Reason": "..."}'
    elif 'reasonleft' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Reason": "...", "Suggestion": "..."}'
    else:
      prompt = prompt + 'Remember to provide your response in the format: {"Suggestion": "..."}'
    
    prompt = prompt.replace('<<AGENT_MEMORY>>', retrieved)
    prompt = prompt.replace('<<CURR_TIME>>', curr_time)

    if is_john_walking == 'yes':
      prompt = prompt.replace('<<USER_ACTION>>', 'John Lin is walking.')
    else:
      prompt = prompt.replace('<<USER_ACTION>>', persona.scratch.user_action_for_suggestion)
    # import pdb; pdb.set_trace()
    return system_prompt, prompt


  def __func_clean_up_rr(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Suggestion": ')[1]
    suggestion_reason = after_suggestion.split(', "Reason": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rr(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Suggestion": ' not in gpt_response:
      return False
    if ', "Reason": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_rl(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Reason": ')[1]
    suggestion_reason = after_suggestion.split(', "Suggestion": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rl(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Reason": ' not in gpt_response:
      return False
    if ', "Suggestion": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_wo_r(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Suggestion": ')[1]
    suggestion = after_suggestion.split('}')[0]
    if 'no' in suggestion.lower() and ('recom' in suggestion.lower() or 'recrea' in suggestion.lower()):
      suggestion = "No Recommendation"
    if len(suggestion) < 7:
      suggestion = "No Recommendation"
    return suggestion, ""

  def __func_validate_wo_r(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Suggestion": ' not in gpt_response:
      return False
    if '"}' not in gpt_response:
      return False
    if re.search(r'[\u4e00-\u9fff]', gpt_response):
      return False
    if re.search(r'[\u0400-\u04FF]', gpt_response):
      return False
    return True
  
  def get_fail_safe(): 
    # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
    fs = "No Recommendation", "No Recommendation"
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  system_prompt, prompt = create_prompt_input(user_persona, retrieved, train_mode)

  fail_safe = get_fail_safe()
  if 'llama' in train_mode or 'mistral' in train_mode:
    persona.current_prompt = prompt
    if 'reasonright' in train_mode:
      suggestion, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_rr, __func_clean_up_rr, persona=persona)
    elif 'reasonleft' in train_mode:
      suggestion, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_rl, __func_clean_up_rl, persona=persona)
    else:
      suggestion, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_wo_r, __func_clean_up_wo_r, persona=persona)
    
  else:
    if 'reasonright' in train_mode:
      suggestion, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rr, __func_clean_up_rr, client=client)
    elif 'reasonleft' in train_mode:
      suggestion, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rl, __func_clean_up_rl, client=client)
    else:
      suggestion, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_wo_r, __func_clean_up_wo_r, client=client)
  
  if 'no recomm' not in suggestion.lower():
    prompt_template = "agent/prompt_template/v1/refine_suggestion.txt"
    f = open(prompt_template, "r")
    prompt_ = f.read()
    f.close()
    prompt_ = prompt_.replace('<<CONTENT>>', suggestion)
    suggestion = safe_generate_response_sugg(None, prompt_, gpt_param, 5, fail_safe,
                                __func_validate_wo_r, __func_clean_up_wo_r, client=client)
    suggestion = suggestion[0]
  
  print('Retrieved: ', prompt)
  print ("Suggestion:", suggestion)
  print("Reason:", reason)
  return suggestion, reason


def run_mmlu_eval(agent_persona, train_mode):
  def format_question(question, choices):
    choice_labels = ['A', 'B', 'C', 'D']
    formatted_choices = '\n'.join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    prompt = f"""Answer the following multiple choice question by selecting the most appropriate option.

Question: {question}

Options:
{formatted_choices}

Answer: """
    return prompt

  def extract_answer(response):
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)
    
    if response and response[0].upper() in ['A', 'B', 'C', 'D']:
        return response[0].upper()
        
    return 'A'

  def get_model_prediction(prompt):
    inputs = agent_persona.suggestion_model[2](prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
      outputs = agent_persona.suggestion_model[0].generate(
          **inputs,
          max_new_tokens=10,
          temperature=0.1,
          do_sample=False,
          pad_token_id=agent_persona.suggestion_model[2].eos_token_id
      )
      
      response = agent_persona.suggestion_model[2].decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
      return response.strip()

  # def get_model_prediction(prompt):
    
  #   def extract_answer(response, prompt=""):
  #     match = re.search(r'\b([ABCD])\b', response.upper())
  #     if match:
  #         return match.group(1)
      
  #     if response and response[0].upper() in ['A', 'B', 'C', 'D']:
  #         return response[0].upper()
          
  #     return 'A'
    
  #   def validate_answer(answer, prompt=""):
  #     if 'A' in answer or 'B' in answer or 'C' in answer or 'D' in answer:
  #       return True
  #     return False
    
  #   def get_fail_safe(): 
  #     # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
  #     fs = "A"
  #     return fs

  #   if 'llama' in train_mode or 'mistral' in train_mode:
  #     answer = opensource_safe_generate_response("You are a helpful assistant.", prompt, train_mode, 5, get_fail_safe,
  #                                 validate_answer, extract_answer, persona=agent_persona)
  #   return answer
  
  def evaluate_subject(subject, max_samples=100):
    print(f"Evaluating subject: {subject}")
    
    dataset = load_dataset("cais/mmlu", subject)
    test_data = dataset['test']
    
    if max_samples:
      test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    correct = 0
    total = len(test_data)
    predictions = []
    
    for i, example in enumerate(tqdm(test_data, desc=f"Processing {subject}")):
      question = example['question']
      choices = example['choices']
      correct_answer_idx = example['answer']
      correct_answer = ['A', 'B', 'C', 'D'][correct_answer_idx]
      
      prompt = format_question(question, choices)
      response = get_model_prediction(prompt)
      predicted_answer = extract_answer(response)
      
      is_correct = predicted_answer == correct_answer
      if is_correct:
          correct += 1
          
      predictions.append({
          'question': question,
          'choices': choices,
          'correct_answer': correct_answer,
          'predicted_answer': predicted_answer,
          'is_correct': is_correct,
          'model_response': response
      })
    
    accuracy = correct / total
    
    return {
        'subject': subject,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions
    }
  
  max_samples_per_subject = 100
  save_results = True
  all_results = {}
  subject_accuracies = {}

  subjects = ['human_aging', 'nutrition', 'world_religions', 'prehistory', 'sociology']

  for subject in subjects:
      try:
          result = evaluate_subject(subject, max_samples_per_subject)
          all_results[subject] = result
          subject_accuracies[subject] = result['accuracy']
          print(f"{subject}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
      except Exception as e:
          print(f"Error evaluating {subject}: {str(e)}")
          continue

  overall_accuracy = np.mean(list(subject_accuracies.values()))

  final_results = {
      'overall_accuracy': overall_accuracy,
      'subject_accuracies': subject_accuracies
  }

  return final_results

def run_gpt_prompt_make_suggestion_divide(persona, user_persona, train_mode, retrieved=None, client=None, is_john_walking=None, curr_time=None):
  def create_prompt_input_no_recommendation_or_not(user_persona, retrieved, train_mode): 
    if 'reasonright' in train_mode:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_w_rr_abs_or_not.txt"
    elif 'reasonleft' in train_mode:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_w_rl_abs_or_not.txt"
    else:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_wo_r_abs_or_not.txt"
    f = open(prompt_template, "r")
    system_prompt = f.read()
    f.close()

    # prompt = prompt.replace('<<<USER_PERSONA>>>', user_persona)
    
    prompt = """[MEMORY]
<<AGENT_MEMORY>>

[ACTION]
<<CURR_TIME>> <<USER_ACTION>>

"""
    if 'reasonright' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Decision": "...", "Reason": "..."}'
    elif 'reasonleft' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Reason": "...", "Decision": "..."}'
    else:
      prompt = prompt + 'Remember to provide your response in the format: {"Decision": "..."}'
    
    prompt = prompt.replace('<<AGENT_MEMORY>>', retrieved)
    prompt = prompt.replace('<<CURR_TIME>>', curr_time)

    if is_john_walking == 'yes':
      prompt = prompt.replace('<<USER_ACTION>>', 'John Lin is walking.')
    else:
      prompt = prompt.replace('<<USER_ACTION>>', persona.scratch.user_action_for_suggestion)

    return system_prompt, prompt
  
  def create_prompt_input_suggestion(user_persona, retrieved, train_mode): 
    if 'reasonright' in train_mode:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_w_rr.txt"
    elif 'reasonleft' in train_mode:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_w_rl.txt"
    else:
      prompt_template = "agent/prompt_template/v1/divide_system_prompt_suggestion_wo_r.txt"
    f = open(prompt_template, "r")
    system_prompt = f.read()
    f.close()

    # prompt = prompt.replace('<<<USER_PERSONA>>>', user_persona)
    
    prompt = """[MEMORY]
<<AGENT_MEMORY>>

[ACTION]
<<CURR_TIME>> <<USER_ACTION>>

"""
    if 'reasonright' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Suggestion": "...", "Reason": "..."}'
    elif 'reasonleft' in train_mode:
      prompt = prompt + 'Remember to provide your response in the format: {"Reason": "...", "Suggestion": "..."}'
    else:
      prompt = prompt + 'Remember to provide your response in the format: {"Suggestion": "..."}'
    
    prompt = prompt.replace('<<AGENT_MEMORY>>', retrieved)
    prompt = prompt.replace('<<CURR_TIME>>', curr_time)

    if is_john_walking == 'yes':
      prompt = prompt.replace('<<USER_ACTION>>', 'John Lin is walking.')
    else:
      prompt = prompt.replace('<<USER_ACTION>>', persona.scratch.user_action_for_suggestion)

    return system_prompt, prompt


  def __func_clean_up_rr_abs_or_not(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Decision": ')[1]
    suggestion_reason = after_suggestion.split(', "Reason": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rr_abs_or_not(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Decision": ' not in gpt_response:
      return False
    if ', "Reason": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_rl_abs_or_not(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Reason": ')[1]
    suggestion_reason = after_suggestion.split(', "Decision": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rl_abs_or_not(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Reason": ' not in gpt_response:
      return False
    if ', "Decision": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_wo_r_abs_or_not(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Decision": ')[1]
    suggestion = after_suggestion.split('}')[0]
    return suggestion, ""

  def __func_validate_wo_r_abs_or_not(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Decision": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_rr(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Suggestion": ')[1]
    suggestion_reason = after_suggestion.split(', "Reason": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rr(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Suggestion": ' not in gpt_response:
      return False
    if ', "Reason": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_rl(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Reason": ')[1]
    suggestion_reason = after_suggestion.split(', "Suggestion": ')
    suggestion = suggestion_reason[0]
    reason = suggestion_reason[1].split('}')[0]
    return suggestion, reason

  def __func_validate_rl(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Reason": ' not in gpt_response:
      return False
    if ', "Suggestion": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def __func_clean_up_wo_r(gpt_response, prompt=""):
    after_suggestion = gpt_response.split('"Suggestion": ')[1]
    suggestion = after_suggestion.split('}')[0]
    return suggestion, ""

  def __func_validate_wo_r(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Suggestion": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True
  
  def get_fail_safe_abs_or_not_wo_r(): 
    # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
    fs = "No Recommendation"
    return fs
  
  def get_fail_safe_abs_or_not(): 
    # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
    fs = "No Recommendation", "No Recommendation"
    return fs
  
  def get_fail_safe_wo_r(): 
    # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
    fs = "May I help you?"
    return fs
  
  def get_fail_safe(): 
    # fs = "May I help you?", "This suggestion could be moderately helpful to anyone."
    fs = "May I help you?", "It is a generally helpful suggestion."
    return fs

  gpt_param = {"engine": "text-davinci-003", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  
  system_prompt, prompt = create_prompt_input_no_recommendation_or_not(user_persona, retrieved, train_mode)

  # fail_safe = get_fail_safe()
  if 'llama' in train_mode or 'mistral' in train_mode:
    persona.current_no_recommendation_prompt = prompt
    if 'reasonright' in train_mode:
      fail_safe = get_fail_safe_abs_or_not()
      decision, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_rr_abs_or_not, __func_clean_up_rr_abs_or_not, persona=persona)
    elif 'reasonleft' in train_mode:
      fail_safe = get_fail_safe_abs_or_not()
      decision, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_rl_abs_or_not, __func_clean_up_rl_abs_or_not, persona=persona)
    else:
      fail_safe = get_fail_safe_abs_or_not_wo_r()
      decision, reason = opensource_safe_generate_response(system_prompt, prompt, train_mode, 5, fail_safe,
                                  __func_validate_wo_r_abs_or_not, __func_clean_up_wo_r_abs_or_not, persona=persona)
    
  else:
    if 'reasonright' in train_mode:
      fail_safe = get_fail_safe_abs_or_not()
      decision, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rr_abs_or_not, __func_clean_up_rr_abs_or_not, client=client)
    elif 'reasonleft' in train_mode:
      fail_safe = get_fail_safe_abs_or_not()
      decision, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rl_abs_or_not, __func_clean_up_rl_abs_or_not, client=client)
    else:
      fail_safe = get_fail_safe_abs_or_not_wo_r()
      decision, reason = safe_generate_response_sugg(system_prompt, prompt, gpt_param, 5, fail_safe,
                                  __func_validate_wo_r_abs_or_not, __func_clean_up_wo_r_abs_or_not, client=client)

  print('Retrieved: ', prompt)
  print ("Decision:", decision)
  print("Reason:", reason)

  if decision.lower() == 'no recommendation':
    return 'No Recommendation', 'No Recommendation'
  
  suggestion_system_prompt, suggestion_prompt = create_prompt_input_suggestion(user_persona, retrieved, train_mode)

  fail_safe = get_fail_safe()
  if 'llama' in train_mode or 'mistral' in train_mode:
    persona.current_prompt = suggestion_prompt
    if 'reasonright' in train_mode:
      suggestion, reason = opensource_safe_generate_response(suggestion_system_prompt, suggestion_prompt, train_mode, 5, fail_safe,
                                  __func_validate_rr, __func_clean_up_rr, persona=persona)
    elif 'reasonleft' in train_mode:
      suggestion, reason = opensource_safe_generate_response(suggestion_system_prompt, suggestion_prompt, train_mode, 5, fail_safe,
                                  __func_validate_rl, __func_clean_up_rl, persona=persona)
    else:
      fail_safe = get_fail_safe_wo_r()
      suggestion, reason = opensource_safe_generate_response(suggestion_system_prompt, suggestion_prompt, train_mode, 5, fail_safe,
                                  __func_validate_wo_r, __func_clean_up_wo_r, persona=persona)
    
  else:
    if 'reasonright' in train_mode:
      suggestion, reason = safe_generate_response_sugg(suggestion_system_prompt, suggestion_prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rr, __func_clean_up_rr, client=client)
    elif 'reasonleft' in train_mode:
      suggestion, reason = safe_generate_response_sugg(suggestion_system_prompt, suggestion_prompt, gpt_param, 5, fail_safe,
                                  __func_validate_rl, __func_clean_up_rl, client=client)
    else:
      fail_safe = get_fail_safe_wo_r()
      suggestion, reason = safe_generate_response_sugg(suggestion_system_prompt, suggestion_prompt, gpt_param, 5, fail_safe,
                                  __func_validate_wo_r, __func_clean_up_wo_r, client=client)

  print('Retrieved: ', suggestion_prompt)
  print ("Decision:", suggestion)
  print("Reason:", reason)

  return suggestion, reason


def run_gpt_prompt_hourly_sum(time_sugg_list, client=None):
  def __func_clean_up(gpt_response, prompt=""):
    after_summ = gpt_response.split('"Summary": ')[1]
    summ = after_summ.split('}')[0]
    return summ

  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 10: 
      return False
    if '{"Summary": ' not in gpt_response:
      return False
    if "}" not in gpt_response:
      return False
    return True

  def get_fail_safe():
    return None

  prompt_template = "agent/prompt_template/v1/summarization_hourly_agent.txt" 
  f = open(prompt_template, "r")
  prompt = f.read()
  f.close()

  histories = ""
  for t, s in time_sugg_list:
    histories += f"{t}\n{s}\n\n"
  
  prompt = prompt.replace('<<HISTORY>>', histories)
  
  fail_safe = get_fail_safe() 
  output = safe_generate_response(prompt, {}, 3, fail_safe,
                        __func_validate, __func_clean_up, client=client)
  print (output)

  return output