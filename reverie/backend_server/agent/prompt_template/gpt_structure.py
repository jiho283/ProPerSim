import json
import random
import openai
import time 
import pickle
import os

from utils import *
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

openai.api_key = openai_api_key

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

# def ChatGPT_single_request(prompt): 
#   temp_sleep()

#   completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo", 
#     messages=[{"role": "user", "content": prompt}]
#   )
#   return completion["choices"][0]["message"]["content"]


# ============================================================================
# ##################[SECTION 0: Customized (GPT-4o-mini)] ####################
# ============================================================================

def ChatGPT_single_request(prompt, client=None): 
  try: 
    print('agent gpt-4o-mini inference')
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature= 0.1
    )
    return completion.choices[0].message.content
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

def ChatGPT_request(prompt, client=None): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    print('agent gpt-4o-mini inference')
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature= 0.1
    )
    return completion.choices[0].message.content
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False, 
                                   client=None): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt, client=client).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response, type(curr_gpt_response))
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      curr_gpt_response = ChatGPT_request(prompt, client=client).strip()
      print("Faild GPT reponse: ", curr_gpt_response)
      assert 0 
      # pass 

  return False

def GPT_request(prompt, gpt_parameter, client=None, system_prompt=None): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  # import pdb;pdb.set_trace()
  if system_prompt:
    try: 
      print('agent gpt-4o-mini inference with system prompt')
      completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "system", "content":system_prompt}, {"role": "user", "content": prompt}],
      temperature= 0.1
      )
      return completion.choices[0].message.content
    except: 
      print ("TOKEN LIMIT EXCEEDED")
      return "TOKEN LIMIT EXCEEDED"
  else:
    try: 
      print('agent gpt-4o-mini inference')
      completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": prompt}],
      temperature= 0.1
      )
      return completion.choices[0].message.content
    except: 
      print ("TOKEN LIMIT EXCEEDED")
      return "TOKEN LIMIT EXCEEDED"


def safe_generate_response_sugg(system_prompt, 
                                prompt, 
                                gpt_parameter,
                                repeat=5,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False,
                                client=None): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter, client=client, system_prompt=system_prompt)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response

def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           client=None): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter, client=client)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response



def opensource_inference(model, tokenizer, train_mode, system_prompt, input_text, max_new_tokens, temperature):
    """
    Performs inference using a Llama3 model.

    Parameters:
        input_text (str): The input text for the model.
        max_length (int): Maximum length of the generated text.
        temperature (float): Sampling temperature; higher values produce more diverse outputs.

    Returns:
        str: The generated text.
    """
    def suggestion_processing(generated_text:str):
      generated_text = generated_text.replace(' Suggestion', 'Suggestion').replace('Suggestion ', 'Suggestion').replace(' Reason', 'Reason').replace('Reason ', 'Reason').replace("'Suggestion'", '"Suggestion"').replace("'Reason'", '"Reason"')
      generated_text = generated_text.replace('\n', '').replace('{ "', '{"').replace('" }', '"}')
      if '{"Suggestion' in generated_text:
        generated_text = '{"Suggestion' + generated_text.split('{"Suggestion')[1]
      if generated_text[0] == '(' or generated_text[0] == '[':
        generated_text = '{'+generated_text[1:]
      if generated_text[-1] == ')' or generated_text[-1] == ']':
        generated_text = generated_text[:-1]+'}'
      if '}' not in generated_text:
        generated_text = generated_text + '}'
      return generated_text

    try:
        input_text = tokenizer.apply_chat_template(conversation=[{'role':'system', 'content':system_prompt}, {'role': 'user', 'content': input_text}], add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to('cuda')
        attention_mask = inputs.attention_mask.to('cuda')  # Create attention mask

        # vllm response
        # sampling_params = SamplingParams(
        #                     temperature=temperature, 
        #                     max_tokens=max_new_tokens, 
        #                     repetition_penalty=1.5) 
        
        # if os.path.exists("./lora_model") and len(os.listdir("./lora_model")) > 0:
        #   outputs = model.generate(
        #       input_text,
        #       sampling_params,
        #       lora_request=LoRARequest("sql_adapter", 1, "./lora_model")
        #   )
        # else:
        #   outputs = model.generate(
        #       input_text,
        #       sampling_params
        #   )
        outputs = model.generate(
            input_ids,
            temperature=temperature,
            use_cache=True,
            num_return_sequences=1,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5
        )
        # import pdb; pdb.set_trace()
        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.split('assistant\n\n')[-1]
        generated_text = suggestion_processing(generated_text)
        # import pdb; pdb.set_trace()
        # generated_text = suggestion_processing(outputs[0].outputs[0].text)
        return generated_text

    except Exception as e:
        # import pdb; pdb.set_trace()
        return f"An error occurred: {str(e)}"


def opensource_request(system_prompt, prompt, train_mode, persona): 
  temp_sleep()
  try: 
    model, base_model, tokenizer = persona.suggestion_model
    response = opensource_inference(model, tokenizer, train_mode, system_prompt, prompt, 128, 0.9)
    # import pdb; pdb.set_trace()
    return response
  except: 
    # import pdb; pdb.set_trace()
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def opensource_safe_generate_response(system_prompt,
                           prompt, 
                           train_mode,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           persona=None): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    
    curr_gpt_response = opensource_request(system_prompt, prompt, train_mode, persona)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response



def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   client=None): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt, client=client).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


# def ChatGPT_request(prompt): 
#   """
#   Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
#   server and returns the response. 
#   ARGS:
#     prompt: a str prompt
#     gpt_parameter: a python dictionary with the keys indicating the names of  
#                    the parameter and the values indicating the parameter 
#                    values.   
#   RETURNS: 
#     a str of GPT-3's response. 
#   """
#   # temp_sleep()
#   try: 
#     completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo", 
#     messages=[{"role": "user", "content": prompt}]
#     )
#     return completion["choices"][0]["message"]["content"]
  
#   except: 
#     print ("ChatGPT ERROR")
#     return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


# def ChatGPT_safe_generate_response(prompt, 
#                                    example_output,
#                                    special_instruction,
#                                    repeat=3,
#                                    fail_safe_response="error",
#                                    func_validate=None,
#                                    func_clean_up=None,
#                                    verbose=False): 
#   # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
#   prompt = '"""\n' + prompt + '\n"""\n'
#   prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
#   prompt += "Example output json:\n"
#   prompt += '{"output": "' + str(example_output) + '"}'

#   if verbose: 
#     print ("CHAT GPT PROMPT")
#     print (prompt)

#   for i in range(repeat): 

#     try: 
#       curr_gpt_response = ChatGPT_request(prompt).strip()
#       end_index = curr_gpt_response.rfind('}') + 1
#       curr_gpt_response = curr_gpt_response[:end_index]
#       curr_gpt_response = json.loads(curr_gpt_response)["output"]

#       # print ("---ashdfaf")
#       # print (curr_gpt_response)
      
#       if func_validate(curr_gpt_response, prompt=prompt): 
#         return func_clean_up(curr_gpt_response, prompt=prompt)
      
#       if verbose: 
#         print ("---- repeat count: \n", i, curr_gpt_response)
#         print (curr_gpt_response)
#         print ("~~~~")

#     except: 
#       pass

#   return False


# def ChatGPT_safe_generate_response_OLD(prompt, 
#                                    repeat=3,
#                                    fail_safe_response="error",
#                                    func_validate=None,
#                                    func_clean_up=None,
#                                    verbose=False): 
#   if verbose: 
#     print ("CHAT GPT PROMPT")
#     print (prompt)

#   for i in range(repeat): 
#     try: 
#       curr_gpt_response = ChatGPT_request(prompt).strip()
#       if func_validate(curr_gpt_response, prompt=prompt): 
#         return func_clean_up(curr_gpt_response, prompt=prompt)
#       if verbose: 
#         print (f"---- repeat count: {i}")
#         print (curr_gpt_response)
#         print ("~~~~")

#     except: 
#       pass
#   print ("FAIL SAFE TRIGGERED") 
#   return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

# def GPT_request(prompt, gpt_parameter): 
#   """
#   Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
#   server and returns the response. 
#   ARGS:
#     prompt: a str prompt
#     gpt_parameter: a python dictionary with the keys indicating the names of  
#                    the parameter and the values indicating the parameter 
#                    values.   
#   RETURNS: 
#     a str of GPT-3's response. 
#   """
#   temp_sleep()
#   try: 
#     response = openai.Completion.create(
#                 model=gpt_parameter["engine"],
#                 prompt=prompt,
#                 temperature=gpt_parameter["temperature"],
#                 max_tokens=gpt_parameter["max_tokens"],
#                 top_p=gpt_parameter["top_p"],
#                 frequency_penalty=gpt_parameter["frequency_penalty"],
#                 presence_penalty=gpt_parameter["presence_penalty"],
#                 stream=gpt_parameter["stream"],
#                 stop=gpt_parameter["stop"],)
#     return response.choices[0].text
#   except: 
#     print ("TOKEN LIMIT EXCEEDED")
#     return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


# def safe_generate_response(prompt, 
#                            gpt_parameter,
#                            repeat=5,
#                            fail_safe_response="error",
#                            func_validate=None,
#                            func_clean_up=None,
#                            verbose=False): 
#   if verbose: 
#     print (prompt)

#   for i in range(repeat): 
#     curr_gpt_response = GPT_request(prompt, gpt_parameter)
#     if func_validate(curr_gpt_response, prompt=prompt): 
#       return func_clean_up(curr_gpt_response, prompt=prompt)
#     if verbose: 
#       print ("---- repeat count: ", i, curr_gpt_response)
#       print (curr_gpt_response)
#       print ("~~~~")
#   return fail_safe_response


def get_embedding(text, model="text-embedding-3-small", client=None):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return client.embeddings.create(input = [text], model=model).data[0].embedding

if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















