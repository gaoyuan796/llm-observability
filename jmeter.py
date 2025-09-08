#!/usr/bin/env python
# coding: utf-8

# ## Initial Setup
# 
# The setup below shows in-code configuration. For most applications, we can also enable LLMObs simply by calling `ddtrace-run` with the appropriate env vars, [as seen here in our quickstart instructions](https://docs.datadoghq.com/tracing/llm_observability/quickstart/).
# 
# The code below requires you to have already created an `.env` file with several configuration variables as explained in the README.
# 

# In[6]:


from dotenv import load_dotenv

load_dotenv()

from ddtrace.llmobs import LLMObs

LLMObs.enable()


# **Note for enterprise customers using secrets:**
# 
# If you are using secrets, you can enable LLM Observability with more specific parameters as demonstrated below.
# 
# ```python
# LLMObs.enable(
#   ml_app="<YOUR_ML_APP_NAME>",
#   api_key="<YOUR_DATADOG_API_KEY>",
#   site="<YOUR_DATADOG_SITE>",
#   agentless_enabled=True,
# )
# ```
# 

# ## Tracing an LLM Call
# 
# An LLM Span represents a call to a model. In this simple example, we are asking `gpt-3.5-turbo` to summarize a provided text and identify a list of topics from the text.
# 
# Because we use OpenAI, the call to the LLM is instrumented automatically by Datadog, with no further action required on our part:
# 

# In[7]:


from openai import OpenAI
import json
import os

oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


sys_prompt = "You are a bot designed to generate JMeter responses. All of your output must be a valid JMeter script. Do not include any additional text or explanations outside of the JMeter."


def summarize(text, prompt=sys_prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    # llm span auto-instrumented via our openai integration
    response_content = (
        oai_client.chat.completions.create(
            messages=messages,
            model="gpt-4",
            response_format={"type": "text"}
        )
        .choices[0]
        .message.content
    )
    return response_content #json.loads(response_content)


# In[8]:


text = 'Generate a JMeter .jmx script for a simple performance test. The test plan should be named "My Load Test" and contain a Thread Group with 10 users and a ramp-up period of 3 seconds. The script should send a GET request to "https://api.example.com/status" and include a Response Assertion to verify that the response code is 200.'


# In[9]:


print(f'{summarize(text)}')


# ## Viewing the trace in Datadog
# 
# Now, check out the [LLM Observability interface](https://app.datadoghq.com/llm) in Datadog. You should see a trace that describes the LLM call, including the system prompt, the user prompt, and the response.
# 

# #### Additional resources
# 
# - [List of all integrations supported by Datadog's LLM Observability product](https://docs.datadoghq.com/tracing/llm_observability/sdk/#llm-integrations)
# - [Instructions for manually instrumenting an LLM Span](https://docs.datadoghq.com/llm_observability/setup/sdk/python/#llm-span)
# 
