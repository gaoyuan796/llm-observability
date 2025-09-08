import os
import requests
import json
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
from ddtrace.llmobs.utils import Messages

# Initialize the LLMObs instance.
llm_obs = LLMObs()
# You must explicitly enable LLM Observability with this command.
LLMObs.enable()

# --- Gemini API Configuration ---
# DO NOT HARD-CODE YOUR API KEY. Use an environment variable.
API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# The Gemini API endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

# The prompt you want to send
prompt_text = 'Generate a JMeter 5.6.3 .jmx script for a simple performance test. The test plan should be named "My Load Test" and contain a Thread Group with 10 users and a ramp-up period of 3 seconds. The script should send a GET request to "https://api.example.com/status" and include a Response Assertion to verify that the response code is 200.'

# --- Main Logic with Datadog Instrumentation ---
@llm(model_name="gemini-2.5-flash", name="invoke_llm", model_provider="gemini")
def llm_call(prompt):

    """
    with llm_obs.llm() as span:
        # Now, set the provider and model name as explicit tags.
        span.set_tag("llm.model_provider", "gemini")
        span.set_tag("llm.model_name", "gemini-2.5-flash")
        span.set_tag("llm.name", "invoke_llm")
    """

    # The request body as a Python dictionary
    request_body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Send the POST request
    response = requests.post(url, json=request_body)

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # Parse the JSON response
    response_data = response.json()

    # Extract the generated text
    generated_text = response_data['candidates'][0]['content']['parts'][0]['text']

    # Explicitly set the input and output on the current trace span
    # This ensures the data is correctly captured by Datadog
    # using the input_data and output_data arguments for compatibility
    LLMObs.annotate(
        span=None,
        input_data=Messages([{"content": prompt}]),
        output_data=Messages([{"content": generated_text}]),
    )

    return generated_text

if __name__ == "__main__":
    try:
        jmeter_script = llm_call(prompt_text)
        print("Generated JMeter Script:")
        print(jmeter_script)
    except Exception as e:
        print(f"An error occurred: {e}")

