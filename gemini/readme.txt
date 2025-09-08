Steps:

0. Datadog Doc Reference https://docs.datadoghq.com/llm_observability/instrumentation/auto_instrumentation/?tab=python

1. config api_key: and site: in datadog agent config file: /opt/datadog-agent/etc/datadog.yaml (leave a space after :)

2. start / stop datadog-agent

launchctl start com.datadoghq.agent
datadog-agent status
launchctl stop com.datadoghq.agent

3. run gemini.py (custom instrument) to generate JMeter test and monitor on Datadog LLM Observability

export GEMINI_API_KEY=<GEMINI API KEY>
DD_LLMOBS_ENABLED=1 DD_LLMOBS_ML_APP="gemini-jmeter" ddtrace-run python3 gemini.py

4. run gemini1.py (auto instrument) to monitor on Datadog LLM Observability

pip3 install google-genai
export GEMINI_API_KEY=<GEMINI API KEY>
DD_LLMOBS_ENABLED=1 DD_LLMOBS_ML_APP="gemini-jmeter" python3 gemini1.py

