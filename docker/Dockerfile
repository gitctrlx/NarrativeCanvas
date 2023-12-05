FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /sky

RUN pip install flask munch openai langchain nemoguardrails gunicorn

CMD ["gunicorn", "-b", "0.0.0.0:3008", "app:app"]
