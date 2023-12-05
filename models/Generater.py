import requests
import base64
import random
from langchain.chat_models import AzureChatOpenAI
from nemoguardrails import LLMRails, RailsConfig
from models.configurations import COLANG_CONFIG, YAML_CONFIG

class ImageGenerator:
    def __init__(self, invoke_url, fetch_url_format, headers, save_dir):
        self.invoke_url = invoke_url
        self.fetch_url_format = fetch_url_format
        self.headers = headers
        self.save_dir = save_dir
        self.session = requests.Session()

    def generate_image(self, prompt, negative_prompt, sampler, seed, unconditional_guidance_scale, inference_steps):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler": sampler,
            "seed": seed,
            "unconditional_guidance_scale": unconditional_guidance_scale,
            "inference_steps": inference_steps
        }

        response = self.session.post(self.invoke_url, headers=self.headers, json=payload)

        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            fetch_url = self.fetch_url_format + request_id
            response = self.session.get(fetch_url, headers=self.headers)

        response.raise_for_status()
        response_body = response.json()

        image_path = self._save_image(response_body['b64_json'])
        return image_path

    
    def _save_image(self, b64_data):
        random_number = random.randint(100000, 999999)
        image_path = f"{self.save_dir}/txt2img_result_{random_number}.jpg"
        with open(image_path, "wb") as fh:
            fh.write(base64.decodebytes(b64_data.encode('utf-8')))
        print(image_path)
        return image_path

class NGCLlama270BSteerLM:
    def __init__(self, invoke_url, fetch_url_format, headers):
        self.invoke_url = invoke_url
        self.fetch_url_format = fetch_url_format
        self.headers = headers
        self.session = requests.Session()

    def send_request(self, payload):
        response = self.session.post(self.invoke_url, headers=self.headers, json=payload)
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            fetch_url = self.fetch_url_format + request_id
            response = self.session.get(fetch_url, headers=self.headers)
        
        response.raise_for_status()
        return response.json()

    def get_content(self, payload):
        response_body = self.send_request(payload)
        content = response_body['choices'][0]['message']['content']
        return content

class AzureChatBot:
    def __init__(self, api_key, api_base, deployment_name, api_version):
        self.llm = AzureChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            deployment_name=deployment_name,
            openai_api_version=api_version
        )
        self.config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)
        self.app = LLMRails(self.config, llm=self.llm)

    def generate_message(self, user_text):
        new_message = self.app.generate_sync(messages=[{  # Assuming generate_sync exists
            "role": "user",
            "content": user_text
        }])
        return new_message['content']


# Usage
if __name__ == '__main__':
    # Example 1

    # First, you would create an instance of the ImageGeneratorClient with the required parameters.
    generator = ImageGenerator(
        
        invoke_url = "" ,
        fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",

        headers = {
            "Authorization": "Bearer ",
            "Accept": "application/json",
        },
        save_dir="../static/generateimg"
    )
    # Then you would call the generate_image method with the parameters for your image.
    image_path = generator.generate_image(
        prompt="A photo of a Shiba Inu dog with a backpack riding a bike",
        negative_prompt="beach",
        sampler="DDIM",
        seed=0,
        unconditional_guidance_scale=5,
        inference_steps=50
    )
    # The image is now saved to the specified directory, and you have the path to the image.
    print(f"Image saved at {image_path}")

    # Example 2
    nvapi = NGCLlama270BSteerLM(
        invoke_url = "",
        fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        headers = {
            "Authorization": "",
            "Accept": "application/json",
    })
    payload = {
        "messages": [
            {"content": "What is the Earth's relationship to the Sun?", "role": "user"},
            {"labels": {"complexity": 9, "creativity": 0, "verbosity": 9}, "role": "assistant"}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }
    print(nvapi.get_content(payload))

    # Example 3
    bot = AzureChatBot('', 'https://nayopenai.openai.azure.com', '', '2023-03-15-preview')
    response = bot.generate_message('how are you.')

