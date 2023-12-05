import asyncio
import requests
import random
import os
import shutil
import json
from flask import Flask, render_template, request, jsonify
from utils import load_cfg

from models.Build import Builder
from models.Infer import ImageClassifier
from models.Generater import ImageGenerator, NGCLlama270BSteerLM


cfg = load_cfg('config.json')

app = Flask(__name__)


@app.context_processor
def inject_config():
    return dict(cfg=cfg)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route('/model/<model_id>', methods=['GET'])
def model_page(model_id):
    model = cfg.models.get(model_id)
    if model:
        return render_template(f'{model_id}.html', title=model['name'], description=model['description'])
    return render_template('index.html', message=f'No such model: {model_id}.', is_warning=True)


@app.route('/clearfix', methods=['POST'])
def clear_files_content():
    # print(cfg.app["upload_folder"])
    try:
        shutil.rmtree(cfg.app["upload_folder"])
        os.makedirs(cfg.app["upload_folder"])
        return jsonify({'message': 'Files content cleared, folder recreated！'})
    except Exception as e:
        return jsonify({'error': 'An error occurred while clearing files content！'}), 500


@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('file')[:8] # Up to 8 images can be uploaded, so only the first eight images will be processed
    image_list = []
    for file in uploaded_files:
        if file:
            random_number = random.randint(100000, 999999)
            filename = f"image{random_number}.jpg"
            file.save(os.path.join(cfg.app["upload_folder"], filename))
            image_list.append(f'static/images/{filename}')
    return jsonify({'image_list': image_list})


@app.route('/api/build', methods=['POST'])
def process_build():
    model = request.form.get('model_name')
    accuracy = request.form.get('accuracy')

    if not model or not accuracy:
        jsonify({'message': 'Missing model name or accuracy!'}), 500

    try:
        # preprocessed onnx: https://huggingface.co/CtrlX/ModelReady-pretrain/tree/main
        # Whether to use PSS: ./models/onnx/{model}-pss.onnx
        builder = Builder(onnxFile=f'./models/onnx/{model}-pss.onnx',
                          trtFile=f'./models/engine/{model}-pss_{accuracy}.plan',
                          accuracy=accuracy, optimization_level=3,
                          calibrationDataPath='./models/calibdata/',
                          int8cacheFile=f'./models/engine/int8Cache/{model}.cache',
                          timingCacheFile=f'./models/engine/timingCache/{model}.TimingCache',
                          removePlanCache=False)

        if builder.build_model():
            return jsonify({'message': '[INFO]Engine build successfully!'}), 200
        else:
            return jsonify({'message': '[ERROR]Engine build failed!'}), 500
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500


@app.route('/api/infer', methods=['POST'])
def process_infer():
    model_name = request.form.get('model_name')
    accuracy = request.form.get('accuracy')
    image_list = request.form.getlist('image_list')
    # image_list = ['static/images/picture1.jpg']

    if not model_name or not accuracy or not image_list:
        return jsonify({'message': 'Missing model name, accuracy, or image URLs.'}), 500

    try:
        classifier = ImageClassifier(
            trt_file=f'./models/engine/{model_name}-pss_{accuracy}.plan',
            labels_file='./models/imagenet_classes.txt'
        )
        predictions = classifier.predict(image_list)

        if predictions:
            return jsonify({'message': 'Inference successful!', 'predictions': predictions})
        else:
            return jsonify({'message': 'No predictions were made.'}), 500  # No Content

    except Exception as e:
        return jsonify({'message': f'Inference failed: {str(e)}'}), 500  # Server Error


@app.route('/api/generateStory', methods=['POST'])
def process_story():
    model_story = request.form.get('model_story')
    style_story = request.form.get('style_story')
    theme_story = request.form.get('theme_story')
    custom_prompt_story = request.form.get('custom_prompt_story')

    predictions = request.form.get('predictions')
    predictions = json.loads(predictions)
    max_lables = [max(d, key=d.get) for d in predictions]
    category = ','.join(max_lables)

    try:
        if model_story == "llama2":
            nvapi = NGCLlama270BSteerLM(
                invoke_url = cfg.llama2["invoke_url"],
                fetch_url_format = cfg.llama2["fetch_url_format"],
                headers = cfg.llama2["headers"])
            content = nvapi.get_content(payload = {
                "messages": [
                    {"content": f"请根据以下要求撰写一个中文故事，故事中必须包含与“{category}”相关的元素，主题围绕“[{theme_story}]”，并采用“{style_story}”的风格来展开。在故事中，请融入以下自定义内容：“{custom_prompt_story}”。", 
                     "role": "user"},
                    {"labels": {"complexity": 9, "creativity": 0, "verbosity": 9}, "role": "assistant"}
                ],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False
            })

        elif model_story == "gpt-3.5-turbo":
            content = "This model is currently not supported, please switch to another model.\n"
            # bot = AzureChatBot(
            #     cfg.azure_openai["api_key"], 
            #     cfg.azure_openai["api_base"],
            #     cfg.azure_openai["deployment_name"],
            #     cfg.azure_openai["api_version"]
            #     )
            # content = bot.generate_message(f"请根据以下要求撰写一个中文故事，故事中必须包含与“{category}”相关的元素，主题围绕“[{theme_story}]”，并采用“{style_story}”的风格来展开。在故事中，请融入以下自定义内容：“{custom_prompt_story}”。")
            
        chinese_response = content
        # print(chinese_response)
    
        if chinese_response:
            return jsonify({'message': 'Successfully generated story!', 'chinese_response': chinese_response})
        else:
            return jsonify({'message': 'Successfully generated story failed!'}), 500
    
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

@app.route('/api/generateSDPrompt', methods=['POST'])
def process_story_SDPrompt():
    model_story = request.form.get('model_story')
    style_story = request.form.get('style_story')
    theme_story = request.form.get('theme_story')
    custom_prompt_story = request.form.get('custom_prompt_story')

    chinese_text = request.form.get('chinese_response')

    try:
        if model_story == "llama2":
            nvapi = NGCLlama270BSteerLM(
                invoke_url = cfg.llama2["invoke_url"],
                fetch_url_format = cfg.llama2["fetch_url_format"],
                headers = cfg.llama2["headers"])
            content = nvapi.get_content(payload = {
                "messages": [
                    {"content": f"提炼下面故事中的的关键人物和事件，限制字数为10字以内，以此制作30字左右的英文的文生图稳定扩散提示词，体现“{theme_story}”主题和“{style_story}”风格。下面是故事内容：“{chinese_text}”" + custom_prompt_story,
                     "role": "user"},
                    {"labels": {"complexity": 9, "creativity": 0, "verbosity": 9}, "role": "assistant"}
                ],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False
            })

        elif model_story == "gpt-3.5-turbo":
            content = "This model is currently not supported, please switch to another model.\n"
            # bot = AzureChatBot(
            #     cfg.azure_openai["api_key"], 
            #     cfg.azure_openai["api_base"],
            #     cfg.azure_openai["deployment_name"],
            #     cfg.azure_openai["api_version"]
            #     )
            # content = bot.generate_message(f"提炼下面故事中的的关键人物和事件，限制字数为10字以内，以此制作30字左右的英文的文生图稳定扩散提示词，体现“{theme_story}”主题和“{style_story}”风格。下面是故事内容：“{chinese_text}”" + custom_prompt_story)   
            
        english_response = content
        # print(english_response)
    
        if english_response:
            return jsonify({'message': 'Successfully generated story!', 'english_response': english_response})
        else:
            return jsonify({'message': 'Successfully generated story failed!'}), 500
    
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

@app.route('/api/continueWriting', methods=['POST'])
def Continuewriting():
    model_story = request.form.get('model_story')
    style_story = request.form.get('style_story')
    theme_story = request.form.get('theme_story')
    custom_prompt_story = request.form.get('custom_prompt_story')

    previous_chinese_story = request.form.get('chinese_response')

    try:
        if model_story == "llama2":
            nvapi = NGCLlama270BSteerLM(
                invoke_url = cfg.llama2["invoke_url"],
                fetch_url_format = cfg.llama2["fetch_url_format"],
                headers = cfg.llama2["headers"])
            content = nvapi.get_content(payload = {
                "messages": [
                    {"content": f"请用中文续写下面的故事，确保续写部分与“[{theme_story}]”主题相符，并采用“[{style_story}]”风格。续写内容不超过100个字符。请根据所提供的故事内容和图像，巧妙地融合这些元素，创作一个流畅且吸引人的故事延续。以下是您需要继续的故事：" + previous_chinese_story + custom_prompt_story,  
                     "role": "user"},
                    {"labels": {"complexity": 9, "creativity": 0, "verbosity": 9}, "role": "assistant"}
                ],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False
            })

        elif model_story == "gpt-3.5-turbo":
            content = "This model is currently not supported, please switch to another model.\n"
            # bot = AzureChatBot(
            #     cfg.azure_openai["api_key"], 
            #     cfg.azure_openai["api_base"],
            #     cfg.azure_openai["deployment_name"],
            #     cfg.azure_openai["api_version"]
            #     )
            # content = bot.generate_message(f"请用中文续写下面的故事，确保续写部分与“[{theme_story}]”主题相符，并采用“[{style_story}]”风格。续写内容不超过100个字符。请根据所提供的故事内容和图像，巧妙地融合这些元素，创作一个流畅且吸引人的故事延续。以下是您需要继续的故事：" + previous_chinese_story + custom_prompt_story)   
  
            
        chinese_response = content
        print(chinese_response)
    
        if chinese_response :
            return jsonify({'message': 'Successfully generated story!', 'chinese_response': chinese_response,})
        else:
            return jsonify({'message': 'Generated story failed!'}), 500
    
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

@app.route('/api/generateImg', methods=['POST'])
def process_generateImg():
    model_image = request.form.get('model_image')
    style_image = request.form.get('style_image')
    negative_prompt = request.form.get('negative_prompt') 
    custom_prompt_image = request.form.get('custom_prompt_image')

    english_response = request.form.get('english_response')
    print(english_response)
    predictions = request.form.get('predictions')
    predictions = json.loads(predictions)
    max_lables = [max(d, key=d.get) for d in predictions]
    category = ','.join(max_lables)

    # if not predictions or not model_image:
    if not model_image:
        return "Model picture and style picture are required.", 400
    
    try:
        generator = ImageGenerator(
            invoke_url=cfg.sdxl["invoke_url"],
            fetch_url_format=cfg.sdxl["fetch_url_format"],
            headers=cfg.sdxl["headers"],
            save_dir=r"static/generateImg"
        )
        
        image_path = generator.generate_image(
            prompt=f"In the style of {style_image}, depicting {category}, envision a scene where {english_response}. {custom_prompt_image}",
            negative_prompt=negative_prompt,
            sampler="DDIM",
            seed=0,
            unconditional_guidance_scale=5,
            inference_steps=50
        )
        print('image_path' + image_path)
        if image_path:
            return jsonify({'message': 'Image generation successful!', 'image_path': image_path})
        else:
            return jsonify({'message': 'Image generation failed!'}), 500
        
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host=cfg.app['host'], port=cfg.app['port'], debug=True)
