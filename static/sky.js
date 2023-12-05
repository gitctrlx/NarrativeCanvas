// Global state variables
var image_list = [];
var currentIndex = 0;
var myChart = null;
var predictions = [];
var chinese_response;
var english_response;

document.addEventListener('DOMContentLoaded', function () {
    initializeEventListeners();
    initializeChart();
});

// Initialize button
function initializeEventListeners() {
    document.getElementById("prevButton").addEventListener("click", navigateImage.bind(null, -1));
    document.getElementById("nextButton").addEventListener("click", navigateImage.bind(null, 1));
    document.getElementById("uploadButton").addEventListener("click", triggerFileUpload);
}

// Initialize bar chart
function initializeChart() {
    const chartOptions = {
        scales: {
            y: { beginAtZero: true }
        },
        plugins: {
            datalabels: {
                anchor: 'end',
                align: 'top',
                formatter: (value) => value.toFixed(3)
            }
        }
    };
    
    var ctx = document.getElementById('myCanvas').getContext('2d');
    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: "IN1K Category",
                backgroundColor: "rgba(75,192,192,0.4)",
                borderColor: "rgba(75,192,192,1)",
                borderWidth: 1,
                data: []
            }]
        },
        options: chartOptions
    });
}

// Update bar chart
function updateChartData(labels, data) {
    myChart.data.labels = labels;
    myChart.data.datasets[0].data = data;
    myChart.update();
}

// page turning
function navigateImage(direction) {
    if (image_list.length > 0) {
        currentIndex = (currentIndex + direction + image_list.length) % image_list.length;
        document.getElementById('showSrcImage').src = '../' + image_list[currentIndex]; // '../static/images/picture1.jpg'
        if (predictions[currentIndex] && Object.keys(predictions[currentIndex]).length > 0) {
            updateChartData(Object.keys(predictions[currentIndex]), Object.values(predictions[currentIndex]));
        }
    }
}

// Function that triggers file upload
function triggerFileUpload() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    fileInput.onchange = () => handleImageUpload(fileInput.files);
    document.body.appendChild(fileInput);
    fileInput.click();
    fileInput.remove();
}

// Functions for handling image uploads
function handleImageUpload(files) {
    // Clear predictions array
    predictions = [];
    updateChartData(0, 0);

    // Check if files are more than 8
    if (files.length > 8) {
        alert('[WARNING]You can only upload up to 8 images. If there are more than 8 images, only the first 8 images will be processed!');
        //return; // Exit the function if more than 8 files
    }

    // Empty file content
    fetch('/clearfix', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error(`[ERROR]Failed to clear file content: ${response.status}`);
            const formData = new FormData();
            for (const file of files) formData.append('file', file);
            return fetch('/upload', { method: 'POST', body: formData });
        })
        .then(response => {
            if (!response.ok) throw new Error(`[ERROR]Failed to upload image: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.image_list) {
                image_list = data.image_list;
                console.log('image_list:', image_list);
                currentIndex = image_list.length - 1;
                document.getElementById('showSrcImage').src = '../' + image_list[currentIndex];// '../static/images/picture1.jpg'
            }
        })
        .catch(error => console.error('[ERROR]Error during upload:', error));
}


function Build() {
    updateChartData(0, 0);

    var model_name = document.getElementById('model_name').value;
    var accuracy = document.getElementsByName('accuracy');
    var accuracy;

    for (var i = 0; i < accuracy.length; i++) {
        if (accuracy[i].checked) {
            accuracy = accuracy[i].value;
            break;
          }
    }

    if (!model_name || !accuracy) {
        alert('[ERROR]Please enter a model name and select accuracy!');
        return; // Exit the function if the inputs are not valid
    }

    var formData = new FormData();
    formData.append('model_name', model_name);
    formData.append('accuracy', accuracy);

    alert('[INFO]Starting to build the engine, the construction time is approximately 3-7 minutes!')

    fetch('/api/build', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('[ERROR]Server responded with status ' + response.status);
          }
          return response.json();
    })
    .then(data => {
        alert(data.message); // Use the message from the server for user feedback
    })
    .catch(error => {
    //   console.error('Error during fetch:', error);
        alert('[ERROR]An error occurred while building the model: ' + error.message);
    });
}


async function Infer() {
    var modelName = document.getElementById('model_name').value;
    var accuracyRadios = document.getElementsByName('accuracy');
    var accuracy;

    for (let i = 0; i < accuracyRadios.length; i++) {
        if (accuracyRadios[i].checked) {
            accuracy = accuracyRadios[i].value;
            break;
        }
    }

    if (!modelName || !accuracy) {
        alert('[ERROR]Please provide a model name and select accuracy.');
        return; // Exit the function if the inputs are not valid
    }

    let formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('accuracy', accuracy);
    // formData.append('image_list', image_list);
    image_list.forEach(url => formData.append('image_list', url));

    try {
        let response = await fetch('/api/infer', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            let result = await response.json();
            if (result.predictions) {
                predictions = result.predictions
                updateChartData(Object.keys(predictions[currentIndex]), Object.values(predictions[currentIndex]));
                // alert('Inference successful.');
            } else {
                alert('[WARNING]Inference was successful, but no predictions were returned.');
            }
        } else {
            alert(`[ERROR]Please build the engine first and then execute the inference.`);
        }
    } catch (error) {
        // console.error('Error during inference:', error);
        alert('[ERROR]An error occurred during inference: '+ error);
    }
}

async function GenerateStory() {
    // empty it
    let textArea = document.getElementById('story-text-box');
    textArea.value = '';

    let form = new FormData();
    
    let model_story = document.getElementById('model_story').value;
    let style_story = document.getElementById('style_story').value;
    let theme_story = document.getElementById('theme_story').value;
    let custom_prompt_story = document.getElementById('custom_prompt_story').value;

    form.append("model_story",model_story );
    form.append("style_story",style_story);
    form.append("theme_story",theme_story);
    form.append("custom_prompt_story",custom_prompt_story);

    form.append("predictions", JSON.stringify(predictions));

    alert('[INFO]Start generating the story, which will take approximately 1 minute!')
    try {
        // Call generateStory API
        let response = await fetch("/api/generateStory", {
            method: 'post',
            body: form
        });
        if (response.ok) {
            let data = await response.json();
            chinese_response = data.chinese_response;
            textArea.value = chinese_response; // Update text box with Chinese response
            form.append("chinese_response", chinese_response);

            // Now call generateSDPrompt API
            let sdPromptResponse = await fetch("/api/generateSDPrompt", {
                method: 'post',
                body: form
            });

            if (sdPromptResponse.ok) {
                let sdData = await sdPromptResponse.json();
                english_response = sdData.english_response;
            } else {
                alert('[ERROR]Generate SD Prompt error: Status code: ' + sdPromptResponse.status);
            }
        } else {
            alert('[ERROR]Generate story error: Status code: ' + response.status);
        }
    } catch (error) {
        alert('[ERROR]An error occurred during Generate Story.');
    }
}

async function ContinueWriting(){

    let form = new FormData();

    let model_story = document.getElementById('model_story').value;
    let style_story = document.getElementById('style_story').value;
    let theme_story = document.getElementById('theme_story').value;
    let custom_prompt_story = document.getElementById('custom_prompt_story').value;

    form.append("model_story",model_story );
    form.append("style_story",style_story);
    form.append("theme_story",theme_story);
    form.append("custom_prompt_story",custom_prompt_story);

    form.append('chinese_response', chinese_response);
    // form.append('english_response', english_response);

    alert('[INFO]Start generating the story, which will take approximately 1 minute!')
    try {
        let response = await fetch('/api/continueWriting', {
            method: 'POST',
            body: form
        });
        if (response.ok) {
            let data = await response.json();
            
            // Differences !
            chinese_response += data.chinese_response;
            // english_response += data.english_response;

            // Update the text box displaying the story
            let textArea = document.getElementById('story-text-box');
            textArea.value = chinese_response; // to display it
        } else {
            alert('Error in continuing the story: ' + response.status);
        }
    } catch (error) {
        alert('An error occurred during Continue Writing.');
    }
}

async function GenerateImg(){
    let form = new FormData();
    let model_image = document.getElementById('model_image').value;
    let style_image = document.getElementById('style_image').value;
    let negative_prompt=document.getElementById('negative_prompt').value;
    let custom_prompt_image=document.getElementById('custom_prompt_image').value;
    form.append("model_image",model_image);
    form.append("style_image",style_image );
    form.append("negative_prompt",negative_prompt );
    form.append("custom_prompt_image",custom_prompt_image);

    form.append("english_response", JSON.stringify(english_response));
    form.append("predictions", JSON.stringify(predictions));

    alert('[INFO]The image is being generated, please wait a few minutes...');
    try {
        let response = await fetch("/api/generateImg", {
            method: 'POST',
            body: form
        });

        if (response.ok) {
            // document.getElementById('generateImg').src = '../static/generateImg/txt2img_result.jpg';
            let data = await response.json();
            document.getElementById('generateImg').src = '../' + data.image_path;
            document.querySelector('.image.is-generateimg > a').href = '../' + data.image_path
            document.getElementById('downloadLink').href ='../' + data.image_path;
        } else {
            // console.log("Generate error: Status code " + response.status);
            alert('[ERROR]Generate error: Status code: ' + response.status);
        }
    } catch (error) {
        // console.error('[ERROR]Error during GenerateImg:', error);
        alert('[ERROR]An error occurred during Generate image.');
    }
}
