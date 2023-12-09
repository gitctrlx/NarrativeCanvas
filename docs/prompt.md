### Story Generation

#### Story generation style, themes, and customized prompts

##### story style prompt

| Parameter    | Prompt                                                       |
| ------------ | ------------------------------------------------------------ |
| 悬疑         | Create a story with an intricate plot that involves a mystery which the main character tries to solve, revealing unexpected twists. |
| 幻想         | Develop a narrative set in a magical world filled with mythical creatures, enchanted objects, and complex lore. |
| 科幻         | Write a story that explores futuristic concepts, advanced technology, space travel, and potentially, alien life. |
| 浪漫         | Narrate a love story that explores the emotional development of a relationship between two individuals. |
| 恐怖         | Craft a tale that induces fear and suspense, involving elements like ghosts, monsters, or psychological thrills. |
| 历史小说     | Weave a story set in a specific historical period, incorporating real historical figures or events with fictional characters. |
| 冒险         | Tell a story of an epic journey or quest, filled with challenges, obstacles, and discoveries. |
| 喜剧         | Create a humorous narrative that includes witty dialogue, funny situations, or satirical elements. |
| 反乌托邦     | Develop a story set in a dystopian future, exploring themes of oppression, rebellion, and survival. |
| 黑色电影     | Write a story in the style of film noir, featuring a cynical hero, a femme fatale, and a dark, moody atmosphere. |
| 魔幻现实主义 | Narrate a story where magical elements are a natural part of an otherwise mundane world. |
| 惊悚         | Craft a fast-paced, high-stakes story that keeps the reader on the edge of their seat, often involving crime or espionage. |
| 文学小说     | Develop a narrative with a strong emphasis on character development, introspective, and thematic depth. |
| 成长         | Tell a story about a young protagonist's journey towards adulthood, exploring themes of identity, belonging, and transformation. |
| 原超现实主义 | Create a story with surreal, dream-like scenes and nonsensical, illogical sequences that challenge the perception of reality. |
| (无)         | Random style                                                 |

##### story theme prompt

| Parameter  | Prompt                     |
| ---------- | -------------------------- |
| 偶然相遇   | A Chance Encounter         |
| 最后一个梦 | The Last Dream             |
| 勇气的瞬间 | A Moment of Courage        |
| 被遗忘的信 | The Forgotten Letter       |
| 神秘礼物   | A Mysterious Gift          |
| 丢失的钥匙 | The Lost Key               |
| 超自然惊悚 | Supernatural Thriller      |
| 古代神话   | Ancient Mythology          |
| 恋爱故事   | Romance Stories -          |
| 推理侦探   | Mystery Detective          |
| 生存挑战   | Survival Challenge         |
| 深海探秘   | Deep Sea Exploration       |
| 太空探险   | Space Exploration          |
| 超级英雄   | Superheroes                |
| 青春校园   | Coming of Age              |
| 时间旅行   | Time Trave                 |
| 另类历史   | Alternative History        |
| 文化探索   | Environmental Preservation |
| 环境保护   | Cultural Exploration       |
| 赛博朋克   | Cyberpunk                  |
| (无)       | Random theme               |

##### Prompt

- `category`：由图像分类引擎推理得到的一个或者多个物体类别
- `theme_story`：主题

- `style_story`：风格
- `custom_prompt_story`：自定义提示词

```py
f"请根据以下要求撰写一个中文故事，故事中必须包含与“{category}”相关的元素，主题围绕“[{theme_story}]”，并采用“{style_story}”的风格来展开。在故事中，请融入以下自定义内容：“{custom_prompt_story}”。"
```



#### Story Generation Supports Story Continuation Feature

> When clicking on "Continuing", the story continuation feature can be activated. **We support the re-specification of the model, story style, and story theme during the continuation**:
>
> Story continuation design logic: The previously generated story is stored in JavaScript and used as the prompt for the next call. For detailed logic, please refer to the architectural diagram and code logic (Relevant code: `app.py`, `static/sky.js`).

##### Prompt

`theme_story`：主题

`style_story`：风格

`previous_chinese_story`：需要续写的中文故事

`custom_prompt_story`：自定义提示词

```py
f"请用中文续写下面的故事，确保续写部分与“[{theme_story}]”主题相符，并采用“[{style_story}]”风格。续写内容不超过100个字符。请根据所提供的故事内容和图像，巧妙地融合这些元素，创作一个流畅且吸引人的故事延续。以下是您需要继续的故事：" + previous_chinese_story + custom_prompt_story
```



### Generated Image

#### Image Generation Prompt Optimization with LLM

> We utilize LLM to transform the Chinese stories we generate into Stable Diffusion prompts, converting them into English prompts suitable for Stable Diffusion input. This process is **parallelized** with the user's reading wait time, significantly reducing their waiting period.
>
> We have implemented temporal parallel processing in JavaScript: by invoking the `/api/generateStory` endpoint to generate stories, and using the `/api/generateSDPrompt` endpoint for creating SD prompts.

##### Prompt

- `theme_story`：主题

- `style_story`：风格

- `chinese_text`：中文故事

- `custom_prompt_story`：自定义提示词

```py
f"提炼下面故事中的的关键人物和事件，限制字数为10字以内，以此制作30字左右的英文的文生图稳定扩散提示词，体现“{theme_story}”主题和“{style_story}”风格。下面是故事内容：“{chinese_text}”" + custom_prompt_story
```



#### Customization of style and type for image generation prompts.

##### image style prompt

| Parameter    | Prompt                                                       |
| ------------ | ------------------------------------------------------------ |
| 超现实主义   | A dreamlike landscape with floating objects and distorted perspectives, rich in symbolism and vivid colors, surrealistic style. |
| 印象主义     | A landscape at sunset with quick, visible brushstrokes capturing the changing light and color, impressionistic style. |
| 装饰艺术     | A cityscape with geometric shapes, sleek lines, and elegant forms, featuring bold and simple colors, art deco style |
| 赛博朋克     | A futuristic city at night, neon lights, high-tech gadgets amidst urban decay, cyberpunk style |
| 浪漫风情     | A serene landscape with dramatic skies, emphasis on nature's beauty and emotional expression, romantic style. |
| 巴洛克       | An elaborate historical scene with grandeur, rich details, dramatic lighting, and intense emotions, baroque style. |
| 极简主义     | A composition with simple geometric forms, limited color palette, and emphasis on space and simplicity, minimalist style. |
| 浮世绘       | A traditional Japanese scene with flat areas of color and graceful lines, featuring nature or daily life, ukiyo-e style. |
| 波普艺术     | A bold and vibrant artwork featuring popular culture elements, bright colors, and a comic-like feel, pop art style. |
| 蒸汽朋克     | A retro-futuristic scene with steam-powered machinery, Victorian era aesthetics, and industrial elements, steampunk style. |
| 古典优雅     | A refined and sophisticated artwork with classical themes, balanced composition, and elegant lines, classically elegant style. |
| 哥特式       | A dark and mysterious scene with gothic architecture, intricate details, and a moody atmosphere, gothic style. |
| 精致细腻     | An artwork with intricate details, delicate textures, and a focus on fine craftsmanship, delicately detailed style. |
| 原始野性     | A wild and untamed landscape with primal elements, vibrant natural colors, and a sense of rawness, primal and wild style |
| 幻想奇幻     | A magical and whimsical scene with mythical creatures, fantastical landscapes, and an otherworldly feel, fantastical style. |
| 都市现代     | A modern urban setting with contemporary elements, dynamic compositions, and a blend of street and fine art, urban contemporary style. |
| 民俗传统     | An artwork depicting traditional folk themes, with vibrant patterns, handmade quality, and cultural motifs, folk and traditional style. |
| 古典文艺复兴 | A historical scene with balanced composition, realism, and depth, inspired by Renaissance art and themes. - |
| 可爱         | A charming and adorable artwork with soft colors, playful subjects, and a heartwarming feel, cute style. |
| (无)         | Random style                                                 |

##### Prompt

- `theme_story`：主题
- `style_story`：风格

- `english_response`：由中文故事经过LLM优化后的英文Stable Diffusion提示词

- `custom_prompt_image`：自定义提示词

```py
f"In the style of {style_image}, depicting {category}, envision a scene where {english_response}. {custom_prompt_image}"
```

