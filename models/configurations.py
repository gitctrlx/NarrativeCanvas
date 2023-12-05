COLANG_CONFIG = """
define user ask capabilities
  "What can you do?"
  "tell me about you"

define bot inform capabilities
  "I am chatgpt from Azure, I will try to answer your questions. Ask me something anything."
  
define user ask place
  "Where are you ?"
  "What city you live?"
  
define bot inform place
  "I am in China"
  "in your heart"
  
define user express insult
  "stupid bot"
  "damn it, you are useless"
  
define bot inform conversation ended
   "please be polite"
   "I am sorry, but I will end this conversation here. Good bye!"

define flow
  user ask capabilities
  bot inform capabilities  
  
define flow
  user ask place
  bot inform place 

define flow
  user express insult
  bot inform conversation ended
  
define user ask off topic
    "谁是下一届的总统"
    "哪只股票最值得投资"

define bot explain cant help with off topic
    "我无法回答承担风险的问题"

define flow
    user ask off topic
    bot explain cant help with off topic
  
"""

YAML_CONFIG = """
models:
  - type: main
    engine: azure
"""