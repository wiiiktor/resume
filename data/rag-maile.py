import autogen
import os
import json

'''
source of most of the code:
https://colab.research.google.com/drive/1vTpaH2EtnXpmKO81EBDCfTGzbDanpf4S#scrollTo=u9P2ve2OVekW

here short article:
https://medium.com/@andysingal/autogen-enabling-next-generation-llm-applications-3fd5c0d3357
'''

KEY_LOC = ""
OAI_CONFIG_LIST = "OAI_CONFIG_LIST"

config_list = autogen.config_list_from_json(
    env_or_file=OAI_CONFIG_LIST,
    file_location=KEY_LOC,
    filter_dict={
        "model": {
            "gpt-4",
            "gpt4",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-35-turbo",
            "gpt-3.5-turbo",
        }
    },
)

with open(os.path.join('../OAI_CONFIG_LIST'), 'r') as key_file:
    data = json.load(key_file)

config_list = [0] * 3
config_list[0] = {"model": "gpt-4", "api_key": data[0]['api_key']}
config_list[1] = {"model": "gpt-35-turbo", "api_key": data[1]['api_key']}  # nie jestem pewien tego gpt-35-turbo
config_list[2] = {"model": "gpt-3.5-turbo", "api_key": data[1]['api_key']}  # nie jestem pewien tego gpt-3.5-turbo

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb

autogen.ChatCompletion.start_logging()

# based on Example 6

PROMPT_MULTIHOP = """You're a retrieve augmented chatbot. 
You answer user's questions based on your own knowledge 
and the context provided by the user. You must think step-by-step.

First, please learn the following examples of question-answer pairs.

QUESTION: I have a Merkava FatFold 500 folding bike with a BB shell 100mm and 1.37x24 threaded cup (L & R).  
The sensor slips on the shell shaft and lock into the cup teeth. From what I can see, everything looks compatible 
with the GTRO but I just want to be certain.I am also not certain I have much space on the left handle for the shifter, 
maybe the Twist-Grip would be a better fitment.So GTRO for Ebike or for folding?Thank you!

ANSWER: Dear Stephane, yes, it all looks compatible, we even sometimes purchase Bottom Brackets from the same company 
(Gineyea); there should be no problems with PAS; we can send you both trigger and grip shifter, 
but one of them would be used (with just a few minor scratches), so that you can change shifter 
if you have no space on the handle bar; folding process of this bike is not creating any conflict with the gearbox; 
in case of any other questions let me know! 

QUESTION: Hello Efneo. Really interesting product.
1) Are the chainrings on the 28T replaceable? (which BCD)
2) What is the weight of the efneo crankset? (please specify if this weight include BottomBracket)

ANSWER: Dear Steffen, in reply to your questions: 
1) Are the chainrings on the 28T replaceable? (which BCD?)
>> we can offer you a version with 28T chainring OR 130mm BCD adapter, but you need to make a decision in advance; however, if you purchase the 28T version, we can offer you a 37T chainring to be installed on top - please, see this manual: http://efneo.com/filesss/manuals/external-chainring-attachment.pdf  
>> In terms of durability, our 28T chainring is made from chromoly steel 30 HRC, which is much above the standard derailleur chainring, so it simply will never wear in practice (we have clients who covered 15 000 miles, and topic of chainring wearing out did not come up) 
2) What is the weight of the efneo crankset? (please specify if this weight include BottomBracket)
>> the gearbox weighs 1200g (without bottom bracket, left crank and shifter)

QUESTION: Hello again, Can one of your cranksets be used with Alfine 11?
With a Gates belt?
With a T47 BB?
Can it be used with Di2 shifters?
Stan, Yours.

ANSWER: Dear Stan, my answers below: 
Can one of your cranksets be used with Alfine 11? -> Yes
With a Gates belt? -> Yes
With a T47 BB? -> Yes, but you need to use this adapter: 
https://candncycles.co.uk/product/8498068/fsa-t47-to-bsa-68-73-83mm-bottom-bracket-adapter-el316/option/
Can it be used with Di2 shifters? -> Yes

QUESTION: I am trying to install one of your systems to a Strida. The drive side crank doesn't seem 
to be going on far enough. The official instructions say that I need 30.5mm of axle but mine is measuring 28mm. 
Where should I go from here? Taylor.

ANSWER: Dear Taylor, 
28mm is ok, please, see the drawing: 
http://efneo.com/filesss/manuals/BB-68mm-dim.png
(30.5mm is when you measure together with the BB cup flange)
If you have any other questions, let me know! 

Second, please answer the question below by thinking step-by-step.
You can ask for appropriate QUESTION-ANSWER examples as your context, if you consider them helpful. 

Context QUESTION-ANSWER examples: {input_context}
QUESTION: {input_question}
A:
"""

# create an RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "request_timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

corpus_file = "../website/docs/wiiiktor-qa.jsonl"

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,  # tu CHYBA wpisujemy, ile razy ragproxyagent ma sie dopominac o wiecej wiedzy
    retrieve_config={
        "task": "qa",
        "docs_path": corpus_file,
        "chunk_token_size": 2000,
        "model": config_list[2]["model"],  # tutaj wybieramy model
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        # "collection_name": "2wikimultihopqa",  # i use my own collection
        "chunk_mode": "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
        "customized_prompt": PROMPT_MULTIHOP,
        "customized_answer_prefix": "the answer is",
    },
)

queries = """
{"question": "Your crank gear box could be perfect upgrade for a great bike.Chainring is 55 tooth gates belt.There is a sensor on non drives side of bottom bracket.Fat bike with 4in tires can measure bottom bracket to know size.Looking to use gearing and belt on bike and gain a low and high gear. Thank you for any help in making this happen.", "answer": "Is it a PAS / Pedal Assist Sensor? Can you maybe send me a photo? we offer 68mm / 73mm / 80mm / 100mm-wide Bottom Brackets"}
{"question": "Hi, I am interested in purchasing one of your Efneo units to replace my too low gears of my front derailleur on a recumbent trike. The trike and I weigh about 122-130kg. I am not a hard masher rider, currently I never shift out of my lowest 30 tooth chainring. I will be above your weight limit, but since I will not be stressing your gearbox, I am asking if it is okay to install the unit on my trike. P.S. Are you thinking of ever going to even more lower gears say a 26 or 24 tooth? I also have a mountain bike ( I don't ride it often because it hurts my knee and hamstring) with a 22 tooth small chainring, and it is very fun to ride and easy to go up hills. Thanks, Bob.", "answer": "Dear Bob, in reply to your questions: 1) The trike and I weigh about 122-130kg. I am not a hard masher rider, currently I never shift out of my lowest 30 tooth chainring. I will be above your weight limit, but since I will not be stressing your gearbox, I am asking if it is okay to install the unit on my trike -> yes, our gearbox should work fine in a recumbent of ~125 kg 2) Are you thinking of ever going to even more lower gears say a 26 or 24 tooth? -> no, this would require squeezing the mechanism inside of the chainring, so everything would become weaker; moreover, some elements can not be made smaller (the BB axle diameter of 17mm is a barrier that we can not move)"}
"""

queries = [json.loads(line) for line in queries.split("\n") if line]  # \n sprawia problemy
questions = [q["question"] for q in queries]
answers = [q["answer"] for q in queries]
print(questions)
print(answers)

for i in range(len(questions)):
    # we can run this process for a few questions, not just one
    print(f"\n\n>>>>>>>>>>>>  Below are outputs of Case {i+1}  <<<<<<<<<<<<\n\n")

    # always reset the assistant
    assistant.reset()

    qa_problem = questions[i]
    ragproxyagent.initiate_chat(assistant, problem=qa_problem, n_results=5)  # n_results means how many text snippets we take
