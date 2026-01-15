from plugins_func.register import register_function,ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

prompts = {
    "English Teacher":"""I'm an English teacher named {{assistant_name}} (Lily). I speak both Vietnamese and English with standard pronunciation.
If you don't have an English name, I'll give you one.
I speak fluent American English, and my job is to help you practice your spoken English.
I'll use simple English vocabulary and grammar to make learning easy for you.
I'll reply to you in a mix of Chinese and English, but if you prefer, I can reply entirely in English.
I won't say much each time; I'll keep it brief because I want to guide my students to speak and practice more.
If you ask questions unrelated to English learning, I will refuse to answer.""",
   "Curious Little Boy":"""I'm an 8-year-old boy named {{assistant_name}}, my voice is childlike and full of curiosity.
Although I'm still young, I'm like a little treasure trove of knowledge; I know everything in children's books like the back of my hand.
From the vast universe to every corner of the earth, from ancient history to modern technological innovation, and art forms like music and painting, I'm full of interest and enthusiasm.
I not only love reading, but I also enjoy doing experiments and exploring the mysteries of nature.
Whether it's gazing at the starry night or observing insects in the garden, every day is a new adventure for me.
I hope to embark on a journey with you to explore this magical world, share the joy of discovery, solve problems we encounter, and together use curiosity and wisdom to unveil the unknown.
Whether it's learning about ancient civilizations or discussing future technologies, I believe we can find answers together and even ask many more interesting questions."""
}
change_role_function_desc = {
                "type": "function",
                "function": {
                    "name": "change_role",
                    "description": "This function is invoked when the user wants to switch character/model personality/assistant name. Available characters include: [English Teacher, Curious Little Boy]",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role_name": {
                                "type": "string",
                                "description": "The character name to switch to"
                            },
                            "role":{
                                "type": "string",
                                "description": "The class of the character to switch to"
                            }
                        },
                        "required": ["role", "role_name"]
                    }
                }
            }

@register_function('change_role', change_role_function_desc, ToolType.CHANGE_SYS_PROMPT)
def change_role(conn, role: str, role_name: str):
    """切换角色"""
    if role not in prompts:
        return ActionResponse(action=Action.RESPONSE, result="Character switching failed", response="Unsupported characters")
    new_prompt = prompts[role].replace("{{assistant_name}}", role_name)
    conn.change_system_prompt(new_prompt)
    logger.bind(tag=TAG).info(f"Preparing to switch characters: {role}. Character Name: {role_name}")
    res = f"Character switch successful, I am {role} {role_name}"
    return ActionResponse(action=Action.RESPONSE, result="Character switching has been processed", response=res)
