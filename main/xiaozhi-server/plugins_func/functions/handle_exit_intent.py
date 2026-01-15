from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

handle_exit_intent_function_desc = {
    "type": "function",
    "function": {
        "name": "handle_exit_intent",
        "description": "Called when the user wants to end the conversation or needs to log out of the system",
        "parameters": {
            "type": "object",
            "properties": {
                "say_goodbye": {
                    "type": "string",
                    "description": "A friendly farewell to end a conversation with the user",
                }
            },
            "required": ["say_goodbye"],
        },
    },
}


@register_function(
    "handle_exit_intent", handle_exit_intent_function_desc, ToolType.SYSTEM_CTL
)
def handle_exit_intent(conn, say_goodbye: str | None = None):
    # 处理退出意图
    try:
        if say_goodbye is None:
            say_goodbye = "Goodbye!"
        conn.close_after_chat = True
        logger.bind(tag=TAG).info(f"Exit intent has been processed: {say_goodbye}")
        return ActionResponse(
            action=Action.RESPONSE, result="Exit intent has been processed", response=say_goodbye
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"Exit intent processing failed: {e}")
        return ActionResponse(
            action=Action.NONE, result="Exit intent processing failed", response=""
        )
