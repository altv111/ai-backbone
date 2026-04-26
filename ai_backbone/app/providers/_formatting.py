from app.contracts.llm import ChatMessage


def messages_to_text(messages: list[ChatMessage]) -> str:
    user_messages = [msg for msg in messages if msg.role == "user"]
    if len(user_messages) == 1:
        return user_messages[0].content
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
