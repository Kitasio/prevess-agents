import asyncio
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage, AIMessage

remote_runnable = RemoteRunnable("http://localhost:8000/openai")

async def chat_with_ai():
    chat_history = []
    while True:
        human = input("Human (Q/q to quit): ")
        if human in {"q", "Q"}:
            print('AI: Bye bye human')
            break

        ai = None
        print("AI: ")
        async for chunk in remote_runnable.astream({"input": human, "chat_history": chat_history}):
            # Agent Action
            if "actions" in chunk:
                for action in chunk["actions"]:
                    print(
                        f"Calling Tool ```{action['tool']}``` with input ```{action['tool_input']}```"
                    )
                    if action["tool"] == "profile_update":
                        _continue = input("Should I update the profile (Y/n)?:\n") or "Y"
                        if _continue.lower() != "y":
                            break
            # Observation
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"Got result: ```{step['observation']}```")
            # Final result
            elif "output" in chunk:
                print(chunk['output'])
                ai = AIMessage(content=chunk['output'])
            else:
                raise ValueError
            print("------")        
        chat_history.extend([HumanMessage(content=human), ai])


async def main():
    await chat_with_ai()

if __name__ == "__main__":
    asyncio.run(main())
