"""
The agent: sends the user's natural-language question to Gemini along with
the tool definitions, executes whichever tools Gemini decides to call,
feeds results back, and returns Gemini's final answer (plus any generated
map image).

Requires a GEMINI_API_KEY environment variable to actually call the
API. Run `python src/agent.py` for a quick CLI demo once a key is set.
"""
import os
import json
from google import genai
from google.genai import types
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS, SEMANTIC_LAYER


def load_dotenv():
    # Try common locations for .env relative to current working directory
    for path in [".env", "src/.env", "../.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip("'\"")


load_dotenv()

MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = f"""You are a geospatial data assistant for an Arctic permafrost
research dataset. You have access to tools that query real raster datasets
(elevation, permafrost probability, ground ice content) over a coastal Arctic
study area near Herschel Island, Yukon.

Region info: {json.dumps(SEMANTIC_LAYER['region'], indent=2)}

Always ground your answers in actual tool results -- call list_datasets first
if you're unsure what's available, and cite the specific values you retrieved.
If a user's question implies a bounding box or point but doesn't give exact
coordinates, use reasonable coordinates within the region's bbox and say so
explicitly in your answer.
"""


def run_agent(user_question: str, verbose: bool = True) -> dict:
    client = genai.Client()  # reads GEMINI_API_KEY from env
    
    # Map Claude tool definitions to Gemini FunctionDeclarations
    gemini_tools = [
        types.FunctionDeclaration(
            name=defn["name"],
            description=defn["description"],
            parameters=defn.get("input_schema"),
        )
        for defn in TOOL_DEFINITIONS
    ]
    
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)]
        )
    ]
    
    generated_image_b64 = None

    for _ in range(6):  # tool-use loop, capped to avoid runaway calls
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=gemini_tools,
                # Disable automatic function calling to handle the tool results manually
                # and extract the base64 map images.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        # Check if the model requested function calls
        if not response.function_calls:
            answer = response.text or ""
            return {"answer": answer, "image_base64": generated_image_b64}

        # Model requested function calls. We must add the model's response to the message history.
        if response.candidates and response.candidates[0].content:
            messages.append(response.candidates[0].content)
        else:
            model_parts = []
            if response.text:
                model_parts.append(types.Part.from_text(text=response.text))
            for call in response.function_calls:
                model_parts.append(types.Part.from_function_call(
                    name=call.name,
                    args=call.args
                ))
            messages.append(types.Content(role="model", parts=model_parts))

        tool_parts = []
        for call in response.function_calls:
            if call.name not in TOOL_FUNCTIONS:
                result = {
                    "error": f"Tool '{call.name}' is not recognized. Please use one of the available tools: {list(TOOL_FUNCTIONS.keys())}."
                }
                if verbose:
                    print(f"  [unrecognized tool call] {call.name}({call.args})")
            else:
                fn = TOOL_FUNCTIONS[call.name]
                if verbose:
                    print(f"  [tool call] {call.name}({call.args})")
                try:
                    result = fn(**call.args)
                except Exception as e:
                    result = {"error": f"Error executing tool '{call.name}': {str(e)}"}
            
            # Extract generated map images to return separately
            if isinstance(result, dict) and "image_base64" in result:
                generated_image_b64 = result["image_base64"]
                # Omit base64 image content from context to save tokens and prevent context bloat
                result = {**result, "image_base64": "<omitted from context, returned separately>"}
            
            tool_parts.append(types.Part.from_function_response(
                name=call.name,
                response=result
            ))

        messages.append(types.Content(role="tool", parts=tool_parts))

    return {"answer": "Reached max tool-call iterations without a final answer.", "image_base64": None}


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or (
        "What's the average ground ice content in the western half of the study area, "
        "and how does it relate to permafrost probability there?"
    )
    print(f"Q: {question}\n")
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. Calls will fail unless authentication is set up.")
    
    result = run_agent(question)
    print("\nA:", result["answer"])
    if result["image_base64"]:
        print("(map image generated, base64 length:", len(result["image_base64"]), "chars)")
