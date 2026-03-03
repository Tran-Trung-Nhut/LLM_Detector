BINARY_PROMPT = (
    "You are given an app store listing. "
    "Based on the screenshot(s) and text, decide whether the app truly integrates a Large Language Model (LLM) feature "
    "such as an AI chat assistant that generates text responses. "
    "Answer with exactly one token: YES or NO.\n\n"
    "{text}\n\n"
    "Answer:"
)

THREE_CLASS_PROMPT = (
    "You are given an app store listing. "
    "Classify the app into exactly one label:\n"
    "LLM = integrates a Large Language Model (chat assistant / text generation).\n"
    "AI = uses AI but not an LLM (e.g., photo enhancement, filters, background removal).\n"
    "NO_LLM = no meaningful AI/LLM feature.\n\n"
    "{text}\n\n"
    "Answer with exactly one label: LLM or AI or NO_LLM.\n"
    "Answer:"
)