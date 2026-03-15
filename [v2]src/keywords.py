"""
LLM-related keyword lists for text and OCR matching.
Reused from V1 with minor additions.
"""

KEYWORDS = [
    # Model names
    "chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "copilot", "llama", "mistral",
    # Core LLM terms
    "llm", "large language model", "ai chat", "chatbot", "ai assistant",
    # Generation features
    "generate text", "text generation", "ai writing", "ai writer", "ai compose",
    "ai draft", "smart reply", "auto-reply", "rewrite", "paraphrase",
    "summar", "ai summary",  # matches summarize, summary, summarization
    # Interaction patterns
    "ask ai", "ai answer", "talk to ai", "chat with ai", "prompt",
    "conversational ai", "ai-powered chat", "ai response",
    # Content creation
    "content generat", "essay generator", "article generator", "story generator",
    "ai copywriting", "ai content",
]

KEYWORD_CATEGORIES = {
    "model_name": ["chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "copilot", "llama", "mistral"],
    "core_llm": ["llm", "large language model", "ai chat", "chatbot", "ai assistant"],
    "generation": ["generate text", "text generation", "ai writing", "ai writer", "ai compose",
                    "ai draft", "smart reply", "auto-reply", "rewrite", "paraphrase", "summar"
                    "ai summary"],
    "interaction": ["ask ai", "ai answer", "talk to ai", "chat with ai", "prompt",
                    "conversational ai", "ai-powered chat", "ai response"],
    "content": ["content generat", "essay generator", "article generator", "story generator",
                "ai copywriting", "ai content"],
}

TOP_CATEGORIES_KEYWORDS = [
    "Education", "Communication", "Business", "Productivity", "Health & Fitness",
    "Tools", "Entertainment", "Lifestyle", "Social", "Finance",
    "Shopping", "Travel & Local", "Medical", "Music & Audio", "Photography",
]