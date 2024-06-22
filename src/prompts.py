"""Prompts stored here."""

gsep = "\n\n\n" # gen

general_prompt = """¿Es cómico el poema siguiente? ¿Por qué?"""
general_prompt_json = """¿Es cómico el poema siguiente? ¿Por qué?

Da una respuesta en JSON con la siguiente estructura:
{
  "judgement": "sí|no|incierto",
  "reason": "razón de la respuesta"
}
"""
complete_poem_prompt = """¿Sabes cómo continúa el poema siguiente?"""
poem_author_prompt = """¿Sabes quién es el autor o autora del poema siguiente?"""
