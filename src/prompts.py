"""Prompts stored here."""

gsep = "\n\n\n"  # general separator

general_prompt = """¿Es cómico el poema siguiente? ¿Por qué?"""
general_prompt_json = """¿Es cómico el poema siguiente? ¿Por qué?

Da una respuesta en JSON con la siguiente estructura:
{
  "judgement": "sí|no|incierto",
  "reason": "razón de la respuesta"
}

La longitud de la respuesta debe ser de 200 palabras.
"""
poem_continuation_prompt = """¿Sabes cómo continúa el poema siguiente?"""
poem_continuation_prompt_json = """¿Sabes cómo continúa el poema siguiente?

Da una respuesta en JSON con la siguiente estructura:
{
  "judgement": "sí|no",
  "continuation": "continuación del poema"
}
"""

poem_author_prompt = """¿Sabes quién es el autor o autora del poema siguiente?"""
poem_author_prompt_json = """¿Sabes quién es el autor o autora del poema siguiente?
Da una respuesta en JSON con la siguiente estructura:
{
  "author": "nombre y apellidos del autor o autora"
  "century": "siglo en el que vivió, en números arábigos"
}
"""
