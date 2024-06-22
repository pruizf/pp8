from openai import OpenAI
client = OpenAI()

all_prompt = """¿Es cómico el poeta siguiente? ¿Por qué?

Llamarla mía y nada todo es uno
aunque naciera en ella y siga a oscuras
fatigando sus tristes espesuras
y ofrendándole un canto inoportuno.
Juré sus fueros en Guernica y Luno
como mandan sus santas escrituras,
y esta tierra feroz, feraz en curas,
me dio un roble, un otero y una muno.
Y una mano –perdón–, mano de hielo,
de nieve no, que crispa y atiranta
yo no sé si el rencor o el desconsuelo.
Y una raza me dio que reza y canta
ante el cántabro mar Cantos de Lelo.
No merecía yo ventura tanta."""

campuzano = """¿Es cómico el poema siguiente? ¿Por qué?

Maestro era de esgrima Campuzano,
de espada y daga diestro a maravilla,
rebanaba narices en Castilla,
y siempre le quedaba el brazo sano.
Quiso pasarse a Indias un verano,
y vino con Montalvo el de Sevilla;
cojo quedó de un pie de la rencilla,
tuerto de un ojo, manco de una mano.
Vínose a recoger a aquesta ermita
con su palo en la mano, y su rosario,
y su ballesta de matar pardales.
Y con su Madalena, que le quita
mil canas, está hecho un San Hilario.
¡Ved cómo nacen bienes de los males!"""


completion = client.chat.completions.create(
  #model="gpt-4-turbo",
  model="gpt-3.5-turbo",
  messages=[
    #{"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": campuzano}
  ]
)

print(completion.choices[0].message)
