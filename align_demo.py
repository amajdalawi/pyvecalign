# align_demo.py (your driver script)
# from huggingface_hub import snapshot_download
# local = snapshot_download(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     local_dir="./models/miniLM-L12", local_dir_use_symlinks=False
# )

# import os
# os.environ["HF_HUB_OFFLINE"] = "1"  # set BEFORE importing SentenceTransformer

# from sentence_transformers import SentenceTransformer
# import torch
from onnx_encoder import OnnxSentenceEncoder


from vecalign import align_in_memory
import logging
logging.getLogger("vecalign").setLevel(logging.INFO)
MODEL_DIR = r".\models\miniLM-L12"
# ---- GPU / CPU setup ----
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_float32_matmul_precision("high")
# print(device)
# # Load a multilingual model
# model_name  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# # model_name = "sentence-transformers/LaBSE"
# model = SentenceTransformer(model_name)
# model = model.to(device).half()   # move to GPU if available
# model.max_seq_length = 256   

# enc = OnnxSentenceEncoder(MODEL_DIR, providers=("CUDAExecutionProvider","CPUExecutionProvider"),
#                           max_seq_length=256, prefer_int8=False)  # use fp32/16 model.onnx on GPU
# GPU (NVIDIA): use the fp32/16 model.onnx; install onnxruntime-gpu
enc = OnnxSentenceEncoder(MODEL_DIR,
                          providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                          max_seq_length=256,
                          prefer_int8=False)
print("Using ONNX:", enc.onnx_path)  # sanity check; should print models\miniLM-L12\onnx\model_qint8_*.onnx or model.onnx
# ---- Your data ----
# src_lines = ["Bonjour le monde.", "Comment allez-vous ?", "Ceci est un test."]
# tgt_lines = ["Hello world.", "How are you?", "This is a test."]

# src_lines = [
#     "Bonjour le monde.",
#     "Comment allez-vous ?",
#     "Je m'appelle Marie.",
#     "Où est la gare ?",
#     "Je voudrais un café.",
#     "Merci beaucoup.",
#     "De rien.",
#     "Quel âge as-tu ?",
#     "J'ai vingt ans.",
#     "Il fait beau aujourd'hui.",
#     "Il pleut.",
#     "À demain.",
#     "Bonne nuit.",
#     "Enchanté de vous rencontrer.",
#     "Pouvez-vous m'aider ?",
#     "Je ne comprends pas.",
#     "Parlez-vous anglais ?",
#     "Oui, un peu.",
#     "Non, pas du tout.",
#     "Quelle heure est-il ?",
#     "Il est trois heures.",
#     "Je suis perdu.",
#     "Tournez à gauche.",
#     "Tournez à droite.",
#     "Allez tout droit.",
#     "Arrêtez-vous ici.",
#     "Combien ça coûte ?",
#     "C’est trop cher.",
#     "Avez-vous une chambre ?",
#     "Pour combien de nuits ?",
#     "Une nuit, s’il vous plaît.",
#     "Où sont les toilettes ?",
#     "Je suis fatigué.",
#     "J’ai faim.",
#     "J’ai soif.",
#     "Je suis malade.",
#     "Appelez un médecin.",
#     "Appelez la police.",
#     "J’adore voyager.",
#     "Je travaille à Paris.",
#     "Je suis étudiant.",
#     "J’aime la musique.",
#     "J’aime lire.",
#     "Quel est ton passe-temps ?",
#     "Mon passe-temps est le sport.",
#     "Bonne chance !",
#     "Félicitations !",
#     "Joyeux anniversaire !",
#     "Bon appétit !",
#     "Au revoir."
# ]
# tgt_lines = [
#     "Hello world.",
#     "How are you?",
#     "My name is Marie.",
#     "Where is the train station?",
#     "I would like a coffee.",
#     "Thank you very much.",
#     "You're welcome.",
#     "How old are you?",
#     "I am twenty years old.",
#     "The weather is nice today.",
#     "It is raining.",
#     "See you tomorrow.",
#     "Good night.",
#     "Nice to meet you.",
#     "Can you help me?",
#     "I don't understand.",
#     "Do you speak English?",
#     "Yes, a little.",
#     "No, not at all.",
#     "What time is it?",
#     "It is three o'clock.",
#     "I am lost.",
#     "Turn left.",
#     "Turn right.",
#     "Go straight ahead.",
#     "Stop here.",
#     "How much does it cost?",
#     "It is too expensive.",
#     "Do you have a room?",
#     "For how many nights?",
#     "One night, please.",
#     "Where is the bathroom?",
#     "I am tired.",
#     "I am hungry.",
#     "I am thirsty.",
#     "I am sick.",
#     "Call a doctor.",
#     "Call the police.",
#     "I love traveling.",
#     "I work in Paris.",
#     "I am a student.",
#     "I like music.",
#     "I like reading.",
#     "What is your hobby?",
#     "My hobby is sports.",
#     "Good luck!",
#     "Congratulations!",
#     "Happy birthday!",
#     "Enjoy your meal!",
#     "Goodbye."
# ]
src_lines = [
    "Je me lève tôt le matin et je vais courir.",
    "J'adore les fruits. Les pommes sont mes préférées.",
    "Nous sommes allés au cinéma hier soir, puis nous avons dîné ensemble.",
    "Il a décidé de quitter son emploi, ce qui a surpris tout le monde.",
    "Elle a pris un taxi.",
    "Ensuite, elle est allée à la gare.",
    "Il faisait très froid hier soir.",
    "J'ai mis un manteau épais, une écharpe et des gants.",
    "C'est une journée spéciale, parce que c'est ton anniversaire.",
    "Il a étudié dur pour l'examen et il a réussi avec une excellente note."
]
tgt_lines = [
    "I wake up early in the morning.",
    "Then I go for a run.",
    "I love fruit.",
    "Apples are my favorite.",
    "We went to the cinema yesterday evening.",
    "After that, we had dinner together.",
    "He decided to quit his job.",
    "This surprised everyone.",
    "She took a taxi to get there, and afterwards she went to the train station.",
    "It was very cold last night, so I wore a thick coat, a scarf, and gloves.",
    "Today is a special day.",
    "It's your birthday.",
    "He studied hard for the exam.",
    "He passed it with an excellent grade."
]

import re 

def normalize_text(s: str) -> str:
    # Strip leading/trailing whitespace
    s = s.strip()
    # Replace non-breaking spaces (U+00A0) with normal spaces
    s = s.replace("\u00A0", " ")
    # Collapse multiple spaces into one
    s = re.sub(r"\s+", " ", s)
    return s

# src_line = """
# Vers cinq heures du matin ou parfois six je me réveille, le besoin est à son comble, c’est le moment le plus douloureux de ma journée. Mon premier geste est de mettre en route la cafetière électrique ; la veille, j’ai rempli le réservoir d’eau et le filtre de café moulu (en général du Malongo, je suis resté assez exigeant sur le café). Je n’allume pas de cigarette avant d’avoir bu une première gorgée ; c’est une contrainte que je m’impose, c’est un succès quotidien qui est devenu ma principale source de fierté (il faut avouer ceci dit que le fonctionnement des cafetières électriques est rapide). Le soulagement que m’apporte la première bouffée est immédiat, d’une violence stupéfiante. La nicotine est une drogue parfaite, une drogue simple et dure, qui n’apporte aucune joie, qui se définit entièrement par le manque, et par la cessation du manque.
# """

# tgt_line = """
# I wake up at about five o’clock in the morning, sometimes six; my need is at its height, it’s the most painful moment in my day. The first thing I do is turn on the electric coffee maker; the previous evening I filled the water container with water and the coffee filter with ground coffee (usually Malongo, I’m still quite particular where coffee is concerned). I don’t smoke a cigarette before taking my first sip, it’s an obligation that I impose upon myself, a daily success that has become my chief source of pride (here I must admit, having said this, that electric coffee makers work quickly). The relief that comes from the first puff is immediate, startlingly violent. Nicotine is a perfect drug, a simple, hard drug that brings no joy, defined entirely by a lack, and by the cessation of that lack.
# """

# src_line = """
# I’m forty-six, my name is Florent-Claude Labrouste and I hate my first name, which I think was inspired by two members of my family that my father and my mother each wished to honour; it’s all the more regrettable that I have nothing else to reproach my parents for, they were excellent parents in every respect, they did their best to arm me with the weapons required in the struggle for life, and if in the end I failed – if my life is ending in sadness and suffering – I can’t hold them responsible, but rather a regrettable sequence of circumstances to which I will return, and which is, in fact, the subject of this book; I have nothing to reproach my parents for apart from the tiny – irritating but tiny – matter of my first name; not only do I find the combination ‘Florent-Claude’ ridiculous, but I find each of its elements disagreeable in itself, in fact I think my first name misses the mark completely. Florent is too gentle, too close to the feminine Florence – in a sense, almost androgynous. It does not correspond in any way to my face, with its energetic features, even brutal when viewed from certain angles, and which has often (by some women in any case) been thought virile – but not at all, really not at all – as the face of a Botticelli queer. As to Claude, let’s not even mention it; it instantly makes me think of the Claudettes, and the terrifying image of a vintage video of Claude François shown on a loop at a party full of old queens comes back to mind as soon as I hear the name Claude.

# It isn’t hard to change your first name, although I don’t mean from a bureaucratic point of view – hardly anything is possible from a bureaucratic point of view. The whole point of bureaucracy is to reduce the possibilities of your life to the greatest possible degree when it doesn’t simply succeed in destroying them; from the bureaucratic point of view, a good citizen is a dead citizen. I am speaking more simply from the point of view of usage: one needs only to present oneself under a new name, and after several months or even just several weeks, everyone gets used to it; it no longer even occurs to people that you might have called yourself by a different first name in the past. In my case the operation would have been even easier since my middle name, Pierre, corresponds perfectly to the image of strength and virility that I wished to convey to the world. But I have done nothing, I have gone on being called by that disgusting first name Florent-Claude, and the best I have had from certain women (Camille and Kate, to be precise, but I’ll come back to that, I’ll come back to that) is that they stick to Florent. From society in general I have had nothing; I have allowed myself to be buffeted by circumstances on this point as on almost everything else, I have demonstrated my inability to take control of my life, the virility that seemed to emanate from my square face with its clear angles and chiselled features is in truth nothing but a decoy, a trick pure and simple – for which, it is true, I was not responsible. God had always disposed of me as he wished but I wasn’t, I really wasn’t; I have only ever been an inconsistent wimp and I’m now forty-six and I’ve never been capable of controlling my own life. In short, it seemed very likely that the second part of my life would be a flabby and painful decline, as the first had been."""

# tgt_line = """
# J’ai quarante-six ans, je m’appelle Florent-Claude Labrouste et je déteste mon prénom, je crois qu’il tient son origine de deux membres de ma famille que mon père et ma mère souhaitaient, chacun de leur côté, honorer ; c’est d’autant plus regrettable que je n’ai par ailleurs rien à reprocher à mes parents, ils furent à tous égards d’excellents parents, ils firent de leur mieux pour me donner les armes nécessaires dans la lutte pour la vie, et si j’ai finalement échoué, si ma vie se termine dans la tristesse et la souffrance, je ne peux pas les en incriminer, mais plutôt un regrettable enchaînement de circonstances sur lequel j’aurai l’occasion de revenir – et qui constitue même, à vrai dire, l’objet de ce livre – je n’ai quoi qu’il en soit rien à reprocher à mes parents mis à part ce minime, ce fâcheux mais minime épisode du prénom, non seulement je trouve la combinaison Florent-Claude ridicule, mais ses éléments en eux-mêmes me déplaisent, en somme je considère mon prénom comme entièrement raté. Florent est trop doux, trop proche du féminin Florence, en un sens presque androgyne. Il ne correspond nullement à mon visage aux traits énergiques, sous certains angles brutaux, qui a souvent (par certaines femmes en tout cas) été considéré comme viril, mais pas du tout, vraiment pas du tout, comme le visage d’une pédale botticellienne. Quant à Claude n’en parlons pas, il me fait instantanément penser aux Claudettes, et l’image d’épouvante d’une vidéo vintage de Claude François repassée en boucle dans une soirée de vieux pédés me revient aussitôt, dès que j’entends prononcer ce prénom de Claude.

# Changer de prénom n’est pas difficile, enfin je ne veux pas dire d’un point de vue administratif, presque rien n’est possible d’un point de vue administratif, l’administration a pour objectif de réduire vos possibilités de vie au maximum quand elle ne parvient pas tout simplement à les détruire, du point de vue de l’administration un bon administré est un administré mort, je parle plus simplement du point de vue de l’usage : il suffit de se présenter sous un prénom nouveau et au bout de quelques mois ou même de quelques semaines tout le monde s’y fait, il ne vient même plus à l’esprit des gens que vous ayez pu, par le passé, vous prénommer différemment. L’opération dans mon cas aurait été d’autant plus simple que mon second prénom, Pierre, correspondait parfaitement à l’image de fermeté et de virilité que j’aurais souhaité communiquer au monde. Mais je n’ai rien fait, j’ai continué à me laisser appeler par ce dégoûtant prénom de Florent-Claude, tout ce que j’ai obtenu de certaines femmes (de Camille et de Kate précisément, mais j’y reviendrai, j’y reviendrai), c’est qu’elles se limitent à Florent, de la société en général je n’ai rien obtenu, sur ce point comme sur presque tous les autres je me suis laissé ballotter par les circonstances, j’ai fait preuve de mon incapacité à reprendre ma vie en main, la virilité qui semblait se dégager de mon visage carré aux arêtes franches, de mes traits burinés n’était en réalité qu’un leurre, une arnaque pure et simple – dont, il est vrai, je n’étais pas responsable, Dieu avait disposé de moi mais je n’étais, je n’étais en réalité, je n’avais jamais été qu’une inconsistante lopette, et j’avais déjà quarante-six ans maintenant, je n’avais jamais été capable de contrôler ma propre vie, bref il paraissait très vraisemblable que la seconde partie de mon existence ne serait, à l’image de la première, qu’un flasque et douloureux effondrement."""


# src_line = """Het verhaal begint in Spanje, in de provincie Almería, precies vijf kilometer ten noorden van El Alquián, op de N340. Het was het begin van de zomer, ongetwijfeld zo rond half juli, ergens tegen het eind van de jaren 2010 – volgens mij was Emmanuel Macron president van Frankrijk. Het was zonnig en extreem heet, zoals altijd in Zuid-Spanje in dat seizoen. Het was het begin van de middag en mijn Mercedes G 350 TD 4x4 stond op de parkeerplaats van het Repsol-benzinestation. Ik had hem net volgegooid met diesel en dronk langzaam een cola zero, leunend tegen de carrosserie, bevangen door toenemende somberheid bij de gedachte dat Yuzu de volgende dag zou aankomen, toen er een Volkswagen Kever voor de bandenpomp tot stilstand kwam.

# Twee meisjes van in de twintig stapten uit, zelfs van veraf kon je zien dat ze verrukkelijk waren, de laatste tijd was ik vergeten hoe verrukkelijk meisjes konden zijn, het gaf me een schok, als een soort overdreven, gekunstelde plotwending. De lucht was zo heet dat hij licht leek te trillen, net als het asfalt van de parkeerplaats, het waren precies de juiste omstandigheden voor een luchtspiegeling. Maar de meisjes waren echt, en ik werd door lichte paniek bevangen toen een van hen op me afkwam. Ze had lang, kastanjeblond haar met een heel lichte slag erin, om haar hoofd droeg ze een dunne leren bandeau met gekleurde geometrische motieven. Haar borsten werden min of meer bedekt door een witte katoenen bandeau, en haar wijde korte rok, eveneens van wit katoen, leek bij het minste zuchtje wind te zullen opwaaien – waarbij moet worden vermeld dat er geen zuchtje wind stond, God is barmhartig en genadig.

# Ze was kalm, glimlachte en leek absoluut niet bang – de angst, laten we duidelijk zijn, bevond zich geheel aan mijn kant. Er stond goedheid en geluk in haar ogen te lezen – ik wist bij de eerste blik dat ze in haar leven alleen maar gelukkige ervaringen met dieren, mannen en zelfs werkgevers had gekend. Waarom kwam ze die zomermiddag op mij af, jong en begeerlijk? Zij en haar vriendin wilden de spanning van hun banden controleren (of nou ja, van de banden van hun auto, ik druk me slecht uit). Dat is een verstandige voorzorgsmaatregel, aanbevolen door de organen voor verkeersveiligheid in bijna alle beschaafde landen en zelfs in sommige andere. Dus dit meisje was niet alleen begeerlijk en goedmoedig, ze was ook verstandig en wijs, mijn bewondering voor haar groeide met de seconde. Mocht ik haar mijn hulp weigeren? Nee, geen sprake van.

# Haar vriendin voldeed meer aan het standaardprofiel van de Spaanse – diepzwart haar, donkerbruine ogen, getinte huid. Met haar look deed ze wat minder aan een bloemenkind denken, of nou ja nog wel tamelijk bloem, maar veel minder kind, ze had iets sletterigs over zich, in haar linkerneusvleugel stak een zilveren ringetje en de bandeau die haar borsten bedekte was bontgekleurd, met een agressief ontwerp vol slogans die je punk of rock kon noemen ik ben het verschil vergeten, laten we voor het gemak maar punkrock zeggen. In tegenstelling tot haar vriendin droeg ze shorts en dat was nog erger, ik weet niet waarom er zulke strakke shorts worden gemaakt, het was onmogelijk om niet te worden gebiologeerd door haar kont. Het was onmogelijk dus ik deed het ook niet, maar vrij snel concentreerde ik me weer op de situatie. Eerst, zo legde ik uit, moest de gewenste bandenspanning worden opgezocht, die afhankelijk was van het betreffende automodel en doorgaans vermeld stond op een klein metalen plaatje dat onderaan op het linker voorportier gesoldeerd zat.

# Het plaatje bevond zich inderdaad op de aangeduide plaats en ik voelde hun waardering voor mijn viriele vaardigheden aanzwellen. Hun auto zat niet erg vol – ze hadden zelfs verrassend weinig bagage, twee lichte tassen die waarschijnlijk wat strings en alledaagse schoonheidsartikelen bevatten – een druk van 2,2 bar was ruim voldoende.

# Restte de oppompoperatie in eigenlijke zin. De druk op de linkervoorband was, zo constateerde ik meteen, maar 1,0 bar. Ik sprak hen vol ernst toe, en zelfs met de lichte strengheid waar mijn leeftijd me recht op gaf: ze hadden er goed aan gedaan zich tot mij te richten, het was hoog tijd, ze verkeerden zonder het te weten in reëel gevaar: onderspanning kon gripverlies of een onvaste koers tot gevolg hebben, wat op den duur bijna zeker tot een ongeluk zou leiden. Ze reageerden emotioneel en naïef, de kastanjeblonde legde een hand op mijn onderarm.

# Het valt niet te ontkennen dat die apparaten strontvervelend zijn om te bedienen, je moet letten op het sissen van het mechanisme en vaak op de tast te werk gaan om het mondstukje op het ventiel te plaatsen, neuken is eerlijk gezegd makkelijker, intuïtiever, ik wist zeker dat ze het daarover met me eens zouden zijn geweest maar ik zag niet hoe ik het onderwerp kon aansnijden, kortom ik deed de linkervoorband en daarna meteen de linkerachterband, ze zaten gehurkt naast me en volgden mijn bewegingen uiterst aandachtig terwijl ze in hun taal het ene na het andere ‘Chulo’ en ‘Claro que si’ tjilpten, daarna gaf ik het stokje aan hen over en gebood hun de andere banden te doen, onder mijn vaderlijk toezicht.

# De brunette, van wie ik wel aanvoelde dat ze impulsiever was, ging direct aan de slag met de rechtervoorband en ik kreeg het erg zwaar toen ze eenmaal op haar hurken zat met haar volmaakt ronde billen strak in haar minishort gegoten, die in de maat bewogen terwijl ze het mondstukje onder controle probeerde te krijgen, de kastanjeblonde voelde volgens mij met me mee, ze sloeg zelfs even een arm om mijn middel, een zusterlijke arm.

# Toen kwam dan eindelijk het moment van de rechterachterband, waarover de kastanjeblonde zich ontfermde. De erotische spanning was minder intens, maar er schoof langzaam een amoureuze spanning overheen, want we wisten alle drie dat het de laatste band was, ze zouden nu geen andere keus meer hebben dan hun reis te hervatten.

# Toch bleven ze nog een paar minuten bij me, met een kluwen van bedankjes en bevallige gebaren, en hun houding was niet puur theoretisch, tenminste dat denk ik nu, een aantal jaren later, wanneer ik me weer eens herinner dat ik in het verleden een liefdesleven heb gehad. Ze vroegen wat mijn nationaliteit was – de Franse, dat heb ik geloof ik nog niet vermeld –, wat me trok in de streek – en met name of ik leuke plekjes kende. In zekere zin, ja, er was een tapasbar, waar je ook overvloedig kon ontbijten, recht tegenover mijn appartementencomplex. Er was ook een nachtclub, iets verder weg, die met enige goede wil als leuk kon worden omschreven. En er was mijn appartement, ze hadden bij mij kunnen logeren, in elk geval voor één nacht, en op dat punt heb ik het gevoel (maar dat is ongetwijfeld gefantaseer achteraf) dat het echt leuk had kunnen zijn. Maar ik zei niets van dat alles, ik hield het bij een synthese en legde uit dat het in de streek goed toeven was (wat klopte) en dat ik me er gelukkig voelde (wat niet klopte, en met Yuzu in aantocht werd het er niet beter op)."""

src_line = """L’histoire commence en Espagne, dans la province d’Almeria, exactement cinq kilomètres au Nord d’Al Alquian, sur la N 340. Nous étions au début de l’été, sans doute vers la mi-juillet, plutôt vers la fin des années 2010 – il me semble qu’Emmanuel Macron était président de la République. Il faisait beau et extrêmement chaud, comme toujours dans le Sud de l’Espagne en cette saison. C’était le début de l’après-midi, et mon 4x4 Mercedes G 350 TD était garé sur le parking de la station Repsol. Je venais de faire le plein de diesel et je buvais lentement un Coca Zéro, appuyé contre la carrosserie, gagné par une morosité croissante à l’idée que Yuzu arriverait le lendemain, lorsqu’une Coccinelle Volkswagen se gara en face de la station de gonflage.

Deux filles dans la vingtaine en sortirent, même de loin on voyait qu’elles étaient ravissantes, ces derniers temps j’avais oublié à quel point les filles pouvaient être ravissantes, ça m’a fait un choc, comme une espèce de coup de théâtre exagéré, factice. L’air était tellement chaud qu’il semblait animé d’une légère vibration, de même que l’asphalte du parking, c’étaient exactement les conditions d’apparition d’un mirage. Les filles étaient réelles pourtant, et je fus saisi par une légère panique lorsque l’une d’elles vint vers moi. Elle avait de longs cheveux châtain clair, très légèrement ondulés, son front était ceint d’un mince bandeau de cuir recouvert de motifs géométriques colorés. Un bandeau de coton blanc recouvrait plus ou moins ses seins, et sa jupe courte, flottante, en coton blanc également, semblait prête à se soulever au moindre souffle d’air – il n’y avait, ceci dit, pas un souffle d’air, Dieu est clément et miséricordieux.

Elle était calme, souriante, et ne semblait pas du tout avoir peur – la peur, disons-le clairement, était de mon côté. Il y avait dans son regard de la bonté et du bonheur – je sus dès le premier regard qu’elle n’avait connu dans sa vie que des expériences heureuses avec les animaux, les hommes, avec les employeurs même. Pourquoi venait-elle à moi, jeune et désirable, en cette après-midi d’été ? Elle et son amie souhaitaient vérifier la pression de gonflage de leurs pneus (enfin des pneus de leur voiture, je m’exprime mal). C’est une mesure prudente, recommandée par les organismes de protection routière dans à peu près tous les pays civilisés, et même dans certains autres. Ainsi, cette jeune fille n’était pas seulement désirable et bonne, elle était également prudente et sage, mon admiration pour elle croissait à chaque seconde. Pouvais-je lui refuser mon aide ? À l’évidence, non.

Sa compagne était plus conforme aux standards attendus de l’Espagnole – cheveux d’un noir profond, yeux d’un brun foncé, peau mate. Son look était un peu moins baba cool, enfin elle semblait une fille assez cool aussi, mais moins baba, avec une petite touche un peu salope, un anneau d’argent perçait sa narine gauche, le bandeau recouvrant ses seins était multicolore, d’un graphisme agressif, traversé de slogans qu’on pouvait qualifier de punk ou de rock j’ai oublié la différence, disons de slogans punk-rock pour simplifier. Contrairement à sa compagne elle portait un short et c’était encore pire, je ne sais pas pourquoi on fabrique des shorts aussi moulants, il était impossible de ne pas être hypnotisé par son cul. C’était impossible, je ne l’ai pas fait, mais je me suis assez vite reconcentré sur la situation. La première chose à rechercher, expliquai-je, était la pression de gonflage souhaitable, compte tenu du modèle automobile considéré : elle figurait en général sur une petite plaque métallique soudée au bas de la portière avant gauche.

La plaque figurait bel et bien à l’endroit indiqué, et je sentis s’enfler leur considération pour mes compétences viriles. Leur voiture n’étant pas très chargée – elles avaient même étonnamment peu de bagages, deux sacs légers qui devaient contenir quelques strings et des produits de beauté usuels – une pression de 2,2 kBars était bien suffisante.

Restait à procéder à l’opération de regonflage proprement dite. La pression du pneu avant gauche, constatai-je d’emblée, n’était que de 1,0 kBar. Je m’adressai à elles avec gravité, voire avec la légère sévérité que m’autorisait mon âge : elles avaient bien fait de s’adresser à moi, il n’était que temps, elles étaient sans le savoir en réel danger : le sous-gonflage pouvait produire des pertes d’adhérence, un flou dans la trajectoire, l’accident à terme était presque certain. Elles réagirent avec émotion et innocence, la châtain posa une main sur mon avant-bras.

Il faut bien reconnaître que ces appareils sont chiants à utiliser, il faut guetter les sifflements du mécanisme et souvent tâtonner avant de positionner l’embout sur la valve, c’est plus facile de baiser en fait, c’est plus intuitif, j’étais sûr qu’elles auraient été d’accord avec moi là-dessus mais je ne voyais pas comment aborder le sujet, bref je fis le pneu avant gauche, puis dans la foulée le pneu arrière gauche, elles étaient accroupies à mes côtés, suivant mes gestes avec une attention extrême, gazouillant dans leur langage des « Chulo » et des « Claro que si », puis je leur passai le relais, leur intimant de s’occuper des autres pneus, sous ma paternelle surveillance.

La brune, plus impulsive je le sentais bien, s’attaqua d’entrée de jeu au pneu avant droit, et là c’est devenu très dur, une fois qu’elle fut agenouillée, ses fesses moulées dans son minishort, d’une rondeur si parfaite, et qui bougeaient à mesure qu’elle cherchait à contrôler l’embout, la châtain je pense compatissait à mon trouble, elle passa même brièvement un bras autour de ma taille, un bras sororal.

Le moment vint, enfin, du pneu arrière droit, dont se chargea la châtain. La tension érotique était moins intense, mais une tension amoureuse s’y superposait doucement, car nous le savions tous les trois c’était le dernier pneu, elles n’auraient d’autre choix, à présent, que de reprendre leur route.

Elles demeurèrent, cependant, avec moi pendant quelques minutes, entrelaçant remerciements et gestes gracieux, et leur attitude n’était pas entièrement théorique, du moins c’est ce que je me dis maintenant, à plusieurs années de distance, lorsqu’il me vient de me remémorer que j’ai eu, par le passé, une vie érotique. Elles m’entreprirent sur ma nationalité – française, je ne crois pas l’avoir mentionné –, sur l’agrément que je trouvais à la région – sur la question de savoir, en particulier, si je connaissais des endroits sympathiques. En un sens, oui, il y avait un bar à tapas, qui servait également de copieux petits déjeuners, juste en face de ma résidence. Il y avait également une boîte de nuit, un peu plus loin, qu’on pouvait en étant large qualifier de sympathique. Il y avait chez moi, j’aurais pu les héberger, au moins une nuit, et là j’ai la sensation (mais je fabule sans doute, avec le recul) que ça aurait pu être vraiment sympathique. Mais je ne dis rien de tout cela, je fis dans la synthèse, leur expliquant en gros que la région était agréable (ce qui était exact) et que je m’y sentais heureux (ce qui était faux, et l’arrivée prochaine de Yuzu n’allait pas arranger les choses)."""

tgt_line = """The story starts in Spain, in the province of Almería, precisely five kilometres north of El Alquián, on the N340. It was early summer, probably about mid-July, some time towards the end of the 2010s – I seem to remember that Emmanuel Macron was President of the Republic. The weather was fine and extremely hot, as it always is in southern Spain at that time of year. It was early afternoon, and my 4x4 Mercedes G350 TD was in the car park of the Repsol service station. I’d just filled up with diesel, and I was slowly drinking a Coke Zero, leaning on the bodywork, prey to a growing sense of gloom at the idea that Yuzu would be arriving the next day, when a Volkswagen Beetle pulled up by the air pump.

Two girls in their twenties got out, and even from a distance you could tell that they were ravishing; lately I’d forgotten how ravishing girls could be, so it came as a shock, like a fake and overdone plot twist. The air was so hot that it seemed to vibrate slightly, so that the tarmac of the car park created the appearance of a mirage. But the girls were real, and I panicked slightly when one of them came towards me. She had long light-chestnut hair, very slightly wavy, and she wore a thin leather band covered with coloured geometrical patterns around her forehead. Her breasts were more or less covered by a white strip of cotton, and her short, floating skirt, also in white cotton, seemed as if it would lift at the slightest gust of wind – having said that, there was no gust of wind; God is merciful and compassionate.

She was calm, and she smiled, and didn’t seem afraid at all – I was the one, let’s be honest, who was afraid. Her expression was one of kindness and happiness – I knew at first sight that in her life she had only had happy experiences, with animals, men, even with employers. Why did she come towards me, young and desirable, that summer afternoon? She and her friend wanted to check the pressure of their tyres (the pressure of the tyres on their car, I’m expressing myself badly). It’s a prudent measure, recommended by roadside assistance organisations in almost all civilised countries, and some others as well. So that girl wasn’t just kind and desirable, she was also prudent and sensible; my admiration for her was growing by the second. Could I refuse her my help? Obviously not.

Her companion was more in line with the standards one expects of a Spanish girl – deep black hair, dark brown eyes, tanned skin. She was a bit less hippie-cool, well, she was certainly cool, but a bit less of a hippie, with a slightly sluttish quality. Her left nostril was pierced by a silver ring, the strip of fabric over her breasts was multicoloured, with very aggressive graphics, run through with slogans that might have been called punk or rock, I’ve forgotten the difference – let’s call them punk-rock slogans for the sake of simplicity. Unlike her companion, she wore shorts, which was even worse; I don’t know why they make shorts so tight, it was impossible not to be hypnotised by her arse. It was impossible, I didn’t do it, but I concentrated again quite quickly on the situation at hand. The first thing to look for, I explained, was the desirable tyre pressure, taking into account the model of the car: it usually appeared on a little metal plate soldered to the bottom of the driver’s seat door.

The plate was indeed in the place I suggested, and I felt their admiration for my manly abilities growing. Since their car wasn’t very full – they even had surprisingly little luggage, only two light bags that must have contained a few thongs and the usual beauty products – a pressure of 2.2 kbars was easily enough.

All that remained was the pumping operation itself. The pressure of the front offside tyre, I observed, was only 1.0 kbars. I spoke to them seriously, indeed with the slight severity afforded to me by my age: they had done the right thing in coming to me, it was only a matter of time and they might have unwittingly put themselves in real danger: under-inflation could lead to a loss of grip, or veering, and in time an accident was almost inevitable. They reacted with naive emotion, and the chestnut-haired girl rested a hand on my forearm.

It must be admitted that those contraptions are tedious to use, you have to check the hiss of the mechanism, and you often have to fiddle about to fit the nozzle over the valve. Fucking’s easier; in fact, it’s more intuitive – I was sure that they would have agreed with me on that point but I couldn’t see how to broach the subject; so in short I did the front offside tyre, then the rear offside tyre, while they were crouching beside me, following my movements with extreme attention, trilling ‘Chulo ’ and ‘Claro que sí ’ in their language; then I passed the task to them, instructing them to attend to the other tyres, under my paternal surveillance.

The darker girl – I sensed she was more impulsive – started off by attacking the front nearside tyre, and it became very hard; once she was kneeling – her bottom swelling in her mini-shorts, so perfectly round, and moving as she tried to control the nozzle – I think the chestnut-haired girl was aware of my unease, and briefly put an arm around my waist, a sisterly arm.

At last the time came for the rear nearside tyre, which the chestnut-haired girl took charge of. The erotic tension this time was less intense, but an amorous tension was gently superimposed upon it, because all three of us knew that it was the last tyre, and now they would have no choice but to continue on their journey.

But they stayed with me for several minutes, twining words of thanks and graceful movements, and their attitude wasn’t entirely theoretical – at least that’s what I tell myself now, a few years on, when I find myself remembering that I did, in the past, have an erotic life. They talked to me about my nationality – French, I don’t think I’d mentioned it – about whether I liked the area, and they particularly wanted to know if I knew of any nice places nearby. In a sense, yes: there was a tapas bar that also served large breakfasts, just opposite my residence. There was also a nightclub a little further off, which one might at a pinch have called nice. There was my place, I would have put them up for at least a night, and I have the feeling (but I may be inventing this in retrospect) that that could have been really nice. But I didn’t say anything about any of that; I gave them a summary, explaining broadly that the region was pleasant (which was true) and that I felt happy there (which was false, and Yuzu’s imminent arrival wasn’t going to make things any better)."""


# src_lines = src_line.strip().split(".")
# tgt_lines = tgt_line.strip().split(".")

src_lines = re.split(r'[.,?!;]|\s-\s',src_line)
# src_lines = re.split(r'',src_line)

tgt_lines= re.split(r'[.,?!;]|\s-\s',tgt_line)


src_lines = [normalize_text(x) for x in src_lines if len(re.findall('\w+',x.strip())) != 0]
tgt_lines = [normalize_text(x) for x in tgt_lines if len(re.findall('\w+',x.strip())) != 0]

# ---- Run alignment ----
alignments, scores = align_in_memory(
    src_lines, tgt_lines,
    model=enc,              # <--- pass model here
    alignment_max_size=8,
    one_to_many=None
)

# ---- Print results ----
for (xs, ys), s in zip(alignments, scores):
    print([src_lines[i] for i in xs], "\n<->\n", [tgt_lines[j] for j in ys], f"(score={s:.3f})")
    print('###')
