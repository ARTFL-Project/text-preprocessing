#!/usr/bin/env python3

import re

FRENCH_WORD_EXCEPTIONS = {
    'aboient', 'adroit', 'ans', 'aperçoient', 'aperçois', 'aperçoit', 'apperçoit", "apperçois", "apperçoient", apitoient', 'assoient',
    'assois', 'assoit', 'atermoient', 'autrefois', 'bois', 'boit', 'boit', 'broient', 'charroient', 'chatoient', 'choient', 'chois',
    'choit', 'choit', 'convoient', 'conçoient', 'conçois', 'conçoit', 'corroient', 'coudoient', 'croient', 'crois', 'croit', 'côtoient',
    'dans', 'decois', 'dois', 'doit', 'droit', 'déchoi', 'déchoient', 'déchois', 'déchoit', 'dénoient', 'déploient', 'dévoient',
    'déçois', 'déçoit', 'emploient', 'endroit', 'entrevoient', 'entrevois', 'entrevoit', 'entrevue', 'envoient', 'escabeau', 'escabeaux',
    'escadre', 'escadres', 'escadrille', 'escadron', 'escalade', 'escalader', 'escalades', 'escale', 'escales', 'escalier', 'escaliers',
    'escalope', 'escamotable', 'escamotage', 'escamotages', 'escamotter', 'escapade', 'escapades', 'escapées', 'escapés', 'escarcelle',
    'escargot', 'escargots', 'escarmouche', 'escarmouches', 'escarpin', 'escarpé', 'escarpée', 'escarre', 'escarres', 'escient', 'esclaffer',
    'esclandre', 'esclavagisme', 'esclavagiste', 'esclavagistes', 'esclave', 'esclaves', 'escompte', 'escompter', 'escorte', 'escorter',
    'escortes', 'escrime', 'escrimer', 'escrimer', 'escrimeur', 'escrimeurs', 'escroc', 'escrocs', 'escroquer', 'escroquerie', 'escroqueries',
    'festoient', 'flamboient', 'fois', 'foudroient', 'fourvoient', 'larmoient', 'lois', 'loit', 'maladroit', 'nettoient', 'noient',
    'noit', 'octroient', 'ondoient', 'perçois', 'perçoit', 'ploient', 'pourvoient', 'pourvois', 'pourvoit', 'prévoient', 'prévois',
    'prévoit', 'recoit', 'redéploient', 'remploient', 'renettoient', 'renvoient', 'renvoient', 'revoient', 'revois', 'revoit', 'reçoient',
    'reçois', 'reçoit', 'rougeoient', 'rudoient', 'réemploient', 'sans', 'soudoient', 'sursoient', 'sursois', 'sursoit', 'toit', 'tournoient',
    'trois', 'tutoient', 'verdict', 'verdoient', 'voient', 'vois', 'voit', 'vouvoient'
}

FRENCH_PATTERNS = [
    (re.compile(r"^esl"), r"el"),
    (re.compile(r"ast\Z"), r"at"),
    (re.compile(r"ust\Z"), r"ut"),
    (re.compile(r"ist\Z"), r"it"),
    (re.compile(r"inst\Z"), r"int"),
    (re.compile(r"osts\Z"), r"ot"),
    (re.compile(r"aincte\Z"), r"ainte"),
    (re.compile(r"ievr\Z"), r"ieur"),
    (re.compile(r"ovr\Z"), r"our"),
    (re.compile(r"cy\Z"), r"ci"),
    (re.compile(r"gy\Z"), r"gi"),
    (re.compile(r"ty\Z"), r"ti"),
    (re.compile(r"sy\Z"), r"si"),
    (re.compile(r"ry\Z"), r"ri"),
    (re.compile(r"ay\Z"), r"ai"),
    (re.compile(r"^croye\b"), r"crois"),
    (re.compile(r"^vraye"), r"vrai"),
    (re.compile(r"^parmy"), r"parmi"),
    (re.compile(r"oy\Z"), r"ai"),
    (re.compile(r"loix\Z"), r"lois"),
    (re.compile(r"^lettrez"), r"lettres"),
    (re.compile(r"^regars"), r"regards"),
    (re.compile(r"^routte"), r"route"),
    (re.compile(r"^faubour"), r"faubourg"),
    (re.compile(r"^quiter"), r"quitter"),
    (re.compile(r"^sergens"), r"sergents"),
    (re.compile(r"dessu\Z"), r"dessus"),
    (re.compile(r"^seulle"), r"seule"),
    (re.compile(r"^absens"), r"absents"),
    (re.compile(r"^petis"), r"petits"),
    (re.compile(r"^suitte"), r"suite"),
    (re.compile(r"grans\Z"), r"grands"),
    (re.compile(r"^effect\Z"), r"effet"),
    (re.compile(r"^hermite"), r"ermite"),
    (re.compile(r"prens\Z"), r"prends"),
    (re.compile(r"temp\Z"), r"temps"),
    (re.compile(r"^espris"), r"esprits"),
    (re.compile(r"^milio"), r"millio"),
    (re.compile(r"^milie"), r"millie"),
    (re.compile(r"^chapp"), r"chap"),
    (re.compile(r"iene\Z"), r"ienne"),
    (re.compile(r"ost\Z"), r"ot"),
    (re.compile(r"eulx\Z"), r"eux"),
    (re.compile(r"est\Z"), r"êt"),
    (re.compile(r"iere\Z"), r"ière"),
    (re.compile(r"iére\Z"), r"ière"),
]

FRENCH_PATTERN_EXCEPTIONS = [
    (re.compile(r"ans\Z"), r"ants"),
    (re.compile(r"ict\Z"), "it"),
    (re.compile(r"^esc"), "ec"),
    (re.compile(r"ois\Z"), "ais"),
    (re.compile(r"oist\Z"), "ait"),
    (re.compile(r"oit\Z"), "ait"),
    (re.compile(r"oient\Z"), "aient"),
]


def french_modernize(word):
    word = word.replace("estre", "être")
    word = word.replace("ostre", "otre")
    word = word.replace("^estat", "état")
    word = word.replace("tost", "tôt")
    word = word.replace("mesme", "même")
    word = word.replace("mesmes", "mêmes")
    word = word.replace("tousjour", "toujour")
    word = word.replace("aysn", "ain")
    word = word.replace("oust", "oût")
    word = word.replace("esf", "ef") # like autresfois : fairly sure
    word = word.replace("cq", "q")
    word = word.replace("sç", "s")
    ## Subtract a c
    word = word.replace("poinct", "point")
    ## Replace u with v
    word = word.replace("ceuo", "cevo")
    word = word.replace("iue", "ive") # fairly sure
    word = word.replace("uiu", "uiv") #like poursuiuit : are there any words in French with the pattern viu ?
    word = word.replace("ouue", "ouve")
    word = word.replace("ouua", "ouva")
    ## Replace y with i
    word = word.replace("quoy", "quoi")
    word = word.replace("suy", "sui")
    word = word.replace("ennuy", "ennui") # Can we assume that the general rule is s/uy\Z/ui/ ?
    word = word.replace("luy", "lui")
    word = word.replace("partys", "partis")
    word = word.replace("nois", "nais") # should check for exceptions
    word = word.replace("oib", "aib") # should check for exceptions
    ## Individual cases
    word = word.replace("agens", "agents")
    word = word.replace("intelligens", "intelligent")
    word = word.replace("droitte", "droite")
    word = word.replace("persone", "personne")
    word = word.replace("maintement", "maintenant")
    word = word.replace("faitte", "faite")
    word = word.replace("trouue", "trouve")
    word = word.replace("tranquile", "tranquille")
    word = word.replace("colomne", "colonne")
    word = word.replace("accens", "accents")
    word = word.replace("horison", "horizon")
    word = word.replace("soufle", "souffle")
    word = word.replace("parolle", "parole")
    word = word.replace("flame", "flamme")
    word = word.replace("suject", "sujet")
    word = word.replace("project", "projet")
    word = word.replace("telz", "tels")
    for pattern, replacement in FRENCH_PATTERNS:
        word = pattern.sub(replacement, word)
    if word not in FRENCH_WORD_EXCEPTIONS:
        for pattern, replacement in FRENCH_PATTERN_EXCEPTIONS:
            word = pattern.sub(replacement, word)
        word = word.replace("euue", "euve")
    return word

def modernize(word, language):
    if language == "french":
        return french_modernize(word)
    else:
        return word