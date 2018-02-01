#!/usr/bin/env python3

import re

FRENCH_WORD_EXCEPTIONS = {
    "dans", "ans", "escalier", "escaliers", "esclave", "esclaves", "escrime", "escorte", "escortes", "escorter", "escapade", "escapades",
    "escamotter", "escarmouche", "escarmouches", "escabeau", "escabeaux", "escadre", "escadres", "escadrille", "escadron",
    "escalade", "escalades", "escalader", "escale", "escales", "escalope", "escamotable", "escamotage", "escamotages", "escarcelle",
    "escargot", "escargots", "escarpé", "escapées", "escapés", "escarpée", "escarpin", "escarre", "escarres", "escient", "esclaffer",
    "esclandre", "esclavagisme", "esclavagiste", "esclavagistes", "escompte", "escompter", "escrimer", "escrimer", "escrimeur",
    "escrimeurs", "escroc", "escrocs", "escroquer", "escroquerie", "escroqueries", "verdict", "reçois", "aperçois", "déçois",
    "conçois", "perçois", "vois", "entrevois", "revois", "prévois", "pourvois", "dois", "assois", "sursois", "chois", "déchois",
    "bois", "crois", "bois", "autrefois", "fois", "lois", "trois", "reçoit", "aperçoit", "déçoit", "conçoit", "perçoit", "voit",
    "entrevoit", "revoit", "prévoit", "pourvoit", "doit", "assoit", "sursoit", "choit", "déchoit", "boit", "croit", "reçoient",
    "aperçoient", "voient", "entrevoient", "revoient", "prévoient", "pourvoient", "assoient", "sursoient", "choient", "déchoient",
    "croient", "envoient", "renvoient", "aboient", "nettoient", "renettoient", "emploient", "noient", "apitoient", "atermoient",
    "broient", "charroient", "chatoient", "convoient", "corroient", "côtoient", "coudoient", "dénoient", "déploient", "dévoient",
    "festoient", "flamboient", "foudroient", "larmoient", "octroient", "ondoient", "ploient", "redéploient", "réemploient",
    "remploient", "renvoient", "rougeoient", "rudoient", "fourvoient", "soudoient", "tournoient", "tutoient", "verdoient",
    "vouvoient", "entrevue"
}


def french_modernize(word):
    word = re.sub(r"estre", r"etre", word)
    word = re.sub(r"ostre", r"otre", word)
    word = re.sub(r"^estat", r"etat", word)
    word = re.sub(r"tost", r"tot", word)
    word = re.sub(r"mesme\Z", r"meme", word)
    word = re.sub(r"mesmes\Z", r"memes", word)
    word = re.sub(r"tousjour", r"toujour", word)
    word = re.sub(r"^esl", r"el", word)
    word = re.sub(r"ast\Z", r"at", word)
    word = re.sub(r"ust\Z", r"ut", word)
    word = re.sub(r"ist\Z", r"it", word)
    word = re.sub(r"inst\Z", r"int", word)
    word = re.sub(r"aysn", r"ain", word)
    word = re.sub(r"oust", r"out", word)
    word = re.sub(r"esf", r"ef", word) # like autresfois : fairly sure
    word = re.sub(r"ost\Z", r"ot", word)
    word = re.sub(r"osts\Z", r"ot", word) #like imposts
    ## Subtract a c
    word = re.sub(r"aincte\Z", r"ainte", word)
    word = re.sub(r"poinct", r"point", word)
    ## Replace u with v
    word = re.sub(r"ceuo", r"cevo", word)
    word = re.sub(r"iue", r"ive", word) # fairly sure
    word = re.sub(r"uiu", r"uiv", word) #like poursuiuit : are there any words in French with the pattern viu ?
    word = re.sub(r"ievr\Z", r"ieur", word)
    word = re.sub(r"ovr\Z", r"our", word)
    word = re.sub(r"ouue", r"ouve", word)
    word = re.sub(r"ouua", r"ouva", word)
    ## Replace y with i
    word = re.sub(r"cy\Z", r"ci", word) # I'm fairly sure about this one (voicy, mercy)
    word = re.sub(r"gy\Z", r"gi", word) #like rougy
    word = re.sub(r"ty\Z", r"ti", word) # like sorty, party
    word = re.sub(r"sy\Z", r"si", word) # like ainsy or aussy
    word = re.sub(r"quoy", r"quoi", word)
    word = re.sub(r"suy", r"sui", word)
    word = re.sub(r"^ennuy", r"ennui", word) # Can we assume that the general rule is s/uy\Z/ui/ ?
    word = re.sub(r"luy", r"lui", word)
    word = re.sub(r"partys", r"partis", word)
    word = re.sub(r"ry\Z", r"ri", word) #like attendry
    word = re.sub(r"ay\Z", r"ai", word)
    word = re.sub(r"^croye\b", r"crois", word)
    word = re.sub(r"^vraye", r"vrai", word)
    word = re.sub(r"^parmy", r"parmi", word)
    ## Replace oi/oy by ai
    word = re.sub(r"oy\Z", r"ai", word)
    word = re.sub(r"nois", r"nais", word) # should check for exceptions
    word = re.sub(r"oib", r"aib", word) # should check for exceptions
    ## Individual cases
    word = re.sub(r"loix\Z", r"lois", word)
    word = re.sub(r"agens", r"agents", word)
    word = re.sub(r"intelligens", r"intelligent", word)
    word = re.sub(r"^lettrez", r"lettres", word)
    word = re.sub(r"^regars", r"regards", word)
    word = re.sub(r"^routte", r"route", word)
    word = re.sub(r"droitte", r"droite", word)
    word = re.sub(r"^faubour", r"faubourg", word) # cannot restrict the rule to bour/bourg because of words like tambour
    word = re.sub(r"^quiter", r"quitter", word)
    word = re.sub(r"^sergens", r"sergents", word)
    word = re.sub(r"^persone", r"personne", word)
    word = re.sub(r"dessu\Z", r"dessus", word)
    word = re.sub(r"^maintement", r"maintenant", word)
    word = re.sub(r"^seulle", r"seule", word)
    word = re.sub(r"^faitte", r"faite", word)
    word = re.sub(r"trouue", r"trouve", word)
    word = re.sub(r"^absens", r"absents", word)
    word = re.sub(r"^petis", r"petits", word)
    word = re.sub(r"^suitte", r"suite", word)
    word = re.sub(r"^tranquile", r"tranquille", word)
    word = re.sub(r"^colomne", r"colonne", word)
    word = re.sub(r"grans\Z", r"grands", word)
    word = re.sub(r"^effect\Z", r"effet", word)
    word = re.sub(r"accens", r"accents", word)
    word = re.sub(r"^hermite", r"ermite", word)
    word = re.sub(r"^horison", r"horizon", word)
    word = re.sub(r"^soufle", r"souffle", word)
    word = re.sub(r"prens", r"prends", word)
    word = re.sub(r"temp\Z", r"temps", word)
    word = re.sub(r"^parolle", r"parole", word)
    word = re.sub(r"flame", r"flamme", word)
    word = re.sub(r"^espris", r"esprits", word)
    word = re.sub(r"suject", r"sujet", word)
    word = re.sub(r"project\Z", r"projet", word)
    ## Random common patterns
    word = re.sub(r"^milio", r"millio", word)
    word = re.sub(r"^milie", r"millie", word)
    word = re.sub(r"^chapp", r"chap", word) # like chappelle, chappeau, chappitre
    word = re.sub(r"iene\Z", r"ienne", word) # like anciene, tiene, siene : but is it reliable ?
    if word not in FRENCH_WORD_EXCEPTIONS:
        word = re.sub(r"ans\Z", r"ants", word) # may be a problem, have to check some more
        word = re.sub(r"ict\Z", "it", word)
        word = re.sub(r"^esc", "ec", word)
        word = re.sub(r"ois\Z", "ais", word)
        word = re.sub(r"oist\Z", "ait", word)
        word = re.sub(r"oient\Z", "aient", word)
        word = re.sub(r"euue", "euve", word)
    return word

def modernize(word, language):
    if language == "french":
        return french_modernize(word)
    else:
        return word