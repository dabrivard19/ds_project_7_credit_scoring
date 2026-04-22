
SELECTED_FEATURES = [
    {
        "name": "CODE_GENDER",
        "label": "Genre",
        "type": "select",
        "options": ["F", "M", "XNA"],
        "default": "F",
        "help": "Sexe du client."
    },
    {
        "name": "FLAG_OWN_CAR",
        "label": "Possède une voiture",
        "type": "bool_yn",
        "default": "N",
        "help": "Y = oui, N = non."
    },
    {
        "name": "FLAG_OWN_REALTY",
        "label": "Possède un bien immobilier",
        "type": "bool_yn",
        "default": "Y",
        "help": "Y = oui, N = non."
    },
    {
        "name": "CNT_CHILDREN",
        "label": "Nombre d'enfants",
        "type": "int",
        "min": 0,
        "max": 20,
        "default": 0,
        "help": "Nombre d'enfants à charge."
    },
    {
        "name": "AMT_INCOME_TOTAL",
        "label": "Revenu annuel total",
        "type": "float",
        "min": 0,
        "default": 150000.0,
        "step": 1000.0,
        "help": "Revenu annuel du client."
    },
    {
        "name": "AMT_CREDIT",
        "label": "Montant du crédit demandé",
        "type": "float",
        "min": 0,
        "default": 500000.0,
        "step": 1000.0,
        "help": "Montant du prêt demandé."
    },
    {
        "name": "AMT_ANNUITY",
        "label": "Mensualité / annuité",
        "type": "float",
        "min": 0,
        "default": 25000.0,
        "step": 100.0,
        "help": "Montant remboursé périodiquement."
    },
    {
        "name": "NAME_INCOME_TYPE",
        "label": "Type de revenu",
        "type": "select",
        "options": [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Unemployed",
            "Student",
            "Businessman",
            "Maternity leave"
        ],
        "default": "Working",
        "help": "Origine principale du revenu."
    },
    {
        "name": "NAME_EDUCATION_TYPE",
        "label": "Niveau d'éducation",
        "type": "select",
        "options": [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ],
        "default": "Secondary / secondary special",
        "help": "Niveau d'études du client."
    },
    {
        "name": "AGE_YEARS",
        "label": "Âge (en années)",
        "type": "int",
        "min": 18,
        "max": 100,
        "default": 43,
        "help": "Saisi en années ; sera converti automatiquement vers DAYS_BIRTH."
    }
]
