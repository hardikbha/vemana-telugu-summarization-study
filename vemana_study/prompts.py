"""One completion format shared by inference and supervised training."""

PROMPT_VERSION = "complaint-summary-v1"
# Hand-written synthetic demonstrations, not reference or training data.
# Do not score these scenarios or close variants as independent held-out cases.
EXAMPLES = (
    {
        "source": "మా ఊరికి ఉదయం బస్సు రావడం లేదు. విద్యార్థులు కళాశాలకు ఆలస్యంగా వెళ్తున్నారు. ఉదయం బస్సు నడపండి.",
        "te": "ఉదయం బస్సు రాకపోవడంతో విద్యార్థులు కళాశాలకు ఆలస్యంగా వెళ్తున్నారు. ఉదయం బస్సు నడపాలని కోరుతున్నారు.",
        "en": "Students are arriving late to college because the morning bus is not running. They request a morning bus service.",
    },
    {
        "source": "గ్రంథాలయంలో రెండు ఫ్యాన్లు పనిచేయడం లేదు. పుస్తకాలు దెబ్బతినలేదు. ఫ్యాన్లను మరమ్మతు చేయండి.",
        "te": "గ్రంథాలయంలో రెండు ఫ్యాన్లు పనిచేయడం లేదు; పుస్తకాలు దెబ్బతినలేదు. ఫ్యాన్ల మరమ్మతు కోరుతున్నారు.",
        "en": "Two library fans are not working, while the books are undamaged. Repairs to the fans are requested.",
    },
    {
        "source": "రైతు సేవా కేంద్రం నిన్న మూసి ఉంది. కారణం తెలియదు. పని వేళలను తెలియజేయండి.",
        "te": "రైతు సేవా కేంద్రం నిన్న మూసి ఉంది; కారణం తెలియదు. పని వేళలను తెలియజేయాలని కోరుతున్నారు.",
        "en": "The farmer service centre was closed yesterday for an unknown reason. Its opening hours are requested.",
    },
)


def build_prompt(source, target_language="en", examples=0):
    if target_language not in {"en", "te"}:
        raise ValueError("target_language must be en or te")
    if examples not in range(4):
        raise ValueError("examples must be between 0 and 3")
    if not source.strip():
        raise ValueError("Source must not be empty")
    if target_language == "te":
        instruction = ("కింది ఫిర్యాదును 1 నుంచి 3 వాక్యాలలో తెలుగులో సంక్షిప్తంగా రాయండి. "
                       "సమస్య, కోరిన చర్య, సంఖ్యలు, నిరాకరణలు నిలుపుకోండి. కొత్త విషయాలు చేర్చవద్దు.")
        source_label, summary_label = "ఫిర్యాదు", "సారాంశం"
    else:
        instruction = ("Summarize the Telugu complaint in 1 to 3 English sentences. "
                       "Preserve the issue, requested action, numbers and negation. Do not add facts.")
        source_label, summary_label = "Telugu complaint", "English summary"
    sections = [instruction]
    for example in EXAMPLES[:examples]:
        sections.append(f"{source_label}:\n{example['source']}\n{summary_label}:\n{example[target_language]}")
    sections.append(f"{source_label}:\n{source}\n{summary_label}:\n")
    return "\n\n".join(sections)
